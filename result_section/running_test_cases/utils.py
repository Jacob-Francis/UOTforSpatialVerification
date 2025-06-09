import netCDF4 as nc
import numpy as np
import torch
from time import perf_counter
import os
from unbalancedsinkhorn import DebiasedUOT, ConvergenceWarning
import pandas as pd

# Change directory to the location of this script,
# so that relative paths work correctly and data is saved in neighbouring directory
# 'data_store'
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_test_fields_bias_scaling(fieldx: str, fieldy: str, L=200, dtype=torch.float64, path_to_data='',cases_scale=1873.5):

    # Load data
    X_i = nc.Dataset(
        path_to_data + str(fieldx) + ".nc"
    )

    # For the 10 grid point increment cases we luse C1 rolled over 10 grid points.
    if type(fieldy) == int:
        # Repeat the x field then shift it through 'roll' later
        Y_j = nc.Dataset(
        path_to_data + str(fieldx) + ".nc"
    )
    else:
        Y_j = nc.Dataset(
            path_to_data + str(fieldy) + ".nc"
        )

    # Extract scaled fields
    X_coordinates = torch.stack(
        torch.meshgrid(
            torch.tensor(X_i["x"][:].__array__(), dtype=dtype) / L,
            torch.tensor(X_i["y"][:].__array__(), dtype=dtype) / L,
            indexing="xy",
        ),
        axis=2,
    )
    X_precipitation = X_i["var2d"][:].__array__()
    Y_coordinates = torch.stack(
        torch.meshgrid(
            torch.tensor(Y_j["x"][:].__array__(), dtype=dtype) / L,
            torch.tensor(Y_j["y"][:].__array__(), dtype=dtype) / L,
            indexing="xy",
        ),
        axis=2,
    )
    Y_precipitation = Y_j["var2d"][:].__array__()

    mass_x, mass_y = np.sum(X_precipitation), np.sum(Y_precipitation)

    X_precipitation /= cases_scale
    Y_precipitation /= cases_scale

    if type(fieldy) == int:
        return (
            X_coordinates,
            X_precipitation,
            Y_coordinates,
            np.roll(Y_precipitation, fieldy, axis=1),
            mass_x,
            mass_y,
        )
    else:
        return (
            X_coordinates,
            X_precipitation,
            Y_coordinates,
            Y_precipitation,
            mass_x,
            mass_y,
        )




class DataCollector:
    def __init__(self, args_epsilon, arg_rho, arg_cuda, file_title, path_to_data):
        self.args_epsilon = args_epsilon
        self.arg_rho = arg_rho
        self.arg_cuda = arg_cuda
        self.file_title = file_title
        self.path_to_data = path_to_data

        self.columns = [
            "case_key",
            "aprox_type",
            "Se",
            "p1",
            "p2",
            "p3",
            "p4",
            "d1",
            "d2",
            "d3",
            "primal",
            "dual",
            "loop_timing",
            "forward_mean_mag_p",
            "forward_median_mag_p",
            "forward_mean_dir_p",
            "forward_median_dir_p",
            "forward_mean_mag_se",
            "forward_median_mag_se",
            "forward_mean_dir_se",
            "forward_median_dir_se",
            "inverse_mean_mag_p",
            "inverse_median_mag_p",
            "inverse_mean_dir_p",
            "inverse_median_dir_p",
            "inverse_mean_mag_se",
            "inverse_median_mag_se",
            "inverse_mean_dir_se",
            "inverse_median_dir_se",
            "mass_x",
            "mass_y",
        ]
        self.rows_to_add = []

    def append_info(self, aprox_type, key, s, p, d, sign, time, temp_dict):
        row = {
            "case_key": key,
            "aprox_type": aprox_type,
            "Se": s.cpu().item(),
            "p1": p[0].cpu().item(),
            "p2": p[1].cpu().item(),
            "p3": p[2].cpu().item(),
            "p4": p[3].cpu().item(),
            "d1": d[0].cpu().item(),
            "d2": d[1].cpu().item(),
            "d3": d[2].cpu().item(),
            "primal": sum(p).cpu().item(),
            "dual": sum(d).cpu().item(),
            "loop_timing": time,
        }

        self.rows_to_add.append({**row, **temp_dict})

    def run_case(self, case):
        key = str(case[0]) + str(case[1])
        print(f"Processing case: {key}")

        # Load data and set up processing
        X, alpha, Y, beta, mass_x, mass_y = load_test_fields_bias_scaling(
            case[0], case[1], path_to_data=self.path_to_data
        )

        for aprox_type in ["tv", "kl"]:
            print(f'------------------{aprox_type} --- {key} -----------------')


            try:
                ot_class = self.initialize_uot(X, alpha, Y, beta, aprox_type)

                tic = perf_counter()
                f_update, g_update, isup = ot_class.sinkhorn_algorithm(
                    sinkhorn_steps=50000,
                    tol=1e-14,
                    verbose=False,
                    aprox=aprox_type,
                    epsilon_annealing=True
                )
                toc = perf_counter()

                s, p = ot_class.sinkhorn_divergence(
                    force_type="pykeops", return_type="breakdown"
                )
                d = ot_class.dual_cost(force_type="pykeops")
                sign = -1 if f_update > 1e-13 or g_update > 1e-13 else 1

                temp_dict = self.mag_direction(ot_class)
                temp_dict["mass_x"] = mass_x
                temp_dict["mass_y"] = mass_y

                self.append_info(aprox_type, key, s, p, d, sign, toc - tic, temp_dict)
            except (ConvergenceWarning, RuntimeWarning) as _:
                print('FAILED TO CONVERGE: Using epsilon annealing')
                try:
                    ot_class = self.initialize_uot(X, alpha, Y, beta, aprox_type)
                    ot_class.epsilon *= 10
                    tic = perf_counter()
                    f_update, g_update, isup = ot_class.sinkhorn_algorithm(
                        sinkhorn_steps=100000,
                        tol=1e-5,
                        verbose=False,
                        aprox=aprox_type,
                        convergence_checks=None,
                        epsilon_annealing=True,
                        epsilon_annealing_const=0.995,
                        convergence_or_fail = False
                    )
                    toc = perf_counter()

                    s, p = ot_class.sinkhorn_divergence(
                        force_type="pykeops", return_type="breakdown"
                    )
                    d = ot_class.dual_cost(force_type="pykeops")
                    sign = -1 if f_update > 1e-13 or g_update > 1e-13 else 1

                    temp_dict = self.mag_direction(ot_class)
                    temp_dict["mass_x"] = mass_x
                    temp_dict["mass_y"] = mass_y

                    self.append_info(aprox_type, key, s, p, d, sign, toc - tic, temp_dict)
                except ConvergenceWarning as _:
                    print(f"FAILED TO CONVERGE: ConvergenceWarning failure {key}: {_}")
                except RuntimeWarning as _:
                    print(f"FAILED TO CONVERGE: RuntimeWarning encountered for {key}: {_}")
            
            finally:
                torch.cuda.empty_cache()

    def mag_direction(self, ot_class):

        temp_dict = {}
        V0 = ot_class.barycentre_map_of_points("target").cpu()
        V1 = ot_class.barycentre_map_of_points("source").cpu()
        u0 = ot_class.debias_f.barycentre_map_of_points("target").cpu()
        u1 = ot_class.debias_g.barycentre_map_of_points("source").cpu()

        # -------------------------------- FORWARD -------------------------------- 
        v0 = -V0
        
        # Calculate x and y average direction of the transport vectors.
        C = torch.nanmean(v0[:,0])
        S = torch.nanmean(v0[:,1])

        # Convert for circular mean average then back to degrees
        direction, mag = torch.arctan2(S, C), torch.sqrt(C**2 + S**2)
        temp_dict["forward_mean_mag_p"] = mag.cpu().item()
        temp_dict["forward_mean_dir_p"] = direction.cpu().item()

        # Calculate median average
        mag = v0.norm(dim=1)
        direction = torch.rad2deg(torch.arctan2(v0[:, 1], v0[:, 0]))

        temp_dict["forward_median_mag_p"] = torch.nanmedian(mag).cpu().item()
        temp_dict["forward_median_dir_p"] = torch.nanmedian(direction).cpu().item()

        # Define the debiased vector: v0 = -(V0 - u0)
        v0_bias = -(V0 - u0)

        # Calculate C and S
        C = torch.nanmean(v0[:,0])
        S = torch.nanmean(v0[:,1])

        # Convert for circular mean average then back to degrees
        direction, mag = torch.arctan2(S, C), torch.sqrt(C**2 + S**2)
        temp_dict["forward_mean_mag_se"] = mag.cpu().item()
        temp_dict["forward_mean_dir_se"] = direction.cpu().item()

        mag = v0_bias.norm(dim=1)
        direction = torch.rad2deg(torch.arctan2(v0_bias[:, 1], v0_bias[:, 0]))

        temp_dict["forward_median_mag_se"] = torch.nanmedian(mag).cpu().item()
        temp_dict["forward_median_dir_se"] = torch.nanmedian(direction).cpu().item()

        # -------------------------------- REVERSE --------------------------------
        v0 = -V1

        # Calculate C and S
        C = torch.nanmean(v0[:,0])
        S = torch.nanmean(v0[:,1])

        # Convert for circular mean average then back to degrees
        direction, mag = torch.arctan2(S, C), torch.sqrt(C**2 + S**2)
        temp_dict["inverse_mean_mag_p"] = mag.cpu().item()
        temp_dict["inverse_mean_dir_p"] = direction.cpu().item()

        mag = v0.norm(dim=1)
        direction = torch.rad2deg(torch.arctan2(v0[:, 1], v0[:, 0]))

        temp_dict["inverse_median_mag_p"] = torch.nanmedian(mag).cpu().item()
        temp_dict["inverse_median_dir_p"] = torch.nanmedian(direction).cpu().item()

        # Define the debiased vector:
        v0_debias = -(V1 - u1)

        # Calculate C and S
        C = torch.nanmean(v0[:,0])
        S = torch.nanmean(v0[:,1])

        # Convert for circular mean average then back to degrees
        direction, mag = torch.arctan2(S, C), torch.sqrt(C**2 + S**2)
        temp_dict["inverse_mean_mag_se"] = mag.cpu().item()
        temp_dict["inverse_mean_dir_se"] = direction.cpu().item()

        mag = v0_debias.norm(dim=1)
        direction = torch.rad2deg(torch.arctan2(v0_debias[:, 1], v0_debias[:, 0]))

        temp_dict["inverse_median_mag_se"] = torch.nanmedian(mag).cpu().item()
        temp_dict["inverse_median_dir_se"] = torch.nanmedian(direction).cpu().item()

        return temp_dict

    def initialize_uot(self, X, alpha, Y, beta, aprox_type):
        
        ot_class = DebiasedUOT(set_fail=False, pykeops=True, cuda_device=self.arg_cuda)
        if self.arg_cuda == None:
            ot_class.device = None
        else:
            ot_class.device = "cuda:" + str(self.arg_cuda)
            torch.cuda.set_device(ot_class.device)
        ot_class.parameters(epsilon=self.args_epsilon, rho=self.arg_rho)
        ot_class.densities(X, Y, alpha, beta)
        return ot_class

    def save_results(self, reload=False):

        # Cehck that the data_store directory exists, if not create it
        if not os.path.exists("data_store"):
            os.makedirs("data_store")

        if self.rows_to_add:
            self.results_df = pd.DataFrame(self.rows_to_add, columns=self.columns)

        file_path = f"data_store/debiased_cases_{self.args_epsilon}_rho_{self.arg_rho}_{self.file_title}.csv"

        if reload:
            data = pd.read_csv(file_path)
            self.results_df = pd.concat([data, self.results_df])
            self.results_df.to_csv(file_path, index=False)
        else:
            self.results_df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")


# Usage; case_list, args_epsilon, arg_rho, arg_cuda, file_title
def running_results(case_list, args_epsilon, arg_rho, arg_cuda, file_title, path_to_data, reload=False):
    collector = DataCollector(args_epsilon, arg_rho, arg_cuda, file_title, path_to_data)
    for case in case_list:
        collector.run_case(case)
    collector.save_results(reload)

