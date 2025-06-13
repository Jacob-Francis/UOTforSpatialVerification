from utils import load_test_fields_bias_scaling, DataCollector, running_results
import torch

# You may be required to install nedCDF4 pandas python package

# -------------------------------------------------------------------
#                     Perturbed/fakeXXX Cases
# -------------------------------------------------------------------

# Set the path to the data
path_to_data = "/home/jacob/PhD_jobs/ICP_Cases/MescoVict_cases/"

# ToDO: I think if a devicce is visible then it is stil using this by default.
if torch.cuda.is_available():
    cuda = 0
else:
    cuda = None  # Default 'cpu' device

for eps in [0.001]:
    for rho in [0.0625, 0.03125, 0.015625]:

        case_list = [
            ("fake000", "fake000"),
            ("fake000", "fake001"),
            ("fake000", "fake002"),
            ("fake000", "fake003"),
            ("fake000", "fake004"),
            ("fake000", "fake005"),
            ("fake000", "fake006"),
            ("fake000", "fake007"),
        ]

        for k, c1, c2 in enumerate(case_list):
            case_list[k] = (c1 + "_g240.txt", c2 + "_g240.txt")

        running_results(
            case_list,
            args_epsilon=eps,
            arg_rho=rho,
            arg_cuda=cuda,
            file_title="scaling_perturbed_fake_cases",
            path_to_data=path_to_data
        )


# -------------------------------------------------------------------
#                     Spring Cases
# -------------------------------------------------------------------

# Set the path to the data
path_to_data = "/home/jjf817/ICP_Cases/Cases/sp2005/"

case_dict = dict(
    tile_2005042500=[
        ("st2/ST2ml_2005042600.g240.txt", "wrf2/wrf2caps_2005042500.g240.f24.txt"),
        ("st2/ST2ml_2005042600.g240.txt", "wrf4ncar/wrf4ncar_2005042500.g240.f24.txt"),
        ("st2/ST2ml_2005042600.g240.txt", "wrf4ncep/wrf4ncep_2005042500.g240.f24.txt"),
        ("st2/ST2ml_2005042600.g240.txt", "st2/ST2ml_2005042600.g240.txt"),
    ],  # 1) These all exist - from figures
    tile_2005051200=[
        ("st2/ST2ml_2005051300.g240.txt", "wrf2/wrf2caps_2005051200.g240.f24.txt"),
        ("st2/ST2ml_2005051300.g240.txt", "wrf4ncar/wrf4ncar_2005051200.g240.f24.txt"),
        ("st2/ST2ml_2005051300.g240.txt", "wrf4ncep/wrf4ncep_2005051200.g240.f24.txt"),
    ],  # 2) These all exist - from figures
    tile_2005051300=[
        ("st2/ST2ml_2005051400.g240.txt", "wrf2/wrf2caps_2005051300.g240.f24.txt"),
        ("st2/ST2ml_2005051400.g240.txt", "wrf4ncar/wrf4ncar_2005051300.g240.f24.txt"),
        ("st2/ST2ml_2005051400.g240.txt", "wrf4ncep/wrf4ncep_2005051300.g240.f24.txt"),
    ],  # 3) These all exist - from figures
    tile_2005051700=[
        ("st2/ST2ml_2005051800.g240.txt", "wrf2/wrf2caps_2005051700.g240.f24.txt"),
        ("st2/ST2ml_2005051800.g240.txt", "wrf4ncar/wrf4ncar_2005051700.g240.f24.txt"),
        ("st2/ST2ml_2005051800.g240.txt", "wrf4ncep/wrf4ncep_2005051700.g240.f24.txt"),
    ],  # 4) These all exist - from figures
    tile_2005051800=[
        ("st2/ST2ml_2005051900.g240.txt", "wrf2/wrf2caps_2005051800.g240.f24.txt"),
        ("st2/ST2ml_2005051900.g240.txt", "wrf4ncar/wrf4ncar_2005051800.g240.f24.txt"),
        ("st2/ST2ml_2005051900.g240.txt", "wrf4ncep/wrf4ncep_2005051800.g240.f24.txt"),
    ],  # 5) These all exist - from figures
    tile_2005052400=[
        ("st2/ST2ml_2005052500.g240.txt", "wrf2/wrf2caps_2005052400.g240.f24.txt"),
        ("st2/ST2ml_2005052500.g240.txt", "wrf4ncar/wrf4ncar_2005052400.g240.f24.txt"),
        ("st2/ST2ml_2005052500.g240.txt", "wrf4ncep/wrf4ncep_2005052400.g240.f24.txt"),
    ],  # 6) These all exist - from figures
    tile_2005053100=[
        ("st2/ST2ml_2005060100.g240.txt", "wrf2/wrf2caps_2005053100.g240.f24.txt"),
        ("st2/ST2ml_2005060100.g240.txt", "wrf4ncar/wrf4ncar_2005053100.g240.f24.txt"),
        ("st2/ST2ml_2005060100.g240.txt", "wrf4ncep/wrf4ncep_2005053100.g240.f24.txt"),
    ],  # 7) These all exist - from figures
    tile_2005060200=[
        ("st2/ST2ml_2005060300.g240.txt", "wrf2/wrf2caps_2005060200.g240.f24.txt"),
        ("st2/ST2ml_2005060300.g240.txt", "wrf4ncar/wrf4ncar_2005060200.g240.f24.txt"),
        ("st2/ST2ml_2005060300.g240.txt", "wrf4ncep/wrf4ncep_2005060200.g240.f24.txt"),
    ],  # 8) These all exist - from figures
    tile_2005060300=[
        ("st2/ST2ml_2005060400.g240.txt", "wrf2/wrf2caps_2005060300.g240.f24.txt"),
        ("st2/ST2ml_2005060400.g240.txt", "wrf4ncar/wrf4ncar_2005060300.g240.f24.txt"),
        ("st2/ST2ml_2005060400.g240.txt", "wrf4ncep/wrf4ncep_2005060300.g240.f24.txt"),
    ],  # 9) These all exist - from figures
)

# case_list, args_epsilon, arg_rho, arg_cuda, file_title
for case in case_dict.keys():
    print(case)
    running_results(
        case_dict[case],
        args_epsilon=0.002,
        arg_rho=1.0,
        arg_cuda=cuda,
        file_title="unit" + case,
        path_to_data=path_to_data
    )
