from utils import running_results, load_test_fields_bias_scaling
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from unbalancedsinkhorn import DebiasedUOT
import pandas as pd
from matplotlib.colors import Normalize
from collections import deque
import pandas as pd
import re

import numpy as np

# Removed to remove seaborn dependence
# import seaborn as sns

from adjustText import adjust_text

import matplotlib.patches as patches


def generate_uot_class(case1, case2, epsilon=0.005, rho=2 ** (-4), aprox="tv"):
    # Initialisation
    ################## binary cass
    PATH_TO_DATA = os.environ.get("PATH_TO_DATA")

    X, alpha, Y, beta, _, _ = load_test_fields_bias_scaling(case1, case2, path_to_data=PATH_TO_DATA)

    alpha = alpha.T
    beta = beta.T

    n1, n2 = alpha.shape

    x, y = np.meshgrid(
        *(
            np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True),
            np.linspace(1 / (2 * n2), n2 / n1 - 1 / (2 * n2), n2, endpoint=True),
        ),
        indexing="ij",
    )
    X = np.stack((x, y), axis=-1)
    Y = np.stack((x, y), axis=-1)

    #  ------------------ kl bias 1 case -------------------------------

    field_testing = DebiasedUOT(set_fail=False, pykeops=True, cuda_device="cuda:0")

    # parameters
    field_testing.parameters(epsilon=epsilon, rho=rho)
    field_testing.densities(X, Y, alpha, beta)

    f_update, g_update, isup = field_testing.sinkhorn_algorithm(
        sinkhorn_steps=5000,
        tol=1e-10,
        verbose=False,
        aprox=aprox,
        epsilon_annealing=False,
    )

    field_testing.sinkhorn_divergence(force_type="pykeops", return_type=None)

    return field_testing, X


def plotting_transport_vectors(
    case1, case2, epsilon=0.005, rho=2 ** (-4), aprox="tv", save_file="test.pdf"
):

    field_testing, X = generate_uot_class(
        case1, case2, epsilon=epsilon, rho=rho, aprox=aprox
    )

    n1, n2, _ = X.shape

    V0 = field_testing.barycentre_map_of_points("target").cpu()
    V1 = field_testing.barycentre_map_of_points("source").cpu()

    u0 = field_testing.debias_f.barycentre_map_of_points("target").cpu()
    u1 = field_testing.debias_g.barycentre_map_of_points("source").cpu()

    fig, axs = plt.subplots(1, 2, figsize=(16, 5), dpi=200)

    plt.rcParams.update({"font.size": 14})

    # First plot
    ax = axs[0]
    ax.set(xlim=(0, n1))
    v0 = (V0 - u0) * n1  # Adjust vector field
    step = 17
    colors = torch.norm(v0[::step, :], dim=1).numpy()
    colors[np.isnan(colors)] = 0
    X_flat = X.reshape(-1, 2) * n1
    index = colors > 0
    quiv = ax.quiver(
        X_flat[::step, 0][index],
        X_flat[::step, 1][index],
        -v0[::step, 0][index],
        -v0[::step, 1][index],
        colors[index],
        cmap="cividis_r",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    fig.colorbar(
        quiv, ax=ax, label="Magnitude"
    )  # Attach colorbar to the correct subplot
    ax.set_facecolor("#f9f6f1")  # Set background color for this subplot
    ax.set_title("Observation to Forecast")
    ax.set(xlim=(0, n1), ylim=(0, n2))

    # Second plot
    ax = axs[1]
    ax.set(xlim=(0, n1))
    v0 = (V1 - u1) * n1  # Adjust vector field
    colors = torch.norm(v0[::step, :], dim=1).numpy()
    colors[np.isnan(colors)] = 0
    X_flat = X.reshape(-1, 2) * n1
    index = colors > 0
    quiv = ax.quiver(
        X_flat[::step, 0][index],
        X_flat[::step, 1][index],
        -v0[::step, 0][index],
        -v0[::step, 1][index],
        colors[index],
        cmap="cividis_r",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    fig.colorbar(
        quiv, ax=ax, label="Magnitude"
    )  # Attach colorbar to the correct subplot
    ax.set_facecolor("#f9f6f1")  # Set background color for this subplot
    ax.set_title("Forecast to Observation")

    ax.set(xlim=(0, n1), ylim=(0, n2))

    # plt.suptitle('C1C1 Transport Vectors', y=0.95)
    plt.tight_layout()
    plt.savefig(save_file)
    # plt.show()


def plotting_2D_histogram(
    case1, case2, epsilon=0.005, rho=2 ** (-4), aprox="tv", save_file="test.pdf"
):

    field_testing, X = generate_uot_class(
        case1, case2, epsilon=epsilon, rho=rho, aprox=aprox
    )

    n1, n2, _ = X.shape[0], X.shape[1]

    V0 = field_testing.barycentre_map_of_points("target").cpu()
    V1 = field_testing.barycentre_map_of_points("source").cpu()

    u0 = field_testing.debias_f.barycentre_map_of_points("target").cpu()
    u1 = field_testing.debias_g.barycentre_map_of_points("source").cpu()

    # Assuming v0 and u0 are already defined tensors
    # Define the first vector: v0 = -V0
    v0 = -(V0 - u0)
    mag1 = v0.norm(dim=1).numpy() * n1
    direction1 = torch.rad2deg(torch.arctan2(v0[:, 1], v0[:, 0])).numpy()

    # Define the second vector: v0 = -(V0 - u0)
    v0_bias = -(V1 - u1)
    mag2 = v0_bias.norm(dim=1).numpy() * n1
    direction2 = torch.rad2deg(torch.arctan2(v0_bias[:, 1], v0_bias[:, 0])).numpy()

    # Define bins (shared for both histograms)
    bins = 50

    # Create a single figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), dpi=200)
    plt.rcParams.update({"font.size": 14})

    # Plot the first histogram (Biased)
    hist1, xedges1, yedges1, img1 = axs[0].hist2d(
        direction1[~np.isnan(direction1)],
        mag1[~np.isnan(mag1)],
        bins=bins,
        cmap="Blues",
    )
    fig.colorbar(img1, ax=axs[0], label="Counts in bin")
    axs[0].set_title("Observation to Forecast")
    axs[0].set_xlabel("Direction (deg)")
    axs[0].set_ylabel("Magnitude")

    # Overlay black points for non-empty bins on the first plot
    xcenters1 = 0.5 * (xedges1[:-1] + xedges1[1:])
    ycenters1 = 0.5 * (yedges1[:-1] + yedges1[1:])
    for i in range(len(xcenters1)):
        for j in range(len(ycenters1)):
            if hist1[i, j] > 0:
                axs[0].plot(
                    xcenters1[i], ycenters1[j], "ko", markersize=0.5
                )  # 'ko' means black circle

    # Plot the second histogram (Debiased)
    hist2, xedges2, yedges2, img2 = axs[1].hist2d(
        direction2[~np.isnan(direction2)],
        mag2[~np.isnan(mag2)],
        bins=bins,
        cmap="Blues",
    )
    fig.colorbar(img2, ax=axs[1], label="Counts in bin")
    axs[1].set_title("Forecast to Observation")
    axs[1].set_xlabel("Direction (deg)")
    axs[1].set_ylabel("Magnitude")

    # Overlay black points for non-empty bins on the second plot
    xcenters2 = 0.5 * (xedges2[:-1] + xedges2[1:])
    ycenters2 = 0.5 * (yedges2[:-1] + yedges2[1:])
    for i in range(len(xcenters2)):
        for j in range(len(ycenters2)):
            if hist2[i, j] > 0:
                axs[1].plot(
                    xcenters2[i], ycenters2[j], "ko", markersize=0.5
                )  # 'ko' means black circle

    # Show the plots
    plt.suptitle(
        "Joint Histogram of transport vectors, case: {} to {}".format(case1, case2)
    )

    plt.tight_layout()
    # plt.show()


def plot_marginals(
    case_from="C1",
    case_to="C3",
    M=1873.5,
    epsilon=0.005,
    rho_exp=-4,
    aprox="tv",
    output_file="output.pdf",
    plot_radius_scale=1.0,
    radius_centre=(120, 100),
):
    # Load data
    PATH_TO_DATA = os.environ.get("PATH_TO_DATA")
    X, alpha, Y, beta, _, _ = load_test_fields_bias_scaling(case_from, case_to,path_to_data=PATH_TO_DATA) 

    alpha = np.transpose(alpha)
    beta = np.transpose(beta)

    n1, n2 = alpha.shape

    # Set up mesh grid
    x, y = np.meshgrid(
        np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True),
        np.linspace(1 / (2 * n2), n2 / n1 - 1 / (2 * n2), n2, endpoint=True),
        indexing="ij",
    )
    X = np.stack((x, y), axis=-1)
    Y = np.stack((x, y), axis=-1)

    # UOT setup
    uot = DebiasedUOT(set_fail=False, pykeops=True, cuda_device="cuda:0")
    uot.parameters(epsilon=epsilon, rho=2**rho_exp)
    uot.densities(X, Y, alpha, beta)
    uot.sinkhorn_algorithm(
        sinkhorn_steps=50000,
        tol=1e-12,
        verbose=False,
        aprox=aprox,
        epsilon_annealing=False,
    )
    uot.sinkhorn_divergence(force_type="pykeops", return_type=None)

    m1 = uot.marginals(1, force_type="pykeops").cpu().numpy()
    print("Marginal 1:", M * m1.max(), M * m1.min())
    norm_alpha = Normalize(vmin=M * np.min(alpha), vmax=M * np.max(alpha))
    norm_beta = Normalize(vmin=M * np.min(m1), vmax=M * np.max(m1))

    fig = plt.figure(figsize=(9, 8), dpi=200)
    plt.rcParams.update({"font.size": 14})
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, 200), ylim=(0, 200))
    ax.set_facecolor("#f9f6f1")

    # Plotting
    ax.scatter(
        200 * X[:, :, 0][alpha > 0],
        200 * X[:, :, 1][alpha > 0],
        c=M * alpha[alpha > 0],
        s=3,
        alpha=0.5,
        cmap="Blues",
        norm=norm_alpha,
    )
    p = ax.scatter(
        200 * X[:, :, 0][m1 > 0],
        200 * X[:, :, 1][m1 > 0],
        c=M * m1[m1 > 0],
        s=2.5,
        alpha=0.4,
        cmap="copper",
        norm=norm_beta,
    )
    plt.colorbar(p, label="Marginal Intensity")

    # Add circles and labels
    center = radius_centre

    radius_scaled = np.sqrt(200**2 * uot.rho.item() * 2) * plot_radius_scale
    ax.add_patch(
        patches.Circle(center, radius_scaled, edgecolor="orange", facecolor="none")
    )
    ax.plot(
        [center[0], center[0] - radius_scaled], [center[1], center[1]], color="orange"
    )
    ax.text(
        center[0] - radius_scaled * 1.1,
        center[1],
        f"Radius = \n {radius_scaled:.3g} (3 s.f)",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    plt.savefig(output_file)
    plt.close()


def timing_average():
    # This function is used to average the timing of the loop
    l = []
    for file in os.listdir(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store"
    ):
        if "debiased_cases_0.001_rho_1_springunittile_" in file:
            data = pd.read_csv(
                f"/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/{file}"
            )
            l.append(np.mean(data.loop_timing))
            print(
                len(data.loop_timing),
                np.mean(data.loop_timing[1:2:]),
                np.mean(data.loop_timing[:2:]),
                np.mean(data.loop_timing),
            )


def loading_data():
    # Load all the CSV files
    data1 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_circles.csv"
    )
    data2 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_ellipse.csv"
    )
    data3 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_pcase.csv"
    )
    data4 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_translation_vs_rho.csv"
    )

    data5 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_relative_scaling_main.csv"
    )

    data6 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_scaling_main.csv"
    )
    # Concatenate the dataframes along rows (axis=0)
    data_combined = pd.concat([data6], ignore_index=True)

    return data_combined


def plot_bartable_plots(case_sets):

    data = loading_data()

    for j, case_list in enumerate(case_sets):
        t0 = deque(case_list)

        cols = len(t0)
        case_count = len(set(t0))

        # Begin figure
        plt.rcParams.update({"font.size": 12})
        fig = plt.figure(
            figsize=(cols * 18 // 4, 5 * ((case_count // cols + 1))), dpi=200
        )

        case_ind = 1  # figure axes index from one
        while len(t0) > 0:
            case = t0.popleft()
            if case in t0:
                pass
            else:
                # Plot this new field
                split_parts = re.split(r"(\d+)", case)
                PATH_TO_DATA = os.environ.get('PATH_TO_DATA')
                X, alpha, Y, beta, a, b = load_test_fields_bias_scaling(
                    split_parts[0] + split_parts[1], split_parts[2] + split_parts[3], path_to_data=PATH_TO_DATA
                )
                ax = fig.add_subplot(3 * (case_count // cols), cols, case_ind)
                # if len(np.unique(alpha)) == 1 and alpha[0, 0] > 0:
                #     # Add fake zero-point for plotting
                #     alpha[0, 0] = 0
                #     print("yepppp")
                # if len(np.unique(alpha)) == 1 and alpha[0, 0] == 0:
                #     # Add fake zero-point for plotting
                #     alpha[0, 0] = 1
                #     print("yepppp")

                # Define the color normalization
                norm_alpha = Normalize(vmin=0, vmax=1 / 1873.5)
                norm_beta = Normalize(vmin=0, vmax=1 / 1873.5)
                # Scatter plot for alpha with normalization
                scatter1 = ax.scatter(
                    X[:, :, 0][alpha > 0],
                    X[:, :, 1][alpha > 0],
                    c=alpha[alpha > 0],
                    s=2,
                    alpha=0.5,
                    cmap="Blues",
                    norm=norm_alpha,
                )

                # Scatter plot for beta with normalization
                scatter2 = ax.scatter(
                    Y[:, :, 0][beta > 0],
                    Y[:, :, 1][beta > 0],
                    c=beta[beta > 0],
                    s=1,
                    alpha=0.3,
                    cmap="copper",
                    norm=norm_beta,
                )

                # ax.set_axis_off()
                ax.set(xlim=[0, 1], ylim=[0, 1])
                ax.set_title(case, y=0.85)
                # ax.set_facecolor((252/255, 250/255, 241/255 ))
                ax.set_facecolor("#f9f6f1")  # Set background color for this subplot
                # for spine in ax.spines.values():
                # spine.set_edgecolor('black')  # Set the color of the box
                # spine.set_linewidth(1)
                ax.set_xticks([])  # Hide the x-axis ticks
                ax.set_yticks([])
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.set_xticklabels([])  # Remove x-axis labels
                ax.set_yticklabels([])  # Remove y-axis labels

                ax.spines["top"].set_visible(False)  # Hide top spine
                ax.spines["right"].set_visible(False)  # Hide right spine
                ax.spines["bottom"].set_visible(False)  # Hide bottom spine
                ax.spines["left"].set_visible(False)

                # Get data for the plot
                if case in data.case_key.values:
                    temp = data.loc[data.case_key == case]
                else:
                    print(f"{case} not found in data.")
                    break
                # print(case, temp.aprox_type.values)
                # assert((temp.aprox_type.values == ['tv', 'kl']).all())

                scale = 200**2
                x = ["KL (p)", "KL (se)", "TV (p)", "TV (se)"]
                try:
                    y = [
                        temp.primal.values[1],
                        temp.Se.values[1],
                        temp.primal.values[0],
                        temp.Se.values[0],
                    ]
                except IndexError:  # KL undefined
                    y = [-0, -0, temp.primal.values[0], temp.Se.values[0]]

                y = np.array(y)
                y[np.abs(y) < 1e-7] = 0

                y *= scale

                # Top subplot for the horizontal bar plot
                ax_bar = fig.add_subplot(
                    3 * (case_count // cols), cols, case_ind + cols
                )
                bar = ax_bar.barh(
                    x, y, color=["darkgrey", "dimgrey", "darkgrey", "dimgrey"]
                )
                ax_bar.set_axis_off()

                # Label bars
                # labeling = [str(f'{xi}={yi:.3g}') for xi, yi in zip(x, y)]
                # kmax = len(max(labeling, key=len))
                # z = [k if len(k) > max(y) / 2 else '' for k in labeling]
                # z = [k if len(k) <= max(y) / 2 else '' for k in labeling]
                pad = -145 if max(y) > 1e3 else -100  # 120 works for close ones
                z = [
                    str(f"{xi}: {yi:.3g}") if yi > max(y) / 2 else ""
                    for xi, yi in zip(x, y)
                ]
                z1 = [
                    str(f"{xi}: {yi:.3g}") if yi <= max(y) / 2 else ""
                    for xi, yi in zip(x, y)
                ]
                ax_bar.bar_label(
                    bar,
                    labels=z,
                    fmt="%g",
                    padding=pad,
                    color="white",
                    fontweight="bold",
                    label_type="edge",
                )
                ax_bar.bar_label(
                    bar,
                    labels=z1,
                    fmt="%g",
                    padding=4,
                    color="black",
                    fontweight="bold",
                    label_type="edge",
                )
                # ax_bar.set_title("Metric Costs")

                # Prepare list information as table data
                try:
                    table_data = [
                        [
                            f"TV: ({np.rad2deg(temp.forward_mean_dir_se.values[0]):.1f}°, {temp.forward_mean_mag_se.values[0]*200:.1f})"
                        ],
                        [
                            rf"TV$^{{-1}}$: ({np.rad2deg(temp.inverse_mean_dir_se.values[0]):.1f}°, {temp.inverse_mean_mag_se.values[0]*200:.1f})"
                        ],
                        [
                            f"KL: ({np.rad2deg(temp.forward_mean_dir_se.values[1]):.1f}°, {temp.forward_mean_mag_se.values[1]*200:.1f})"
                        ],
                        [
                            rf"KL$^{{-1}}$: ({np.rad2deg(temp.inverse_mean_dir_se.values[1]):.1f}°, {temp.inverse_mean_mag_se.values[1]*200:.1f})"
                        ],
                    ]
                except IndexError:  # KL undefined
                    table_data = [
                        [
                            f"TV: ({np.rad2deg(temp.forward_mean_dir_se.values[0]):.1f}°, {temp.forward_mean_mag_se.values[0]*200:.1f})"
                        ],
                        [
                            rf"TV$^{{-1}}$: ({np.rad2deg(temp.inverse_mean_dir_se.values[0]):.1f}°, {temp.inverse_mean_mag_se.values[0]*200:.1f})"
                        ],
                        [f"KL: Undefined"],
                        [rf"KL$^{{-1}}$: Undefined"],
                    ]

                # Bottom subplot for the table (spans the full width)
                ax_table = fig.add_subplot(
                    3 * (case_count // cols), cols, case_ind + 2 * cols
                )
                ax_table.axis("off")

                # Create the table
                table = ax_table.table(
                    cellText=table_data,
                    cellLoc="center",
                    loc="center",
                    cellColours=[["darkgrey"], ["dimgrey"], ["darkgrey"], ["dimgrey"]],
                )

                # Style the table cells for alternating row colors
                for i, key in enumerate(table.get_celld().keys()):
                    cell = table.get_celld()[key]
                    cell.set_edgecolor("w")
                    if key[0] >= 0:  # Data rows
                        if key[0] % 2 == 1:
                            cell.set_text_props(color="white", weight="bold")
                        else:
                            cell.set_text_props(color="white", weight="bold")

                # Adjust table size
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(
                    1.0, 3.0
                )  # Adjust scale to make it fill more of the subplot
                # ax_table.set_title("Mean Average Transport Vector", va='bottom')

                # Next case
                if case_ind % cols == 0:
                    case_ind += cols + 1
                else:
                    case_ind += 1

            os.makedirs("figs", exist_ok=True)

            plt.savefig(f"figs/{''.join(case_list)}.pdf")


def plotting_cost_decomposition_ellipses(
    aprox_type="tv", save_file="tv_ellipse_test_spread_rho.pdf", log_scale=[False, True]
):

    texts = []
    # Define the directory containing data
    data_dir = (
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store"
    )

    # Initialize storage lists
    transport_dict = {"E3E11": [], "E7E3": [], "E7E11": []}
    balance_dict = {"E3E11": [], "E7E3": [], "E7E11": []}
    rho_values = []

    # Loop through files and extract data
    for file in os.listdir(data_dir):
        if "ellipse_disintergration_test" in file:
            rho = float(file.split("_")[4])  # Extract rho value as float
            rho_values.append(rho)

            data = pd.read_csv(f"{data_dir}/{file}")

            for case in ["E3E11", "E7E3", "E7E11"]:
                transport_value = data.loc[
                    (data.case_key == case) & (data.aprox_type == aprox_type)
                ].p1.values
                balance_value = (
                    data.loc[
                        (data.case_key == case) & (data.aprox_type == aprox_type)
                    ].p3.values
                    + data.loc[
                        (data.case_key == case) & (data.aprox_type == aprox_type)
                    ].p4.values
                ) / rho

                transport_dict[case].append(transport_value)
                balance_dict[case].append(balance_value)

    # Sort data by increasing rho values
    sorted_indices = np.argsort(rho_values)
    rho_values = np.array(rho_values)[sorted_indices]

    # The adjust text doesn't seem to work with the log scale,
    # so we need to set the x and y functions accordingly
    x_scale = ""
    y_scale = ""
    xfunc = lambda x: x
    yfunc = lambda x: x

    if log_scale[0]:
        xfunc = lambda x: np.log10(x)
        x_scale = r"(log$_{10}$ scale)"

    if log_scale[1]:
        yfunc = lambda x: np.log10(x)
        y_scale = r"(log$_{10}$ scale)"

    for case in ["E3E11", "E7E3", "E7E11"]:
        transport_dict[case] = yfunc(
            np.abs(np.array(transport_dict[case])[sorted_indices]) * 200**2
        )

        balance_dict[case] = xfunc(
            np.array(balance_dict[case])[sorted_indices] * 200**2
        )

    # Plotting
    plt.figure(figsize=(8, 6), dpi=200)
    
    # Removed to remove seaborn dependence
    # sns.set_style("whitegrid")

    colors = {
        "E3E11": "#0072B2",  # Blue (CUD Safe)
        "E7E3": "#D55E00",  # Orange (CUD Safe, replaces red)
        "E7E11": "#009E73",  # Green (CUD Safe)
    }
    markers = {"E3E11": "o", "E7E3": "s", "E7E11": "^"}

    for case in ["E3E11", "E7E3", "E7E11"]:
        plt.scatter(
            balance_dict[case],
            transport_dict[case],
            label=case,
            color=colors[case],
            marker=markers[case],
        )
        plt.plot(
            balance_dict[case], transport_dict[case], linestyle="--", color=colors[case]
        )

        # Add arrows to indicate increasing rho direction
        # Annotate points with rho values
        for i, rho in enumerate(rho_values):

            txt = plt.text(
                balance_dict[case][i],
                transport_dict[case][i],
                f"{-8+i}",
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="left",
            )  # , color=colors[case])

            texts.append(txt)

    adjust_text(texts)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Marginal Mass Imbalance Term " + x_scale)
    plt.ylabel("Transport Term " + y_scale)
    plt.title(f"Cost Decomposition ({aprox_type.upper()})")
    plt.legend(title="Case", loc="lower left")
    plt.savefig(save_file)
    # plt.show()


def plotting_cost_decomposition_circles(
    aprox_type="tv", save_file="tv_circle_test_spread_rho.pdf", log_scale=[False, True]
):

    # Load all the CSV files
    data1 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_circles.csv"
    )
    data2 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_ellipse.csv"
    )
    data3 = pd.read_csv(
        "/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/rank_cases/data_store/debiased_cases_0.005_rho_1_new_paper_pcase.csv"
    )

    # Concatenate the dataframes along rows (axis=0)
    data_combined = pd.concat([data1, data2, data3], ignore_index=True)

    # Optionally, you can now view or work with the combined data
    print(data_combined.head())

    data = data_combined

    # Get unique case keys
    cases = np.unique(data.case_key)

    # Drop specific values from the cases array
    cases = np.setdiff1d(
        cases, ["P1C1", "P1P1", "P1P2", "P1P3", "P1P4", "P1P5", "P2P5", "P2P6"]
    )

    print(cases)
    cases
    colours = [
        "x",
        "o",
        "x",
        "x",
        "o",
        "o",
        "o",
        "x",
        "x",
        "x",
        "x",
        "x",
        "x",
        "x",
        "x",
        "o",
        "x",
        "o",
        "o",
        "o",
        "x",
        "x",
        "x",
        "o",
        "o",
        "o",
        "x",
        "o",
        "x",
        "o",
        "x",
        "x",
        "o",
        "o",
        "x",
        "x",
        "x",
        "o",
        "o",
        "x",
        "x",
    ]

    # Initialize storage dictionaries
    transport_dict = {k: [] for k in cases}
    balance_dict = {k: [] for k in cases}

    # Extract transport and balance values
    for case in cases:
        transport_value = data.loc[
            (data.case_key == case) & (data.aprox_type == aprox_type)
        ].p1.values
        balance_value = (
            data.loc[
                (data.case_key == case) & (data.aprox_type == aprox_type)
            ].p3.values
            + data.loc[
                (data.case_key == case) & (data.aprox_type == aprox_type)
            ].p4.values
        )

        if transport_value[0] <= 0:
            print(case)
        if balance_value[0] <= 0:
            print(case)
        transport_dict[case].append(transport_value)
        balance_dict[case].append(balance_value)

    # Convert to NumPy arrays
    for case in cases:
        transport_dict[case] = np.log10(
            np.abs(np.array(transport_dict[case]).flatten()) * 200**2
        )
        balance_dict[case] = np.log10(np.array(balance_dict[case]).flatten() * 200**2)

    # Plotting
    plt.figure(figsize=(8, 6), dpi=200)

    #  Removed to remove seaborn dependence
    # sns.set_style("whitegrid")

    texts = []

    for j, case in enumerate(cases):
        x_vals = balance_dict[case]
        y_vals = transport_dict[case]

        if x_vals.size == 0:
            print(x_vals, case)
        # Ensure we are dealing with scalars
        for i in range(len(x_vals)):
            plt.scatter(
                x_vals[i],
                y_vals[i],
                marker=colours[j],
                c="r" if colours[j] == "x" else "k",
                alpha=0.8,
            )

            txt = plt.annotate(f"{case}", (x_vals[i], y_vals[i]), fontsize=10)
            texts.append(txt)  # Store text objects for later adjustment

    adjust_text(texts)
    # Labels and title
    plt.xlabel("Marginal Mass Imbalance Term ")
    plt.ylabel("Transport Term")
    plt.title(
        f"Cost Decomposition ({aprox_type.upper()}, " + r"$\rho=L^2$, loglog scale)"
    )
    # plt.legend(title='Case')
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="r",
            linestyle="None",
            markersize=8,
            label="Balanced Cases",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linestyle="None",
            markersize=8,
            label="Unbalanced Cases",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper center")
    plt.savefig(save_file)
    # plt.show()

