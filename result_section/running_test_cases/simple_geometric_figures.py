from plotting_utils import *

global PATH_TO_DATA

PATH_TO_DATA = "/home/jacob/PhD_jobs/ICP_Cases/MescoVict_cases/"

plotting_transport_vectors("C1", "C3", epsilon=0.005, rho=1, aprox="tv")
plotting_transport_vectors("C6", "C8", epsilon=0.005, rho=2 ** (-4), aprox="tv")
plotting_transport_vectors(
    "E19",
    "E20",
    epsilon=0.005,
    rho=1,
    aprox="kl",
    save_file="e19e20_kl_rho1_transportvectors.pdf",
)

plot_marginals("C1", "C3", aprox="kl", output_file="c1c3_kl_marginal.pdf")
plot_marginals("C1", "C3", aprox="tv", output_file="c1c3_tv_marginal.pdf")

plot_marginals(
    "C1",
    "C6",
    aprox="kl",
    output_file="c1c6_kl_marginal.pdf",
    plot_radius_scale=0.5,
    radius_centre=(100, 100),
)
plot_marginals(
    "C1",
    "C6",
    aprox="tv",
    output_file="c1c6_tv_marginal.pdf",
    plot_radius_scale=0.5,
    radius_centre=(100, 100),
)

plot_marginals(
    "C1",
    "C8",
    aprox="kl",
    output_file="c1c8_kl_marginal.pdf",
    radius_centre=(100, 100),
)
plot_marginals(
    "C1",
    "C8",
    aprox="tv",
    output_file="c1c8_tv_marginal.pdf",
    radius_centre=(110, 80),
)

plot_marginals(
    "C6",
    "C8",
    aprox="kl",
    output_file="c6c8_kl_marginal.pdf",
    radius_centre=(120, 60),
)
plot_marginals(
    "C6",
    "C8",
    aprox="tv",
    output_file="c6c8_tv_marginal.pdf",
    radius_centre=(120, 60),
)

plot_marginals(
    "C13",
    "C14",
    aprox="kl",
    rho_exp=-6,
    output_file="c13c14_kl_marginal.pdf",
    radius_centre=(82, 25),
)
plot_marginals(
    "C13",
    "C14",
    aprox="tv",
    rho_exp=-6,
    output_file="c13c14_tv_marginal.pdf",
    radius_centre=(82, 25),
)

case_sets = [
    ["C6C12", "C13C14"],
    ["E6E14", "E2E10"],
    ["C1C7", "C1C8"],
    ["C1C9", "E19E20"],
    ["E1E4", "E2E4"],
    ["C2C11", "C1C6"],
    ["C6C7", "C6C8"],
    ["S1S2", "S1S3", "H1H2"],
    ["C1C4", "C1N3", "C1N4", "N1N2"],
    ["C3C5", "C1C10", "C3C4"],
    ["E4E8", "E6E16", "E2E17", "E4E12"],
    ["E4E10", "E4E14", "E1E3", "E1E13"],
    ["E5E7", "E2E18", "E2E6", "E1E11"],
    ["E2E16", "E1E14"],
    ["P1C1", "P2P6", "P1P5", "P1P3"],
    ["P2P2", "P2P5", "P1P2", "P1P1"],
    ["P2C1", "P6P7", "P1P4"],
]

plot_bartable_plots(case_sets)

case_sets = [
    ["C2C3", "C2C4", "E1E9", "E2E10"],
    ["C1C2", "C2C3", "C2C4"],
    ["E1E9", "E2E10", "E3E11", "E4E12"],
    ["C1C1", "C2C5", "C1C2", "C1C3"],
    ["E3E11", "E7E3", "E7E11"],
    ["C2C3", "C2C4", "E1E9", "E2E10"],
    ["C1C1", "C2C5", "C1C2", "C1C3"],
    ["E3E11", "E7E3", "E7E11"],
]


plot_bartable_plots(case_sets)

plotting_cost_decomposition_circles(aprox_type="tv", save_file="tv_spreadofcases.pdf")
plotting_cost_decomposition_circles(aprox_type="kl", save_file="kl_spreadofcases.pdf")

plotting_cost_decomposition_ellipses(
    aprox_type="tv", save_file="tv_ellipse_test_spread_rho.pdf"
)
plotting_cost_decomposition_ellipses(
    aprox_type="kl", save_file="kl_ellipse_test_spread_rho.pdf", log_scale=[True, True]
)
