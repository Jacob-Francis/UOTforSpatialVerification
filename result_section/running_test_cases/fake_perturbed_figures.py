from fake_plotting_utils import *

import os

os.environ["PATH_TO_DATA"] = "/home/jacob/PhD_jobs/ICP_Cases/Fake/Perturbed/"



plotting_transport_vectors('fake000', 'fake007', epsilon=0.001, rho=2**(-6), aprox='tv', save_file='007_rho2-6_transportvectors.pdf')
plotting_2D_histogram('fake000', 'fake007', epsilon=0.001, rho=2**(-6), aprox='tv', save_file='007_rho2-6_2dhist.pdf')
plotting_2D_histogram('fake000', 'fake007', epsilon=0.001, rho=2**(1), aprox='tv', save_file='007_rho1_2dhist.pdf')
plot_decomposition_fake_cases(aprox_type="tv", save_file="perturbed_tv_spreadofcases.pdf")
plot_decomposition_fake_cases(aprox_type="kl", save_file="perturbed_kl_spreadofcases.pdf")
plot_primal_spread_KL_vs_TV()
plotting_metrics_extended_rho()