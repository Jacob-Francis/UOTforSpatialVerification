from utils import load_test_fields_bias_scaling, DataCollector, running_results
import torch

# You may be required to install nedCDF4 pandas python package

# Set epsilon and rho
# recall epsilon is set by he grid mesh size, not the user.
# Whilst rho is set by the user.
eps = 0.005
rho = 1

# Set the path to the data
path_to_data = '/home/jacob/PhD_jobs/ICP_Cases/MescoVict_cases/'

# ToDO: I think if a devicce is visible then it is stil using this by default.
if torch.cuda.is_available():
    cuda = 0
else:
    cuda = None # Default 'cpu' device

# Load the test fields
case_list = [
    ("C1", "N3"),
    ("C1", "C2"),
    ("C1", "C3"),
    ("C1", "C5"),
    ("C2", "C3"),
    ("C2", "C5"),
    ("C3", "C5"),
    ("C2", "C2"),
    ("C3", "C3"),
    ("C5", "C5"),
    ("C1", "C6"),
    ("C1", "C7"),
    ("C1", "C8"),
    ("C6", "C7"),
    ("C6", "C8"),
    ("E3", "E11"),
    ("E3", "E7"),
    ("E7", "E11"),
    ("C1", "C4"),
    ("C2", "C4"),
    ("C1", "C1"),
    ("C6", "C12"),
    ("C13", "C14"),
    ("C2", "C11"),
    ("C1", "C9"),
    ("C1", "N4"),
    ("E1", "E9"),
    ("E2", "E10"),
    ("E4", "E12"),
    ("E6", "E14"),
    ("E1", "E4"),
    ("E2", "E4"),
    ("E7", "E3"),
    ("E19", "E20"),
    ("H1", "H2"),
    ("N1", "N2"),
    ("S1", "S2"),
    ("S1", "S3"),
    ("P2", "C1"),
    ("P2", "P5"),
    ("P2", "P6"),
    ("P3", "P4"),
    ("P6", "P7"),
    ("C1", 10),
    ("C1", 20),
    ("C1", 30),
    ("C1", 40),
    ("C1", 50),
    ("C1", 60),
    ("C1", 70),
    ("C1", 80)
]

running_results(
    case_list,
    args_epsilon=eps,
    arg_rho=1,
    arg_cuda=None,
    file_title="case_list",
    path_to_data=path_to_data
)
        
