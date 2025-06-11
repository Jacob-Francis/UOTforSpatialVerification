from setuptools import setup, find_packages

setup(
    name="unbalanced_ot_metric",
    version="0.1",
    packages=find_packages(include=["unbalancedsinkhorn"]),
    install_requires=[
        "tensoring>=0.1",       # Dependency on the 'tensoring' package
        "torch_numpy_process>=0.1",  # Dependency on the 'torch_numpy_process' package
        "flipflops>=0.1",       # Dependency on the 'flipflops' package
        "scipy>=1.10.1", 
        "geomloss==0.2.6",
        "numpy>=1.24.3",
        "pykeops==2.1.2",
        "pytest==7.2.0"
    ],
)
