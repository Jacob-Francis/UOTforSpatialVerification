from setuptools import setup, find_packages

setup(
    name="torch_numpy_process",
    version="0.1",
    packages=find_packages(include=["torchnumpyprocess"]),
    install_requires=["torch>=1.13.1"],  # Add dependencies here if needed
)
