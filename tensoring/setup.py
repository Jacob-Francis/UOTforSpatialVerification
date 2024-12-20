from setuptools import setup, find_packages

setup(
    name="tensoring",
    version="0.1",
    packages=find_packages(include=["tensorisation"]),
    install_requires=["torch>=1.13.1", "torch_numpy_process"],  
)
