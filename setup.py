from setuptools import setup, find_packages

setup(
    name="compositional_rl_synth_data",
    version="0.1.0",
    packages=find_packages(
        include=[
            "CompoSuite*",
            "CORL*",
            "diffusion*",
            "offline_compositional_rl_datasets*",
        ]
    )
)
