import pathlib
import setuptools


here = pathlib.Path(__file__).resolve().parent

with open(here / 'RMC/README.md', 'r') as f:
    readme = f.read()
install_requirements = [
    "numpy==1.24.3",
    "six",
    "paramz==0.9.5",
    "cython>=0.29",
    "GPy==1.10.0",
    "scikit-learn==1.3.0",
    "scipy==1.11.4",
    "matplotlib==3.7.1",
    "python==3.10.12"
]

setuptools.setup(name='RMCgp',
                 version='0.0.30',
                 author='Thiha Aung',
                 author_email='taung@ucsb.edu',
                 maintainer='Thiha Aung',
                 maintainer_email='taung@ucsb.edu',
                 description='RMC for stochastic control.',
                 long_description=readme,
                 url='https://github.com/thihaa2019/RMCgp',
                 license='Apache-2.0',
                 zip_safe=False,
                 python_requires='>=3.9',
                 install_requires=install_requirements,
                packages=[
                    "RMC",
                    "RMC.costfunctions",
                    "RMC.design",
                    "RMC.emulator",  
                    "RMC.model",
                    "RMC.optimization",
                    "RMC.simulate"
                ],    
                extras_require={
                "dev": ["pytest>=7.0", "twine>=4.0.2"],
                },
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: Apache Software License"])
