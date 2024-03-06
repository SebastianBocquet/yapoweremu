import setuptools
import os

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('yapoweremu/VERSION', 'r') as version_file:
    version = version_file.read().strip()

setuptools.setup(
    name="yapoweremu",
    version=version,
    author="Sebastian Bocquet",
    author_email="sebastian.bocquet@gmail.com",
    description="yet another emulator for the linear matter power spectrum",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/SebastianBocquet/yapoweremu",
    packages=['yapoweremu'],
    package_data = {'yapoweremu': ['VERSION', '*.npy', '*.pt', '*.txt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ['numpy', 'torch'],
)
