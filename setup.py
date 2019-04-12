from setuptools import setup, find_packages

version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="scout",
    version=version,
    description=
    "Multiscale hyperdimensional phentypic analysis of organoids",
    long_description=long_description,
    install_requires=[
        "matplotlib",
        "scipy",
        "scikit-image",
        "zarr",
        "numcodecs",
        "numpy",
        "h5py",
        "tqdm",
        "scikit-learn",
        "pandas",
        "tifffile",
        "lapsolver",
        "pytorch",
    ],
    author="Kwanghun Chung Lab",
    packages=["scout",
              ],
    entry_points={'console_scripts': [
    ]},
    url="https://github.com/chunglabmit/scout",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.6',
    ]
)
