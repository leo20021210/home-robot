# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy<1.24",
    "scipy==1.10.1",
    "hydra-core",
    "yacs",
    "h5py==3.9.0",
    "pybullet",
    "pygifsicle",
    "open3d",
    "numpy-quaternion==2022.4.3",
    "pybind11-global",
    "sophuspy==0.0.8",
    #"trimesh",
    "pin>=2.6.20",
    "torch==2.0.1",
    "torch_cluster==1.6.2",
    "torch_scatter==2.1.2",
    "pillow==9.5.0",  # For Detic compatibility
    "pyqt6"
]

setup(
    name="home-robot",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
