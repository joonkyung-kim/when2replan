# NOTE: This is an UPDATED setup.py for Python 3.11.
#       The original (Python 3.8) version is kept in setup_original.py.
#
# Changes from the original:
#   - hydra-core   1.2.0  -> 1.3.2   (1.2.0 crashes on import under Python 3.11)
#   - numba        0.56.4 -> 0.57.1  (0.56 supports Python <= 3.10 only)
#   - scikit-image 0.19.3 -> 0.20.0  (no Python 3.11 wheels for 0.19)
#   - protobuf     3.12.0 -> 3.20.3  (no Python 3.11 wheels for 3.12)
#   - removed autorom and gym[atari] (Atari only, not used by this repo)
#   - added pygame and psutil (imported by scripts but missing before)
#   - added stable-baselines3 pinned to the commit used by the README
#
# Install (conda):
#   conda create -n when2replan311 python=3.11 -y && conda activate when2replan311
#   pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40"   # needed to build gym==0.21.0
#   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
#   pip install -e .
#
# numpy must stay on 1.x: gym==0.21.0 does not work with numpy 2.

import setuptools

requirements = [
    "numpy==1.23.5",
    "gym==0.21.0",
    "stable-baselines3 @ git+https://github.com/DLR-RM/stable-baselines3@0532a5719c2bb46fd96b61a7e03dd8cb180c00fc",
    "imitation==0.3.1",
    "protobuf==3.20.3",
    "pyglet==2.0.4",
    "pymap2d==0.1.15",
    "pytest==7.2.1",
    "fire==0.5.0",
    "cython",
    "scikit-image==0.20.0",
    "tensorboard==2.11.2",
    "hydra-core==1.3.2",
    "seaborn==0.11.2",
    "hydra-optuna-sweeper==1.2.0",
    "numba==0.57.1",
    "packaging==22.0",
    "mlflow==2.1.1",
    "opencv-python==4.7.0.68",
    "pygame",
    "psutil",
]

setuptools.setup(
    name="navigation_stack_py",
    version="0.0.0",
    author="Kohei Honda",
    author_email="0905honda@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.11,<3.12",
    platforms=["Linux"],
    install_requires=requirements,
    include_package_data=True,
)
