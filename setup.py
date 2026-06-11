from setuptools import setup, find_packages

# NOTE: runtime dependencies are installed by scripts/setup_env.sh (Blackwell-
# compatible: torch cu128, transformers 4.28, etc.). install_requires is left
# empty on purpose so `pip install -e .` only registers the `hawk` package on the
# path and does NOT try to (re)install the pinned, CUDA-incompatible versions in
# requirements.txt.
setup(
    name='hawk',
    version='0.1.0',
    python_requires='>=3.8.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
)
