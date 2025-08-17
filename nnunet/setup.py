
from setuptools import setup, find_packages



if __name__ == "__main__":
    setup(
        # ... 其他参数 ...
        packages=find_packages(include=["nnunetv2", "nnunetv2.*"]),
        # ... 其他参数 ...
    )
