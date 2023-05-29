from setuptools import setup

setup(
    name="quadcopter",
    version="0.0.1",
    author="Mahdi Nasser",
    author_email="mhnasser23@gmail.com",
    install_requires=[
        "numpy",
        "scipy",
        "mujoco",
    ],
    packages=[
        "quadcopter",
    ],
)
