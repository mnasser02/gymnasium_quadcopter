from setuptools import setup

extras = {
    "mujoco": [],
}

# dependency
all_dependencies = []
for group_name in extras:
    all_dependencies += extras[group_name]
extras["all"] = all_dependencies

setup(
    name="quadcopter",
    version="0.0.1",
    author="Mahdi Nasser",
    author_email="mhnasser23@gmail.com",
    install_requires=[
        "matplotlib",
        "scipy",
        "numpy",
    ],
    packages=[
        "quadcopter",
    ],
    extras_require=extras,
)
