from setuptools import find_namespace_packages, setup
import os.path as osp

with open('%s/%s' % (osp.dirname(osp.realpath(__file__)), 'requirements.txt')) as f:
    requirements = [line.strip() for line in f]

setup(
    name="prl_baselines",
    author="hellovertex",
    author_email="hellovertex@outlook.com",
    description="Baseline Agents - Reinforcement Learning training partners.",
    license='MIT',
    url="https://github.com/hellovertex/prl_baselines",
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages=find_namespace_packages(),
)