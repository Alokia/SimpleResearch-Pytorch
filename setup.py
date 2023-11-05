from setuptools import find_packages, setup

setup(
    name="SimpleResearch",
    version="0.0.1",
    author="zhao di",
    author_email="zhaodi0817@163.com",
    description=(
        "a pytorch implementation of some models, help your research"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "Attention"
        "Convolutional"
        "Backbone"
        "Deep Learning"
    ),
    license="Apache",
    url="https://github.com/Alokia/component",
    package_dir={"": "."},
    packages=find_packages("."),
    python_requires=">=3.10.0",
    install_requires=["torch>=2.0.1", "torchvision>=0.15.2"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
