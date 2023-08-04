import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transport-cba",
    version="0.1.1",
    author="Peter Vanya",
    author_email="peter.vanya@gmail.com",
    description="A Python package for cost-benefit analysis of infrastructure projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/transport-cba/transport-cba",
    keywords="cost benefit analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    packages=setuptools.find_packages(),
    package_data={"": ["examples/*.csv", "examples/cba_sample_bypass.xlsx", "parameters/*/*.csv"]},
    install_requires=["numpy>=1.16, <2", "pandas>=0.24, <1.3"],
)
