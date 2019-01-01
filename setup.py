import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepdeface",
    version="0.0.1",
    author="Anish Khazane",
    author_email="akhazane@stanford.edu",
    description="Automatic Removal of Facial Features from MRI Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AKhazane/ARFF-CNN.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)