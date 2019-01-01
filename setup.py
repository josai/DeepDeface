from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deepdefacer",
    packages=find_packages(),
    version="0.0.2",
    author="Anish Khazane",
    author_email="akhazane@stanford.edu",
    description="Automatic Removal of Facial Features from MRI Images",
    download_url='https://github.com/AKhazane/DeepDeface/archive/v_01.tar.gz',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AKhazane/DeepDeface",
    entry_points='''
    [console_scripts]
    deepdefacer=deepdefacer.defacer:deface_3D_MRI
    ''',
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "keras==2.2.2",
        "tensorflow-gpu==1.8.0",
        "nilearn",
        "nibabel",
        "sklearn",
        "scipy",
        "ipython" 
        "matplotlib"
    ]
)
