from setuptools import setup, find_packages

setup(
    name="coco_augmentation",
    version="1.0.0",
    author="Sharath",
    author_email="copy.to.sharath@gmail.com",
    description="A library for augmenting COCO datasets with Albumentations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coco_augmentation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "albumentations",
        "pycocotools",
        "torch",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
