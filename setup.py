from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text("utf-8")

setup(
    name="rffickle",
    description="Roboflow fork of fickle - load pickled data as safely as possible",
    version="0.2.2",
    author="Eduard Christian Dumitrescu, Roboflow Inc.",
    author_email="help@roboflow.com",
    url="https://github.com/roboflow/fickle",
    install_requires=["attrs"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
