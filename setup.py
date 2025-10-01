from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="baremetalml",                 
    version="1.0.0",                   
    author="Askari Abidi",              
    author_email="askari.abidi.2005@gmail.com",  
    description="Lightweight modular ML library from scratch in Python",
    long_description=long_description,  
    long_description_content_type="text/markdown",
    url="https://github.com/AskariAbidi18/ML_from_Scratch", 
    packages=find_packages(),         
    include_package_data=True,        # Include files listed in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",      
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',         
    install_requires=[                 
        "numpy>=1.24.0"
    ],
    license="MIT",
    keywords="machine learning, ML, numpy, educational, baremetalml",
)
