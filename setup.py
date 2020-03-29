from setuptools import setup, find_packages

setup(
    name="scilmm",
    version="0.2.2",
    description="Sparse Cholesky factorIzation Linear Mixed model",
    url="https://github.com/TalShor/SciLMM.git",
    author="TalShor, kalkairis",
    author_email="talihow@gmail.com, iris.kalka@weizmann.ac.il",
    license="GNU",
    packages=find_packages(),
    zip_safe=False,
    setup_requires=[
        "numpy>=1.15.1"
    ],
    install_requires=[
        "click",
        "nose",
        "numpy>=1.15.1",
        "scipy>=1.1.0",
        "scikit-learn>=0.19.2",
        "scikit-sparse>=0.4.3",
        "pandas>=0.23.4",
        "networkx>=2.1",
    ],
    extras_require={"dev": ["pylint", "black>=19.3b0", "wheel"],},
    scripts=[
        "scilmm/IBDCompute.py",
        "scilmm/SparseCholesky.py",
        "scilmm/he_estimator"
    ],
)
