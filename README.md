# STAMPede - STAMP data Exploration and Differential Expression

[![Documentation](https://github.com/mhlangalab/stampede/actions/workflows/docs.yml/badge.svg)](https://github.com/mhlangalab/stampede/actions/workflows/docs.yml)
pypi: soon
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/stampede/README.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Process & analyse [STAMP data](https://doi.org/10.1016/j.cell.2025.05.027) with ease!
STAMPede is built to handle huge datasets with shallow depth, using syntax familiar to `scanpy` users. 



## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)
3.  [Citation](#citation)



## Installation

### Using conda

```bash
conda install -c conda-forge -c bioconda stampede
```

### Using pip

Please note: the Pypi package name different. 
After installing the `stampede-sc` package, it can be imported as `stampede`

```bash
pip install stampede-sc
```

### Developer installation

Clone the repo
```bash
git clone https://github.com/mhlangalab/stampede.git
```

Change directory into the repo
```bash
cd stampede
```

Create a conda environment
```bash
conda env create -n stampede -f requirements.yaml
```

Activate the conda environment
```bash
conda activate stampede
```

Install the package
```bash
pip install -e .
```



## Documentation

Check out our [GitHub pages](https://mhlangalab.github.io/stampede/index.html)!



## Usage

Follow along with the [tutorial.ipynb](tutorial.ipynb)!



## Citation

If you used STAMPede in your work, please cite it.
