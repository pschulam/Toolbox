# Toolbox
Python module for including in research code.

## Usage

This repository is meant to be included as a [Git
submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) in a
Python package repository. In particular, it provides functions and
classes that are useful when implementing machine learning
methods. Suppose we're working on an implementation of logistic
regression in a repository where the top level of the repo is the
python package, then within the repo we can type the following:

```{bash}
git submodule add https://github.com/pschulam/Toolbox tbx
```

After cloning Toolbox into your Python directory structure, you can
access utilities from within the package using:

```{python}
from .tbx import bsplines
```
