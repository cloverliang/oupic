## oupic

A Phylogenetic Independent Contrast Method Under the OU Model

*oupic* is a python package for calculating phylogenetic independent contrasts under the Ornstein-Uhlenbeck model. The package is under active development. 

### Install
One way to install the package is to run
`python setup.py install`
in the package folder. (Dependencies may need to be resolved manually.)

Alternatively, to resolve dependency automatically using conda, follow the steps:
1. Create an environment that includes `python` and `invoke`.
```sh
$ conda create -n oupic python=3.7 invoke --yes
```
2. Activate the environment.
```sh
$ conda activate oupic
```
3. Install required packages with `bootstrap` in `develop` mode. To install `dendropy` via conda, include channel `bioconda` in `.condarc`.
```sh
$ invoke bootstrap develop
```

### Run example
A simple example of calculating the contrasts is provided in `example/example_pic.py`.

### Reference
A Phylogenetic Independent Contrast Method under the Ornstein-Uhlenbeck Model and its Applications in Correlated Evolution.
