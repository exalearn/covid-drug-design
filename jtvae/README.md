# Junction Tree Variational Autoencoder for COVID-19
Python 3 Version of Fast Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018) amended from [https://github.com/Bibyutatsu/FastJTNNpy3](https://github.com/Bibyutatsu/FastJTNNpy3)

Junction Tree Variational Autoencoder paper [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Requirements
* RDKit (version >= 2017.09)    : Tested on 2019.09.1
* Python (version >= 3.6)       : Tested on 3.7.4
* PyTorch (version >= 0.2)      : Tested on 1.0.1

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)


# Quick Start

## Code for Accelerated Training
This repository contains the Python 3 implementation of the Fast Junction Tree Variational Autoencoder code.

* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.
* `fast_jtnn/` contains codes for model implementation.
* `fast_bo/` contains codes for Bayesian Optimisation with custom weighted scoring functions.


