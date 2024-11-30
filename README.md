# CORN (Code of Relaxing Networks)

Visit the [CORN Wiki](https://github.com/willwng/CORN/wiki)! This code generates and minimizes the energy of a lattice/network:

<img src="https://github.com/willwng/CORN/assets/8275672/922ac19e-9e6c-4860-852b-6e5d909308b6" width="300">

## Required Libraries
See [requirements.txt](requirements.txt)

These can be installed with `pip install -r requirements.txt.`

An older, but optimized (including GPU-accelerated) version of the code can be found in the `optimized` branch, but this requires installation of additional libraries

## Running CORN

    python main.py

`main.py` will requires a folder named `outputs` in the same directory as it. 
It will then try to create a folder based on the initial time run and 
subsequent subfolders for each bond occupation probability.

Alternatively, there are pre-existing bash scripts which can perform batch runs

