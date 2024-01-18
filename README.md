# CORN (Code of Relaxing Networks)

This code generates and minimizes the energy of a lattice/network

<img src="https://media.github.coecis.cornell.edu/user/9399/files/84d5f6f3-08db-4fad-ae37-a1d51e081906" width="300">

## Required Libraries
See [requirements.txt](requirements.txt)

These can be installed with `pip install -r requirements.txt`

(It is not important to get these version numbers - the latest versions of 
each library should work fine. However, early versions of `scipy` may not have
the newest solvers/mimization methods)

## Running CORN

    python main.py

`main.py` will requires a folder named `outputs` in the same directory as it. 
It will then try to create a folder based on the initial time run and 
subsequent subfolders for each bond occupation probability.

Alternatively, there are pre-existing bash scripts which can perform batch runs

