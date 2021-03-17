#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k "] = ["life"]
options["-v "] = ["tiled_omp", "lazy", "lazy_ji", "ft"]

options["-ts "] = [4, 8, 16, 32]
options["-i "] = [960,1920,3840,8000]  

# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS="]= [46]
ompenv["OMP_PLACES="] = ["cores"]

nbrun = 4
# Lancement des experiences pour ./run -k life
execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v "] = ["seq"]
ompenv["OMP_NUM_THREADS="] = [1]
execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")