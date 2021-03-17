#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k "] = ["life"]
options["-v "] = ["tiled_omp", "lazy", "lazy_ji", "ft"]

options["-ts "] = [4, 8, 16, 32]
options["-a "] = ["random"]

size = [512, 1024, 2048, 4096, 8192] 

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
# options["-of "] = ["./plots/data/perf_data.csv"]

# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS="]= [46]
ompenv["OMP_PLACES="] = ["cores"]

nbrun = 4
# Lancement des experiences pour random
for s in size:
    options["-s "] = s
    options["-i "] = (100*8192*8192)//(s*s)
    execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
options["-v "] = ["seq"]
ompenv["OMP_NUM_THREADS="] = [1]
execute('./run', ompenv, options, nbrun, verbose=False, easyPath=".")


