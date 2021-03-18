#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os
# 
# # Dictionnaire avec les options de compilations d'apres commande
# options = {}
# options["-k "] = ["life"]
# 
# options["-v "] = ["omp", "lazy", "lazy_ji", "lazy_ft -ft"]
# 
# options["-tw "] = [256]
# options["-th "] = [64]
# options["-a "] = ["random"]
# 
# # Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
# #options["-of "] = ["./plots/data/perf_data.csv"]
# 
# 
# # Dictionnaire avec les options OMP
# ompenv = {}
# ompenv["OMP_NUM_THREADS="] = [46]
# ompenv["OMP_PLACES="] = ["cores"] 
# 
# nbrun = 1
# # Lancement des experiences
# 
# 
# for size in [512, 1024, 2048, 4096, 8192]:
#     options["-i "] = [100*(8192*8192)/(size*size)]
#     options["-s "] =  [size]
#     execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")
# 
# 
# 
# 
# # Lancement de la version seq avec le nombre de thread impose a 1
# options["-v "] = ["seq"]
# ompenv["OMP_NUM_THREADS="] = [1]
# 
# for size in [512, 1024, 2048, 4096, 8192]:
#     options["-i "] = [100*(8192*8192)/(size*size)]
#     options["-s "] =  [size]
#     execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k "] = ["life"]

options["-v "] = ["omp", "lazy", "lazy_ji", "lazy_ft -ft"]

options["-i "] = [960, 1920, 3840, 8000] 
options["-s "] = [4096]
options["-tw "] = [256]
options["-th "] = [64]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
#options["-of "] = ["./plots/data/perf_data.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS="] = [46]
ompenv["OMP_PLACES="] = ["cores"] 

nbrun = 3
# Lancement des experiences
execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")




# Lancement de la version seq avec le nombre de thread impose a 1
options["-v "] = ["seq"]
ompenv["OMP_NUM_THREADS="] = [1]
execute('./run ', ompenv, options, nbrun, verbose=True, easyPath=".")