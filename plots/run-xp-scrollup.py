#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

easyspap_options = {}
easyspap_options["--kernel "] = ["scrollup"]
easyspap_options["--variant "] = ["seq", "ji"]

omp_icv = {}  # OpenMP Internal Control Variables
omp_icv["OMP_NUM_THREADS="] = [1]
omp_icv["OMP_SCHEDULE="] = ["static"]

for size in [2**i for i in range(5, 13)]:
    easyspap_options["--iterations "] = [4096*4096//(size*size)]
    easyspap_options["--size "] = [size]
    execute('./run', omp_icv, easyspap_options, nbrun=5)
