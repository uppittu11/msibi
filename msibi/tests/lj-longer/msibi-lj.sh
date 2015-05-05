#!/bin/bash
#SBATCH --mail-user=tcmoore3@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH -o runs.log
#SBATCH --partition=gpu
#SBATCH --account=mccabe_gpu
#SBATCH --constrain=cuda42
#SBATCH --mem=40G

setpkgs -a python
setpkgs -a openmpi_gcc
setpkgs -a hoomd_1.0.1

rm rdfs/pair*
rm potentials/*
rm figures/*

python opt.py > opt.out
