#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name="eitF10k"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module purge
module load matlab/r2019a
seed=220301
N_loop=20
N_cond=500
N_solves=512
tau_m=20
tau_p=20
al_m=4.5
al_p=4.5
rho_m=0.7
rho_p=0.7
cr_m=100
cr_p=100

for (( i=1; i<=$N_loop; i++))
do
   srun --nodes=1 --exclusive --ntasks=1 matlab -nosoftwareopengl -batch "data_generation_fND_script $seed$i $N_cond $N_solves $tau_m $tau_p $al_m $al_p $rho_m $rho_p $cr_m $cr_p" &
done
wait
echo done
