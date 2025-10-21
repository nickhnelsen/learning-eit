#!/bin/bash

# Exit on error, unset var use, or failed pipe
set -euo pipefail

# Define parameter arrays
N_ar=(256 1024 4096)
Noise_ar=(1 5 15 20)
Seed_ar=(0 1 2 3 4)

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu

# Loop over all parameter combinations
for N in "${N_ar[@]}"; do
  for Noise in "${Noise_ar[@]}"; do
    for Seed in "${Seed_ar[@]}"; do
      echo "Running: N=${N}, Noise=${Noise}, Seed=${Seed}"
      python -u driver.py "$N" "$Noise" "$Seed" | tee "log_N${N}_Noise${Noise}_Seed${Seed}.out"
    done
  done
done

echo "All runs complete."
