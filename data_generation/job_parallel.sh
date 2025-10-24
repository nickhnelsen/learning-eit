#!/usr/bin/env bash

# Exit on error (including in pipelines) and on unset vars
set -euo pipefail

# ---- parameters ----
seed=2025          # base seed; actual seed will be "${seed}${i}"
N_loop=12
N_cond=2
N_solves=32
tau_m=20
tau_p=20
al_m=4.5
al_p=4.5
rho_m=0.7
rho_p=0.7
cr_m=100
cr_p=100

# Concurrency: use 8 physical cores
MAX_JOBS=8

LOGDIR="logs"
mkdir -p "$LOGDIR"

# Keep BLAS/OMP to 1 thread per MATLAB to avoid oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1

running=0

for (( i=1; i<=N_loop; i++ )); do
  job_seed="${seed}${i}"
  log="${LOGDIR}/run_${i}.log"
  echo "Starting job ${i} (seed=${job_seed}). Logging to ${log}"

  matlab -nosplash -nodesktop -noFigureWindows -nosoftwareopengl -singleCompThread \
    -batch "data_generation_fND_script ${job_seed} ${N_cond} ${N_solves} ${tau_m} ${tau_p} ${al_m} ${al_p} ${rho_m} ${rho_p} ${cr_m} ${cr_p}" \
    >"$log" 2>&1 &

  # Throttle to MAX_JOBS using wait -n (Bash 5.1+)
  (( ++running >= MAX_JOBS )) && { wait -n || { echo "A job failed; see ${LOGDIR}/*.log. Tailing:"; tail -n 50 ${LOGDIR}/*.log || true; exit 1; }; ((running--)); }
done

# Wait for the rest
while (( running > 0 )); do
  wait -n || { echo "A job failed; see ${LOGDIR}/*.log"; exit 1; }
  ((running--))
done

echo "All runs complete."
