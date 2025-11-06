#!/usr/bin/env bash

# Exit on error (including in pipelines) and on unset vars
set -euo pipefail

# ---- parameters ----
seed=2025          # base seed; actual seed will be "${seed}${i}"
N_loop=40
N_cond=250
N_solves=256
tau_m=7
tau_p=9
al_m=3
al_p=4
rhom_m=0.5
rhom_p=0.55
rhop_m=0.85
rhop_p=0.95
scale_m=7.5
scale_p=8.5

MAX_JOBS=8

LOGDIR="logs_lognormal"
mkdir -p "$LOGDIR"

# Keep BLAS/OMP to 1 thread per MATLAB to avoid oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1

running=0

for (( i=1; i<=N_loop; i++ )); do
  job_seed="${seed}${i}"
  log="${LOGDIR}/run_${i}.log"
  echo "Starting job ${i} (seed=${job_seed}). Logging to ${log}"

  matlab -nosplash -nodesktop -noFigureWindows -nosoftwareopengl -singleCompThread \
    -batch "data_generation_fND_lognormal_script ${job_seed} ${N_cond} ${N_solves} ${tau_m} ${tau_p} ${al_m} ${al_p} ${rhom_m} ${rhom_p} ${rhop_m} ${rhop_p} ${scale_m} ${scale_p}" \
    >"$log" 2>&1 &

  # Throttle to MAX_JOBS using wait -n (Bash 5.1+)
  (( ++running >= MAX_JOBS )) && {
    if ! wait -n; then
      echo "A job failed; see ${LOGDIR}/*.log. Tailing recent lines:"
      tail -n 30 "$LOGDIR"/*.log || true
      exit 1
    fi
    ((running--))
  }
done

# Drain remaining jobs
while (( running > 0 )); do
  if ! wait -n; then
    echo "A job failed; see ${LOGDIR}/*.log"
    exit 1
  fi
  ((running--))
done

echo "All runs complete."
