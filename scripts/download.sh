#!/usr/bin/env bash
# Fetch the BEIR datasets used in the benchmark into data/.
set -euo pipefail
mkdir -p data && cd data
for d in "${@:-scifact nfcorpus}"; do
  for name in $d; do
    curl -sSfLO "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${name}.zip"
    unzip -qo "${name}.zip" && rm "${name}.zip"
  done
done
