#!/usr/bin/bash
# Run Dijkstras for all directories/different configurations

# Arg 1: directory to targets
# Arg 2: directory for guess_ext


for pa in 0 1 5 10 100; do 
    for s in 0 1 5 10 100; do
        dirname=europe/costs/pa$pa-s$s
        echo mv/$dirname results/$dirname 
        ./runner.py dijk_extended --guess_ext_dir mv/$dirname --pa $pa --s $s 
        ./runner.py validate --guess_ext_dir mv/$dirname --results_dir Data/results/$dirname 
    done
done