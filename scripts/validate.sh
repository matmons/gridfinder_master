#!/usr/bin/bash
# Validate all files in dir

# Arg 1: directory to results
# Arg 2: directory to guess in

for f in $1/*; do 
    base=${f: -7}
    prefix=${2: 5}
    echo $f $prefix/$base 
    #./runner.py validate --results_dir $f --guess_ext_dir $prefix/$base
done