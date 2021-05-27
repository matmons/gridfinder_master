#!/usr/bin/bash
# Run Dijkstras for all directories/different configurations

# Arg 1: directory to targets
# Arg 2: directory for guess_ext


for f in Data/$1/*; do 
	#dirname=$(echo $f | sed -r "s/.+\/(.+)\..+/\1/");
    dirname=${f: -7}
    fa=${f:5}
    guess=$2/$dirname/
    echo current $fa $guess
    ./runner.py dijk_extended --targets_dir $fa/ --guess_ext_dir $guess
done