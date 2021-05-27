#!/usr/bin/bash
# Use OSM gpkg files to create files with only powerlines

# Arg 1: directory to input gpkg
# Arg 2: directory for output gpkg


for f in $1/*; do
	name=$(echo $f | sed -r "s/.+\/(.+)\..+/\1/");
    out=$2/$name.gpkg
    ogr2ogr -where "\"power\"=\"line\"" -f GPKG $out $f
done
