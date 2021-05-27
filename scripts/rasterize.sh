#!/usr/bin/bash
# Rasterize all in dir using targets for shape and extents
# First arg: directory containing files to rasterize
# Second arg: directory for output

dir=$(dirname $0)
echo $dir

for f in $1/*; do
	if [[ $f == *"merged"* ]]; then
		continue;
	fi
	name=$(echo $f | sed -r "s/.+\/(.+)\..+/\1/");
	targ=~/Data/targets/$name.tif;
	out=$2/$name.tif
	if [ ! -f $out ]; then
		echo "Doing $name"
        # -at (ALL_TOUCHED) breaks with IRN, USA, ZMB
		gdal_rasterize -burn 1 -a_nodata 0 -init 0 -ts $($dir/getres.sh $targ) -te $($dir/getextents.sh $targ) $f $2/$name.tif
	fi
done

