#!/usr/bin/env python

"""
Clip a supplied raster to match the admin boundaries supplied,
and then warp it to match another set of rasters.
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

import geopandas as gpd
import rasterio
import numpy as np
from gridfinder import clip_raster, save_raster
from rasterio.warp import Resampling, reproject

def make_same_as(curr_arr, curr_aff, curr_crs, dest_arr_like, dest_affine, dest_crs, resampling):
    """

    """
    
    dest_arr = np.empty_like(dest_arr_like)
    with rasterio.Env():
        reproject(
            source=curr_arr,
            destination=dest_arr,
            src_transform=curr_aff,
            dst_transform=dest_affine,
            src_crs=curr_crs,
            dst_crs=dest_crs,
            resampling=resampling,
        )

    return dest_arr


def clip_all(raster_in, admin_in, raster_shape_dir, dir_out, code="adm0_a3", resampling=Resampling.nearest):
    raster_in = Path(raster_in).expanduser()
    admin_in = Path(admin_in).expanduser()
    raster_shape_dir = Path(raster_shape_dir).expanduser()
    dir_out = Path(dir_out).expanduser()

    admin = gpd.read_file(admin_in)
    countries = admin[code].tolist()

    for c in countries:
        print(f"Doing {c}")
        c_out = dir_out / f"{c}.tif"

        aoi = admin[admin[code] == c]
        try:
            arr, aff, crs = clip_raster(raster_in, aoi)
        except ValueError:
            print("No input data for this AOI - skipped")
            continue
        targets_in = raster_shape_dir / f"{c}.tif"
        targets_rd = rasterio.open(targets_in)
        dest_arr_like = targets_rd.read(1)
        dest_affine = targets_rd.transform
        dest_crs = targets_rd.crs

        new_arr = make_same_as(arr, aff, crs, dest_arr_like, dest_affine, dest_crs, resampling)

        save_raster(c_out, new_arr, dest_affine, dest_crs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("raster_in", help="Input raster")
    parser.add_argument("dir_out", help="Output directory")
    parser.add_argument("--admin_in", "-a")
    parser.add_argument("--raster_shape_dir", "-s")
    parser.add_argument("--code", default="ADM0_A3")
    parser.add_argument("--resampling", default="nearest", help="nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, gauss, max, min, med")
    args = parser.parse_args()

    switch = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "gauss": Resampling.gauss,
        "min": Resampling.min,
        "max": Resampling.max,
        "med": Resampling.med
    }
    func = switch.get(args.resampling)
    
    clip_all(
        raster_in=args.raster_in,
        admin_in=args.admin_in,
        raster_shape_dir=args.raster_shape_dir,
        dir_out=args.dir_out,
        code=args.code,
        resampling=func
    )
