
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from rasterio.warp import reproject, Resampling

import geopandas as gpd
import rasterio

from gridfinder import clip_raster, save_raster


def make_same_as(curr_arr, curr_aff, curr_crs, dest_arr_like, dest_affine, dest_crs):
    dest_arr = np.empty_like(dest_arr_like)

    with rasterio.Env():
        reproject(
            source=curr_arr,
            destination=dest_arr,
            src_transform=curr_aff,
            dst_transform=dest_affine,
            src_crs=curr_crs,
            dst_crs=dest_crs,
            resampling=Resampling.med,
        )

    return dest_arr



def clip_all(raster_in, admin_in, raster_shape_dir, dir_out, code="ADM0_A3"):
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
            arr, aff, crs = gf.clip_raster(raster_in, aoi)
        except ValueError:
            print("No input data for this AOI - skipped")
            continue
        targets_in = raster_shape_dir / f"{c}.tif"
        targets_rd = rasterio.open(targets_in)
        dest_arr_like = targets_rd.read(1)
        dest_affine = targets_rd.transform
        dest_crs = targets_rd.crs

        new_arr = make_same_as(arr, aff, crs, dest_arr_like, dest_affine, dest_crs)

        gf.save_raster(c_out, new_arr, dest_affine, dest_crs)
#clip_all("Data/lccs.tif", "admin_boundaries.gpkg", "Data/targets_filt/", "Data/LandCover_rs/")


def make_cost_func(lc1, lc2, truth1, truth2):
    uganda_cover = rasterio.open(lc1)
    uganda_covers = uganda_cover.read(1)
    uganda_truth = rasterio.open(truth1)
    uganda_truths = uganda_truth.read(1)
    ken_cover = rasterio.open(lc2)
    ken_covers = ken_cover.read(1)
    kenya_truth = rasterio.open(truth2)
    kenya_truths = kenya_truth.read(1)
    collections = {}
    for i in range(uganda_covers.shape[0]):
        for j in range(uganda_covers.shape[1]):
            if uganda_covers[i,j] in collections.keys():
                collections[uganda_covers[i,j]] +=1
            else:
                collections[uganda_covers[i,j]] =1


    collections2 = {}
    for i in range(uganda_covers.shape[0]):
        for j in range(uganda_covers.shape[1]):
            if uganda_truths[i,j]:
                if uganda_covers[i,j] in collections2.keys():
                    collections2[uganda_covers[i,j]] +=1
                else:
                    collections2[uganda_covers[i,j]] =1

    collections2 = {}
    for i in range(uganda_covers.shape[0]):
        for j in range(uganda_covers.shape[1]):
            if uganda_truths[i,j]:
                if uganda_covers[i,j] in collections2.keys():
                    collections2[uganda_covers[i,j]] +=1
                else:
                    collections2[uganda_covers[i,j]] =1
    for key in collections2.keys():
        collections2[key]/=collections[key]
    for key in collections2.keys():
        
        collections2[key] = 1/collections2[key]

    col = {}
    for i in range(ken_covers.shape[0]):
        for j in range(ken_covers.shape[1]):
            if ken_covers[i,j] in col.keys():
                col[ken_covers[i,j]] +=1
            else:
                col[ken_covers[i,j]] =1

    col2 = {}
    for i in range(ken_covers.shape[0]):
        for j in range(ken_covers.shape[1]):
            if kenya_truths[i,j]:
                if ken_covers[i,j] in col2.keys():
                    col2[ken_covers[i,j]] +=1
                else:
                    col2[ken_covers[i,j]] =1
    for key in col2.keys():
        col2[key]/=col[key]
    for key in col2.keys():
        col2[key]=1/col2[key]

    for key, values in col2.items():
        try:
            col2[key] = values + collections2[key]
        except:
            collections2[key] = sum(collections2.values())/len(collections2.keys())
            col2[key] = values + collections2[key]

    col2[0]=0
    liste = [70,71,72, 80, 81, 82, 121, 140,151]
    for i in liste:
        col[i] = max(col2.values())
    fsums = sum(col2.values())
    for key, values in col2.items():
        col2[key] = 10*col2[key]/fsums
    col2[0]=10000

    return col2

def replace_with_dict2(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]



