#!/usr/bin/env python

"""
Script to control running all energy-infra algorithms.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import yaml
import csv

import geopandas as gpd
import rasterio
import accessestimator as ea
import gridfinder as gf

from helpers import (
    prepare_roads_no_powerlines, 
    merge_rasters_local, 
    get_targets_costs_extended, 
    prepare_powerlines,
    prepare_groundtruth
)
from validation import validation

script_dir = Path(os.path.dirname(__file__))
with open(script_dir / "config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

admin_in = Path(cfg["inputs"]["admin"]).expanduser()
code = cfg["inputs"]["admin_code"]
ntl_in = Path(cfg["inputs"]["ntl_monthly"]).expanduser()
ntl_ann_in = Path(cfg["inputs"]["ntl_annual"]).expanduser()
pop_in = Path(cfg["inputs"]["pop"]).expanduser()
urban_in = Path(cfg["inputs"]["urban"]).expanduser()
data = Path(cfg["outputs"]["base"]).expanduser()
scratch = Path(cfg["outputs"]["scratch"]).expanduser()

targets_dir = cfg["outputs"]["targets"]
costs_dir = cfg["outputs"]["costs"]
guess_dir = cfg["outputs"]["guess"]

vector_dir = cfg["outputs"]["vector"]
pop_elec_dir = cfg["outputs"]["pop_elec"]
local_dir = cfg["outputs"]["local"]

percentile = cfg["options"]["percentile"]
ntl_threshold = cfg["options"]["ntl_threshold"]
threads = cfg["options"]["threads"]
raise_errors = False

# assert str(ntl_threshold).replace('.', '') == targets_dir[-3:-1], "NTL Threshold and Targets Dir output do not match"

# Costs Extended
pa_dir = cfg["inputs"]["protected_areas"]
slope_dir = cfg["inputs"]["slope"]
costs_ext_dir = cfg["outputs"]["costs_ext"]

# Dijk Extended
costs_roads_dir = cfg["costs"]["costs_roads"]
costs_pa_dir = cfg["costs"]["costs_protected_areas"]
costs_slope_dir = cfg["costs"]["costs_slope"]
costs_hv_dir = cfg["costs"]["costs_hv"]

r = cfg["costs"]["r"]
pa = cfg["costs"]["pa"]
s = cfg["costs"]["s"]

# Ground Truth
guess_ext_dir = cfg["outputs"]["guess_ext"]
gt_dir = cfg["outputs"]["gt"] 
gt_buff_dir = cfg["outputs"]["gt_buff"] 
gt_vec_dir = cfg["outputs"]["gt_vec"]

# Results
results_dir = cfg["results"]["dirname"] 

admin = gpd.read_file(admin_in)

def get_dirname(tool):
    if tool == targets:
        return targets_dir
    elif tool == costs:
        return costs_dir
    elif tool == costs_extended:
        return costs_ext_dir
    elif tool == dijk:
        return guess_dir
    elif tool == dijk_extended:
        return guess_ext_dir
    elif tool == vector:
        return vector_dir
    elif tool == pop_elec:
        return pop_elec_dir
    elif tool == local:
        return local_dir
    elif tool == ground_truth:
        return gt_dir
    elif tool == validate:
        return results_dir
    else:
        return ValueError(f"{tool} not supported")


def get_filename(dirname, country, ext="tif"):
    return data / dirname / f"{country}.{ext}"


def get_filename_auto(tool, country):
    dirname = get_dirname(tool)
    ext = "tif"
    if tool == vector:
        ext = "gpkg"
    return get_filename(dirname, country, ext)


def spawn(tool, countries):
    if countries is None:
        countries = admin[code].tolist()
    countries[:] = [c for c in countries if not get_filename_auto(tool, c).is_file()]
    print("Will process", len(countries), "countries")

    with Pool(processes=threads) as pool:
        pool.map(tool, countries)


def targets(country):
    this_scratch = scratch / f"targets_{country}"
    ntl_out = this_scratch / "ntl"
    ntl_merged_out = this_scratch / "ntl_merged.tif"
    ntl_thresh_out = this_scratch / "ntl_thresh.tif"
    targets_out = get_filename(targets_dir, country)

    try:
        print(f"Targets\tstart\t{country}")
        this_scratch.mkdir(parents=True, exist_ok=True)
        aoi = admin.loc[admin[code] == country]
        buff = aoi.copy()
        buff.geometry = buff.buffer(0.1)

        # Clip NTL rasters and calculate nth percentile values
        # WARNING! merge rasters local was originally gf.merge_raster()
        gf.clip_rasters(ntl_in, ntl_out, buff)
        raster_merged, affine = merge_rasters_local(ntl_out, percentile=percentile)
        gf.save_raster(ntl_merged_out, raster_merged, affine)

        # Apply filter to NTL
        ntl_filter = gf.create_filter()
        ntl_thresh, affine = gf.prepare_ntl(
            ntl_merged_out,
            buff,
            ntl_filter=ntl_filter,
            upsample_by=1,
            threshold=ntl_threshold,
        )
        gf.save_raster(ntl_thresh_out, ntl_thresh, affine)

        # Clip to actual AOI
        targets, affine, _ = gf.clip_raster(ntl_thresh_out, aoi)
        gf.save_raster(targets_out, targets, affine)
        msg = f"Targets\tDONE\t{country}"

    except Exception as e:
        msg = f"Targets\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        shutil.rmtree(this_scratch)
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def costs(country):
    targets_in = get_filename(targets_dir, country)
    costs_in = get_filename("costs_vec/europe/", country, ext="gpkg")
    costs_out = get_filename(costs_dir, country)

    try:
        print(f"Costs\tstart\t{country}")
        aoi = admin.loc[admin[code] == country]
        roads_raster, affine = gf.prepare_roads(costs_in, aoi, targets_in)
        gf.save_raster(costs_out, roads_raster, affine, nodata=-1)

        msg = f"Costs\tDONE\t{country}"
    except Exception as e:
        msg = f"Costs\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def costs_extended(country):
    """
    A cost raster with roads is the default case for costs. Other sources of weighting
    are multiplied into the base cost layer.

    The dir for protected areas and slopes must be set in the config_file. This function
    generates separated files for each layer of weights.
    """
    roads_dir = "costs_vec/europe/"
    targets_in = get_filename(targets_dir, country)
    roads_in = get_filename(roads_dir, country, ext="gpkg")
    pa_in = get_filename(pa_dir, country)
    slope_in = get_filename(slope_dir, country)
    costs_roads_out = get_filename(costs_roads_dir, country)
    costs_pa_out = get_filename(costs_pa_dir, country)
    costs_slope_out = get_filename(costs_slope_dir, country)
    
    try:
        print(f"Costs\tstart\t{country}")
        aoi = admin.loc[admin[code] == country]
        roads_raster, affine = prepare_roads_no_powerlines(roads_in, aoi, targets_in)
        gf.save_raster(costs_roads_out, roads_raster, affine, nodata=-1)
        
        pa_raster = rasterio.open(pa_in)
        pa_costs = pa_raster.read(1)
        gf.save_raster(costs_pa_out, pa_costs, affine, nodata=-1)

        slope_raster = rasterio.open(slope_in)
        slope_costs = slope_raster.read(1)
        slope_costs[slope_costs <= 20] = 1
        slope_costs[(slope_costs > 20) & (slope_costs <= 30)] = 2
        slope_costs[slope_costs > 30] = 3
        gf.save_raster(costs_slope_out, slope_costs, affine, nodata=-1)

        msg = f"Costs\tDONE\t{country}"
    except Exception as e:
        msg = f"Costs\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def dijk(country):
    this_scratch = scratch / f"dijk_{country}"
    dist_out = this_scratch / "dist.tif"
    targets_in = get_filename(targets_dir, country)
    costs_in = get_filename(costs_dir, country)
    guess_out = get_filename(guess_dir, country)

    try:
        print(f"Dijk\tstart\t{country}")
        this_scratch.mkdir(parents=True, exist_ok=True)

        targets, costs, start, affine = gf.get_targets_costs(targets_in, costs_in)
        dist = gf.optimise(targets, costs, start, silent=True)
        gf.save_raster(dist_out, dist, affine)
        guess, affine = gf.threshold(dist_out)
        guess_skel = gf.thin(guess)
        gf.save_raster(guess_out, guess_skel, affine)

        msg = f"Dijk\tDONE\t{country}"
    except Exception as e:
        msg = f"Dijk\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        shutil.rmtree(this_scratch)
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def dijk_extended(country):
    this_scratch = scratch / f"dijk_{country}"
    dist_out = this_scratch / "dist.tif"
    targets_in = get_filename(targets_dir, country)

    costs_roads_in = get_filename(costs_roads_dir, country)
    costs_protected_areas_in = get_filename(costs_pa_dir, country)
    costs_slope_in = get_filename(costs_slope_dir, country)
    costs_hv_in = get_filename(costs_hv_dir, country)

    guess_out = get_filename(guess_ext_dir, country)
    #assert guess_ext_dir.split("/")[-2] == targets_dir.split("/")[-2], f"Mismatch in {targets_dir} and {guess_ext_dir}"  
    try:
        print(f"Dijk\tstart\t{country}")
        this_scratch.mkdir(parents=True, exist_ok=True)

        targets, costs, start, affine = get_targets_costs_extended(
            targets_in,
            costs_roads_in,
            costs_protected_areas_in,
            costs_slope_in,
            costs_hv_in,
            r,
            pa,
            s
        )
        dist = gf.optimise(targets, costs, start, silent=True)
        gf.save_raster(dist_out, dist, affine)
        guess, affine = gf.threshold(dist_out)
        guess_skel = gf.thin(guess)
        gf.save_raster(guess_out, guess_skel, affine)

        msg = f"Dijk\tDONE\t{country}"
    except Exception as e:
        msg = f"Dijk\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        shutil.rmtree(this_scratch)
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def vector(country):
    guess_in = get_filename(guess_dir, country)
    guess_vec_out = get_filename(vector_dir, country, ext="gpkg")

    try:
        print(f"Vector\tstart\t{country}")
        guess_gdf = gf.raster_to_lines(guess_in)
        guess_gdf.to_file(guess_vec_out, driver="GPKG")

        msg = f"Vector\tDONE\t{country}"
    except Exception as e:
        msg = f"Vector\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def pop_elec(country):
    targets_in = get_filename(targets_dir, country)
    pop_elec_out = get_filename(pop_elec_dir, country)
    weight_out = pop_elec_out.parents[0] / (str(pop_elec_out.stem) + "_W.tif")

    try:
        msg = ""
        print(f"\n\nPopElec\tstart\t{country}")
        aoi = admin.loc[admin[code] == country]
        access = aoi[["total", "urban", "rural"]].iloc[0].to_dict()
        if access["total"] == 1:
            return

        pop, urban, ntl, targets, affine, crs = ea.regularise(
            country, aoi, pop_in, urban_in, ntl_ann_in, targets_in
        )
        pop_elec, access_model_total, weights = ea.estimate(
            pop, urban, ntl, targets, access
        )
        gf.save_raster(pop_elec_out, pop_elec, affine, crs)
        # gf.save_raster(weight_out, weights, affine, crs)

        msg = f"PopElec\tDONE\t{country}\t\treal: {access['total']:.2f}\tmodel: {access_model_total:.2f}"
    except Exception as e:
        msg = f"PopElec\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def local(country):
    pop_elec_in = get_filename(pop_elec_dir, country)
    lv_out = get_filename(local_dir, country)

    try:
        print(f"Local\tstart\t{country}")

        pop_elec_rd = rasterio.open(pop_elec_in)
        pop_elec = pop_elec_rd.read(1)
        affine = pop_elec_rd.transform
        crs = pop_elec_rd.crs

        aoi = admin.loc[admin[code] == country]
        access = aoi[["total", "urban", "rural"]].iloc[0].to_dict()
        peak_kw_pp = 0.1
        people_per_hh = 5
        if access["total"] >= 0.95:
            peak_kw_pp = 2
            people_per_hh = 3

        lengths = ea.apply_lv_length(
            pop_elec, peak_kw_pp=peak_kw_pp, people_per_hh=people_per_hh
        )
        gf.save_raster(lv_out, lengths, affine, crs)
        total_length = np.sum(lengths)
        msg = f"Local\tDONE\t{country}\tTot length: {total_length} km"
    except Exception as e:
        msg = f"Local\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def ground_truth(country):
    """
    Creates a ground truth raster with powerlines for each country.

    Minor modifications must be made to this functions for the creation of a cost raster
    containing only hv lines. Comment out the buffered-related lines and swap out
    prepare_groundtruth() with prepare_powerlines()

    """
    
    targets_in = get_filename(targets_dir, country)
    grid_in = get_filename(gt_vec_dir, country, ext="gpkg")
    truth_out = get_filename(gt_dir, country)
    truth_buff_out = get_filename(gt_buff_dir, country)
    
    try:
        print(f"Costs\tstart\t{country}")
        aoi_in = admin.loc[admin[code] == country]
        #grid_raster, affine = prepare_powerlines(grid_in, targets_in, aoi_in)
        grid_raster, grid_buff_raster, affine = prepare_groundtruth(grid_in, targets_in, aoi_in, buffer_amount=0.01)
        gf.save_raster(truth_out, grid_raster, affine)
        gf.save_raster(truth_buff_out, grid_buff_raster, affine)
        msg = f"Costs\tDONE\t{country}"
    except Exception as e:
        msg = f"Costs\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)
        if log:
            with open(log, "a") as f:
                print(msg, file=f)

def validate(country):
    guess_in = get_filename(guess_ext_dir, country)
    grid_in = get_filename(gt_dir, country)
    grid_buff_in = get_filename(gt_buff_dir, country)
    hv_in = get_filename(costs_hv_dir, country)
    aoi = admin.loc[admin[code] == country]
    try: 
        print(f"Validation\tstart\t{country}")
        tp, fp, fn, precision, recall, iou, f1 = validation(grid_in, grid_buff_in, guess_in, hv_in, aoi)
        logging_data = {
            "country": country,
            "percentile": percentile,
            "threshold": ntl_threshold,
            "r": r,
            "pa": pa,
            "s": s,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "f1": f1
        }
        header_switch = os.path.isfile(results_dir + '/results.csv')
        if header_switch:
            with open(results_dir + '/results.csv', 'a', newline='') as csvfile:
                fieldnames = ["country", "percentile", "threshold", "r", "pa", "s", "tp", "fp", "fn", "precision", "recall", "iou", "f1"]  
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(logging_data)
        else:
            r_dir = Path(results_dir)
            r_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir + '/results.csv', 'w', newline='') as csvfile:
                fieldnames = ["country", "percentile", "threshold", "r", "pa", "s", "tp", "fp", "fn", "precision", "recall", "iou", "f1"]  
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(logging_data)
        msg = f"Validation\tDONE\t{country}"
    except Exception as e:
        msg = f"Validation\tFAILED\t{country}\t{e}"
        if raise_errors:
            raise
    finally:
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tool", help="One of targets, costs, costs_extended, dijk, vector, pop_elec, local"
    )
    parser.add_argument("--countries")
    parser.add_argument("--targets_dir", default=targets_dir)
    parser.add_argument("--costs_dir", default=costs_dir)
    
    parser.add_argument("--costs_ext_dir", default=costs_ext_dir)
    parser.add_argument("--pa_dir", default=pa_dir)
    parser.add_argument("--slope_dir", default=slope_dir)

    parser.add_argument("--costs_roads_dir", default=costs_roads_dir)
    parser.add_argument("--costs_pa_dir", default=costs_pa_dir)
    parser.add_argument("--costs_slope_dir", default=costs_slope_dir)
    
    parser.add_argument("--guess_ext_dir", default=guess_ext_dir)

    parser.add_argument("--r", default=r, type=int) # Very similar to raise errors
    parser.add_argument("--pa", default=pa, type=int)
    parser.add_argument("--s", default=s, type=int)

    parser.add_argument("--gt_dir", default=gt_dir)
    parser.add_argument("--results_dir", default=results_dir)
    
    parser.add_argument("--guess_dir", default=guess_dir)
    parser.add_argument("--vector_dir", default=vector_dir)
    parser.add_argument("--pop_elec_dir", default=pop_elec_dir)
    parser.add_argument("--local_dir", default=local_dir)
    parser.add_argument("--percentile", default=percentile, type=int)
    parser.add_argument("--ntl_threshold", default=ntl_threshold, type=float)
    parser.add_argument(
        "-r",
        "--raise_errors",
        action="store_true",
        default=False,
        help="Whether to raise errors",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=None,
        help="If supplied, logs will be written to this file",
    )
    args = parser.parse_args()

    switch = {
        "targets": targets,
        "costs": costs,
        "costs_extended": costs_extended,
        "dijk": dijk,
        "dijk_extended": dijk_extended,
        "vector": vector,
        "pop_elec": pop_elec,
        "local": local,
        "ground_truth": ground_truth,
        "validate": validate
    }

    func = switch.get(args.tool)
    if func is None:
        sys.exit(f"Option {args.tool} not supported")

    countries = None
    if args.countries:
        if "," in args.countries:
            countries = args.countries.split(",")
        else:
            countries = [args.countries]

    targets_dir = args.targets_dir
    costs_dir = args.costs_dir
    guess_dir = args.guess_dir
    vector_dir = args.vector_dir
    pop_elec_dir = args.pop_elec_dir
    local_dir = args.local_dir
    percentile = args.percentile
    ntl_threshold = args.ntl_threshold
    raise_errors = args.raise_errors
    log = args.log

    pa_dir = args.pa_dir
    slope_dir = args.slope_dir
    costs_ext_dir = args.costs_ext_dir
    gt_dir = args.gt_dir
    results_dir = args.results_dir

    costs_roads_dir = args.costs_roads_dir
    costs_pa_dir = args.costs_pa_dir
    costs_slope_dir = args.costs_slope_dir
    guess_ext_dir = args.guess_ext_dir

    r = args.r
    pa = args.pa
    s = args.s
    
    spawn(func, countries)
