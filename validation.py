"""
Post-processing for gridfinder package.

Functions:

- threshold
- thin
- raster_to_lines
- accuracy
- true_positives
- false_negatives
- flip_arr_values
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from skimage.morphology import skeletonize
import shapely.wkt
from shapely.geometry import Point, LineString
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy

def raster_to_lines(arr):
    """
    Convert thinned raster to linestring geometry.

    Parameters
    ----------
    guess_skel_in : path-like
        Output from thin().

    Returns
    -------
    guess_gdf : GeoDataFrame
        Converted to geometry.
    """

    rast = rasterio.open(guess_skel_in)
    arr = rast.read(1)
    affine = rast.transform

    max_row = arr.shape[0]
    max_col = arr.shape[1]
    lines = []

    for row in range(0, max_row):
        for col in range(0, max_col):
            loc = (row, col)
            if arr[loc] == 1:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        next_row = row + i
                        next_col = col + j
                        next_loc = (next_row, next_col)

                        # ensure we're within bounds
                        # ensure we're not looking at the same spot
                        if (
                            next_row < 0
                            or next_col < 0
                            or next_row >= max_row
                            or next_col >= max_col
                            or next_loc == loc
                        ):
                            continue

                        if arr[next_loc] == 1:
                            line = (loc, next_loc)
                            rev = (line[1], line[0])
                            if line not in lines and rev not in lines:
                                lines.append(line)

    real_lines = []
    for line in lines:
        real = (xy(affine, line[0][0], line[0][1]), xy(affine, line[1][0], line[1][1]))
        real_lines.append(real)

    shapes = []
    for line in real_lines:
        shapes.append(LineString([Point(line[0]), Point(line[1])]).wkt)

    guess_gdf = pd.DataFrame(shapes)
    geometry = guess_gdf[0].map(shapely.wkt.loads)
    guess_gdf = guess_gdf.drop(0, axis=1)
    guess_gdf = gpd.GeoDataFrame(guess_gdf, crs=rast.crs, geometry=geometry)

    guess_gdf["same"] = 0
    guess_gdf = guess_gdf.dissolve(by="same")
    guess_gdf = guess_gdf.to_crs(epsg=4326)

    return guess_gdf


def validation(grid_in, grid_buff_in, guess_in, hv_in, aoi_in):
    """Measure accuracy against a specified grid 'truth' file.

    Parameters
    ----------
    grid_in : str, Path
        Path to vector truth file.
    guess_in : str, Path
        Path to guess output from guess2geom.
    aoi_in : str, Path
        Path to AOI feature.
    buffer_amount : float, optional (default 0.01.)
        Leeway in decimal degrees in calculating equivalence.
        0.01 DD equals approximately 1 mile at the equator.
    """

    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    guess = rasterio.open(guess_in)
    guesses = guess.read(1)

    g = rasterio.open(grid_in)
    grid_raster = g.read(1)
    grid_raster = flip_arr_values(grid_raster)

    g_buff = rasterio.open(grid_buff_in)
    grid_buff_raster = g_buff.read(1)
    grid_buff_raster = flip_arr_values(grid_buff_raster)

    hv = rasterio.open(hv_in)
    hv_raster = hv.read(1)

    guesses = np.multiply(guesses, hv_raster)
    grid_raster = np.multiply(grid_raster, hv_raster)
    grid_buff_raster = np.multiply(grid_buff_raster, hv_raster)

    assert grid_raster.shape == grid_buff_raster.shape, "Ground truth rasters are not same shape"
    assert guesses.shape ==  grid_raster.shape, "Shapes of guesses and groundt truth do not match"

    tp, fp = positives(guesses, grid_buff_raster)
    fn = negatives(guesses, grid_raster)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    iou = tp/(tp+fp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return tp, fp, fn, precision, recall, iou, f1


def positives(guesses, truths):
    """Calculate true positives, used by accuracy().

    Parameters
    ----------
    guesses : numpy array
        Output from model.
    truths : numpy array
        Truth feature converted to array.

    Returns
    -------
    tp : float
        Ratio of true positives.
    """

    yes_guesses = 0
    yes_guesses_correct = 0
    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x, y]
            truth = truths[x, y]
            if guess == 1:
                yes_guesses += 1
                if guess == truth:
                    yes_guesses_correct += 1

    tp = yes_guesses_correct
    fp = yes_guesses - yes_guesses_correct

    return tp, fp


def negatives(guesses, truths):
    """Calculate false negatives, used by accuracy().

    Parameters
    ----------
    guesses : numpy array
        Output from model.
    truths : numpy array
        Truth feature converted to array.

    Returns
    -------
    fn : float
        Ratio of false negatives.
    """

    actual_grid = 0
    actual_grid_missed = 0

    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x, y]
            truth = truths[x, y]

            if truth == 1:
                actual_grid += 1
                if guess != truth:
                    found = False
                    for i in range(-5, 6):
                        for j in range(-5, 6):
                            if i == 0 and j == 0:
                                continue

                            shift_x = x + i
                            shift_y = y + j
                            if shift_x < 0 or shift_y < 0:
                                continue
                            if shift_x >= rows or shift_y >= cols:
                                continue

                            other_guess = guesses[shift_x, shift_y]
                            if other_guess == 1:
                                found = True
                    if not found:
                        actual_grid_missed += 1

    fn = actual_grid_missed

    return fn


def flip_arr_values(arr):
    """
    Simple helper function used by accuracy()
    The ground truth datasets (buffered and non-buffered) are binary matrices
    where 0 indicates grid and 1 indicates no-grid.
    
    This function returns a matrix where grid = 1 & no-grid = 0 
    """

    arr[arr == 1] = 2
    arr[arr == 0] = 1
    arr[arr == 2] = 0
    return arr
