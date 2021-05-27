import numpy as np
import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd


def get_targets_costs_extended(targets_in, costs_roads_in, costs_protected_areas_in, costs_slope_in, costs_hv_in, r=1, pa=1, s=1):
    """Load the targets and costs arrays from the given file paths.

    Parameters
    ----------
    targets_in : str
        Path for targets raster.
    costs_roads_in : str
        Path for road costs raster.
    costs_protected_areas_in : str
        Path for protected areas costs raster.
    costs_slope_in : str
        Path for slope costs raster.
    r : float
        Scalar for road costs
    pa : float
        Scalar for protected areas costs
    s : float
        Scalar for slope costs
    Returns
    -------
    targets : numpy array
        2D array of targets
    costs: numpy array
        2D array of costs
    start: tuple
        Two-element tuple with row, col of starting point.
    affine : affine.Affine
        Affine transformation for the rasters.
    """

    targets_ra = rasterio.open(targets_in)
    affine = targets_ra.transform
    targets = targets_ra.read(1)

    costs_roads_ra = rasterio.open(costs_roads_in)
    costs_roads = costs_roads_ra.read(1) * r

    costs_protected_areas_ra = rasterio.open(costs_protected_areas_in)
    costs_protected_areas = costs_protected_areas_ra.read(1) * pa

    costs_slope_ra = rasterio.open(costs_slope_in)
    costs_slope = costs_slope_ra.read(1) * s

    costs_hv_ra = rasterio.open(costs_hv_in)
    costs_hv = costs_hv_ra.read(1)

    assert costs_roads.shape == costs_protected_areas.shape, "Shape mismatch roads and protected areas"
    assert costs_roads.shape == costs_slope.shape, "Shape mismatch roads and slope"
    assert costs_roads.shape == costs_hv.shape, "Shape mismatch roads and hv"
    
    # Elementwise multiplication of HV-lines to properly set all hv containing cells
    # to zero
    costs = np.multiply(costs_roads + costs_protected_areas + costs_slope, costs_hv)
    target_list = np.argwhere(targets == 1.0)
    start = tuple(target_list[0].tolist())

    targets = targets.astype(np.int8)
    costs = costs.astype(np.float16)

    return targets, costs, start, affine

def merge_rasters_local(folder, percentile=70):
    """Merge a set of monthly rasters keeping the nth percentile value.

    Used to remove transient features from time-series data.

    Parameters
    ----------
    folder : str, Path
        Folder containing rasters to be merged.
    percentile : int, optional (default 70.)
        Percentile value to use when merging using np.nanpercentile.
        Lower values will result in lower values/brightness.

    Returns
    -------
    raster_merged : numpy array
        The merged array.
    affine : affine.Affine
        The affine transformation for the merged raster.
    """

    affine = None
    rasters = []

    for file in os.listdir(folder):
        if file.endswith(".tif"):
            ntl_rd = rasterio.open(os.path.join(folder, file))
            rd = ntl_rd.read(1)
            zero_pct = np.count_nonzero(rd==0)/(rd.shape[0]*rd.shape[1])
            # Added check for blank file. 
            # Multiple summer months lack data for E   uropean countries.
            # Will not append months where zero-values make up 70 % of all pixels.
            # The average zero-percentage varies from country to country, but no observations above 70% have been made (by random sampling).
            if zero_pct < 0.7:
                rasters.append(rd)
            if not affine:
                affine = ntl_rd.transform

    raster_arr = np.array(rasters)
    raster_merged = np.percentile(raster_arr, percentile, axis=0)
    return raster_merged, affine

def prepare_roads_no_powerlines(roads_in, aoi_in, ntl_in):
    """Prepare a roads feature layer for use in algorithm.

    Parameters
    ----------
    roads_in : str, Path
        Path to a roads feature layer. This implementation is specific to
        OSM data and won't assign proper weights to other data inputs.
    aoi_in : str, Path or GeoDataFrame
        AOI to clip roads.
    ntl_in : str, Path
        Path to a raster file, only used for correct shape and
        affine of roads raster.

    Returns
    -------
    roads_raster : numpy array
        Roads as a raster array with the value being the cost of traversing.
    affine : affine.Affine
        Affine raster transformation for the new raster (same as ntl_in).
    """

    ntl_rd = rasterio.open(ntl_in)
    shape = ntl_rd.read(1).shape
    affine = ntl_rd.transform

    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)
    roads_masked = gpd.read_file(roads_in, mask=aoi)
    roads = gpd.sjoin(roads_masked, aoi, how="inner", op="intersects")
    roads = roads[roads_masked.columns]

    roads["weight"] = 1
    roads.loc[roads["highway"] == "motorway", "weight"] = 1 / 10
    roads.loc[roads["highway"] == "trunk", "weight"] = 1 / 9
    roads.loc[roads["highway"] == "primary", "weight"] = 1 / 8
    roads.loc[roads["highway"] == "secondary", "weight"] = 1 / 7
    roads.loc[roads["highway"] == "tertiary", "weight"] = 1 / 6
    roads.loc[roads["highway"] == "unclassified", "weight"] = 1 / 5
    roads.loc[roads["highway"] == "residential", "weight"] = 1 / 4
    roads.loc[roads["highway"] == "service", "weight"] = 1 / 3

    # Power lines get weight 0
    #if "power" in roads:
    #    roads.loc[roads["power"] == "line", "weight"] = 0

    # Removes all rows with weight = 1, that is all roads which did not
    # match any of the categories above
    roads = roads[roads.weight != 1]

    # sort by weight descending so that lower weight (bigger roads) are
    # processed last and overwrite higher weight roads
    roads = roads.sort_values(by="weight", ascending=False)

    roads_for_raster = [(row.geometry, row.weight) for _, row in roads.iterrows()]
    roads_raster = rasterize(
        roads_for_raster,
        out_shape=shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=affine,
    )

    return roads_raster, affine

def prepare_powerlines(grid_in, targets_in, aoi_in):
    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    grid_masked = gpd.read_file(grid_in, mask=aoi)
    grid = gpd.sjoin(grid_masked, aoi, how="inner", op="intersects")
    grid = grid[grid_masked.columns]

    grid.voltage = grid.voltage.apply(lambda x: x.split(';')[0] if (type(x) == str) else x)
    grid.voltage = pd.to_numeric(grid.voltage, errors="coerce")
    grid = grid[grid["voltage"].notna()]
    grid["weight"] = 1

    grid.loc[grid["voltage"] > 70000, "weight"] = 0

    targets_reader = rasterio.open(targets_in)
    targets = targets_reader.read(1)

    grid_for_raster = [(row.geometry, row.weight) for _, row in grid.iterrows()]
    grid_raster = rasterize(
        grid_for_raster,
        out_shape=targets.shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=targets_reader.transform,
    )

    return grid_raster, targets_reader.transform

def prepare_groundtruth(grid_in, targets_in, aoi_in, buffer_amount=0.01):
    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    grid_masked = gpd.read_file(grid_in, mask=aoi)
    grid = gpd.sjoin(grid_masked, aoi, how="inner", op="intersects")
    grid = grid[grid_masked.columns]

    grid_buff = grid.buffer(buffer_amount)

    targets_reader = rasterio.open(targets_in)
    targets = targets_reader.read(1)

    grid_for_raster = [(row.geometry) for _, row in grid.iterrows()]
    grid_raster = rasterize(
        grid_for_raster,
        out_shape=targets_reader.shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=targets_reader.transform,
    )

    grid_buff_raster = rasterize(
        grid_buff,
        out_shape=targets_reader.shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=targets_reader.transform,
    )
    return grid_raster, grid_buff_raster, targets_reader.transform