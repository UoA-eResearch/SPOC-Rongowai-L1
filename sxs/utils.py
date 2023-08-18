import math
import numpy as np
from rasterio.windows import Window
from scipy.interpolate import interp1d
from datetime import datetime

# L = 18030
# grid_res = 30  # L may need to be updated in the future

# define constants once, used in LOCAL_DEM function
LOCAL_DEM_L = 90
LOCAL_DEM_RES = 30
LOCAL_DEM_MARGIN = 0
LOCAL_NUM_PIXELS = int(LOCAL_DEM_L / LOCAL_DEM_RES)
LOCAL_HALF_NP = int(LOCAL_NUM_PIXELS // 2)


def timeit(f):
    """timer decorator"""
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = f(*args, **kwargs)
        span = datetime.now() - start
        print(f"{f.__name__}: runtime {span}")
        return result
    return wrapper


def expand_to_RHCP(array, J_2, J):
    array[:, J_2:J] = array[:, 0:J_2]
    return array


def interp_ddm(x, y, x_ddm):
    """Interpolate DDM data onto new grid of points.

    Parameters
    ----------
    x : numpy.array()
        array of x values to create interpolation
    y : numpy.array()
        array of y values to create interpolation
    x_ddm : numpy.array()
        new x data to interpolate

    Returns
    -------
    y_ddm : numpy.array()
        interpolated y values corresponding to x_ddm
    """
    # regrid ddm data using 1d interpolator
    interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate")
    return interp_func(x_ddm)


def get_local_dem(sx_pos_lla, dem, dtu10, dist):
    lon_index = np.argmin(abs(dem["lon"] - sx_pos_lla[1]))
    lat_index = np.argmin(abs(dem["lat"] - sx_pos_lla[0]))

    local_lon = dem["lon"][lon_index - LOCAL_HALF_NP : lon_index + LOCAL_HALF_NP + 1]
    local_lat = dem["lat"][lat_index - LOCAL_HALF_NP : lat_index + LOCAL_HALF_NP + 1]

    if dist > LOCAL_DEM_MARGIN:
        local_ele = dem["ele"][
            lat_index - LOCAL_HALF_NP : lat_index + LOCAL_HALF_NP + 1,
            lon_index - LOCAL_HALF_NP : lon_index + LOCAL_HALF_NP + 1,
        ]
    else:
        local_ele = dtu10(
            (
                np.tile(local_lon, LOCAL_NUM_PIXELS),
                np.repeat(local_lat, LOCAL_NUM_PIXELS),
            )
        ).reshape(-1, LOCAL_NUM_PIXELS)

    return {"lat": local_lat, "lon": local_lon, "ele": local_ele}


def get_pek_value(lat, lon, water_mask):
    # minus 1 to account for 0-base indexing
    lat_index = math.ceil((water_mask["lat_max"] - lat) / water_mask["res_deg"]) - 1
    lon_index = math.ceil((lon - water_mask["lon_min"]) / water_mask["res_deg"]) - 1

    data = water_mask["file"].read(1, window=Window(lat_index, lon_index, 1, 1))
    return data


def get_surf_type2(P, cst_mask, lcv_mask, water_mask):
    # this function returns the surface type of a coordinate P <lat lon>
    # P[0] = lat, P[1] = lon
    landcover_type = get_landcover_type2(P[0], P[1], lcv_mask)

    lat_pek = int(abs(P[0]) // 10 * 10)
    lon_pek = int(abs(P[1]) // 10 * 10)

    file_id = str(lon_pek) + "E_" + str(lat_pek) + "S"
    # water_mask1 = water_mask[file_id]
    pek_value = get_pek_value(P[0], P[1], water_mask[file_id])

    dist_coast = cst_mask((P[1], P[0]))

    if all([pek_value > 0, landcover_type != -1, dist_coast > 0.5]):
        surface_type = 3  # not consistent with matlab code
        # surface_type = 0  # coordinate on inland water
    elif all([pek_value > 0, dist_coast < 0.5]):
        surface_type = -1
    else:
        surface_type = landcover_type

    return surface_type


def get_landcover_type2(lat_P, lon_P, lcv_mask):
    """% this function returns the landcover type of the coordinate P (lat lon)
    % over landsurface"""

    # bounding box is hardcoded, so N/M dimensions should be too...
    lat_max, lat_range, lat_M = -34, 13.5, 21000
    lat_res = lat_range / lat_M
    lon_min, lon_range, lon_N = 165.75, 13.5, 21000
    lon_res = lon_range / lon_N

    # -1 to account for 1-based (matlab) vs 0-base indexing
    lat_index = math.ceil((lat_max - lat_P) / lat_res) - 1
    lon_index = math.ceil((lon_P - lon_min) / lon_res) - 1

    lcv_RGB1 = lcv_mask.getpixel((lon_index, lat_index))
    # drop alpha channel in index 3
    lcv_RGB = tuple([z / 255 for z in lcv_RGB1[:3]])
    color = [
        (0.8, 0, 0.8),  # 1: artifical
        (0.6, 0.4, 0.2),  # 2: barely vegetated
        (0, 0, 1),  # 3: inland water
        (1, 1, 0),  # 4: crop
        (0, 1, 0),  # 5: grass
        (0.6, 0.2, 0),  # 6: shrub
        (0, 0.2, 0),  # 7: forest
    ]

    landcover_type = 0

    if sum(lcv_RGB) == 3:
        landcover_type = -1
    else:
        for idx, val in enumerate(color):
            if lcv_RGB == val:
                landcover_type = idx + 1  # match matlab indexes
            # else:
            #     raise Exception("landcover type not found")

    assert (
        landcover_type != 0
    ), f"landcover type not find. landcover_type = {landcover_type} lcv_RGB = {lcv_RGB}."

    return landcover_type
