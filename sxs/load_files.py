# mike.laverick@auckland.ac.nz
# load_files.py
# Functions relating to the finding, loading, and processing of input files

import netCDF4
import array
import math
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.interpolate import interp1d, RegularGridInterpolator
from rasterio.windows import Window

# binary types for loading of gridded binary files
grid_type_list = [
    ("lat_min", "d"),
    ("lat_max", "d"),
    ("num_lat", "H"),
    ("lon_min", "d"),
    ("lon_max", "d"),
    ("num_lon", "H"),
]

# L = 18030
# grid_res = 30  # L may need to be updated in the future

# define constants once, used in LOCAL_DEM function
LOCAL_DEM_L = 90
LOCAL_DEM_RES = 30
LOCAL_DEM_MARGIN = 0
LOCAL_NUM_PIXELS = int(LOCAL_DEM_L / LOCAL_DEM_RES)
LOCAL_HALF_NP = int(LOCAL_NUM_PIXELS // 2)


def load_netcdf(netcdf_variable):
    """Unpack netcdf variable to python variable.
       Removes masked rows from 1D, 2D, & 4D NetCDF variables.
       The masks smuggle NaN values into the code which throw off things.
    Parameters
    ----------
    netcdf4.variable
        Specified variable from a netcdf4 dataset

    Returns
    -------
    netcdf_variable as N-D numpy.array
    """
    if len(netcdf_variable.shape[:]) == 1:
        return netcdf_variable[:].compressed()
    if len(netcdf_variable.shape[:]) == 2:
        return np.ma.compress_rows(np.ma.masked_invalid(netcdf_variable[:]))
    if len(netcdf_variable.shape[:]) == 4:
        # note: this results in a masked array that needs special treatment
        # before use with scipy
        count_mask = ~netcdf_variable[:, 0, 0, 0].mask
        return netcdf_variable[count_mask, :, :, :]


# function to load a specified type of binary data from  file
def load_dat_file(file, typecode, size):
    """Load data from generic binary dat file.

    Parameters
    ----------
    file : open() instance
        Opened file instance to read from
    typecode : str
        String designation for byte type code
    size : int
        Number of byte code types to read.

    Returns
    -------
    List of bytecode variables
    """
    value = array.array(typecode)
    value.fromfile(file, size)
    if size == 1:
        return value.tolist()[0]
    else:
        return value.tolist()


# load antenna binary files
def load_antenna_pattern(filepath):
    """Load data from antenna pattern dat file.

    Parameters
    ----------
    filepath : pathlib.Path
        path to file

    Returns
    -------
    2D numpy.array of antenna pattern data
    """
    with open(filepath, "rb") as f:
        ignore_values = load_dat_file(f, "d", 5)
        ant_data = load_dat_file(f, "d", 3601 * 1201)
    return np.reshape(ant_data, (-1, 3601))


# load antenna binary files
def load_A_phy_LUT(filepath):
    """This function retrieves A_phy LUT and the input matrices

    Parameters
    ----------
    filepath : pathlib.Path
        path to file

    Returns
    -------
    rx_alt_bins, inc_angle_bins, az_angle_bins, A_phy_LUT_all
    """
    with open(filepath, "rb") as f:
        min_rx_alt = load_dat_file(f, "H", 1)  # uint16
        res_rx_alt = load_dat_file(f, "H", 1)
        num_rx_alt = load_dat_file(f, "H", 1)

        min_inc_angle = load_dat_file(f, "H", 1)
        res_inc_angle = load_dat_file(f, "H", 1)
        num_inc_angle = load_dat_file(f, "H", 1)

        min_az_angle = load_dat_file(f, "H", 1)
        res_az_angle = load_dat_file(f, "H", 1)
        num_az_angle = load_dat_file(f, "H", 1)

        rx_alt_bins = (np.asarray(range(num_rx_alt)) * res_rx_alt) + min_rx_alt
        inc_angle_bins = (
            np.asarray(range(num_inc_angle)) * res_inc_angle
        ) + min_inc_angle
        az_angle_bins = (np.asarray(range(num_az_angle)) * res_az_angle) + min_az_angle

        A_phy_LUT_all = np.full(
            [num_rx_alt, num_inc_angle, num_az_angle, 7, 41], np.nan
        )

        for m in range(num_rx_alt):
            for n in range(num_inc_angle):
                for k in range(num_az_angle):
                    data = np.reshape(
                        load_dat_file(f, "I", 7 * 41), (41, 7)
                    ).T  # uint32
                    A_phy_LUT_all[m, n, k] = data

    return rx_alt_bins, RegularGridInterpolator(
        (rx_alt_bins, inc_angle_bins, az_angle_bins), A_phy_LUT_all, bounds_error=True
    )
    # return rx_alt_bins, inc_angle_bins, az_angle_bins, A_phy_LUT_all


# calculate which orbit file to load
# TODO automate retrieval of orbit files for new days
def get_orbit_file(gps_week, gps_tow, start_obj, end_obj, change_idx=0):
    """Determine which orbital file to use based upon gps_week and gps_tow.

    Parameters
    ----------
    gps_week : int
        GPS week number, i.e. 1866.
    gps_tow : int
        Number of seconds since the beginning of week.
    start_obj : str
        String representation of datetime of start of flight segment
    end_obj : str
        String representation of datetime of end of flight segment

    Optional parameters
    ----------
    change_idx : int
        Index of change of day in gps_tow. Default = 0

    Returns
    -------
    sp3_filename1_full: pathlib.Path
    sp3_filename2_full: pathlib.Path

    """
    orbit_path = Path().absolute().joinpath(Path("./dat/orbits/"))
    # determine gps_week and day of the week (1-7)
    gps_week1, gps_dow1 = int(gps_week[0]), int(gps_tow[0] // 86400)
    # try loading in latest file name for data
    sp3_filename1 = (
        "IGS0OPSRAP_"
        + str(start_obj.year)
        + "{:03d}".format(start_obj.timetuple().tm_yday)  # match for the dropbox data
        + "0000_01D_15M_ORB.SP3"
    )
    month_year = start_obj.strftime("%B %Y")
    sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
    if not os.path.isfile(sp3_filename1_full):
        # try loading in alternate name
        sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".SP3"
        sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
        if not os.path.isfile(sp3_filename1_full):
            # try loading in ealiest format name
            sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".sp3"
            sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
            if not os.path.isfile(sp3_filename1_full):
                # TODO implement a mechanism for last valid file?
                raise Exception("Orbit file not found...")
    if change_idx:
        # if change_idx then also determine the day priors orbit file and return both
        # substitute in last gps_week/gps_tow values as first, end_obj as start_obj
        sp3_filename2_full = get_orbit_file(
            gps_week[-1:], gps_tow[-1:], end_obj, end_obj, change_idx=0
        )
        return sp3_filename1_full, sp3_filename2_full
    return sp3_filename1_full


# load in map data binary files
def load_dat_file_grid(filepath):
    """Load data from geospatial dat file.

    Parameters
    ----------
    filepath : pathlib.Path
        path to file

    Returns
    -------
    dict containing the following:
       "lat" 1D numpy array of latitude coordinates
       "lon" 1D numpy array of longitude coordinates
       "ele" 2D numpy array of elevations at lat/lon coordinates
    """
    # type_list = [(lat_min,"d"),(num_lat,"H"), etc] + omit last grid type
    temp = {}
    with open(filepath, "rb") as f:
        for field, field_type in grid_type_list:
            temp[field] = load_dat_file(f, field_type, 1)
        map_data = load_dat_file(f, "d", temp["num_lat"] * temp["num_lon"])
    data = {
        "lat": np.linspace(temp["lat_min"], temp["lat_max"], temp["num_lat"]),
        "lon": np.linspace(temp["lon_min"], temp["lon_max"], temp["num_lon"]),
        "ele": np.reshape(map_data, (-1, temp["num_lat"])),
    }

    # create and return interpolator model for the grid file
    return RegularGridInterpolator(
        (data["lon"], data["lat"]), data["ele"], bounds_error=True
    )


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
        # surface_type = 3  # not consistent with matlab code
        surface_type = 0  # coordinate on inland water
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
    lon_res = lat_range / lon_N

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


def get_datatype(data_series, value=None):
    datatype = data_series["Data_type"].values[0]
    if datatype == "single":
        return np.single
    elif datatype == "double":
        return np.double
    elif datatype == "int8":
        return np.int8
    elif datatype == "int16":
        return np.int16
    elif datatype == "int32":
        return np.int32
    elif datatype == "int64":
        return np.int64
    elif datatype == "uint8":
        return np.uint8
    elif datatype == "uint16":
        return np.uint16
    elif datatype == "uint32":
        return np.uint32
    elif datatype == "uint64":
        return np.uint64
    elif datatype == "string":
        if isinstance(value, str):
            return "S" + str(len(value))
    else:
        raise Exception(f"datatype '{datatype}' not supported")


def get_dimensions(data_series):
    dim = data_series["Dimensions"].values[0].split(",")
    return tuple([x.strip() for x in dim])


def write_netcdf(dict_in, definition_file, output_file):
    assert isinstance(dict_in, dict), "input must be a dictionary"
    assert (
        Path(definition_file).suffix == ".xlsx"
    ), "definition file must be a .xlsx file"

    # read definition file
    df = pd.read_excel(definition_file)

    # open netcdf file
    with netCDF4.Dataset(output_file, mode="w") as ncfile:
        # create dimensions
        ncfile.createDimension("sample", None)
        ncfile.createDimension("ddm", None)
        ncfile.createDimension("delay", None)
        ncfile.createDimension("doppler", None)

        for k, v in dict_in.items():
            print("writing: ", k)
            ds_k = df[df["Name"] == k]

            if ds_k.empty:
                print(
                    f"Warning: variable {k} not found in definition file, skip this variable."
                )
                continue
            elif len(ds_k) > 1:
                print(
                    f"Warning: find multiple variable {k} definition in definition file, skip this variable."
                )
                continue

            # if ds_k["Data_type"].str.contains("attribute").any():  # attribute
            if ds_k["Dimensions"].item() == "<none>":
                if ds_k["Units"].item() == "<none>":  # scalar
                    ncfile.setncattr(k, str(v))
                else:
                    var_k = ncfile.createVariable(
                        k, get_datatype(ds_k, v), (), zlib=True
                    )
                    var_k.units = ds_k["Units"].values[0]
                    var_k.long_name = ds_k["Long_name"].values[0]
                    var_k.comment = ds_k["Comment"].values[0]
                    var_k[()] = v
            else:  # variable
                var_k = ncfile.createVariable(
                    k, get_datatype(ds_k), get_dimensions(ds_k), zlib=True
                )
                var_k.units = ds_k["Units"].values[0]
                var_k.long_name = ds_k["Long_name"].values[0]
                var_k.comment = ds_k["Comment"].values[0]
                if len(get_dimensions(ds_k)) == len(v.shape) == 1:
                    var_k[:] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 2:
                    var_k[:, :] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 3:
                    var_k[:, :, :] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 4:
                    var_k[:, :, :, :] = v
                elif (
                    len(get_dimensions(ds_k)) == 3
                    and len(v.shape) == 4
                    and v.shape[3] == 1
                ):  # norm_refl_waveform
                    var_k[:, :, :] = np.squeeze(v, axis=3)
                else:
                    raise Exception(f"variable {k} has unsupported dimensions")

        # print the Dataset object to see what we've got
        print(ncfile)
