# mike.laverick@auckland.ac.nz
# load_files.py
# Functions relating to the finding, loading, and processing of input files

import array
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d, interpn

# binary types for loading of gridded binary files
grid_type_list = [
    ("lat_min", "d"),
    ("lat_max", "d"),
    ("num_lat", "H"),
    ("lon_min", "d"),
    ("lon_max", "d"),
    ("num_lon", "H"),
]

# define constants once, used in LOCAL_DEM function
LOCAL_DEM_L = 90
LOCAL_DEM_RES = 30
LOCAL_DEM_MARGIN = 0
LOCAL_NUM_PIXELS = int(LOCAL_DEM_L / LOCAL_DEM_RES)
LOCAL_HALF_NP = int(LOCAL_NUM_PIXELS // 2)


def load_netcdf(netcdf_variable):
    """Unpack netcdf variable to python variable.
       Removes masked rows from 1D, 2D, & 4D NetCDF variables.
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
        + str(start_obj.timetuple().tm_yday)
        + "0000_01D_15M_ORB.sp3"
    )
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
    return {
        "lat": np.linspace(temp["lat_min"], temp["lat_max"], temp["num_lat"]),
        "lon": np.linspace(temp["lon_min"], temp["lon_max"], temp["num_lon"]),
        "ele": np.reshape(map_data, (-1, temp["num_lat"])),
    }


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

    lon_index = np.argmin(abs(dem["lon"] - sx_pos_lla[0]))
    lat_index = np.argmin(abs(dem["lat"] - sx_pos_lla[1]))

    local_lon = dem["lon"][lon_index - LOCAL_HALF_NP : lon_index + LOCAL_HALF_NP + 1]
    local_lat = dem["lat"][lat_index - LOCAL_HALF_NP : lat_index + LOCAL_HALF_NP + 1]

    if dist > LOCAL_DEM_MARGIN:
        local_ele = dem["ele"][
            lon_index - LOCAL_HALF_NP : lon_index + LOCAL_HALF_NP + 1,
            lat_index - LOCAL_HALF_NP : lat_index + LOCAL_HALF_NP + 1,
        ]
    else:

        local_ele = interpn(
            points=(dtu10["lon"], dtu10["lat"]),
            values=dtu10["ele"],
            xi=(
                np.tile(local_lon, LOCAL_NUM_PIXELS),
                np.repeat(local_lat, LOCAL_NUM_PIXELS),
            ),
            method="linear",
        ).reshape(-1, LOCAL_NUM_PIXELS)

    return {"lat": local_lat, "lon": local_lon, "ele": local_ele}
