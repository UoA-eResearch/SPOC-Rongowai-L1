# mike.laverick@auckland.ac.nz
#
# load_files.py
#
# Functions relating to the finding, loading, and processing of input files

import array
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d

# binary types for loading of gridded binary files
grid_type_list = [
    ("lat_min", "d"),
    ("lat_max", "d"),
    ("num_lat", "H"),
    ("lon_min", "d"),
    ("lon_max", "d"),
    ("num_lon", "H"),
]


# Removes masked data from 1D, D, & 4D NetCDF variables using their masks
def load_netcdf(netcdf_variable):
    if len(netcdf_variable.shape[:]) == 1:
        return netcdf_variable[:].compressed()
    if len(netcdf_variable.shape[:]) == 2:
        return np.ma.compress_rows(np.ma.masked_invalid(netcdf_variable[:]))
    if len(netcdf_variable.shape[:]) == 4:
        count_mask = ~netcdf_variable[:, 0, 0, 0].mask
        return netcdf_variable[count_mask, :, :, :]


def load_dat_file(file, typecode, size):
    value = array.array(typecode)
    value.fromfile(file, size)
    if size == 1:
        return value.tolist()[0]
    else:
        return value.tolist()


def load_antenna_pattern(filename):
    with open(filename, "rb") as f:
        ignore_values = load_dat_file(f, "d", 5)
        ant_data = load_dat_file(f, "d", 3601 * 1201)
    return np.reshape(ant_data, (-1, 3601))


def get_orbit_file(gps_week, gps_tow, start_obj, end_obj, change_idx=0):
    orbit_path = Path().absolute().joinpath(Path("./dat/orbits/"))
    gps_week1, gps_dow1 = int(gps_week[0]), int(gps_tow[0] // 86400)
    sp3_filename1 = (
        "IGS0OPSRAP_"
        + str(start_obj.year)
        + str(start_obj.timetuple().tm_yday)
        + "0000_01D_15M_ORB.sp3"
    )
    sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
    if not os.path.isfile(sp3_filename1_full):
        sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".SP3"
        sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
        if not os.path.isfile(sp3_filename1_full):
            sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".sp3"
            sp3_filename1_full = orbit_path.joinpath(Path(sp3_filename1))
            if not os.path.isfile(sp3_filename1_full):
                raise Exception("Orbit file not found...")
    if change_idx:
        # substitute in last gps_week/gps_tow values as first, end_obj as start_obj
        sp3_filename2_full = get_orbit_file(
            gps_week[-1:], gps_tow[-1:], end_obj, end_obj, change_idx=0
        )
        return sp3_filename1_full, sp3_filename2_full
    return sp3_filename1_full


def load_dat_file_grid(filepath):
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
    interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate")
    return interp_func(x_ddm)
