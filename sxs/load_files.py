# mike.laverick@auckland.ac.nz
# load_files.py
# Functions relating to the finding, loading, and processing of input files

import netCDF4 as nc
import array
import numpy as np
import os
from pathlib import Path
from PIL import Image
import rasterio
from scipy.interpolate import interp1d, RegularGridInterpolator

# Required to load the land cover mask file
Image.MAX_IMAGE_PIXELS = None

# binary types for loading of gridded binary files
grid_type_list = [
    ("lat_min", "d"),
    ("lat_max", "d"),
    ("num_lat", "H"),
    ("lon_min", "d"),
    ("lon_max", "d"),
    ("num_lon", "H"),
]


class L0_file:
    """
    Basic class to hold variables from the input L0 file
    """

    def __init__(self, filename):
        ds = nc.Dataset(filename)
        # load in rx-related variables
        # PVT GPS week and sec
        self.pvt_gps_week = load_netcdf(ds["/science/GPS_week_of_SC_attitude"])
        self.pvt_gps_sec = load_netcdf(ds["/science/GPS_second_of_SC_attitude"])
        # rx positions in ECEF, metres
        self.rx_pos_x_pvt = load_netcdf(ds["/geometry/receiver/rx_position_x_ecef_m"])
        self.rx_pos_y_pvt = load_netcdf(ds["/geometry/receiver/rx_position_y_ecef_m"])
        self.rx_pos_z_pvt = load_netcdf(ds["/geometry/receiver/rx_position_z_ecef_m"])
        # rx velocity in ECEF, m/s
        self.rx_vel_x_pvt = load_netcdf(ds["/geometry/receiver/rx_velocity_x_ecef_mps"])
        self.rx_vel_y_pvt = load_netcdf(ds["/geometry/receiver/rx_velocity_y_ecef_mps"])
        self.rx_vel_z_pvt = load_netcdf(ds["/geometry/receiver/rx_velocity_z_ecef_mps"])
        # rx attitude, deg | TODO this is actually radians and will be updated
        self.rx_pitch_pvt = load_netcdf(ds["/geometry/receiver/rx_attitude_pitch_deg"])
        self.rx_roll_pvt = load_netcdf(ds["/geometry/receiver/rx_attitude_roll_deg"])
        self.rx_yaw_pvt = load_netcdf(ds["/geometry/receiver/rx_attitude_yaw_deg"])
        # rx clock bias and drifts
        self.rx_clk_bias_m_pvt = load_netcdf(ds["/geometry/receiver/rx_clock_bias_m"])
        self.rx_clk_drift_mps_pvt = load_netcdf(
            ds["/geometry/receiver/rx_clock_drift_mps"]
        )

        # TODO: Some processing required here to fix leading/trailing/sporadic "zero" values?

        # load in ddm-related variables
        # tx ID/satellite PRN
        self.transmitter_id = load_netcdf(ds["/science/ddm/transmitter_id"])
        # raw counts and ddm parameters
        self.first_scale_factor = load_netcdf(ds["/science/ddm/first_scale_factor"])
        # raw counts, uncalibrated
        self.raw_counts = load_netcdf(ds["/science/ddm/counts"])
        self.zenith_i2q2 = load_netcdf(ds["/science/ddm/zenith_i2_plus_q2"])
        self.rf_source = load_netcdf(ds["/science/ddm/RF_source"])
        # binning standard deviation
        self.std_dev_rf1 = load_netcdf(ds["/science/ddm/RF1_zenith_RHCP_std_dev"])
        self.std_dev_rf2 = load_netcdf(ds["/science/ddm/RF2_nadir_LHCP_std_dev"])
        self.std_dev_rf3 = load_netcdf(ds["/science/ddm/RF3_nadir_RHCP_std_dev"])

        # delay bin resolution
        self.delay_bin_res = load_netcdf(ds["/science/ddm/delay_bin_res_narrow"])
        self.delay_bin_res = self.delay_bin_res[~np.isnan(self.delay_bin_res)][0]
        # doppler bin resolution
        self.doppler_bin_res = load_netcdf(ds["/science/ddm/doppler_bin_res_narrow"])
        self.doppler_bin_res = self.doppler_bin_res[~np.isnan(self.doppler_bin_res)][0]

        # delay and Doppler center bin
        self.center_delay_bin = load_netcdf(ds["/science/ddm/ddm_center_delay_bin"])
        self.center_delay_bin = self.center_delay_bin[~np.isnan(self.center_delay_bin)][
            0
        ]
        self.center_doppler_bin = load_netcdf(ds["/science/ddm/ddm_center_doppler_bin"])
        self.center_doppler_bin = self.center_doppler_bin[
            ~np.isnan(self.center_doppler_bin)
        ][0]

        # absolute ddm center delay and doppler
        self.delay_center_chips = load_netcdf(
            ds["/science/ddm/center_delay_bin_code_phase"]
        )
        self.doppler_center_hz = load_netcdf(
            ds["/science/ddm/center_doppler_bin_frequency"]
        )

        # coherent duration and noncoherent integration
        self.coherent_duration = (
            load_netcdf(ds["/science/ddm/L1_E1_coherent_duration"]) / 1000
        )
        self.non_coherent_integrations = (
            load_netcdf(ds["/science/ddm/L1_E1_non_coherent_integrations"]) / 1000
        )

        # NGRx estimate additional delay path
        self.add_range_to_sp_pvt = load_netcdf(
            ds["/science/ddm/additional_range_to_SP"]
        )

        # antenna temperatures and engineering timestamp
        self.eng_timestamp = load_netcdf(ds["/eng/packet_creation_time"])
        self.zenith_ant_temp_eng = load_netcdf(ds["/eng/zenith_ant_temp"])
        self.nadir_ant_temp_eng = load_netcdf(ds["/eng/nadir_ant_temp"])

        # Define important file shape/ength variables
        self.I = self.transmitter_id.shape[0]
        self.J = self.transmitter_id.shape[1]
        self.J_2 = int(self.J / 2)
        self.shape_2d = self.transmitter_id.shape
        self.shape_4d = self.raw_counts.shape


class input_files:
    """
    Basic class to hold data from input files
    """

    def __init__(
        self,
        L1a_cal_ddm_counts_db_filename,
        L1a_cal_ddm_power_dbm_filename,
        dem_filename,
        dtu_filename,
        landmask_filename,
        lcv_filename,
        water_mask_paths,
        pek_path,
        SV_PRN_filename,
        SV_eirp_filename,
        rng_filenames,
        A_phy_LUT_path,
    ):
        self.L1a_cal_ddm_counts_db = np.loadtxt(L1a_cal_ddm_counts_db_filename)
        self.L1a_cal_ddm_power_dbm = np.loadtxt(L1a_cal_ddm_power_dbm_filename)

        # create the interpolation functions for the 3 ports
        self.L1a_cal_1dinterp = {}
        for i in range(3):
            self.L1a_cal_1dinterp[i] = interp1d(
                self.L1a_cal_ddm_counts_db[i, :],
                self.L1a_cal_ddm_power_dbm[i, :],
                kind="cubic",
                fill_value="extrapolate",
            )

        self.dem = rasterio.open(dem_filename)
        self.dem = {
            "ele": self.dem.read(1),
            "lat": np.linspace(
                self.dem.bounds.top, self.dem.bounds.bottom, self.dem.height
            ),
            "lon": np.linspace(
                self.dem.bounds.left, self.dem.bounds.right, self.dem.width
            ),
        }

        self.dtu10 = load_dat_file_grid(dtu_filename)
        self.landmask_nz = load_dat_file_grid(landmask_filename)
        self.lcv_mask = Image.open(lcv_filename)

        self.water_mask = {}
        for path in water_mask_paths:
            self.water_mask[path] = {}
            pek_file = rasterio.open(pek_path.joinpath("occurrence_" + path + ".tif"))
            self.water_mask[path]["lon_min"] = pek_file._transform[0]
            self.water_mask[path]["res_deg"] = pek_file._transform[1]
            self.water_mask[path]["lat_max"] = pek_file._transform[3]
            self.water_mask[path]["file"] = pek_file

        self.SV_PRN_LUT = np.loadtxt(SV_PRN_filename, usecols=(0, 1))
        self.SV_eirp_LUT = np.loadtxt(SV_eirp_filename)

        self.LHCP_pattern = {
            "LHCP": load_antenna_pattern(rng_filenames[0]),
            "RHCP": load_antenna_pattern(rng_filenames[1]),
        }
        self.RHCP_pattern = {
            "LHCP": load_antenna_pattern(rng_filenames[2]),
            "RHCP": load_antenna_pattern(rng_filenames[3]),
        }
        self.rx_alt_bins, self.A_phy_LUT_interp = load_A_phy_LUT(A_phy_LUT_path)


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
def load_orbit_file(gps_week, gps_tow, start_obj, end_obj, change_idx=0):
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
        sp3_filename2_full = load_orbit_file(
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
