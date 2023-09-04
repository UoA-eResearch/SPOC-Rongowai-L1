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
import urllib
import os
from http.cookiejar import CookieJar
import gzip
import shutil

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
    Basic class to load in, preprocess, and store variables from the input L0 file.
    rx-related variables contain zeros as
    """

    def __init__(self, filename):
        ds = nc.Dataset(filename)

        # load in nan indexes (in mask form) for eng/sci/ddm fields
        self.eng_nans = ds["/eng/packet_creation_time"][:].mask
        self.sci_nans = ds["/science/clk_bias_rate_pvt"][:].mask
        self.ddm_nans = ds["/science/ddm/delay_bin_res_narrow"][:].mask
        # combine masks OR style to find total mask for all data
        self.mask = np.ma.mask_or(self.eng_nans, self.sci_nans)
        self.mask = np.ma.mask_or(self.mask, self.ddm_nans)
        # load in rx-related variables
        # PVT GPS week and sec
        self.pvt_gps_week = self.compress(ds["/science/GPS_week_of_SC_attitude"])
        self.pvt_gps_sec = self.compress(ds["/science/GPS_second_of_SC_attitude"])
        # rx positions in ECEF, metres
        self.rx_pos_x_pvt = self.compress(ds["/geometry/receiver/rx_position_x_ecef_m"])
        self.rx_pos_y_pvt = self.compress(ds["/geometry/receiver/rx_position_y_ecef_m"])
        self.rx_pos_z_pvt = self.compress(ds["/geometry/receiver/rx_position_z_ecef_m"])
        # rx velocity in ECEF, m/s
        self.rx_vel_x_pvt = self.compress(
            ds["/geometry/receiver/rx_velocity_x_ecef_mps"]
        )
        self.rx_vel_y_pvt = self.compress(
            ds["/geometry/receiver/rx_velocity_y_ecef_mps"]
        )
        self.rx_vel_z_pvt = self.compress(
            ds["/geometry/receiver/rx_velocity_z_ecef_mps"]
        )
        # rx attitude, rad
        self.rx_pitch_pvt = self.compress(ds["/geometry/receiver/rx_attitude_pitch"])
        self.rx_roll_pvt = self.compress(ds["/geometry/receiver/rx_attitude_roll"])
        self.rx_heading_pvt = self.compress(
            ds["/geometry/receiver/rx_attitude_heading"]
        )
        # rx clock bias and drifts
        self.rx_clk_bias_m_pvt = self.compress(ds["/geometry/receiver/rx_clock_bias_m"])
        self.rx_clk_drift_mps_pvt = self.compress(
            ds["/geometry/receiver/rx_clock_drift_mps"]
        )

        # self.remove_leading_trailing_zeros()
        # self.interpolate_zero_values()

        # load in ddm-related variables
        # tx ID/satellite PRN
        self.transmitter_id = self.compress(ds["/science/ddm/transmitter_id"])
        # raw counts and ddm parameters
        self.first_scale_factor = self.compress(ds["/science/ddm/first_scale_factor"])
        # raw counts, uncalibrated
        self.raw_counts = self.compress(ds["/science/ddm/counts"])
        self.zenith_i2q2 = self.compress(ds["/science/ddm/zenith_i2_plus_q2"])
        self.rf_source = self.compress(ds["/science/ddm/RF_source"])
        # binning standard deviation
        self.std_dev_rf1 = self.compress(ds["/science/ddm/RF1_zenith_RHCP_std_dev"])
        self.std_dev_rf2 = self.compress(ds["/science/ddm/RF2_nadir_LHCP_std_dev"])
        self.std_dev_rf3 = self.compress(ds["/science/ddm/RF3_nadir_RHCP_std_dev"])

        # delay bin resolution
        self.delay_bin_res = self.compress(ds["/science/ddm/delay_bin_res_narrow"])
        self.delay_bin_res = self.delay_bin_res[~np.isnan(self.delay_bin_res)][0]
        # doppler bin resolution
        self.doppler_bin_res = self.compress(ds["/science/ddm/doppler_bin_res_narrow"])
        self.doppler_bin_res = self.doppler_bin_res[~np.isnan(self.doppler_bin_res)][0]

        # delay and Doppler center bin
        self.center_delay_bin = self.compress(ds["/science/ddm/ddm_center_delay_bin"])
        self.center_delay_bin = self.center_delay_bin[~np.isnan(self.center_delay_bin)][
            0
        ]
        self.center_doppler_bin = self.compress(
            ds["/science/ddm/ddm_center_doppler_bin"]
        )
        self.center_doppler_bin = self.center_doppler_bin[
            ~np.isnan(self.center_doppler_bin)
        ][0]

        # absolute ddm center delay and doppler
        self.delay_center_chips = self.compress(
            ds["/science/ddm/center_delay_bin_code_phase"]
        )
        self.doppler_center_hz = self.compress(
            ds["/science/ddm/center_doppler_bin_frequency"]
        )

        # coherent duration and noncoherent integration
        self.coherent_duration = (
            self.compress(ds["/science/ddm/L1_E1_coherent_duration"]) / 1000
        )
        self.non_coherent_integrations = (
            self.compress(ds["/science/ddm/L1_E1_non_coherent_integrations"]) / 1000
        )

        # NGRx estimate additional delay path
        self.add_range_to_sp_pvt = self.compress(
            ds["/science/ddm/additional_range_to_SP"]
        )

        # antenna temperatures and engineering timestamp
        self.eng_timestamp = self.compress(ds["/eng/packet_creation_time"])
        self.zenith_ant_temp_eng = self.compress(ds["/eng/zenith_ant_temp"])
        self.nadir_ant_temp_eng = self.compress(ds["/eng/nadir_ant_temp"])

        # Define important file shape/ength variables
        self.I = self.transmitter_id.shape[0]
        self.J = self.transmitter_id.shape[1]
        self.J_2 = int(self.J / 2)
        self.shape_2d = self.transmitter_id.shape
        self.shape_4d = self.raw_counts.shape

    def remove_leading_trailing_zeros(self):
        """
        This function removes leading/trailing zeros, replaces sporadic
        zeros with interpolated values
        """
        # get index of first and last zeros using a bit of Python array magic
        # idx_min = first non-zero index, idx_max = index of first trailing zero
        # thus end of range that we need to specify
        temp = np.array(self.pvt_gps_week)
        temp != 0
        idx_min, idx_max = temp.argmax(), temp.size - temp[::-1].argmax()

        self.pvt_gps_week = self.pvt_gps_week[idx_min:idx_max]
        self.pvt_gps_sec = self.pvt_gps_sec[idx_min:idx_max]
        self.rx_pos_x_pvt = self.rx_pos_x_pvt[idx_min:idx_max]
        self.rx_pos_y_pvt = self.rx_pos_y_pvt[idx_min:idx_max]
        self.rx_pos_z_pvt = self.rx_pos_z_pvt[idx_min:idx_max]
        self.rx_vel_x_pvt = self.rx_vel_x_pvt[idx_min:idx_max]
        self.rx_vel_y_pvt = self.rx_vel_y_pvt[idx_min:idx_max]
        self.rx_vel_z_pvt = self.rx_vel_z_pvt[idx_min:idx_max]
        self.rx_roll_pvt = self.rx_roll_pvt[idx_min:idx_max]
        self.rx_pitch_pvt = self.rx_pitch_pvt[idx_min:idx_max]
        self.rx_heading_pvt = self.rx_heading_pvt[idx_min:idx_max]
        self.rx_clk_bias_m_pvt = self.rx_clk_bias_m_pvt[idx_min:idx_max]
        self.rx_clk_drift_mps_pvt = self.rx_clk_drift_mps_pvt[idx_min:idx_max]

    def interpolate_zero_values(self):
        """
        This function finds the zero values in the L0 craft data and
        replaces them with simple interpolations from nearest non-zero values.
        Note: the matlab code only interpolates from +/- 1 index, which may produce
        poor results if two zeros occur next to each other.
        """

        def interp_funct(array):
            # 1D array of all indexes for linear interp 0:len(array)
            x = np.arange(len(array))
            # indexes of non-zero values that we wish to interp
            idx = np.where(array != 0)[0]
            # if all values are zero, just return the original array
            # TODO not sure how this will handle sparse data...
            # Nor how well the Matlab implementation would!
            if not idx.any():
                return array
            # set up interp function using non-zero indexes and values
            f = interp1d(x[idx], array[idx])
            # return interpolated values for 1D array of all indexes 0:len(array)
            return f(x)

        self.pvt_gps_week = interp_funct(self.pvt_gps_week)
        self.pvt_gps_sec = interp_funct(self.pvt_gps_sec)
        self.rx_pos_x_pvt = interp_funct(self.rx_pos_x_pvt)
        self.rx_pos_y_pvt = interp_funct(self.rx_pos_y_pvt)
        self.rx_pos_z_pvt = interp_funct(self.rx_pos_z_pvt)
        self.rx_vel_x_pvt = interp_funct(self.rx_vel_x_pvt)
        self.rx_vel_y_pvt = interp_funct(self.rx_vel_y_pvt)
        self.rx_vel_z_pvt = interp_funct(self.rx_vel_z_pvt)
        self.rx_roll_pvt = interp_funct(self.rx_roll_pvt)
        self.rx_pitch_pvt = interp_funct(self.rx_pitch_pvt)
        self.rx_heading_pvt = interp_funct(self.rx_heading_pvt)
        self.rx_clk_bias_m_pvt = interp_funct(self.rx_clk_bias_m_pvt)
        self.rx_clk_drift_mps_pvt = interp_funct(self.rx_clk_drift_mps_pvt)

    def compress(self, netcdf_variable):
        """compress netcdf masked array variable.
        Removes masked rows from 1D, 2D, & 4D NetCDF variables.
        The masks smuggle NaN values into the code which throw off things,
        so we compress the arrays to drop NaN values
        Parameters
        ----------
        netcdf4.variable
            Specified variable from a netcdf4 dataset

        Returns
        -------
        netcdf_variable as N-D numpy.array
        """
        if len(netcdf_variable.shape[:]) == 1:
            return netcdf_variable[~self.mask].compressed()
        if len(netcdf_variable.shape[:]) == 2:
            return np.ma.compress_rows(
                np.ma.masked_invalid(netcdf_variable[~self.mask])
            )
        if len(netcdf_variable.shape[:]) == 4:
            # note: this results in a masked array that needs special treatment
            # before use with scipy
            return netcdf_variable[~self.mask, :, :, :]


class input_files:
    """
    Basic class to load in and set up interpolator functions for input files
    """

    def __init__(
        self,
        L1a_cal_ddm_counts_filename,
        L1a_cal_ddm_power_filename,
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
        orbit_path,
    ):
        self.L1a_cal_ddm_counts = np.loadtxt(L1a_cal_ddm_counts_filename)
        self.L1a_cal_ddm_power = np.loadtxt(L1a_cal_ddm_power_filename)

        # create the interpolation functions for the 3 ports
        self.L1a_cal_1dinterp = {}
        # adjustments to LHCP and RHCP values - June 27th 2023
        offsets = [0, 0, 0]
        for i in range(1, 3):  # 0 is all nan
            self.L1a_cal_1dinterp[i] = interp1d(
                self.L1a_cal_ddm_counts[i, :],
                self.L1a_cal_ddm_power[i, :] + offsets[i],
                kind="cubic",
                fill_value="extrapolate",
            )

        self.dem = load_dem_file(dem_filename)
        # self.dem = rasterio.open(dem_filename)
        # self.dem = {
        #    "ele": self.dem.read(1),
        #    "lat": np.linspace(
        #        self.dem.bounds.top, self.dem.bounds.bottom, self.dem.height
        #    ),
        #    "lon": np.linspace(
        #        self.dem.bounds.left, self.dem.bounds.right, self.dem.width
        #    ),
        # }

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

        # scattering area LUT - load in 4D array from range of files
        A_PHY_LUT_paths = {
            "A_phy_LUT_-0.5.dat": -0.5,
            "A_phy_LUT_-0.4.dat": -0.4,
            "A_phy_LUT_-0.3.dat": -0.3,
            "A_phy_LUT_-0.2.dat": -0.2,
            "A_phy_LUT_-0.1.dat": -0.1,
            "A_phy_LUT_0.0.dat": 0.0,
            "A_phy_LUT_0.1.dat": 0.1,
            "A_phy_LUT_0.2.dat": 0.2,
            "A_phy_LUT_0.3.dat": 0.3,
            "A_phy_LUT_0.4.dat": 0.4,
            "A_phy_LUT_0.5.dat": 0.5,
        }
        A_PHY_LUT_INPUTS = A_phy_LUT_path.joinpath(Path("input_variables.dat"))
        (
            self.rx_alt_bins,
            self.inc_angle_bins,
            self.az_angle_bins,
        ) = load_A_phy_LUT_inputs(A_PHY_LUT_INPUTS)
        self.A_phy_LUT_all = []
        for filename, value in A_PHY_LUT_paths.items():
            A_PHY_LUT_DATA = A_phy_LUT_path.joinpath(Path(filename))
            LUT = load_A_phy_LUT(
                A_PHY_LUT_DATA,
                self.rx_alt_bins,
                self.inc_angle_bins,
                self.az_angle_bins,
            )
            self.A_phy_LUT_all.append({"LUT": LUT, "sp_doppler_frac": value})
        # self.rx_alt_bins, self.A_phy_LUT_interp = load_A_phy_LUT(A_phy_LUT_path)
        # rx_alt_bins, inc_angle_bins, az_angle_bins, A_phy_LUT_all = load_A_phy_LUT(
        #    A_phy_LUT_path
        # )

        self.orbit_path = orbit_path


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
        # The Matlab code does not use the azim_min, azim_max,
        # elev_max, elev_min, values so ignore them here
        ignore_values = load_dat_file(f, "H", 4)
        ignore_values = load_dat_file(f, "d", 1)
        ant_data = load_dat_file(f, "d", 3601 * 1201)
    return np.reshape(ant_data, (-1, 3601))


# load antenna binary files
def load_A_phy_LUT_inputs(filepath):
    """This function retrieves A_phy LUT variables

    Parameters
    ----------
    filepath : pathlib.Path
        path to file

    Returns
    -------
    rx_alt_bins, inc_angle_bins, az_angle_bins
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

    return rx_alt_bins, inc_angle_bins, az_angle_bins


# load antenna binary files
def load_A_phy_LUT(filepath, rx_alt_bins, inc_angle_bins, az_angle_bins):
    """This function retrieves A_phy LUT and the input matrices

    Parameters
    ----------
    filepath : pathlib.Path
        path to file

    Returns
    -------
    A_phy_LUT
    """
    with open(filepath, "rb") as f:
        num_rx_alt = len(rx_alt_bins)
        num_inc_angle = len(inc_angle_bins)
        num_az_angle = len(az_angle_bins)

        A_phy_LUT = np.full([num_rx_alt, num_inc_angle, num_az_angle, 7, 41], np.nan)

        for m in range(num_rx_alt):
            for n in range(num_inc_angle):
                for k in range(num_az_angle):
                    data = np.reshape(
                        load_dat_file(f, "I", 7 * 41), (41, 7)
                    ).T  # uint32
                    A_phy_LUT[m, n, k] = data

    return RegularGridInterpolator(
        (rx_alt_bins, inc_angle_bins, az_angle_bins), A_phy_LUT, bounds_error=True
    )


def retrieve_and_extract_orbit_file(settings, gps_week, gps_filename, orbit_path):
    """
    This code was adapted from https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    """
    # The user credentials that will be used to authenticate access to the data
    username = settings["L1_CDDIS_USERNAME"]
    password = settings["L1_CDDIS_PASSWORD"]

    base_url = "https://cddis.nasa.gov/archive/gnss/products/"
    file_name = gps_filename + ".gz"
    url = base_url + str(gps_week) + "/" + file_name

    gz_file_full = orbit_path.joinpath(Path(file_name))
    sp3_file_full = orbit_path.joinpath(Path(gps_filename))

    # Create a password manager to deal with the 401 reponse that is returned from
    # Earthdata Login

    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(
        None, "https://urs.earthdata.nasa.gov", username, password
    )

    # Create a cookie jar for storing cookies. This is used to store and return
    # the session cookie given to use by the data server (otherwise it will just
    # keep sending us back to Earthdata Login to authenticate).  Ideally, we
    # should use a file based cookie jar to preserve cookies between runs. This
    # will make it much more efficient.

    cookie_jar = CookieJar()

    # Install all the handlers.
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_manager),
        # urllib.request.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
        # urllib.request.HTTPSHandler(debuglevel=1),   # details of the requests/responses
        urllib.request.HTTPCookieProcessor(cookie_jar),
    )
    urllib.request.install_opener(opener)

    # Create and submit the requests. There are a wide range of exceptions that
    # can be thrown here, including HTTPError and URLError. These should be
    # caught and handled.

    # Open a request for the data, and download a specific file
    DataRequest = urllib.request.Request(url)
    DataRequest.add_header(
        "Cookie", str(cookie_jar)
    )  # Pass the saved cookie into a second HTTP request
    DataResponse = urllib.request.urlopen(DataRequest)

    # Get the redirect url and append 'app_type=401'
    # to do basic http auth
    DataRedirect_url = DataResponse.geturl()
    DataRedirect_url += "&app_type=401"

    # Request the resource at the modified redirect url
    try:
        DataRequest = urllib.request.Request(DataRedirect_url)
        DataResponse = urllib.request.urlopen(DataRequest)
    except urllib.error.HTTPError:
        return False

    DataBody = DataResponse.read()
    # Save file to working directory
    file_ = open(gz_file_full, "wb")
    file_.write(DataBody)
    file_.close()

    # unpack .gz file into .SP3 file
    with gzip.open(gz_file_full, "rb") as f_in:
        with open(sp3_file_full, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # delete .gz file
    os.remove(gz_file_full)
    return True


def load_orbit_file(settings, inp, gps_week, gps_tow, start_obj, end_obj, change_idx=0):
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
    # determine gps_week and day of the week (1-7)
    gps_week1, gps_dow1 = int(gps_week[0]), int(gps_tow[0] // 86400)
    # try loading in latest file name for data
    sp3_filename1 = (
        "IGS0OPSRAP_"
        + str(start_obj.year)
        + "{:03d}".format(start_obj.timetuple().tm_yday)  # match for the dropbox data
        + "0000_01D_15M_ORB.SP3"
    )
    sp3_filename1_full = inp.orbit_path.joinpath(Path(sp3_filename1))
    if not os.path.isfile(sp3_filename1_full):
        # try downloading file from NASA (not implemented for old format...)
        success = retrieve_and_extract_orbit_file(
            settings, gps_week1, sp3_filename1, inp.orbit_path
        )
        if not success:
            # try loading in alternate name from local
            sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".SP3"
            sp3_filename1_full = inp.orbit_path.joinpath(Path(sp3_filename1))
            if not os.path.isfile(sp3_filename1_full):
                # try loading in earliest format name from local
                sp3_filename1 = "igr" + str(gps_week1) + str(gps_dow1) + ".sp3"
                sp3_filename1_full = inp.orbit_path.joinpath(Path(sp3_filename1))
                if not os.path.isfile(sp3_filename1_full):
                    # TODO implement a mechanism for last valid file?
                    raise Exception(
                        "Orbit file not found locally or via NASA. Too soon for file release?"
                    )
    if change_idx:
        # if change_idx then also determine the day priors orbit file and return both
        # substitute in last gps_week/gps_tow values as first, end_obj as start_obj
        sp3_filename2_full = load_orbit_file(
            settings, inp, gps_week[-1:], gps_tow[-1:], end_obj, end_obj, change_idx=0
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
        "lat": np.linspace(temp["lat_max"], temp["lat_min"], temp["num_lat"]),
        "lon": np.linspace(temp["lon_min"], temp["lon_max"], temp["num_lon"]),
        "ele": np.rot90(np.reshape(map_data, (-1, temp["num_lat"])), 1),
    }

    # create and return interpolator model for the grid file
    return RegularGridInterpolator(
        (data["lat"], data["lon"]), data["ele"], bounds_error=True
    )


# load in map data binary files
def load_dem_file(filepath):
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
    # binary types for loading of gridded binary files
    type_list = [
        ("lat_min", "f"),
        ("lat_res", "f"),
        ("num_lat", "f"),
        ("lon_min", "f"),
        ("lon_res", "f"),
        ("num_lon", "f"),
    ]
    # type_list = [(lat_min,"d"),(num_lat,"H"), etc] + omit last grid type
    temp = {}
    with open(filepath, "rb") as f:
        for field, field_type in type_list:
            temp[field] = load_dat_file(f, field_type, 1)
        map_data = load_dat_file(f, "H", int(temp["num_lat"]) * int(temp["num_lon"]))
    data = {
        # "lat": np.linspace(temp["lat_min"], temp["lat_max"], int(temp["num_lat"])),
        # "lon": np.linspace(temp["lon_min"], temp["lon_max"], int(temp["num_lon"])),
        "lat": np.arange(0, int(temp["num_lat"])) * temp["lat_res"] + temp["lat_min"],
        "lon": np.arange(0, int(temp["num_lon"])) * temp["lon_res"] + temp["lon_min"],
        "ele": np.reshape(map_data, (-1, int(temp["num_lat"]))).T,
    }
    return data
    # create and return interpolator model for the grid file
    # return RegularGridInterpolator(
    #    (data["lon"], data["lat"]), data["ele"], bounds_error=True
    # )
