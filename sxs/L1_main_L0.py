from pathlib import Path
import netCDF4 as nc
import numpy as np
import array
import rasterio

# from rasterio.plot import show
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

### ---------------------- Prelaunch 1: Load L0 data


raw_data_path = Path().absolute().joinpath(Path("./dat/raw/"))
L0_filename = Path("20221103-121416_NZNV-NZCH.nc")
# L0_filenames = glob.glob("*.nc")

L0_dataset = nc.Dataset(raw_data_path.joinpath(L0_filename))

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


### rx -related variables
# PVT GPS week and sec
pvt_gps_week = load_netcdf(L0_dataset["/science/GPS_week_of_SC_attitude"])
pvt_gps_sec = load_netcdf(L0_dataset["/science/GPS_second_of_SC_attitude"])
# rx positions in ECEF, metres
rx_pos_x_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_x_ecef_m"])
rx_pos_y_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_y_ecef_m"])
rx_pos_z_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_z_ecef_m"])
# rx velocity in ECEF, m/s
rx_vel_x_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_x_ecef_mps"])
rx_vel_y_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_y_ecef_mps"])
rx_vel_z_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_z_ecef_mps"])
# rx attitude, deg
rx_pitch_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_pitch_deg"])
rx_roll_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_roll_deg"])
rx_yaw_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_yaw_deg"])
# rx clock bias and drifts
rx_clk_bias_m_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_bias_m"])
rx_clk_drift_mps_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_drift_mps"])

#
# Some processing required here to fix sporadic "zero" values??
#

### ddm-related variables
# tx ID/satellite PRN
transmitter_id = load_netcdf(L0_dataset["/science/ddm/transmitter_id"])
# raw counts and ddm parameters
first_scale_factor = load_netcdf(L0_dataset["/science/ddm/first_scale_factor"])
raw_counts = load_netcdf(L0_dataset["/science/ddm/counts"])  # raw counts, uncalibrated
zenith_i2q2 = load_netcdf(L0_dataset["/science/ddm/zenith_i2_plus_q2"])  # zenith counts
rf_source = load_netcdf(L0_dataset["/science/ddm/RF_source"])  # RF source
# binning standard deviation
std_dev_rf1 = load_netcdf(L0_dataset["/science/ddm/RF1_zenith_RHCP_std_dev"])
std_dev_rf2 = load_netcdf(L0_dataset["/science/ddm/RF2_nadir_LHCP_std_dev"])
std_dev_rf3 = load_netcdf(L0_dataset["/science/ddm/RF3_nadir_RHCP_std_dev"])
# absolute ddm center delay and doppler
delay_center_chips = load_netcdf(L0_dataset["/science/ddm/center_delay_bin_code_phase"])
doppler_center_hz = load_netcdf(L0_dataset["/science/ddm/center_doppler_bin_frequency"])
# coherent duration and noncoherent integration
coherent_duration = (
    load_netcdf(L0_dataset["/science/ddm/L1_E1_coherent_duration"]) / 1000
)
non_coherent_integrations = (
    load_netcdf(L0_dataset["/science/ddm/L1_E1_non_coherent_integrations"]) / 1000
)
# NGRx estimate additional delay path
add_range_to_sp_pvt = load_netcdf(L0_dataset["/science/ddm/additional_range_to_SP"])

# print(len(pvt_gps_week) - len(transmitter_id))
#
# Additional processing if ddm- and rx- related varaibles aren't the same length
#

### temperatures from engineering data
# antenna temperatures and engineering timestamp
eng_timestamp = load_netcdf(L0_dataset["/eng/packet_creation_time"])
zenith_ant_temp_eng = load_netcdf(L0_dataset["/eng/zenith_ant_temp"])
nadir_ant_temp_eng = load_netcdf(L0_dataset["/eng/nadir_ant_temp"])


### ---------------------- Prelaunch 2 - define external data paths and filenames

# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path("./dat/L1a_cal/"))
L1a_cal_ddm_counts_db_filename = Path("L1A_cal_ddm_counts_dB.dat")
L1a_cal_ddm_power_dbm_filename = Path("L1A_cal_ddm_power_dBm.dat")
L1a_cal_ddm_counts_db = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_counts_db_filename))
L1a_cal_ddm_power_dbm = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_power_dbm_filename))

# load SRTM_30 DEM
dem_path = Path().absolute().joinpath(Path("./dat/dem/"))
dem_filename = Path("nzsrtm_30_v1.tif")
dem = rasterio.open(dem_path.joinpath(dem_filename))
dem = {
    "ele": dem.read(),
    "lat": np.linspace(dem.bounds.top, dem.bounds.bottom, dem.height),
    "lon": np.linspace(dem.bounds.left, dem.bounds.right, dem.width),
}

# load DTU10 model
dtu_path = Path().absolute().joinpath(Path("./dat/dtu/"))
dtu_filename = Path("dtu10_v1.dat")
with open(dtu_path.joinpath(dtu_filename), "rb") as f:
    lat_min = load_dat_file(f, "d", 1)
    lat_max = load_dat_file(f, "d", 1)
    num_lat = load_dat_file(f, "H", 1)
    lon_min = load_dat_file(f, "d", 1)
    lon_max = load_dat_file(f, "d", 1)
    num_lon = load_dat_file(f, "H", 1)
    map_data = load_dat_file(f, "d", num_lat * num_lon)
dtu10 = {
    "lat": np.linspace(lat_min, lat_max, num_lat),
    "lon": np.linspace(lon_min, lon_max, num_lon),
    "ele": np.reshape(map_data, (-1, num_lat)),
}

# load ocean/land (distance to coast) mask
landmask_path = Path().absolute().joinpath(Path("./dat/cst/"))
landmask_filename = Path("dist_to_coast_nz_v1.dat")
with open(landmask_path.joinpath(landmask_filename), "rb") as f:
    lat_min = load_dat_file(f, "d", 1)
    lat_max = load_dat_file(f, "d", 1)
    num_lat = load_dat_file(f, "H", 1)
    lon_min = load_dat_file(f, "d", 1)
    lon_max = load_dat_file(f, "d", 1)
    num_lon = load_dat_file(f, "H", 1)
    map_data = load_dat_file(f, "d", num_lat * num_lon)
landmask_nz = {
    "lat": np.linspace(lat_min, lat_max, num_lat),
    "lon": np.linspace(lon_min, lon_max, num_lon),
    "ele": np.reshape(map_data, (-1, num_lat)),
}

# process landcover mask
lcv_path = Path().absolute().joinpath(Path("./dat/lcv/"))
lcv_filename = Path("lcv.png")
lcv_mask = Image.open(lcv_path.joinpath(lcv_filename))

# -------------- This isn't actually used??
# process inland water mask
# pek_path = Path().absolute().joinpath(Path("./dat/pek/"))
# pek_filename_1 = Path("occurrence_160E_40S.tif")
# pek_filename_2 = Path("occurrence_170E_30S.tif")
# pek_filename_3 = Path("occurrence_170E_40S.tif")
# water_mask = {
#    "water_mask_160E_40S": rasterio.open(pek_path.joinpath(pek_filename_1)),
#    "water_mask_170E_30S": rasterio.open(pek_path.joinpath(pek_filename_2)),
#    "water_mask_170E_40S": rasterio.open(pek_path.joinpath(pek_filename_3)),
# }

# load PRN-SV and SV-EIRP(static) LUT
gps_path = Path().absolute().joinpath(Path("./dat/gps/"))
SV_PRN_filename = Path("PRN_SV_LUT_v1.dat")
SV_eirp_filename = Path("GPS_SV_EIRP_Params_v7.dat")
SV_PRN_LUT = np.loadtxt(gps_path.joinpath(SV_PRN_filename), usecols=(0, 1))
SV_eirp_LUT = np.loadtxt(gps_path.joinpath(SV_eirp_filename))

# load and process nadir NGRx-GNSS antenna patterns
rng_path = Path().absolute().joinpath(Path("./dat/rng/"))
LHCP_L_filename = Path("GNSS_LHCP_L_gain_db_i_v1.dat")
LHCP_R_filename = Path("GNSS_LHCP_R_gain_db_i_v1.dat")
RHCP_L_filename = Path("GNSS_RHCP_L_gain_db_i_v1.dat")
RHCP_R_filename = Path("GNSS_RHCP_R_gain_db_i_v1.dat")
LHCP_pattern = {
    "LHCP": load_antenna_pattern(rng_path.joinpath(LHCP_L_filename)),
    "RHCP": load_antenna_pattern(rng_path.joinpath(LHCP_R_filename)),
}
RHCP_pattern = {
    "LHCP": load_antenna_pattern(rng_path.joinpath(RHCP_L_filename)),
    "RHCP": load_antenna_pattern(rng_path.joinpath(RHCP_R_filename)),
}

# load physical scattering area LUT
phy_ele_filename = Path("phy_ele_size.dat")  # same path as DEM
phy_ele_size = np.loadtxt(dem_path.joinpath(phy_ele_filename))
