# mike.laverick@auckland.ac.nz
# L1_main_L0.py
# Initial draft L1 script

from pathlib import Path
import netCDF4 as nc
import numpy as np
import rasterio
from scipy.interpolate import interpn
import pyproj
from datetime import datetime
from PIL import Image

from cal_functions import ddm_calibration
from gps_functions import gps2utc, utc2gps, satellite_orbits
from load_files import (
    load_netcdf,
    load_antenna_pattern,
    interp_ddm,
    get_orbit_file,
    load_dat_file_grid,
)
from specular import sp_solver


# Required to load the land cover mask file
Image.MAX_IMAGE_PIXELS = None


### ---------------------- Prelaunch 1: Load L0 data

# specify input L0 netcdf file
raw_data_path = Path().absolute().joinpath(Path("./dat/raw/"))
L0_filename = Path("20221103-121416_NZNV-NZCH.nc")
L0_dataset = nc.Dataset(raw_data_path.joinpath(L0_filename))


# load in rx-related variables
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
# rx attitude, deg | TODO this is actually radians and will be updated
rx_pitch_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_pitch_deg"])
rx_roll_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_roll_deg"])
rx_yaw_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_yaw_deg"])
# rx clock bias and drifts
rx_clk_bias_m_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_bias_m"])
rx_clk_drift_mps_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_drift_mps"])

# TODO: Some processing required here to fix leading/trailing/sporadic "zero" values?

# load in ddm-related variables
# tx ID/satellite PRN
transmitter_id = load_netcdf(L0_dataset["/science/ddm/transmitter_id"])
# raw counts and ddm parameters
first_scale_factor = load_netcdf(L0_dataset["/science/ddm/first_scale_factor"])
# raw counts, uncalibrated
raw_counts = load_netcdf(L0_dataset["/science/ddm/counts"])
zenith_i2q2 = load_netcdf(L0_dataset["/science/ddm/zenith_i2_plus_q2"])
rf_source = load_netcdf(L0_dataset["/science/ddm/RF_source"])
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

#
# TODO: Additional processing if ddm- and rx- related varaibles aren't the same length
#

# antenna temperatures and engineering timestamp
eng_timestamp = load_netcdf(L0_dataset["/eng/packet_creation_time"])
zenith_ant_temp_eng = load_netcdf(L0_dataset["/eng/zenith_ant_temp"])
nadir_ant_temp_eng = load_netcdf(L0_dataset["/eng/nadir_ant_temp"])


### ---------------------- Prelaunch 2 - define external data paths and filenames


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
dtu10 = load_dat_file_grid(dtu_path.joinpath(dtu_filename))

# load ocean/land (distance to coast) mask
landmask_path = Path().absolute().joinpath(Path("./dat/cst/"))
landmask_filename = Path("dist_to_coast_nz_v1.dat")
landmask_nz = load_dat_file_grid(landmask_path.joinpath(landmask_filename))

# load landcover mask
lcv_path = Path().absolute().joinpath(Path("./dat/lcv/"))
lcv_filename = Path("lcv.png")
lcv_mask = Image.open(lcv_path.joinpath(lcv_filename))

# TODO -------------- This isn't actually used??
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


### ---------------------- Part 1: General processing
# This part derives global constants, timestamps, and all the other
# parameters at ddm timestamps


# make array (ddm_pvt_bias) of non_coherent_integrations divided by 2
ddm_pvt_bias = non_coherent_integrations / 2
# make array (pvt_utc) of gps to unix time (see above)
pvt_utc = np.array(
    [gps2utc(week, pvt_gps_sec[i]) for i, week in enumerate(pvt_gps_week)]
)
# make array (ddm_utc) of ddm_pvt_bias + pvt_utc
ddm_utc = pvt_utc + ddm_pvt_bias
# make arrays (gps_week, gps_tow) of ddm_utc to gps week/sec (inc. 1/2*integration time)
gps_week, gps_tow = utc2gps(ddm_utc)

# interpolate rx positions onto new time grid
rx_pos_x = interp_ddm(pvt_utc, rx_pos_x_pvt, ddm_utc)
rx_pos_y = interp_ddm(pvt_utc, rx_pos_y_pvt, ddm_utc)
rx_pos_z = interp_ddm(pvt_utc, rx_pos_z_pvt, ddm_utc)
rx_pos_xyz = [rx_pos_x, rx_pos_y, rx_pos_z]
# interpolate rx velocities onto new time grid
rx_vel_x = interp_ddm(pvt_utc, rx_vel_x_pvt, ddm_utc)
rx_vel_y = interp_ddm(pvt_utc, rx_vel_y_pvt, ddm_utc)
rx_vel_z = interp_ddm(pvt_utc, rx_vel_z_pvt, ddm_utc)
# interpolate rx roll/pitch/yaw onto new time grid
rx_roll = interp_ddm(pvt_utc, rx_roll_pvt, ddm_utc)
rx_pitch = interp_ddm(pvt_utc, rx_pitch_pvt, ddm_utc)
rx_yaw = interp_ddm(pvt_utc, rx_yaw_pvt, ddm_utc)
# interpolate bias+drift onto new time grid
rx_clk_bias_m = interp_ddm(pvt_utc, rx_clk_bias_m_pvt, ddm_utc)
rx_clk_drift_mps = interp_ddm(pvt_utc, rx_clk_drift_mps_pvt, ddm_utc)

# define maximum NGRx signal capacity, and half
J = 20
J_2 = int(J / 2)

# interpolate "additional_range_to_SP" to new time grid
add_range_to_sp = np.full([*add_range_to_sp_pvt.shape], np.nan)
for ngrx_channel in range(J):
    add_range_to_sp[:, ngrx_channel] = interp_ddm(
        pvt_utc, add_range_to_sp_pvt[:, ngrx_channel], ddm_utc
    )
# interpolate temperatures onto new time grid
ant_temp_zenith = interp_ddm(eng_timestamp, zenith_ant_temp_eng, ddm_utc)
ant_temp_nadir = interp_ddm(eng_timestamp, nadir_ant_temp_eng, ddm_utc)


# ecef2lla Matlab function
# define projections and transform
# TODO function is depreciated,see following url
# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
# ecef2ella
lon, lat, alt = pyproj.transform(ecef, lla, *rx_pos_xyz, radians=False)
rx_pos_lla = [lat, lon, alt]

# determine specular point "over land" flag from landmask
# replaces get_map_value function
status_flags_one_hz = interpn(
    points=(landmask_nz["lon"], landmask_nz["lat"]),
    values=landmask_nz["ele"],
    xi=(lon, lat),
    method="linear",
)
status_flags_one_hz[status_flags_one_hz > 0] = 5
status_flags_one_hz[status_flags_one_hz <= 0] = 4

# determine time coverage
time_coverage_start_obj = datetime.utcfromtimestamp(ddm_utc[0])
time_coverage_start = time_coverage_start_obj.strftime("%Y-%m-%d %H:%M:%S")
time_coverage_end_obj = datetime.utcfromtimestamp(ddm_utc[-1])
time_coverage_end = time_coverage_end_obj.strftime("%d-%m-%Y %H:%M:%S")
time_coverage_resolution = ddm_utc[1] - ddm_utc[0]
hours, remainder = divmod((ddm_utc[-1] - ddm_utc[0] + 1), 3600)
minutes, seconds = divmod(remainder, 60)
time_coverage_duration = f"P0DT{int(hours)}H{int(minutes)}M{int(seconds)}S"

# specify L1 netcdf information and write algorithm + LUT versions
aircraft_reg = "ZK-NFA"  # default value
ddm_source = 2  # 1 = GPS signal simulator, 2 = aircraft
ddm_time_type_selector = 1  # 1 = middle of DDM sampling period
delay_resolution = 0.25  # unit in chips
dopp_resolution = 500  # unit in Hz
dem_source = "SRTM30"
l1_algorithm_version = "1.1"
l1_data_version = "1"
l1a_sig_LUT_version = "1"
l1a_noise_LUT_version = "1"
ngrx_port_mapping_version = "1"
nadir_ant_data_version = "1"
zenith_ant_data_version = "1"
prn_sv_maps_version = "1"
gps_eirp_param_version = "7"
land_mask_version = "1"
surface_type_version = "1"
mean_sea_surface_version = "1"
per_bin_ant_version = "1"

# write timestamps and ac-related variables
# 0-indexed sample and DDM
# TODO Skipping these right now to avoid dupication of variables


### ---------------------- Part 2: Derive TX related variables
# This part derives TX positions and velocities, maps between PRN and SVN,
# and gets track ID

# determine unique satellite transponder IDs
trans_id_unique = np.unique(transmitter_id)
trans_id_unique = trans_id_unique[trans_id_unique > 0]

# create data arrays for C++ code to populate
tx_pos_x = np.full([*transmitter_id.shape], np.nan)
tx_pos_y = np.full([*transmitter_id.shape], np.nan)
tx_pos_z = np.full([*transmitter_id.shape], np.nan)
tx_vel_x = np.full([*transmitter_id.shape], np.nan)
tx_vel_y = np.full([*transmitter_id.shape], np.nan)
tx_vel_z = np.full([*transmitter_id.shape], np.nan)
tx_clk_bias = np.full([*transmitter_id.shape], np.nan)
prn_code = np.full([*transmitter_id.shape], np.nan)
sv_num = np.full([*transmitter_id.shape], np.nan)
track_id = np.full([*transmitter_id.shape], np.nan)
orbit_bundle = [
    tx_pos_x,
    tx_pos_y,
    tx_pos_z,
    tx_vel_x,
    tx_vel_y,
    tx_vel_z,
    tx_clk_bias,
    prn_code,
    sv_num,
    track_id,
    trans_id_unique,
]

# determine whether flight spans a UTC day
if time_coverage_start_obj.day == time_coverage_end_obj.day:
    # determine single orbit file of that day
    orbit_file1 = get_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
    )
    # calculate satellite orbits, data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2, gps_week, gps_tow, transmitter_id, SV_PRN_LUT, orbit_file1, *orbit_bundle
    )
else:
    # find idx of day change in timestamps
    # np.diff does "arr_new[i] = arr[i+1] - arr[i]" thus +1 to find changed idx
    change_idx = np.where(np.diff(np.floor(gps_tow / 86400)) > 0)[0][0] + 1
    # determine day_N and day_N+1 orbit files to use
    orbit_file1, orbit_file2 = get_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
        change_idx=change_idx,
    )
    # calculate first chunk of specular points using 1st orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2,
        gps_week,
        gps_tow,
        transmitter_id,
        SV_PRN_LUT,
        orbit_file1,
        *orbit_bundle,
        end=change_idx,
    )
    # calculate last chunk of specular points using 2nd orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2,
        gps_week,
        gps_tow,
        transmitter_id,
        SV_PRN_LUT,
        orbit_file2,
        *orbit_bundle,
        start=change_idx,
    )

### ----------------------  Part 3: L1a calibration
# this part converts from raw counts to signal power in watts and complete
# L1a calibration


# create data arrays to hold DDM power/count arrays
ddm_power_counts = np.full([*raw_counts.shape], np.nan)
power_analog = np.full([*raw_counts.shape], np.nan)
ddm_ant = np.full([*transmitter_id.shape], np.nan)
ddm_noise_counts = np.full([*transmitter_id.shape], np.nan)
ddm_noise_watts = np.full([*transmitter_id.shape], np.nan)
peak_ddm_counts = np.full([*transmitter_id.shape], np.nan)
peak_ddm_watts = np.full([*transmitter_id.shape], np.nan)
peak_delay_bin = np.full([*transmitter_id.shape], np.nan)

# invoke calibration function which populates above arrays
ddm_calibration(
    std_dev_rf1,
    std_dev_rf2,
    std_dev_rf3,
    J,
    prn_code,
    raw_counts,
    rf_source,
    first_scale_factor,
    ddm_power_counts,
    power_analog,
    ddm_ant,
    ddm_noise_counts,
    peak_ddm_counts,
    peak_ddm_watts,
    peak_delay_bin,
)


# --------------------- Part 4A: SP solver and geometries
# initialise a huge amount of empty arrays

sx_pos_x = np.full([*transmitter_id.shape], np.nan)
sx_pos_y = np.full([*transmitter_id.shape], np.nan)
sx_pos_z = np.full([*transmitter_id.shape], np.nan)

sx_lat = np.full([*transmitter_id.shape], np.nan)
sx_lon = np.full([*transmitter_id.shape], np.nan)
sx_alt = np.full([*transmitter_id.shape], np.nan)

sx_vel_x = np.full([*transmitter_id.shape], np.nan)
sx_vel_y = np.full([*transmitter_id.shape], np.nan)
sx_vel_z = np.full([*transmitter_id.shape], np.nan)

sx_inc_angle = np.full([*transmitter_id.shape], np.nan)
sx_d_snell_angle = np.full([*transmitter_id.shape], np.nan)
dist_to_coast_km = np.full([*transmitter_id.shape], np.nan)
surface_type = np.full([*transmitter_id.shape], np.nan)

LOS_flag = np.full([*transmitter_id.shape], np.nan)

tx_to_sp_range = np.full([*transmitter_id.shape], np.nan)
rx_to_sp_range = np.full([*transmitter_id.shape], np.nan)

gps_boresight = np.full([*transmitter_id.shape], np.nan)

sx_theta_body = np.full([*transmitter_id.shape], np.nan)
sx_az_body = np.full([*transmitter_id.shape], np.nan)

sx_theta_enu = np.full([*transmitter_id.shape], np.nan)
sx_az_enu = np.full([*transmitter_id.shape], np.nan)

gps_tx_power_db_w = np.full([*transmitter_id.shape], np.nan)
gps_ant_gain_db_i = np.full([*transmitter_id.shape], np.nan)
static_gps_eirp = np.full([*transmitter_id.shape], np.nan)

sx_rx_gain_copol = np.full([*transmitter_id.shape], np.nan)
sx_rx_gain_xpol = np.full([*transmitter_id.shape], np.nan)


# iterate over each second of flight
for sec in range(len(transmitter_id)):

    # bundle up craft pos/vel/attitude data into per sec, and rx1
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    rx_attitude1 = np.array([rx_roll[sec], rx_pitch[sec], rx_yaw[sec]])
    rx1 = {
        "rx_pos_xyz": rx_pos_xyz1,
        "rx_vel_xyz": rx_vel_xyz1,
        "rx_attitude": rx_attitude1,
    }

    # variables are solved only for LHCP channels
    # RHCP channels share the same vales except RX gain solved for each channel
    for ngrx_channel in range(J_2):
        # bundle up satellite position and velocity data into per sec, and tx1
        tx_pos_xyz1 = np.array(
            [
                tx_pos_x[sec][ngrx_channel],
                tx_pos_y[sec][ngrx_channel],
                tx_pos_z[sec][ngrx_channel],
            ]
        )
        tx_vel_xyz1 = np.array(
            [
                tx_vel_x[sec][ngrx_channel],
                tx_vel_y[sec][ngrx_channel],
                tx_vel_z[sec][ngrx_channel],
            ]
        )
        trans_id1 = prn_code[sec][ngrx_channel]
        sv_num1 = sv_num[sec][ngrx_channel]
        ddm_ant1 = ddm_ant[sec][ngrx_channel]
        tx1 = {"tx_pos_xyz": tx_pos_xyz1, "tx_vel_xyz": tx_vel_xyz1, "sv_num": sv_num1}

        # TODO is checking only pos_x enough? it could be.
        if not np.isnan(tx_pos_x[sec][ngrx_channel]):
            (
                sx_pos_xyz1,
                inc_angle_deg1,
                d_snell_deg1,
                dist_to_coast_km1,
                LOS_flag1,
            ) = sp_solver(tx_pos_xyz1, rx_pos_xyz1, dem, dtu10, landmask_nz)

            sys.exit()
