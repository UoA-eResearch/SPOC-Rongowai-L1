# mike.laverick@auckland.ac.nz
# L1_main_L0.py
import math
import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import netCDF4 as nc
import numpy as np
import rasterio
from datetime import datetime
from PIL import Image
from timeit import default_timer as timer

from cal_functions import ddm_calibration, db2power, power2db, get_quality_flag
from gps_functions import gps2utc, utc2gps, satellite_orbits
from load_files import (
    load_netcdf,
    load_antenna_pattern,
    interp_ddm,
    get_orbit_file,
    load_dat_file_grid,
    write_netcdf,
    load_A_phy_LUT,
    get_surf_type2,
)
from specular import (
    sp_solver,
    sp_related,
    get_chi2,
    get_ddm_Aeff4,
    ddm_brcs2,
    ddm_refl2,
    get_fresnel,
    coh_det,
    meter2chips,
    delay_correction,
    deldop,
    los_status,
    get_sx_rx_gain,
)
from projections import ecef2lla

# Required to load the land cover mask file
Image.MAX_IMAGE_PIXELS = None

### ---------------------- Prelaunch 1: Load L0 data

# specify input L0 netcdf file
raw_data_path = Path().absolute().joinpath(Path("./dat/raw/"))
L0_filename = Path("20221103-121416_NZNV-NZCH.nc")
# L0_filename = Path("20230404-065056_NZTU-NZWN.nc")
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

# delay bin resolution
delay_bin_res = load_netcdf(L0_dataset["/science/ddm/delay_bin_res_narrow"])
delay_bin_res = delay_bin_res[~np.isnan(delay_bin_res)][0]
# doppler bin resolution
doppler_bin_res = load_netcdf(L0_dataset["/science/ddm/doppler_bin_res_narrow"])
doppler_bin_res = doppler_bin_res[~np.isnan(doppler_bin_res)][0]

# delay and Doppler center bin
center_delay_bin = load_netcdf(L0_dataset["/science/ddm/ddm_center_delay_bin"])
center_delay_bin = center_delay_bin[~np.isnan(center_delay_bin)][0]
center_doppler_bin = load_netcdf(L0_dataset["/science/ddm/ddm_center_doppler_bin"])
center_doppler_bin = center_doppler_bin[~np.isnan(center_doppler_bin)][0]

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

# antenna temperatures and engineering timestamp
eng_timestamp = load_netcdf(L0_dataset["/eng/packet_creation_time"])
zenith_ant_temp_eng = load_netcdf(L0_dataset["/eng/zenith_ant_temp"])
nadir_ant_temp_eng = load_netcdf(L0_dataset["/eng/nadir_ant_temp"])


### ---------------------- Prelaunch 1.5: Filter valid timestampes

# identify and compensate the value equal to 0 (randomly happens)
assert not (pvt_gps_week == 0).any(), "pvt_gps_week contains 0, need to compensate."

# the below is to process when ddm-related and rx-related variables do not
# have the same length, which happens for some of the L0 products
assert (
    pvt_gps_week.shape[0] == transmitter_id.shape[0]
), "pvt_gps_week and transmitter_id do not have the same length."
#
# TODO: Additional processing if ddm- and rx- related varaibles aren't the same length
#

integration_duration = coherent_duration * non_coherent_integrations * 1000


### ---------------------- Prelaunch 2 - define external data paths and filenames

# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path(r"./dat/L1a_cal/"))
L1a_cal_ddm_counts_db = np.loadtxt(L1a_path.joinpath(r"L1A_cal_ddm_counts_dB.dat"))
L1a_cal_ddm_power_dbm = np.loadtxt(L1a_path.joinpath(r"L1A_cal_ddm_power_dBm.dat"))


# load SRTM_30 DEM
dem_path = Path().absolute().joinpath(Path("./dat/dem/"))
dem_filename = Path("nzsrtm_30_v1.tif")
dem = rasterio.open(dem_path.joinpath(dem_filename))
dem = {
    "ele": dem.read(1),
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

# process inland water mask
pek_path = Path().absolute().joinpath(Path("./dat/pek/"))

water_mask = {}
for path in [
    "160E_40S",
    "170E_30S",
    "170E_40S",
]:
    water_mask[path] = {}
    pek_file = rasterio.open(pek_path.joinpath("occurrence_" + path + ".tif"))
    water_mask[path]["lon_min"] = pek_file._transform[0]
    water_mask[path]["res_deg"] = pek_file._transform[1]
    water_mask[path]["lat_max"] = pek_file._transform[3]
    water_mask[path]["file"] = pek_file

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
# phy_ele_filename = Path("phy_ele_size.dat")  # same path as DEM
# phy_ele_size = np.loadtxt(dem_path.joinpath(phy_ele_filename))

# scattering area LUT
A_phy_LUT_path = "./dat/A_phy_LUT/A_phy_LUT.dat"
# rx_alt_bins, inc_angle_bins, az_angle_bins, A_phy_LUT_all = load_A_phy_LUT(
#    A_phy_LUT_path
# )
rx_alt_bins, A_phy_LUT_interp = load_A_phy_LUT(A_phy_LUT_path)


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
rx_vel_xyz = [rx_vel_x, rx_vel_y, rx_vel_z]
# interpolate rx roll/pitch/yaw onto new time grid
rx_roll = interp_ddm(pvt_utc, rx_roll_pvt, ddm_utc)
rx_pitch = interp_ddm(pvt_utc, rx_pitch_pvt, ddm_utc)
rx_yaw = interp_ddm(pvt_utc, rx_yaw_pvt, ddm_utc)
rx_attitude = [rx_roll, rx_pitch, rx_yaw]
# interpolate bias+drift onto new time grid
rx_clk_bias_m = interp_ddm(pvt_utc, rx_clk_bias_m_pvt, ddm_utc)
rx_clk_drift_mps = interp_ddm(pvt_utc, rx_clk_drift_mps_pvt, ddm_utc)
rx_clk = [rx_clk_bias_m, rx_clk_drift_mps]

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

# ecef2ella
lon, lat, alt = ecef2lla.transform(*rx_pos_xyz, radians=False)
rx_pos_lla = [lat, lon, alt]

# determine specular point "over land" flag from landmask
# replaces get_map_value function
status_flags_one_hz = landmask_nz((lon, lat))
status_flags_one_hz[status_flags_one_hz > 0] = 5
status_flags_one_hz[status_flags_one_hz <= 0] = 4

# write global variables

L1_postCal = {}

time_coverage_start_obj = datetime.utcfromtimestamp(ddm_utc[0])
L1_postCal["time_coverage_start"] = time_coverage_start_obj.strftime(
    "%Y-%m-%d %H:%M:%S"
)
time_coverage_end_obj = datetime.utcfromtimestamp(ddm_utc[-1])
L1_postCal["time_coverage_end"] = time_coverage_end_obj.strftime("%d-%m-%Y %H:%M:%S")
L1_postCal["time_coverage_resolution"] = ddm_utc[1] - ddm_utc[0]

# time coverage
hours, remainder = divmod((ddm_utc[-1] - ddm_utc[0] + 1), 3600)
minutes, seconds = divmod(remainder, 60)

# below is new for algorithm version 1.1
ref_timestamp_utc = ddm_utc[0]

pvt_timestamp_utc = pvt_utc - ref_timestamp_utc
ddm_timestamp_utc = ddm_utc - ref_timestamp_utc

L1_postCal[
    "time_coverage_duration"
] = f"P0DT{int(hours)}H{int(minutes)}M{int(seconds)}S"

L1_postCal["aircraft_reg"] = "ZK-NFA"  # default value
L1_postCal["ddm_source"] = 2  # 1 = GPS signal simulator, 2 = aircraft
L1_postCal["ddm_time_type_selector"] = 1  # 1 = middle of DDM sampling period
L1_postCal["delay_resolution"] = delay_bin_res  # unit in chips
L1_postCal["dopp_resolution"] = doppler_bin_res  # unit in Hz
L1_postCal["dem_source"] = "SRTM30"

# write algorithm and LUT versions
L1_postCal["l1_algorithm_version"] = "2.0"
L1_postCal["l1_data_version"] = "2.0"
L1_postCal["l1a_sig_LUT_version"] = "1"
L1_postCal["l1a_noise_LUT_version"] = "1"
L1_postCal["A_LUT_version"] = "1"
L1_postCal["ngrx_port_mapping_version"] = "1"
L1_postCal["nadir_ant_data_version"] = "1"
L1_postCal["zenith_ant_data_version"] = "1"
L1_postCal["prn_sv_maps_version"] = "1"
L1_postCal["gps_eirp_param_version"] = "7"
L1_postCal["land_mask_version"] = "1"
L1_postCal["surface_type_version"] = "1"
L1_postCal["mean_sea_surface_version"] = "1"
L1_postCal["per_bin_ant_version"] = "1"

# write timestamps and ac-related variables
L1_postCal["pvt_timestamp_gps_week"] = pvt_gps_week
L1_postCal["pvt_timestamp_gps_sec"] = pvt_gps_sec
L1_postCal["pvt_timestamp_utc"] = pvt_timestamp_utc

L1_postCal["ddm_timestamp_gps_week"] = gps_week
L1_postCal["ddm_timestamp_gps_sec"] = gps_tow
L1_postCal["ddm_timestamp_utc"] = ddm_timestamp_utc

L1_postCal["ddm_pvt_bias"] = ddm_pvt_bias

# 0-indexed sample and DDM
L1_postCal["sample"] = np.arange(0, len(pvt_gps_sec))
L1_postCal["ddm"] = np.arange(0, J)

L1_postCal["sp_fsw_delay"] = delay_center_chips
L1_postCal["sp_ngrx_dopp"] = doppler_center_hz

L1_postCal["add_range_to_sp"] = add_range_to_sp
L1_postCal["add_range_to_sp_pvt"] = add_range_to_sp_pvt

L1_postCal["ac_lat"] = rx_pos_lla[0]
L1_postCal["ac_lon"] = rx_pos_lla[1]
L1_postCal["ac_alt"] = rx_pos_lla[2]

L1_postCal["ac_pos_x_pvt"] = rx_pos_x_pvt
L1_postCal["ac_pos_y_pvt"] = rx_pos_y_pvt
L1_postCal["ac_pos_z_pvt"] = rx_pos_z_pvt

L1_postCal["ac_pos_x"] = rx_pos_x
L1_postCal["ac_pos_y"] = rx_pos_y
L1_postCal["ac_pos_z"] = rx_pos_z

L1_postCal["ac_vel_x_pvt"] = rx_vel_x_pvt
L1_postCal["ac_vel_y_pvt"] = rx_vel_y_pvt
L1_postCal["ac_vel_z_pvt"] = rx_vel_z_pvt

L1_postCal["ac_vel_x"] = rx_vel_x
L1_postCal["ac_vel_y"] = rx_vel_y
L1_postCal["ac_vel_z"] = rx_vel_z

L1_postCal["ac_roll_pvt"] = rx_roll_pvt
L1_postCal["ac_pitch_pvt"] = rx_pitch_pvt
L1_postCal["ac_yaw_pvt"] = rx_yaw_pvt

L1_postCal["ac_roll"] = rx_attitude[0]
L1_postCal["ac_pitch"] = rx_attitude[1]
L1_postCal["ac_yaw"] = rx_attitude[2]

L1_postCal["rx_clk_bias_pvt"] = rx_clk_bias_m_pvt
L1_postCal["rx_clk_drift_pvt"] = rx_clk_drift_mps_pvt

L1_postCal["rx_clk_bias"] = rx_clk_bias_m
L1_postCal["rx_clk_drift"] = rx_clk_drift_mps

L1_postCal["ant_temp_nadir"] = ant_temp_nadir
L1_postCal["ant_temp_zenith"] = ant_temp_zenith

L1_postCal["status_flags_one_hz"] = status_flags_one_hz

# part 1 ends

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

# write TX variables
L1_postCal["tx_pos_x"] = tx_pos_x
L1_postCal["tx_pos_y"] = tx_pos_y
L1_postCal["tx_pos_z"] = tx_pos_z
L1_postCal["tx_vel_x"] = tx_vel_x
L1_postCal["tx_vel_y"] = tx_vel_y
L1_postCal["tx_vel_z"] = tx_vel_z
L1_postCal["tx_clk_bias"] = tx_clk_bias
L1_postCal["prn_code"] = prn_code
L1_postCal["sv_num"] = sv_num
L1_postCal["track_id"] = track_id

### ----------------------  Part 3: L1a calibration
# this part converts from raw counts to signal power in watts and complete
# L1a calibration


# create data arrays to hold DDM power/count arrays
# initialise variables for L1a results
ddm_power_counts = np.full([*raw_counts.shape], np.nan)
power_analog = np.full([*raw_counts.shape], np.nan)

ddm_ant = np.full([*transmitter_id.shape], np.nan)
inst_gain = np.full([*transmitter_id.shape], np.nan)

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
)

# save outputs to L1 structure
L1_postCal["raw_counts"] = ddm_power_counts
L1_postCal["l1a_power_ddm"] = power_analog
L1_postCal["zenith_sig_i2q2"] = zenith_i2q2  # read from file

L1_postCal["inst_gain"] = inst_gain
L1_postCal["ddm_ant"] = ddm_ant  # 0-based


#   # --------------------- Part 4A: SP solver and geometries
# initialise variables
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
    t0 = timer()
    tn = 0
    # retrieve rx positions, velocities and attitdues
    # bundle up craft pos/vel/attitude data into per sec, and rx1
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    # Euler angels are now in radians and yaw is resp. North
    # Hard-code of 0 due to alignment of antenna and craft
    rx_attitude1 = np.array([rx_roll[sec], rx_pitch[sec], 0])  # rx_yaw[sec]])
    rx1 = {
        "rx_pos_xyz": rx_pos_xyz1,
        "rx_vel_xyz": rx_vel_xyz1,
        "rx_attitude": rx_attitude1,
    }

    # variables are solved only for LHCP channels
    # RHCP channels share the same vales except RX gain solved for each channel
    for ngrx_channel in range(J_2):
        # retrieve tx positions and velocities
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

        # only process these with valid TX positions
        # TODO is checking only pos_x enough? it could be.
        if not np.isnan(tx_pos_x[sec][ngrx_channel]):
            LOS_flag1 = los_status(tx_pos_xyz1, rx_pos_xyz1)

            LOS_flag[sec][ngrx_channel] = int(LOS_flag1)

            # only process samples with valid sx positions, i.e., LOS = True
            if LOS_flag1:
                tn1 = timer()

                # Part 4.1: SP solver
                # derive SP positions, angle of incidence and distance to coast
                # returning sx_pos_lla1 in Py version to avoid needless coord conversions
                # derive sx velocity
                # time step in second
                dt = 1
                tx_pos_xyz_dt = tx_pos_xyz1 + (dt * tx_vel_xyz1)
                rx_pos_xyz_dt = rx_pos_xyz1 + (dt * rx_vel_xyz1)
                (
                    sx_pos_xyz1,
                    inc_angle_deg1,
                    d_snell_deg1,
                    dist_to_coast_km1,
                    # LOS_flag1,
                ) = sp_solver(tx_pos_xyz1, rx_pos_xyz1, dem, dtu10, landmask_nz)

                (
                    sx_pos_xyz_dt,
                    _,
                    _,
                    _,
                    # _,
                ) = sp_solver(tx_pos_xyz_dt, rx_pos_xyz_dt, dem, dtu10, landmask_nz)
                tn += timer() - tn1

                lon, lat, alt = ecef2lla.transform(*sx_pos_xyz1, radians=False)
                sx_pos_lla1 = [lat, lon, alt]
                # <lon,lat,alt> of the specular reflection
                # algorithm version 1.11
                surface_type1 = get_surf_type2(
                    sx_pos_lla1, landmask_nz, lcv_mask, water_mask
                )

                sx_vel_xyz1 = np.array(sx_pos_xyz_dt) - np.array(sx_pos_xyz1)

                # save sx values to variables
                sx_pos_x[sec][ngrx_channel] = sx_pos_xyz1[0]
                sx_pos_y[sec][ngrx_channel] = sx_pos_xyz1[1]
                sx_pos_z[sec][ngrx_channel] = sx_pos_xyz1[2]

                sx_lat[sec][ngrx_channel] = sx_pos_lla1[0]
                sx_lon[sec][ngrx_channel] = sx_pos_lla1[1]
                sx_alt[sec][ngrx_channel] = sx_pos_lla1[2]

                sx_vel_x[sec][ngrx_channel] = sx_vel_xyz1[0]
                sx_vel_y[sec][ngrx_channel] = sx_vel_xyz1[1]
                sx_vel_z[sec][ngrx_channel] = sx_vel_xyz1[2]
                surface_type[sec][ngrx_channel] = surface_type1

                sx_inc_angle[sec][ngrx_channel] = inc_angle_deg1
                sx_d_snell_angle[sec][ngrx_channel] = d_snell_deg1
                dist_to_coast_km[sec][ngrx_channel] = dist_to_coast_km1

                # Part 4.2: SP-related variables - 1
                # this part derives tx/rx gains, ranges and other related variables
                # derive SP related geo-parameters, including angles in various frames, ranges and antenna gain/GPS EIRP
                (
                    sx_angle_body1,
                    sx_angle_enu1,
                    sx_angle_ant1,
                    theta_gps1,
                    ranges1,
                    gps_rad1,
                ) = sp_related(tx1, rx1, sx_pos_xyz1, SV_eirp_LUT)

                # get values for deriving BRCS and reflectivity
                R_tsx1 = ranges1[0]
                R_rsx1 = ranges1[1]
                gps_eirp_watt1 = gps_rad1[2]

                # get active antenna gain for LHCP and RHCP channels
                sx_rx_gain_LHCP1 = get_sx_rx_gain(sx_angle_ant1, LHCP_pattern)
                sx_rx_gain_RHCP1 = get_sx_rx_gain(sx_angle_ant1, RHCP_pattern)

                # save to variables
                sx_theta_body[sec, ngrx_channel] = sx_angle_body1[0]
                sx_az_body[sec, ngrx_channel] = sx_angle_body1[1]

                sx_theta_enu[sec, ngrx_channel] = sx_angle_enu1[0]
                sx_az_enu[sec, ngrx_channel] = sx_angle_enu1[1]

                gps_boresight[sec, ngrx_channel] = theta_gps1

                tx_to_sp_range[sec, ngrx_channel] = ranges1[0]
                rx_to_sp_range[sec, ngrx_channel] = ranges1[1]

                gps_tx_power_db_w[sec, ngrx_channel] = gps_rad1[0]
                gps_ant_gain_db_i[sec, ngrx_channel] = gps_rad1[1]
                static_gps_eirp[sec, ngrx_channel] = gps_rad1[2]

                # copol gain
                # LHCP channel rx gain
                sx_rx_gain_copol[sec, ngrx_channel] = sx_rx_gain_LHCP1[0]
                # RHCP channel rx gain
                sx_rx_gain_copol[sec, ngrx_channel + J_2] = sx_rx_gain_RHCP1[1]
                # xpol gain gain
                # LHCP channel rx gain
                sx_rx_gain_xpol[sec, ngrx_channel] = sx_rx_gain_LHCP1[1]
                # RHCP channel rx gain
                sx_rx_gain_xpol[sec, ngrx_channel + J_2] = sx_rx_gain_RHCP1[0]
    print(
        f"******** start processing part 4A {sec} second data with {timer() - t0} ********"
    )
    print(f"*{tn}*")

# expand to RHCP channels
sx_pos_x[:, J_2:J] = sx_pos_x[:, 0:J_2]
sx_pos_y[:, J_2:J] = sx_pos_y[:, 0:J_2]
sx_pos_z[:, J_2:J] = sx_pos_z[:, 0:J_2]

sx_lat[:, J_2:J] = sx_lat[:, 0:J_2]
sx_lon[:, J_2:J] = sx_lon[:, 0:J_2]
sx_alt[:, J_2:J] = sx_alt[:, 0:J_2]

sx_vel_x[:, J_2:J] = sx_vel_x[:, 0:J_2]
sx_vel_y[:, J_2:J] = sx_vel_y[:, 0:J_2]
sx_vel_z[:, J_2:J] = sx_vel_z[:, 0:J_2]

surface_type[:, J_2:J] = surface_type[:, 0:J_2]
dist_to_coast_km[:, J_2:J] = dist_to_coast_km[:, 0:J_2]
LOS_flag[:, J_2:J] = LOS_flag[:, 0:J_2]

rx_to_sp_range[:, J_2:J] = rx_to_sp_range[:, 0:J_2]
tx_to_sp_range[:, J_2:J] = tx_to_sp_range[:, 0:J_2]

sx_inc_angle[:, J_2:J] = sx_inc_angle[:, 0:J_2]
sx_d_snell_angle[:, J_2:J] = sx_d_snell_angle[:, 0:J_2]

sx_theta_body[:, J_2:J] = sx_theta_body[:, 0:J_2]
sx_az_body[:, J_2:J] = sx_az_body[:, 0:J_2]

sx_theta_enu[:, J_2:J] = sx_theta_enu[:, 0:J_2]
sx_az_enu[:, J_2:J] = sx_az_enu[:, 0:J_2]

gps_boresight[:, J_2:J] = gps_boresight[:, 0:J_2]

static_gps_eirp[:, J_2:J] = static_gps_eirp[:, 0:J_2]

gps_tx_power_db_w[:, J_2:J] = gps_tx_power_db_w[:, 0:J_2]
gps_ant_gain_db_i[:, J_2:J] = gps_ant_gain_db_i[:, 0:J_2]

# save variables
L1_postCal["sp_pos_x"] = sx_pos_x  # checked value diff < 1 / e5
L1_postCal["sp_pos_y"] = sx_pos_y  # checked value diff < 1 / e5
L1_postCal["sp_pos_z"] = sx_pos_z  # checked value diff < 1 / e6

L1_postCal["sp_lat"] = sx_lat  # checked ok
L1_postCal["sp_lon"] = sx_lon  # checked ok
L1_postCal["sp_alt"] = sx_alt  # checked ok

L1_postCal["sp_vel_x"] = sx_vel_x  # checked value diff < 10
L1_postCal["sp_vel_y"] = sx_vel_y  # checked value diff < 10
L1_postCal["sp_vel_z"] = sx_vel_z  # checked value diff < 10

L1_postCal["sp_surface_type"] = surface_type  # checked ok
L1_postCal["sp_dist_to_coast_km"] = dist_to_coast_km  # checked ok
L1_postCal["LOS_flag"] = LOS_flag  # checked ok

L1_postCal["rx_to_sp_range"] = rx_to_sp_range  # checked value diff < 1 / e2
L1_postCal["tx_to_sp_range"] = tx_to_sp_range  # checked value diff < 1 / e7

L1_postCal["sp_inc_angle"] = sx_inc_angle  # checked ok
L1_postCal["sp_d_snell_angle"] = sx_d_snell_angle  # checked ok

L1_postCal["sp_theta_body"] = sx_theta_body  # checked value diff < 0.1
L1_postCal["sp_az_body"] = sx_az_body  # checked value diff < 0.01
L1_postCal["sp_theta_enu"] = sx_theta_enu  # checked value diff < 0.1 / e2
L1_postCal["sp_az_enu"] = sx_az_enu  # checked ok

L1_postCal["sp_rx_gain_copol"] = sx_rx_gain_copol
L1_postCal["sp_rx_gain_xpol"] = sx_rx_gain_xpol

L1_postCal["gps_off_boresight_angle_deg"] = gps_boresight  # checked ok

L1_postCal["static_gps_eirp"] = static_gps_eirp  # checked ok
L1_postCal["gps_tx_power_db_w"] = gps_tx_power_db_w  # checked ok
L1_postCal["gps_ant_gain_db_i"] = gps_ant_gain_db_i  # checked ok


############# to save debug time, save and restore variables ##########
#
np.save("debug_4a.npy", L1_postCal)
"""
#
# L1_postCal_loaded = np.load("debug.npy", allow_pickle=True).item()
# ##############


# def dic_to_keys_values(dic):
#     keys, values = list(dic.keys()), list(dic.values())
#     return keys, values


# #
# #
# def numpy_assert_almost_dict_values(dict1, dict2):
#     keys1, values1 = dic_to_keys_values(dict1)
#     keys2, values2 = dic_to_keys_values(dict2)
#     np.testing.assert_equal(keys1, keys2)
#     np.testing.assert_equal(values1, values2)


# #
# #
# # numpy_assert_almost_dict_values(L1_postCal, L1_postCal_loaded)
# ##############

L1_postCal = np.load("debug_4a.npy", allow_pickle=True).item()

sx_pos_x = L1_postCal["sp_pos_x"]
sx_pos_y = L1_postCal["sp_pos_y"]
sx_pos_z = L1_postCal["sp_pos_z"]
# #
sx_lat = L1_postCal["sp_lat"]
sx_lon = L1_postCal["sp_lon"]
sx_alt = L1_postCal["sp_alt"]
# #
sx_vel_x = L1_postCal["sp_vel_x"]
sx_vel_y = L1_postCal["sp_vel_y"]
sx_vel_z = L1_postCal["sp_vel_z"]
# #
surface_type = L1_postCal["sp_surface_type"]
dist_to_coast_km = L1_postCal["sp_dist_to_coast_km"]
LOS_flag = L1_postCal["LOS_flag"]
# #
rx_to_sp_range = L1_postCal["rx_to_sp_range"]
tx_to_sp_range = L1_postCal["tx_to_sp_range"]
# #
sx_inc_angle = L1_postCal["sp_inc_angle"]
sx_d_snell_angle = L1_postCal["sp_d_snell_angle"]
# #
sx_theta_body = L1_postCal["sp_theta_body"]
sx_az_body = L1_postCal["sp_az_body"]
sx_theta_enu = L1_postCal["sp_theta_enu"]
sx_az_enu = L1_postCal["sp_az_enu"]
# #
# sx_rx_gain = L1_postCal["sp_rx_gain"]
# #
gps_boresight = L1_postCal["gps_off_boresight_angle_deg"]
# #
static_gps_eirp = L1_postCal["static_gps_eirp"]
gps_tx_power_db_w = L1_postCal["gps_tx_power_db_w"]
gps_ant_gain_db_i = L1_postCal["gps_ant_gain_db_i"]

sx_rx_gain_copol = L1_postCal["sp_rx_gain_copol"]
sx_rx_gain_xpol = L1_postCal["sp_rx_gain_xpol"]
# ##############
"""
# -------------------- Part 4B: BRCS/NBRCS, reflectivity, coherent status and fresnel zone
# initialise variables
peak_delay_row = np.full([*transmitter_id.shape], np.nan)
peak_doppler_col = np.full([*transmitter_id.shape], np.nan)

sp_delay_row = np.full([*transmitter_id.shape], np.nan)
sp_delay_error = np.full([*transmitter_id.shape], np.nan)

sp_doppler_col = np.full([*transmitter_id.shape], np.nan)
sp_doppler_error = np.full([*transmitter_id.shape], np.nan)

zenith_code_phase = np.full([*transmitter_id.shape], np.nan)

noise_floor_all_LHCP = np.full([transmitter_id.shape[0], J_2], np.nan)
noise_floor_all_RHCP = np.full([transmitter_id.shape[0], J_2], np.nan)

delay_offset = 4

t0 = timer()
# derive floating SP bin location and effective scattering area A_eff
for sec in range(len(transmitter_id)):
    # retrieve rx positions and velocities
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])

    for ngrx_channel in range(J_2):
        # retrieve tx positions and velocities
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

        # retrieve sx-related parameters
        sx_pos_xyz1 = np.array(
            [
                sx_pos_x[sec][ngrx_channel],
                sx_pos_y[sec][ngrx_channel],
                sx_pos_z[sec][ngrx_channel],
            ]
        )

        counts_LHCP1 = ddm_power_counts[sec, ngrx_channel, :, :]
        # from onboard tracker
        add_range_to_sp1 = add_range_to_sp[sec][ngrx_channel]
        delay_center_chips1 = delay_center_chips[sec][ngrx_channel]

        # zenith code phase
        add_range_to_sp_chips1 = meter2chips(add_range_to_sp1)
        zenith_code_phase1 = delay_center_chips1 + add_range_to_sp_chips1
        zenith_code_phase1 = delay_correction(zenith_code_phase1, 1023)

        # Part 3B: noise floor here to avoid another interation over [sec,J_2]
        nf_counts_LHCP1 = ddm_power_counts[sec, ngrx_channel, :, :]
        nf_counts_RHCP1 = ddm_power_counts[sec, ngrx_channel + J_2, :, :]

        # delay_offset+1 due to difference between Matlab and Python indexing
        noise_floor_bins_LHCP1 = nf_counts_LHCP1[-(delay_offset + 1) :, :]
        noise_floor_bins_RHCP1 = nf_counts_RHCP1[-(delay_offset + 1) :, :]

        if (not np.isnan(tx_pos_x[sec][ngrx_channel])) and (
            not np.isnan(counts_LHCP1).all()
        ):
            # peak delay and doppler location
            # assume LHCP and RHCP DDMs have the same peak location
            peak_counts1 = np.max(counts_LHCP1)
            # invert order compared to Matlab
            [peak_delay_row1, peak_doppler_col1] = np.where(
                counts_LHCP1 == peak_counts1
            )

            # tx to rx range
            r_trx1 = np.linalg.norm(np.array(tx_pos_xyz1) - np.array(rx_pos_xyz1), 2)

            # SOC derived more accurate additional range to SP
            r_tsx1 = np.linalg.norm(np.array(tx_pos_xyz1) - np.array(sx_pos_xyz1), 2)
            r_rsx1 = np.linalg.norm(np.array(rx_pos_xyz1) - np.array(sx_pos_xyz1), 2)

            add_range_to_sp_soc1 = r_tsx1 + r_rsx1 - r_trx1
            d_add_range1 = add_range_to_sp_soc1 - add_range_to_sp1

            d_delay_chips1 = meter2chips(d_add_range1)
            d_delay_bin1 = d_delay_chips1 / delay_bin_res

            sp_delay_row1 = center_delay_bin - d_delay_bin1

            # SP doppler value
            _, sp_doppler_hz1, _ = deldop(
                tx_pos_xyz1, rx_pos_xyz1, tx_vel_xyz1, rx_vel_xyz1, sx_pos_xyz1
            )

            doppler_center_hz1 = doppler_center_hz[sec][ngrx_channel]

            d_doppler_hz1 = doppler_center_hz1 - sp_doppler_hz1 + 250
            d_doppler_bin1 = d_doppler_hz1 / doppler_bin_res

            sp_doppler_col1 = center_doppler_bin - d_doppler_bin1

            # SP delay and doppler location
            peak_delay_row[sec][ngrx_channel] = peak_delay_row1[0]  # 0-based index
            peak_doppler_col[sec][ngrx_channel] = peak_doppler_col1[0]

            sp_delay_row[sec][ngrx_channel] = sp_delay_row1
            sp_delay_error[sec][ngrx_channel] = d_delay_chips1

            sp_doppler_col[sec][ngrx_channel] = sp_doppler_col1
            sp_doppler_error[sec][ngrx_channel] = d_doppler_hz1

            noise_floor_all_LHCP[sec][ngrx_channel] = np.nanmean(noise_floor_bins_LHCP1)
            noise_floor_all_RHCP[sec][ngrx_channel] = np.nanmean(noise_floor_bins_RHCP1)

        zenith_code_phase[sec][ngrx_channel] = zenith_code_phase1

print(f"******** finish processing part 4B data with {timer() - t0}********")

# extend to RHCP channels
peak_delay_row[:, J_2:J] = peak_delay_row[:, 0:J_2]
peak_doppler_col[:, J_2:J] = peak_doppler_col[:, 0:J_2]

sp_delay_row[:, J_2:J] = sp_delay_row[:, 0:J_2]
sp_doppler_col[:, J_2:J] = sp_doppler_col[:, 0:J_2]

sp_delay_error[:, J_2:J] = sp_delay_error[:, 0:J_2]
sp_doppler_error[:, J_2:J] = sp_doppler_error[:, 0:J_2]

zenith_code_phase[:, J_2:J] = zenith_code_phase[:, 0:J_2]

# save variables
L1_postCal["brcs_ddm_peak_bin_delay_row"] = peak_delay_row
L1_postCal["brcs_ddm_peak_bin_dopp_col"] = peak_doppler_col

L1_postCal["brcs_ddm_sp_bin_delay_row"] = sp_delay_row
L1_postCal["brcs_ddm_sp_bin_dopp_col"] = sp_doppler_col

L1_postCal["sp_delay_error"] = sp_delay_error
L1_postCal["sp_dopp_error"] = sp_doppler_error

L1_postCal["zenith_code_phase"] = zenith_code_phase


# Part 3B: noise floor and SNR
sp_safe_margin = 9  # safe space between SP and DDM end

# single noise floor from valid DDMs
sp_delay_row_LHCP = sp_delay_row[:, :10]  # reference to LHCP delay row

valid_idx = np.where(
    (sp_delay_row_LHCP > 0)
    & (sp_delay_row_LHCP < (39 - sp_safe_margin))
    & ~np.isnan(noise_floor_all_LHCP)
)


# Part 3C: confidence flag of the SP solved
confidence_flag = np.full([*transmitter_id.shape], np.nan)

# noise floor is the median of the average counts
noise_floor_LHCP = np.nanmedian(noise_floor_all_LHCP[valid_idx])
noise_floor_RHCP = np.nanmedian(noise_floor_all_RHCP[valid_idx])

# SNR of SP
# flag 0 for signal < 0
ddm_snr = np.full([*transmitter_id.shape], np.nan)
snr_flag = np.full([*transmitter_id.shape], np.nan)


for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J_2):
        counts_LHCP1 = ddm_power_counts[sec, ngrx_channel, :, :]
        counts_RHCP1 = ddm_power_counts[sec, ngrx_channel + J_2, :, :]

        # Removed +1 due to Python 0-based indexing
        sp_delay_row1 = np.floor(sp_delay_row_LHCP[sec][ngrx_channel])  # + 1
        sp_doppler_col1 = np.floor(sp_doppler_col[sec][ngrx_channel])  # + 1

        if (0 < sp_delay_row1 < 40) and (0 < sp_doppler_col1 < 5):
            sp_counts_LHCP1 = counts_LHCP1[int(sp_delay_row1), int(sp_doppler_col1)]
            sp_counts_RHCP1 = counts_RHCP1[int(sp_delay_row1), int(sp_doppler_col1)]

            signal_counts_LHCP1 = sp_counts_LHCP1 - noise_floor_LHCP
            snr_LHCP1 = signal_counts_LHCP1 / noise_floor_LHCP
            signal_counts_RHCP1 = sp_counts_RHCP1 - noise_floor_RHCP
            snr_RHCP1 = signal_counts_RHCP1 / noise_floor_RHCP

            if signal_counts_LHCP1 > 0:
                snr_LHCP_db1 = power2db(snr_LHCP1)
                snr_flag_LHCP1 = 1
            else:
                snr_LHCP_db1 = np.nan
                snr_flag_LHCP1 = 0
            ddm_snr[sec][ngrx_channel] = snr_LHCP_db1
            snr_flag[sec][ngrx_channel] = snr_flag_LHCP1

            if signal_counts_RHCP1 > 0:
                snr_RHCP_db1 = power2db(snr_RHCP1)
                snr_flag_RHCP1 = 1
            else:
                snr_RHCP_db1 = np.nan
                snr_flag_RHCP1 = 0
            ddm_snr[sec][ngrx_channel + J_2] = snr_RHCP_db1
            snr_flag[sec][ngrx_channel + J_2] = snr_flag_RHCP1

            sx_delay_error1 = abs(sp_delay_error[sec][ngrx_channel])
            sx_doppler_error1 = abs(sp_doppler_error[sec][ngrx_channel])
            sx_d_snell_angle1 = abs(sx_d_snell_angle[sec][ngrx_channel])

            if not np.isnan(tx_pos_x[sec][ngrx_channel]):
                # criteria may change at a later stage
                delay_doppler_snell1 = (
                    (sx_delay_error1 < 1.25)
                    & (abs(sx_doppler_error1) < 250)
                    & (sx_d_snell_angle1 < 2)
                )

                # Python snr_LHCP_db1 == Matlab snr_LHCP1 for this step,
                # Python does this all in one loop whereas Matlab does this in multiple
                # loops and re-uses the snr_LHCP1 variable to refer to snr_LHCP_db1 values
                if (snr_LHCP_db1 >= 2.0) and not delay_doppler_snell1:
                    confidence_flag1 = 0
                elif (snr_LHCP_db1 < 2.0) and not delay_doppler_snell1:
                    confidence_flag1 = 1
                elif (snr_LHCP_db1 < 2.0) and delay_doppler_snell1:
                    confidence_flag1 = 2
                elif (snr_LHCP_db1 >= 2.0) and delay_doppler_snell1:
                    confidence_flag1 = 3
                else:
                    confidence_flag1 = np.nan

                confidence_flag[sec][ngrx_channel] = confidence_flag1

noise_floor = np.hstack(
    (
        np.full([raw_counts.shape[2], raw_counts.shape[3]], noise_floor_LHCP),
        np.full([raw_counts.shape[2], raw_counts.shape[3]], noise_floor_RHCP),
    )
)
L1_postCal["ddm_noise_floor"] = noise_floor

confidence_flag[:, J_2:J] = confidence_flag[:, 0:J_2]

L1_postCal["ddm_snr"] = ddm_snr
L1_postCal["ddm_snr_flag"] = snr_flag

L1_postCal["sp_confidence_flag"] = confidence_flag


L1_postCal["sp_ngrx_delay_correction"] = sp_delay_error
L1_postCal["sp_ngrx_dopp_correction"] = sp_doppler_error


# Part 5: Copol and xpol BRCS, reflectivity, peak reflectivity

# separate copol and xpol gain for using later
rx_gain_copol_LL = sx_rx_gain_copol[:, :10]
rx_gain_copol_RR = sx_rx_gain_copol[:, 10:20]

rx_gain_xpol_RL = sx_rx_gain_xpol[:, :10]
rx_gain_xpol_LR = sx_rx_gain_xpol[:, 10:20]

# BRCS, reflectivity
pol_shape = [*raw_counts.shape]
# pol_shape[1] = J_2

# brcs_copol = np.full([*pol_shape], np.nan)
# brcs_xpol = np.full([*pol_shape], np.nan)
brcs = np.full([*pol_shape], np.nan)

# refl_copol = np.full([*pol_shape], np.nan)
# refl_xpol = np.full([*pol_shape], np.nan)
surface_reflectivity = np.full([*pol_shape], np.nan)

sp_refl = np.full([*transmitter_id.shape], np.nan)
norm_refl_waveform = np.full([*transmitter_id.shape, 40, 1], np.nan)

# TODO draw these from a config file, here and in other places
cable_loss_db_LHCP = 0.6600
cable_loss_db_RHCP = 0.5840
powloss_LHCP = db2power(cable_loss_db_LHCP)
powloss_RHCP = db2power(cable_loss_db_RHCP)

t0 = timer()
for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J_2):
        # compensate cable loss
        power_analog_LHCP1 = power_analog[sec, ngrx_channel, :, :] * powloss_LHCP
        power_analog_RHCP1 = power_analog[sec, ngrx_channel + J_2, :, :] * powloss_RHCP

        R_tsx1 = tx_to_sp_range[sec][ngrx_channel]
        R_rsx1 = rx_to_sp_range[sec][ngrx_channel]
        rx_gain_dbi_1 = [
            rx_gain_copol_LL[sec][ngrx_channel],
            rx_gain_xpol_RL[sec][ngrx_channel],
            rx_gain_xpol_LR[sec][ngrx_channel],
            rx_gain_copol_RR[sec][ngrx_channel],
        ]
        gps_eirp1 = static_gps_eirp[sec][ngrx_channel]

        if not np.isnan(power_analog_LHCP1).all():
            brcs_copol1, brcs_xpol1 = ddm_brcs2(
                power_analog_LHCP1,
                power_analog_RHCP1,
                gps_eirp1,
                rx_gain_dbi_1,
                R_tsx1,
                R_rsx1,
            )
            refl_copol1, refl_xpol1 = ddm_refl2(
                power_analog_LHCP1,
                power_analog_RHCP1,
                gps_eirp1,
                rx_gain_dbi_1,
                R_tsx1,
                R_rsx1,
            )

            # reflectivity at SP
            # ignore +1 as Python is 0-base not 1-base
            sp_delay_row1 = np.floor(sp_delay_row_LHCP[sec][ngrx_channel])  # + 1
            sp_doppler_col1 = np.floor(sp_doppler_col[sec][ngrx_channel])  # + 1

            if (0 < sp_delay_row1 < 40) and (0 < sp_doppler_col1 < 5):
                sp_refl_copol1 = refl_copol1[int(sp_delay_row1), int(sp_doppler_col1)]
                sp_refl_xpol1 = refl_xpol1[int(sp_delay_row1), int(sp_doppler_col1)]
            else:
                sp_refl_copol1 = np.nan
                sp_refl_xpol1 = np.nan

            refl_waveform_copol1 = np.sum(refl_copol1, axis=1)
            norm_refl_waveform_copol1 = np.divide(
                refl_waveform_copol1, np.nanmax(refl_waveform_copol1)
            ).reshape(40, -1)

            refl_waveform_xpol1 = np.sum(refl_xpol1, axis=1)
            norm_refl_waveform_xpol1 = np.divide(
                refl_waveform_xpol1, np.nanmax(refl_waveform_xpol1)
            ).reshape(40, -1)

            brcs[sec][ngrx_channel] = brcs_copol1
            brcs[sec][ngrx_channel + J_2] = brcs_xpol1

            surface_reflectivity[sec][ngrx_channel] = refl_copol1
            surface_reflectivity[sec][ngrx_channel + J_2] = refl_xpol1

            sp_refl[sec][ngrx_channel] = sp_refl_copol1
            sp_refl[sec][ngrx_channel + J_2] = sp_refl_xpol1

            norm_refl_waveform[sec][ngrx_channel] = norm_refl_waveform_copol1
            norm_refl_waveform[sec][ngrx_channel + J_2] = norm_refl_waveform_xpol1
print(f"******** finish processing part 5 data with {timer() - t0}********")

L1_postCal["brcs"] = brcs

L1_postCal["surface_reflectivity"] = surface_reflectivity
L1_postCal["surface_reflectivity_peak"] = sp_refl
L1_postCal["norm_refl_waveform"] = norm_refl_waveform

# Part 5 ends

# Part 6: NBRCS and other related parameters

A_eff = np.full([*raw_counts.shape], np.nan)
nbrcs_scatter_area = np.full([*transmitter_id.shape], np.nan)

pol_shape = [*raw_counts.shape]
# pol_shape[1] = J_2
# nbrcs_copol = np.full([*pol_shape], np.nan)
# nbrcs_xpol = np.full([*pol_shape], np.nan)
nbrcs = np.full([*transmitter_id.shape], np.nan)

coherency_ratio = np.full([*transmitter_id.shape], np.nan)
coherency_state = np.full([*transmitter_id.shape], np.nan)
# derive amb-function (chi2) to be used in computing A_eff
# % Matlab corrects delay/Doppler index by adding +1, Python doesn't
chi2 = get_chi2(
    40, 5, center_delay_bin, center_doppler_bin, delay_bin_res, doppler_bin_res
)  # 0-based

t0 = timer()
# iterate over each second of flight
for sec in range(len(transmitter_id)):
    # retrieve velocities and altitdues
    # bundle up craft vel data into per sec
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    rx_alt1 = rx_pos_lla[2][sec]

    # variables are solved only for LHCP channels
    for ngrx_channel in range(J_2):
        # retrieve tx velocities
        # bundle up velocity data into per sec
        tx_vel_xyz1 = np.array(
            [
                tx_vel_x[sec][ngrx_channel],
                tx_vel_y[sec][ngrx_channel],
                tx_vel_z[sec][ngrx_channel],
            ]
        )

        # azimuth angle between TX and RX velocity
        unit_rx_vel1 = rx_vel_xyz1 / np.linalg.norm(rx_vel_xyz1, 2)
        unit_tx_vel1 = tx_vel_xyz1 / np.linalg.norm(tx_vel_xyz1, 2)

        # 1st input of A_eff
        az_angle1 = math.degrees(math.acos(np.dot(unit_rx_vel1, unit_tx_vel1)))

        sx_pos_xyz1 = [
            sx_pos_x[sec][ngrx_channel],
            sx_pos_y[sec][ngrx_channel],
            sx_pos_z[sec][ngrx_channel],
        ]
        sx_lla1 = ecef2lla.transform(*sx_pos_xyz1, radians=False)

        # 2nd input of A_eff
        rx_alt_corrected1 = rx_alt1 - sx_lla1[2]

        # % 3rd input of A_eff
        inc_angle1 = sx_inc_angle[sec][ngrx_channel]

        brcs_copol1 = brcs[sec][ngrx_channel]
        brcs_xpol1 = brcs[sec][ngrx_channel + J_2]
        counts_LHCP1 = ddm_power_counts[sec][ngrx_channel]
        snr_LHCP1 = ddm_snr[sec][ngrx_channel]

        # evaluate delay and Doppler bin location at SP
        # Matlab uses +1, not required in Python 0-based indexing
        sp_delay_row1 = sp_delay_row[sec][ngrx_channel]  # +1;
        sp_doppler_col1 = sp_doppler_col[sec][ngrx_channel]  # +1;

        # ensure the SP is within DDM range (account for python vs Matlab indexing)
        SP_cond = (0 <= sp_delay_row1 <= 38) and (0 <= sp_doppler_col1 <= 4)
        # ensure interpolate within reasonable range
        interp_cond = rx_alt_bins[0] <= rx_alt_corrected1 <= rx_alt_bins[-1]
        angle_cond = 0 <= inc_angle1 <= 80

        if SP_cond and interp_cond and angle_cond:
            # note that the A_eff1 is transposed due to shape inconsistencies
            #  (40,5) vs (5,40)
            A_eff1 = get_ddm_Aeff4(
                rx_alt_corrected1,
                inc_angle1,
                az_angle1,
                sp_delay_row1,
                sp_doppler_col1,
                chi2,
                A_phy_LUT_interp,
            ).T

            # derive NBRCS - single theoretical SP bin
            # formerly delay_intg1 and delay_frac1, shortened for code clarity
            del_intg1 = int(np.floor(sp_delay_row1) + 1)
            del_frac1 = sp_delay_row1 - np.floor(sp_delay_row1)

            # formerly delay_intg1 and delay_frac1, shortened for code clarity
            dop_intg1 = int(np.floor(sp_doppler_col1) + 1)
            dop_frac1 = sp_doppler_col1 - np.floor(sp_doppler_col1)

            term1 = 1 - dop_frac1
            term2 = 1 - del_frac1

            if dop_intg1 <= 4:
                brcs_copol_ddma1 = (
                    (term1 * term2 * brcs_copol1[del_intg1, dop_intg1])
                    + (term1 * del_frac1 * brcs_copol1[del_intg1 + 1, dop_intg1])
                    + (dop_frac1 * term2 * brcs_copol1[del_intg1, dop_intg1 + 1])
                    + (
                        dop_frac1
                        * del_frac1
                        * brcs_copol1[del_intg1 + 1, dop_intg1 + 1]
                    )
                )

                brcs_xpol_ddma1 = (
                    (term1 * term2 * brcs_xpol1[del_intg1, dop_intg1])
                    + (term1 * del_frac1 * brcs_xpol1[del_intg1 + 1, dop_intg1])
                    + (dop_frac1 * term2 * brcs_xpol1[del_intg1, dop_intg1 + 1])
                    + (dop_frac1 * del_frac1 * brcs_xpol1[del_intg1 + 1, dop_intg1 + 1])
                )

                A_eff_ddma1 = (
                    (term1 * term2 * A_eff1[del_intg1, dop_intg1])
                    + (term1 * del_frac1 * A_eff1[del_intg1 + 1, dop_intg1])
                    + (dop_frac1 * term2 * A_eff1[del_intg1, dop_intg1 + 1])
                    + (dop_frac1 * del_frac1 * A_eff1[del_intg1 + 1, dop_intg1 + 1])
                )

            else:
                brcs_copol_ddma1 = (1 - del_frac1) * brcs_copol1[
                    del_intg1, dop_intg1
                ] + (del_frac1 * brcs_copol1[del_intg1 + 1, dop_intg1])

                brcs_xpol_ddma1 = (1 - del_frac1) * brcs_xpol1[del_intg1, dop_intg1] + (
                    del_frac1 * brcs_xpol1[del_intg1 + 1, dop_intg1]
                )
                A_eff_ddma1 = (1 - del_frac1) * A_eff1[del_intg1, dop_intg1] + (
                    del_frac1 * A_eff1[del_intg1 + 1, dop_intg1]
                )

            nbrcs_copol1 = brcs_copol_ddma1 / A_eff_ddma1
            nbrcs_xpol1 = brcs_xpol_ddma1 / A_eff_ddma1

            # coherent reflection
            CR1, CS1 = coh_det(counts_LHCP1, snr_LHCP1)

            A_eff[sec][ngrx_channel] = A_eff1
            nbrcs_scatter_area[sec][ngrx_channel] = A_eff_ddma1

            nbrcs[sec][ngrx_channel] = nbrcs_copol1
            nbrcs[sec][ngrx_channel + J_2] = nbrcs_xpol1

            coherency_ratio[sec][ngrx_channel] = CR1
            coherency_state[sec][ngrx_channel] = CS1
print(f"******** finish processing part 6 data with {timer() - t0}********")

A_eff[:, J_2:J] = A_eff[:, 0:J_2]
nbrcs_scatter_area[:, J_2:J] = nbrcs_scatter_area[:, 0:J_2]

coherency_ratio[:, J_2:J] = coherency_ratio[:, 0:J_2]
coherency_state[:, J_2:J] = coherency_state[:, 0:J_2]

L1_postCal["eff_scatter"] = A_eff
L1_postCal["nbrcs_scatter_area"] = nbrcs_scatter_area
L1_postCal["ddm_nbrcs"] = nbrcs

# L1_postCal["coherency_ratio"] = coherency_ratio
L1_postCal["coherence_metric"] = coherency_ratio
L1_postCal["coherence_state"] = coherency_state

# Part 7: fresnel dimensions and cross Pol

fresnel_coeff = np.full([*transmitter_id.shape], np.nan)
fresnel_minor = np.full([*transmitter_id.shape], np.nan)
fresnel_major = np.full([*transmitter_id.shape], np.nan)
fresnel_orientation = np.full([*transmitter_id.shape], np.nan)

nbrcs_cross_pol = np.full([*transmitter_id.shape], np.nan)

t0 = timer()
# TODO can probably condense this loop into thre above loop
for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J):
        tx_pos_xyz1 = [
            tx_pos_x[sec][ngrx_channel],
            tx_pos_y[sec][ngrx_channel],
            tx_pos_z[sec][ngrx_channel],
        ]
        rx_pos_xyz1 = [rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]]
        sx_pos_xyz1 = [
            sx_pos_x[sec][ngrx_channel],
            sx_pos_y[sec][ngrx_channel],
            sx_pos_z[sec][ngrx_channel],
        ]

        inc_angle1 = sx_inc_angle[sec][ngrx_channel]
        dist_to_coast1 = dist_to_coast_km[sec][ngrx_channel]
        ddm_ant1 = ddm_ant[sec][ngrx_channel]

        if not np.isnan(ddm_ant1):
            fresnel_coeff1, fresnel_axis1, fresnel_orientation1 = get_fresnel(
                tx_pos_xyz1,
                rx_pos_xyz1,
                sx_pos_xyz1,
                dist_to_coast1,
                inc_angle1,
                ddm_ant1,
            )

            fresnel_coeff[sec][ngrx_channel] = fresnel_coeff1
            fresnel_major[sec][ngrx_channel] = fresnel_axis1[0]
            fresnel_minor[sec][ngrx_channel] = fresnel_axis1[1]
            fresnel_orientation[sec][ngrx_channel] = fresnel_orientation1

        # Do this once here rather than another loop over Sec and J_2
        if ngrx_channel < J_2:
            nbrcs_LHCP1 = nbrcs[sec][ngrx_channel]
            nbrcs_RHCP1 = nbrcs[sec][ngrx_channel + J_2]
            CP1 = nbrcs_LHCP1 / nbrcs_RHCP1
            if CP1 > 0:
                CP_db1 = power2db(CP1)
                nbrcs_cross_pol[sec][ngrx_channel] = CP_db1

print(f"******** finish processing part 7 data with {timer() - t0}********")

nbrcs_cross_pol[:, J_2:J] = nbrcs_cross_pol[:, 0:J_2]

L1_postCal["fresnel_coeff"] = fresnel_coeff
L1_postCal["fresnel_major"] = fresnel_major
L1_postCal["fresnel_minor"] = fresnel_minor
L1_postCal["fresnel_orientation"] = fresnel_orientation

# LNA noise figure is 3 dB according to the specification
L1_postCal["nbrcs_cross_pol"] = nbrcs_cross_pol
L1_postCal["lna_noise_figure"] = np.full([*transmitter_id.shape], 3)


# Quality Flags

quality_flags1 = np.full([*transmitter_id.shape], np.nan)

for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J):
        quality_flag1_1 = np.full([26, 1], 0)

        # flag 1, 2 and 22  0-based indexing
        rx_roll1 = rx_roll[sec]
        rx_pitch1 = rx_pitch[sec]
        rx_yaw1 = rx_yaw[sec]

        # TODO is this an index issue???
        if (rx_roll1 >= 29) or (rx_pitch1 >= 9) or (rx_yaw1 >= 4):  # 0-based indexing
            quality_flag1_1[2] = 1
        else:
            quality_flag1_1[1] = 1

        if rx_roll1 > 1:
            quality_flag1_1[22] = 1

        # flag 3   0-based indexing
        quality_flag1_1[3] = 0

        # flag 4 and 5
        trans_id1 = transmitter_id[sec][ngrx_channel]
        if trans_id1 == 0:
            quality_flag1_1[4] = 1

        if trans_id1 == 28:
            quality_flag1_1[5] = 1

        # flag 6 and 9
        snr_db1 = ddm_snr[sec][ngrx_channel]

        if sec > 0:  # 0-based indexing
            snr_db2 = ddm_snr[sec - 1][ngrx_channel]
            diff1 = (db2power(snr_db1) - db2power(snr_db2)) / db2power(snr_db1)
            diff2 = snr_db1 - snr_db2

            if abs(diff1) > 0.1:
                quality_flag1_1[6] = 1

            if abs(diff2) > 0.24:
                quality_flag1_1[9] = 1

        # flag 7 and 8
        dist_to_coast1 = dist_to_coast_km[sec][ngrx_channel]

        if dist_to_coast1 > 0:
            quality_flag1_1[7] = 1

        if dist_to_coast1 > -25:
            quality_flag1_1[8] = 1

        # flag 10
        ant_temp1 = ant_temp_nadir[sec]
        if sec > 0:
            ant_temp2 = ant_temp_nadir[sec - 1]
            rate = (ant_temp2 - ant_temp1) * 60

            if rate > 1:
                quality_flag1_1[10] = 1

        # flag 11
        zenith_code_phase1 = zenith_code_phase[sec][ngrx_channel]
        signal_code_phase1 = delay_correction(
            meter2chips(add_range_to_sp[sec][ngrx_channel]), 1023
        )
        diff1 = zenith_code_phase1 - signal_code_phase1
        if diff1 >= 10:
            quality_flag1_1[11] = 1

        # flag 14 and 15
        sp_delay_row1 = sp_delay_row[sec][ngrx_channel]
        sp_dopp_col = sp_doppler_col[sec][ngrx_channel]

        if not np.isnan(sp_delay_row1):
            if (sp_delay_row1 < 15) or (sp_delay_row1 > 35):
                quality_flag1_1[14] = 1

        if not np.isnan(sp_dopp_col):
            if (sp_dopp_col < 2) or (sp_dopp_col > 4):
                quality_flag1_1[15] = 1

        # flag 16
        if not np.isnan(sp_delay_row1) and not np.isnan(sp_dopp_col):
            if (
                (math.floor(sp_delay_row1) < 38)
                and (math.floor(sp_delay_row1) > 0)
                and (math.floor(sp_dopp_col) < 5)
                and (math.floor(sp_dopp_col) > 1)
            ):
                sp_dopp_col_range = list(
                    range(math.floor(sp_dopp_col) - 1, math.floor(sp_dopp_col) + 2)
                )
                sp_delay_raw_range = list(
                    range(math.floor(sp_delay_row1), math.floor(sp_dopp_col) + 4)
                )  # TODO: sp_dopp_col, again?
                brcs_ddma = brcs[sp_delay_raw_range, :][:, sp_dopp_col_range]
                det = brcs_ddma[brcs_ddma < 0]
                if len(det) > 0:
                    quality_flag1_1[16] = 1

        # flag 17
        tx_pos_x1 = tx_pos_x[sec][ngrx_channel]
        prn_code1 = prn_code[sec][ngrx_channel]
        if (tx_pos_x1 == 0) and (not np.isnan(prn_code1)):
            quality_flag1_1[17] = 1

        # flag 18
        sx_pos_x1 = sx_pos_x[sec][ngrx_channel]
        if np.isnan(sx_pos_x1) and (not np.isnan(prn_code1)):
            quality_flag1_1[18] = 1

        # flag 19
        rx_gain1 = sx_rx_gain_copol[sec][ngrx_channel]
        if np.isnan(rx_gain1) and (not np.isnan(prn_code1)):
            quality_flag1_1[19] = 1

        quality_flag1_1[20] = 1

        # flag 21 and 25
        rx_alt = rx_pos_lla[2][sec]
        if rx_alt > 15000:
            quality_flag1_1[21] = 1
        if rx_alt < 700:
            quality_flag1_1[25] = 1

        # flag 23
        prn1 = prn_code[sec][ngrx_channel]
        if prn1 == 28:
            quality_flag1_1[23] = 1

        # flag 24
        # rx_vel_xyz1 = rx_vel_xyz[sec][ngrx_channel]
        rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
        rx_speed1 = np.linalg.norm(rx_vel_xyz1, 2)
        if rx_speed1 > 150:
            quality_flag1_1[24] = 1

        # TODO flag 12 and flag 13 missing from matlab?

        # flag 1
        if (
            quality_flag1_1[2] == 1
            or quality_flag1_1[3] == 1
            or quality_flag1_1[4] == 1
            or quality_flag1_1[5] == 1
            or quality_flag1_1[6] == 1
            or quality_flag1_1[9] == 1
            or quality_flag1_1[10] == 1
            or quality_flag1_1[11] == 1
            or quality_flag1_1[12] == 1
            or quality_flag1_1[13] == 1
            or quality_flag1_1[14] == 1
            or quality_flag1_1[15] == 1
            or quality_flag1_1[16] == 1
            or quality_flag1_1[17] == 1
            or quality_flag1_1[19] == 1
            or quality_flag1_1[21] == 1
            or quality_flag1_1[22] == 1
        ):
            quality_flag1_1[0] = 1

        quality_flags1[sec][ngrx_channel] = get_quality_flag(quality_flag1_1)


L1_postCal["quality_flags1"] = quality_flags1

definition_file = "./dat/L1_Dict/L1_Dict_v2_1m.xlsx"
output_file = "./out/mike_test.nc"

# to netcdf
write_netcdf(L1_postCal, definition_file, output_file)
