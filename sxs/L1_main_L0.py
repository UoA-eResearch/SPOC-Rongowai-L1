# mike.laverick@auckland.ac.nz
# L1_main_L0.py
import math
import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import numpy as np
from datetime import datetime

from aeff import aeff_and_nbrcs
from brcs import brcs_calculations
from calibration import ddm_calibration
from fresnel import fresnel_calculations
from gps import gps2utc, utc2gps, satellite_orbits
from load_files import L0_file, input_files, load_orbit_file
from noise import noise_floor_prep, noise_floor
from projections import ecef2lla
from quality_flags import quality_flag_calculations
from specular import specular_calculations
from utils import interp_ddm
from output import L1_file, write_netcdf

### ---------------------- Prelaunch 1: Load L0 data

# specify input L0 netcdf file
raw_data_path = Path().absolute().joinpath(Path("./dat/raw/"))
L0_filename = raw_data_path.joinpath(Path("20221103-121416_NZNV-NZCH.nc"))
# L0_filename = Path("20230404-065056_NZTU-NZWN.nc")
# L0_dataset = nc.Dataset(raw_data_path.joinpath(L0_filename))
L0 = L0_file(L0_filename)

### ---------------------- Prelaunch 1.5: Filter valid timestampes

# identify and compensate the value equal to 0 (randomly happens)
assert not (L0.pvt_gps_week == 0).any(), "pvt_gps_week contains 0, need to compensate."

# the below is to process when ddm-related and rx-related variables do not
# have the same length, which happens for some of the L0 products
assert (
    L0.pvt_gps_week.shape[0] == L0.I
), "pvt_gps_week and transmitter_id do not have the same length."
#
# TODO: Additional processing if ddm- and rx- related varaibles aren't the same length
#

integration_duration = L0.coherent_duration * L0.non_coherent_integrations * 1000


### ---------------------- Prelaunch 2 - define external data paths and filenames

# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path("./dat/L1a_cal/"))
L1a_cal_ddm_counts_db_filename = L1a_path.joinpath("L1A_cal_ddm_counts_dB.dat")
L1a_cal_ddm_power_dbm_filename = L1a_path.joinpath("L1A_cal_ddm_power_dBm.dat")

# load SRTM_30 DEM
dem_path = Path().absolute().joinpath(Path("./dat/dem/"))
dem_filename = dem_path.joinpath(Path("nzsrtm_30_v1.tif"))

# load DTU10 model
dtu_path = Path().absolute().joinpath(Path("./dat/dtu/"))
dtu_filename = dtu_path.joinpath(Path("dtu10_v1.dat"))

# load ocean/land (distance to coast) mask
landmask_path = Path().absolute().joinpath(Path("./dat/cst/"))
landmask_filename = landmask_path.joinpath(Path("dist_to_coast_nz_v1.dat"))

# load landcover mask
lcv_path = Path().absolute().joinpath(Path("./dat/lcv/"))
lcv_filename = lcv_path.joinpath(Path("lcv.png"))

# process inland water mask
pek_path = Path().absolute().joinpath(Path("./dat/pek/"))

water_mask_paths = ["160E_40S", "170E_30S", "170E_40S"]

# load PRN-SV and SV-EIRP(static) LUT
gps_path = Path().absolute().joinpath(Path("./dat/gps/"))
SV_PRN_filename = gps_path.joinpath(Path("PRN_SV_LUT_v1.dat"))
SV_eirp_filename = gps_path.joinpath(Path("GPS_SV_EIRP_Params_v7.dat"))


# load and process nadir NGRx-GNSS antenna patterns
rng_path = Path().absolute().joinpath(Path("./dat/rng/"))
LHCP_L_filename = rng_path.joinpath(Path("GNSS_LHCP_L_gain_db_i_v1.dat"))
LHCP_R_filename = rng_path.joinpath(Path("GNSS_LHCP_R_gain_db_i_v1.dat"))
RHCP_L_filename = rng_path.joinpath(Path("GNSS_RHCP_L_gain_db_i_v1.dat"))
RHCP_R_filename = rng_path.joinpath(Path("GNSS_RHCP_R_gain_db_i_v1.dat"))
rng_filenames = [LHCP_L_filename, LHCP_R_filename, RHCP_L_filename, RHCP_R_filename]


# scattering area LUT
A_phy_LUT_path = "./dat/A_phy_LUT/A_phy_LUT.dat"
# rx_alt_bins, inc_angle_bins, az_angle_bins, A_phy_LUT_all = load_A_phy_LUT(
#    A_phy_LUT_path
# )

inp = input_files(
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
)


### ---------------------- Part 1: General processing
# This part derives global constants, timestamps, and all the other
# parameters at ddm timestamps


# make array (ddm_pvt_bias) of non_coherent_integrations divided by 2
ddm_pvt_bias = L0.non_coherent_integrations / 2
# make array (pvt_utc) of gps to unix time (see above)
pvt_utc = np.array(
    [gps2utc(week, L0.pvt_gps_sec[i]) for i, week in enumerate(L0.pvt_gps_week)]
)
# make array (ddm_utc) of ddm_pvt_bias + pvt_utc
ddm_utc = pvt_utc + ddm_pvt_bias
# make arrays (gps_week, gps_tow) of ddm_utc to gps week/sec (inc. 1/2*integration time)
gps_week, gps_tow = utc2gps(ddm_utc)

# interpolate rx positions onto new time grid
rx_pos_x = interp_ddm(pvt_utc, L0.rx_pos_x_pvt, ddm_utc)
rx_pos_y = interp_ddm(pvt_utc, L0.rx_pos_y_pvt, ddm_utc)
rx_pos_z = interp_ddm(pvt_utc, L0.rx_pos_z_pvt, ddm_utc)
rx_pos_xyz = [rx_pos_x, rx_pos_y, rx_pos_z]
# interpolate rx velocities onto new time grid
rx_vel_x = interp_ddm(pvt_utc, L0.rx_vel_x_pvt, ddm_utc)
rx_vel_y = interp_ddm(pvt_utc, L0.rx_vel_y_pvt, ddm_utc)
rx_vel_z = interp_ddm(pvt_utc, L0.rx_vel_z_pvt, ddm_utc)
rx_vel_xyz = [rx_vel_x, rx_vel_y, rx_vel_z]
# interpolate rx roll/pitch/yaw onto new time grid
rx_roll = interp_ddm(pvt_utc, L0.rx_roll_pvt, ddm_utc)
rx_pitch = interp_ddm(pvt_utc, L0.rx_pitch_pvt, ddm_utc)
rx_yaw = interp_ddm(pvt_utc, L0.rx_yaw_pvt, ddm_utc)
rx_attitude = [rx_roll, rx_pitch, rx_yaw]
# interpolate bias+drift onto new time grid
rx_clk_bias_m = interp_ddm(pvt_utc, L0.rx_clk_bias_m_pvt, ddm_utc)
rx_clk_drift_mps = interp_ddm(pvt_utc, L0.rx_clk_drift_mps_pvt, ddm_utc)
rx_clk = [rx_clk_bias_m, rx_clk_drift_mps]

# interpolate "additional_range_to_SP" to new time grid
add_range_to_sp = np.full([*L0.add_range_to_sp_pvt.shape], np.nan)
for ngrx_channel in range(L0.J):
    add_range_to_sp[:, ngrx_channel] = interp_ddm(
        pvt_utc, L0.add_range_to_sp_pvt[:, ngrx_channel], ddm_utc
    )
# interpolate temperatures onto new time grid
ant_temp_zenith = interp_ddm(L0.eng_timestamp, L0.zenith_ant_temp_eng, ddm_utc)
ant_temp_nadir = interp_ddm(L0.eng_timestamp, L0.nadir_ant_temp_eng, ddm_utc)

# ecef2ella
lon, lat, alt = ecef2lla.transform(*rx_pos_xyz, radians=False)
rx_pos_lla = [lat, lon, alt]

# determine specular point "over land" flag from landmask
# replaces get_map_value function
status_flags_one_hz = inp.landmask_nz((lon, lat))
status_flags_one_hz[status_flags_one_hz > 0] = 5
status_flags_one_hz[status_flags_one_hz <= 0] = 4

# write global variables
output_file = "./out/mike_test.nc"
L1 = L1_file(output_file, "config_file", L0)

time_coverage_start_obj = datetime.utcfromtimestamp(ddm_utc[0])
L1.postCal["time_coverage_start"] = time_coverage_start_obj.strftime(
    "%Y-%m-%d %H:%M:%S"
)
time_coverage_end_obj = datetime.utcfromtimestamp(ddm_utc[-1])
L1.postCal["time_coverage_end"] = time_coverage_end_obj.strftime("%d-%m-%Y %H:%M:%S")
L1.postCal["time_coverage_resolution"] = ddm_utc[1] - ddm_utc[0]

# time coverage
hours, remainder = divmod((ddm_utc[-1] - ddm_utc[0] + 1), 3600)
minutes, seconds = divmod(remainder, 60)

# below is new for algorithm version 1.1
ref_timestamp_utc = ddm_utc[0]

pvt_timestamp_utc = pvt_utc - ref_timestamp_utc
ddm_timestamp_utc = ddm_utc - ref_timestamp_utc

L1.postCal[
    "time_coverage_duration"
] = f"P0DT{int(hours)}H{int(minutes)}M{int(seconds)}S"

# write timestamps and ac-related variables
L1.postCal["pvt_timestamp_utc"] = pvt_timestamp_utc

L1.postCal["ddm_timestamp_gps_week"] = gps_week
L1.postCal["ddm_timestamp_gps_sec"] = gps_tow
L1.postCal["ddm_timestamp_utc"] = ddm_timestamp_utc

L1.postCal["ddm_pvt_bias"] = ddm_pvt_bias

# 0-indexed sample and DDM
L1.postCal["sample"] = np.arange(0, len(L0.pvt_gps_sec))
L1.postCal["ddm"] = np.arange(0, L0.J)

L1.postCal["add_range_to_sp"] = add_range_to_sp

L1.postCal["ac_lat"] = rx_pos_lla[0]
L1.postCal["ac_lon"] = rx_pos_lla[1]
L1.postCal["ac_alt"] = rx_pos_lla[2]

L1.postCal["ac_pos_x"] = rx_pos_x
L1.postCal["ac_pos_y"] = rx_pos_y
L1.postCal["ac_pos_z"] = rx_pos_z

L1.postCal["ac_vel_x"] = rx_vel_x
L1.postCal["ac_vel_y"] = rx_vel_y
L1.postCal["ac_vel_z"] = rx_vel_z

L1.postCal["ac_roll"] = rx_attitude[0]
L1.postCal["ac_pitch"] = rx_attitude[1]
L1.postCal["ac_yaw"] = rx_attitude[2]

L1.postCal["rx_clk_bias"] = rx_clk_bias_m
L1.postCal["rx_clk_drift"] = rx_clk_drift_mps

L1.postCal["ant_temp_nadir"] = ant_temp_nadir
L1.postCal["ant_temp_zenith"] = ant_temp_zenith

L1.postCal["status_flags_one_hz"] = status_flags_one_hz

# part 1 ends

### ---------------------- Part 2: Derive TX related variables
# This part derives TX positions and velocities, maps between PRN and SVN,
# and gets track ID

# determine unique satellite transponder IDs
trans_id_unique = np.unique(L0.transmitter_id)
trans_id_unique = trans_id_unique[trans_id_unique > 0]

# create data arrays for C++ code to populate
orbit_bundle = [
    L1.postCal["tx_pos_x"],
    L1.postCal["tx_pos_y"],
    L1.postCal["tx_pos_z"],
    L1.postCal["tx_vel_x"],
    L1.postCal["tx_vel_y"],
    L1.postCal["tx_vel_z"],
    L1.postCal["tx_clk_bias"],
    L1.postCal["prn_code"],
    L1.postCal["sv_num"],
    L1.postCal["track_id"],
    trans_id_unique,
]

# determine whether flight spans a UTC day
if time_coverage_start_obj.day == time_coverage_end_obj.day:
    # determine single orbit file of that day
    orbit_file1 = load_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
    )
    # calculate satellite orbits, data assigned to orbit_bundle arrays
    satellite_orbits(
        L0.J_2,
        gps_week,
        gps_tow,
        L0.transmitter_id,
        inp.SV_PRN_LUT,
        orbit_file1,
        *orbit_bundle,
    )
else:
    # find idx of day change in timestamps
    # np.diff does "arr_new[i] = arr[i+1] - arr[i]" thus +1 to find changed idx
    change_idx = np.where(np.diff(np.floor(gps_tow / 86400)) > 0)[0][0] + 1
    # determine day_N and day_N+1 orbit files to use
    orbit_file1, orbit_file2 = load_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
        change_idx=change_idx,
    )
    # calculate first chunk of specular points using 1st orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        L0.J_2,
        gps_week,
        gps_tow,
        L0.transmitter_id,
        inp.SV_PRN_LUT,
        orbit_file1,
        *orbit_bundle,
        end=change_idx,
    )
    # calculate last chunk of specular points using 2nd orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        L0.J_2,
        gps_week,
        gps_tow,
        L0.transmitter_id,
        inp.SV_PRN_LUT,
        orbit_file2,
        *orbit_bundle,
        start=change_idx,
    )


# Part 3: L1a calibration
# this part converts from raw counts to signal power in watts and complete
# L1a calibration
ddm_calibration(
    L0.std_dev_rf1,
    L0.std_dev_rf2,
    L0.std_dev_rf3,
    L0.J,
    L1.postCal["prn_code"],
    L0.raw_counts,
    L0.rf_source,
    L0.first_scale_factor,
    L1.ddm_power_counts,
    L1.power_analog,
    L1.postCal["ddm_ant"],
    L1.postCal["inst_gain"],
)


# Part 4A: SP solver and geometries
specular_calculations(
    L0,
    L1,
    inp,
    rx_pos_x,
    rx_pos_y,
    rx_pos_z,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
    rx_roll,
    rx_pitch,
)

# Part 3B and 3C: noise floor, SNR, confidence flag of the SP solved
noise_floor_prep(
    L0,
    L1,
    add_range_to_sp,
    rx_pos_x,
    rx_pos_y,
    rx_pos_z,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
)
noise_floor(L0, L1)

# Part 5: Copol and xpol BRCS, reflectivity, peak reflectivity
brcs_calculations(L0, L1)

# Part 6: NBRCS and other related parameters
aeff_and_nbrcs(L0, L1, inp, rx_vel_x, rx_vel_y, rx_vel_z, rx_pos_lla)

# Part 7: fresnel dimensions and cross Pol
fresnel_calculations(L0, L1, rx_vel_x, rx_vel_y, rx_vel_z)

# Quality Flags
quality_flag_calculations(
    L0,
    L1,
    rx_roll,
    rx_pitch,
    rx_yaw,
    ant_temp_nadir,
    add_range_to_sp,
    rx_pos_lla,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
)

definition_file = "./dat/L1_Dict/L1_Dict_v2_1m.xlsx"

# to netcdf
write_netcdf(L1.postCal, definition_file, output_file)
