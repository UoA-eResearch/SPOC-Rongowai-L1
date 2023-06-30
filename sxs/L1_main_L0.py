# mike.laverick@auckland.ac.nz
# L1_main_L0.py
import math
import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import netCDF4 as nc
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

from aeff import meter2chips, delay_correction, deldop, get_ddm_Aeff4, get_chi2
from brcs import ddm_brcs2, ddm_refl2, get_fresnel, coh_det
from calibration import ddm_calibration, db2power, power2db, get_quality_flag
from gps import gps2utc, utc2gps, satellite_orbits
from load_files import (
    L0_file,
    input_files,
    load_orbit_file,
)
from projections import ecef2lla
from specular import specular_calculations
from utils import interp_ddm
from write_files import L1_file, write_netcdf

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


### ----------------------  Part 3: L1a calibration
# this part converts from raw counts to signal power in watts and complete
# L1a calibration


# invoke calibration function which populates above arrays
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


#   # --------------------- Part 4A: SP solver and geometries

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

# expand to RHCP channels
L1.expand_sp_arrays()

# -------------------- Part 4B: BRCS/NBRCS, reflectivity, coherent status and fresnel zone
# initialise variables
peak_delay_row = np.full([*L0.shape_2d], np.nan)
peak_doppler_col = np.full([*L0.shape_2d], np.nan)

sp_delay_row = np.full([*L0.shape_2d], np.nan)
sp_delay_error = np.full([*L0.shape_2d], np.nan)

sp_doppler_col = np.full([*L0.shape_2d], np.nan)
sp_doppler_error = np.full([*L0.shape_2d], np.nan)

zenith_code_phase = np.full([*L0.shape_2d], np.nan)

noise_floor_all_LHCP = np.full([L0.I, L0.J_2], np.nan)
noise_floor_all_RHCP = np.full([L0.I, L0.J_2], np.nan)

delay_offset = 4

t0 = timer()
# derive floating SP bin location and effective scattering area A_eff
for sec in range(L0.I):
    # retrieve rx positions and velocities
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])

    for ngrx_channel in range(L0.J_2):
        # retrieve tx positions and velocities
        tx_pos_xyz1 = np.array(
            [
                L1.postCal["tx_pos_x"][sec][ngrx_channel],
                L1.postCal["tx_pos_y"][sec][ngrx_channel],
                L1.postCal["tx_pos_z"][sec][ngrx_channel],
            ]
        )
        tx_vel_xyz1 = np.array(
            [
                L1.postCal["tx_vel_x"][sec][ngrx_channel],
                L1.postCal["tx_vel_y"][sec][ngrx_channel],
                L1.postCal["tx_vel_z"][sec][ngrx_channel],
            ]
        )

        # retrieve sx-related parameters
        sx_pos_xyz1 = np.array(
            [
                L1.postCal["sx_pos_x"][sec][ngrx_channel],
                L1.postCal["sx_pos_y"][sec][ngrx_channel],
                L1.postCal["sx_pos_z"][sec][ngrx_channel],
            ]
        )

        counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
        # from onboard tracker
        add_range_to_sp1 = add_range_to_sp[sec][ngrx_channel]
        delay_center_chips1 = L0.delay_center_chips[sec][ngrx_channel]

        # zenith code phase
        add_range_to_sp_chips1 = meter2chips(add_range_to_sp1)
        zenith_code_phase1 = delay_center_chips1 + add_range_to_sp_chips1
        zenith_code_phase1 = delay_correction(zenith_code_phase1, 1023)

        # Part 3B: noise floor here to avoid another interation over [sec,L0.J_2]
        nf_counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
        nf_counts_RHCP1 = L1.ddm_power_counts[sec, ngrx_channel + L0.J_2, :, :]

        # delay_offset+1 due to difference between Matlab and Python indexing
        noise_floor_bins_LHCP1 = nf_counts_LHCP1[-(delay_offset + 1) :, :]
        noise_floor_bins_RHCP1 = nf_counts_RHCP1[-(delay_offset + 1) :, :]

        if (not np.isnan(L1.postCal["tx_pos_x"][sec][ngrx_channel])) and (
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
            d_delay_bin1 = d_delay_chips1 / L0.delay_bin_res

            sp_delay_row1 = L0.center_delay_bin - d_delay_bin1

            # SP doppler value
            _, sp_doppler_hz1, _ = deldop(
                tx_pos_xyz1, rx_pos_xyz1, tx_vel_xyz1, rx_vel_xyz1, sx_pos_xyz1
            )

            doppler_center_hz1 = L0.doppler_center_hz[sec][ngrx_channel]

            d_doppler_hz1 = doppler_center_hz1 - sp_doppler_hz1 + 250
            d_doppler_bin1 = d_doppler_hz1 / L0.doppler_bin_res

            sp_doppler_col1 = L0.center_doppler_bin - d_doppler_bin1

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
peak_delay_row[:, L0.J_2 : L0.J] = peak_delay_row[:, 0 : L0.J_2]
peak_doppler_col[:, L0.J_2 : L0.J] = peak_doppler_col[:, 0 : L0.J_2]

sp_delay_row[:, L0.J_2 : L0.J] = sp_delay_row[:, 0 : L0.J_2]
sp_doppler_col[:, L0.J_2 : L0.J] = sp_doppler_col[:, 0 : L0.J_2]

sp_delay_error[:, L0.J_2 : L0.J] = sp_delay_error[:, 0 : L0.J_2]
sp_doppler_error[:, L0.J_2 : L0.J] = sp_doppler_error[:, 0 : L0.J_2]

zenith_code_phase[:, L0.J_2 : L0.J] = zenith_code_phase[:, 0 : L0.J_2]

# save variables
L1.postCal["brcs_ddm_peak_bin_delay_row"] = peak_delay_row
L1.postCal["brcs_ddm_peak_bin_dopp_col"] = peak_doppler_col

L1.postCal["brcs_ddm_sp_bin_delay_row"] = sp_delay_row
L1.postCal["brcs_ddm_sp_bin_dopp_col"] = sp_doppler_col

L1.postCal["sp_delay_error"] = sp_delay_error
L1.postCal["sp_dopp_error"] = sp_doppler_error

L1.postCal["zenith_code_phase"] = zenith_code_phase


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
confidence_flag = np.full([*L0.shape_2d], np.nan)

# noise floor is the median of the average counts
noise_floor_LHCP = np.nanmedian(noise_floor_all_LHCP[valid_idx])
noise_floor_RHCP = np.nanmedian(noise_floor_all_RHCP[valid_idx])

# SNR of SP
# flag 0 for signal < 0
ddm_snr = np.full([*L0.shape_2d], np.nan)
snr_flag = np.full([*L0.shape_2d], np.nan)


for sec in range(L0.I):
    for ngrx_channel in range(L0.J_2):
        counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
        counts_RHCP1 = L1.ddm_power_counts[sec, ngrx_channel + L0.J_2, :, :]

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
            ddm_snr[sec][ngrx_channel + L0.J_2] = snr_RHCP_db1
            snr_flag[sec][ngrx_channel + L0.J_2] = snr_flag_RHCP1

            sx_delay_error1 = abs(sp_delay_error[sec][ngrx_channel])
            sx_doppler_error1 = abs(sp_doppler_error[sec][ngrx_channel])
            sx_d_snell_angle1 = abs(L1.postCal["sx_d_snell_angle"][sec][ngrx_channel])

            if not np.isnan(L1.postCal["tx_pos_x"][sec][ngrx_channel]):
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
        np.full([L0.shape_4d[2], L0.shape_4d[3]], noise_floor_LHCP),
        np.full([L0.shape_4d[2], L0.shape_4d[3]], noise_floor_RHCP),
    )
)
L1.postCal["ddm_noise_floor"] = noise_floor

confidence_flag[:, L0.J_2 : L0.J] = confidence_flag[:, 0 : L0.J_2]

L1.postCal["ddm_snr"] = ddm_snr
L1.postCal["ddm_snr_flag"] = snr_flag

L1.postCal["sp_confidence_flag"] = confidence_flag


L1.postCal["sp_ngrx_delay_correction"] = sp_delay_error
L1.postCal["sp_ngrx_dopp_correction"] = sp_doppler_error


# Part 5: Copol and xpol BRCS, reflectivity, peak reflectivity

# separate copol and xpol gain for using later
rx_gain_copol_LL = L1.sx_rx_gain_copol[:, :10]
rx_gain_copol_RR = L1.sx_rx_gain_copol[:, 10:20]

rx_gain_xpol_RL = L1.sx_rx_gain_xpol[:, :10]
rx_gain_xpol_LR = L1.sx_rx_gain_xpol[:, 10:20]

# BRCS, reflectivity
pol_shape = [*L0.shape_4d]

# brcs_copol = np.full([*pol_shape], np.nan)
# brcs_xpol = np.full([*pol_shape], np.nan)
brcs = np.full([*pol_shape], np.nan)

# refl_copol = np.full([*pol_shape], np.nan)
# refl_xpol = np.full([*pol_shape], np.nan)
surface_reflectivity = np.full([*pol_shape], np.nan)

sp_refl = np.full([*L0.shape_2d], np.nan)
norm_refl_waveform = np.full([*L0.shape_2d, 40, 1], np.nan)

# TODO draw these from a config file, here and in other places
cable_loss_db_LHCP = 0.6600
cable_loss_db_RHCP = 0.5840
powloss_LHCP = db2power(cable_loss_db_LHCP)
powloss_RHCP = db2power(cable_loss_db_RHCP)

t0 = timer()
for sec in range(L0.I):
    for ngrx_channel in range(L0.J_2):
        # compensate cable loss
        power_analog_LHCP1 = L1.power_analog[sec, ngrx_channel, :, :] * powloss_LHCP
        power_analog_RHCP1 = (
            L1.power_analog[sec, ngrx_channel + L0.J_2, :, :] * powloss_RHCP
        )

        R_tsx1 = L1.postCal["tx_to_sp_range"][sec][ngrx_channel]
        R_rsx1 = L1.postCal["rx_to_sp_range"][sec][ngrx_channel]
        rx_gain_dbi_1 = [
            rx_gain_copol_LL[sec][ngrx_channel],
            rx_gain_xpol_RL[sec][ngrx_channel],
            rx_gain_xpol_LR[sec][ngrx_channel],
            rx_gain_copol_RR[sec][ngrx_channel],
        ]
        gps_eirp1 = L1.postCal["static_gps_eirp"][sec][ngrx_channel]

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
            brcs[sec][ngrx_channel + L0.J_2] = brcs_xpol1

            surface_reflectivity[sec][ngrx_channel] = refl_copol1
            surface_reflectivity[sec][ngrx_channel + L0.J_2] = refl_xpol1

            sp_refl[sec][ngrx_channel] = sp_refl_copol1
            sp_refl[sec][ngrx_channel + L0.J_2] = sp_refl_xpol1

            norm_refl_waveform[sec][ngrx_channel] = norm_refl_waveform_copol1
            norm_refl_waveform[sec][ngrx_channel + L0.J_2] = norm_refl_waveform_xpol1
print(f"******** finish processing part 5 data with {timer() - t0}********")

L1.postCal["brcs"] = brcs

L1.postCal["surface_reflectivity"] = surface_reflectivity
L1.postCal["surface_reflectivity_peak"] = sp_refl
L1.postCal["norm_refl_waveform"] = norm_refl_waveform

# Part 5 ends

# Part 6: NBRCS and other related parameters

A_eff = np.full([*L0.shape_4d], np.nan)
nbrcs_scatter_area = np.full([*L0.shape_2d], np.nan)

pol_shape = [*L0.shape_4d]
# nbrcs_copol = np.full([*pol_shape], np.nan)
# nbrcs_xpol = np.full([*pol_shape], np.nan)
nbrcs = np.full([*L0.shape_2d], np.nan)

coherency_ratio = np.full([*L0.shape_2d], np.nan)
coherency_state = np.full([*L0.shape_2d], np.nan)
# derive amb-function (chi2) to be used in computing A_eff
# % Matlab corrects delay/Doppler index by adding +1, Python doesn't
chi2 = get_chi2(
    40,
    5,
    L0.center_delay_bin,
    L0.center_doppler_bin,
    L0.delay_bin_res,
    L0.doppler_bin_res,
)  # 0-based

t0 = timer()
# iterate over each second of flight
for sec in range(L0.I):
    # retrieve velocities and altitdues
    # bundle up craft vel data into per sec
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    rx_alt1 = rx_pos_lla[2][sec]

    # variables are solved only for LHCP channels
    for ngrx_channel in range(L0.J_2):
        # retrieve tx velocities
        # bundle up velocity data into per sec
        tx_vel_xyz1 = np.array(
            [
                L1.postCal["tx_vel_x"][sec][ngrx_channel],
                L1.postCal["tx_vel_y"][sec][ngrx_channel],
                L1.postCal["tx_vel_z"][sec][ngrx_channel],
            ]
        )

        # azimuth angle between TX and RX velocity
        unit_rx_vel1 = rx_vel_xyz1 / np.linalg.norm(rx_vel_xyz1, 2)
        unit_tx_vel1 = tx_vel_xyz1 / np.linalg.norm(tx_vel_xyz1, 2)

        # 1st input of A_eff
        az_angle1 = math.degrees(math.acos(np.dot(unit_rx_vel1, unit_tx_vel1)))

        sx_pos_xyz1 = [
            L1.postCal["sx_pos_x"][sec][ngrx_channel],
            L1.postCal["sx_pos_y"][sec][ngrx_channel],
            L1.postCal["sx_pos_z"][sec][ngrx_channel],
        ]
        sx_lla1 = ecef2lla.transform(*sx_pos_xyz1, radians=False)

        # 2nd input of A_eff
        rx_alt_corrected1 = rx_alt1 - sx_lla1[2]

        # % 3rd input of A_eff
        inc_angle1 = L1.postCal["sx_inc_angle"][sec][ngrx_channel]

        brcs_copol1 = brcs[sec][ngrx_channel]
        brcs_xpol1 = brcs[sec][ngrx_channel + L0.J_2]
        counts_LHCP1 = L1.ddm_power_counts[sec][ngrx_channel]
        snr_LHCP1 = ddm_snr[sec][ngrx_channel]

        # evaluate delay and Doppler bin location at SP
        # Matlab uses +1, not required in Python 0-based indexing
        sp_delay_row1 = sp_delay_row[sec][ngrx_channel]  # +1;
        sp_doppler_col1 = sp_doppler_col[sec][ngrx_channel]  # +1;

        # ensure the SP is within DDM range (account for python vs Matlab indexing)
        SP_cond = (0 <= sp_delay_row1 <= 38) and (0 <= sp_doppler_col1 <= 4)
        # ensure interpolate within reasonable range
        interp_cond = inp.rx_alt_bins[0] <= rx_alt_corrected1 <= inp.rx_alt_bins[-1]
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
                inp.A_phy_LUT_interp,
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
            nbrcs[sec][ngrx_channel + L0.J_2] = nbrcs_xpol1

            coherency_ratio[sec][ngrx_channel] = CR1
            coherency_state[sec][ngrx_channel] = CS1
print(f"******** finish processing part 6 data with {timer() - t0}********")

A_eff[:, L0.J_2 : L0.J] = A_eff[:, 0 : L0.J_2]
nbrcs_scatter_area[:, L0.J_2 : L0.J] = nbrcs_scatter_area[:, 0 : L0.J_2]

coherency_ratio[:, L0.J_2 : L0.J] = coherency_ratio[:, 0 : L0.J_2]
coherency_state[:, L0.J_2 : L0.J] = coherency_state[:, 0 : L0.J_2]

L1.postCal["eff_scatter"] = A_eff
L1.postCal["nbrcs_scatter_area"] = nbrcs_scatter_area
L1.postCal["ddm_nbrcs"] = nbrcs

# L1.postCal["coherency_ratio"] = coherency_ratio
L1.postCal["coherence_metric"] = coherency_ratio
L1.postCal["coherence_state"] = coherency_state

# Part 7: fresnel dimensions and cross Pol

fresnel_coeff = np.full([*L0.shape_2d], np.nan)
fresnel_minor = np.full([*L0.shape_2d], np.nan)
fresnel_major = np.full([*L0.shape_2d], np.nan)
fresnel_orientation = np.full([*L0.shape_2d], np.nan)

nbrcs_cross_pol = np.full([*L0.shape_2d], np.nan)

t0 = timer()
# TODO can probably condense this loop into thre above loop
for sec in range(L0.I):
    for ngrx_channel in range(L0.J):
        tx_pos_xyz1 = [
            L1.postCal["tx_pos_x"][sec][ngrx_channel],
            L1.postCal["tx_pos_y"][sec][ngrx_channel],
            L1.postCal["tx_pos_z"][sec][ngrx_channel],
        ]
        rx_pos_xyz1 = [rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]]
        sx_pos_xyz1 = [
            L1.postCal["sx_pos_x"][sec][ngrx_channel],
            L1.postCal["sx_pos_y"][sec][ngrx_channel],
            L1.postCal["sx_pos_z"][sec][ngrx_channel],
        ]

        inc_angle1 = L1.postCal["sx_inc_angle"][sec][ngrx_channel]
        dist_to_coast1 = L1.dist_to_coast_km[sec][ngrx_channel]
        ddm_ant1 = L1.postCal["ddm_ant"][sec][ngrx_channel]

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

        # Do this once here rather than another loop over Sec and L0.J_2
        if ngrx_channel < L0.J_2:
            nbrcs_LHCP1 = nbrcs[sec][ngrx_channel]
            nbrcs_RHCP1 = nbrcs[sec][ngrx_channel + L0.J_2]
            CP1 = nbrcs_LHCP1 / nbrcs_RHCP1
            if CP1 > 0:
                CP_db1 = power2db(CP1)
                nbrcs_cross_pol[sec][ngrx_channel] = CP_db1

print(f"******** finish processing part 7 data with {timer() - t0}********")

nbrcs_cross_pol[:, L0.J_2 : L0.J] = nbrcs_cross_pol[:, 0 : L0.J_2]

L1.postCal["fresnel_coeff"] = fresnel_coeff
L1.postCal["fresnel_major"] = fresnel_major
L1.postCal["fresnel_minor"] = fresnel_minor
L1.postCal["fresnel_orientation"] = fresnel_orientation

# LNA noise figure is 3 dB according to the specification
L1.postCal["nbrcs_cross_pol"] = nbrcs_cross_pol
L1.postCal["lna_noise_figure"] = np.full([*L0.shape_2d], 3)


# Quality Flags

quality_flags1 = np.full([*L0.shape_2d], np.nan)

for sec in range(L0.I):
    for ngrx_channel in range(L0.J):
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
        trans_id1 = L0.transmitter_id[sec][ngrx_channel]
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
        dist_to_coast1 = L1.dist_to_coast_km[sec][ngrx_channel]

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
        tx_pos_x1 = L1.postCal["tx_pos_x"][sec][ngrx_channel]
        prn_code1 = L1.postCal["prn_code"][sec][ngrx_channel]
        if (tx_pos_x1 == 0) and (not np.isnan(prn_code1)):
            quality_flag1_1[17] = 1

        # flag 18
        sx_pos_x1 = L1.postCal["sx_pos_x"][sec][ngrx_channel]
        if np.isnan(sx_pos_x1) and (not np.isnan(prn_code1)):
            quality_flag1_1[18] = 1

        # flag 19
        rx_gain1 = L1.sx_rx_gain_copol[sec][ngrx_channel]
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
        if prn_code1 == 28:
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


L1.postCal["quality_flags1"] = quality_flags1

definition_file = "./dat/L1_Dict/L1_Dict_v2_1m.xlsx"

# to netcdf
write_netcdf(L1.postCal, definition_file, output_file)
