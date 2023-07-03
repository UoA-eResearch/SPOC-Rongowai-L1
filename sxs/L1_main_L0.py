# mike.laverick@auckland.ac.nz
# L1_main_L0.py
import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path

from aeff import aeff_and_nbrcs
from brcs import brcs_calculations
from calibration import ddm_calibration
from fresnel import fresnel_calculations
from gps import calculate_satellite_orbits
from load_files import L0_file, input_files
from noise import noise_floor_prep, noise_floor
from quality_flags import quality_flag_calculations
from specular import specular_calculations
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

# write global variables
output_file = "./out/mike_test.nc"
L1 = L1_file(output_file, "config_file", L0, inp)
# part 1 ends

### ---------------------- Part 2: Derive TX related variables
# This part derives TX positions and velocities, maps between PRN and SVN,
# and gets track ID

calculate_satellite_orbits(L0, L1, inp)

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
    L1.rx_pos_x,
    L1.rx_pos_y,
    L1.rx_pos_z,
    L1.rx_vel_x,
    L1.rx_vel_y,
    L1.rx_vel_z,
    L1.rx_roll,
    L1.rx_pitch,
)

# Part 3B and 3C: noise floor, SNR, confidence flag of the SP solved
noise_floor_prep(
    L0,
    L1,
    L1.postCal["add_range_to_sp"],
    L1.rx_pos_x,
    L1.rx_pos_y,
    L1.rx_pos_z,
    L1.rx_vel_x,
    L1.rx_vel_y,
    L1.rx_vel_z,
)
noise_floor(L0, L1)

# Part 5: Copol and xpol BRCS, reflectivity, peak reflectivity
brcs_calculations(L0, L1)

# Part 6: NBRCS and other related parameters
aeff_and_nbrcs(L0, L1, inp, L1.rx_vel_x, L1.rx_vel_y, L1.rx_vel_z, L1.rx_pos_lla)

# Part 7: fresnel dimensions and cross Pol
fresnel_calculations(L0, L1, L1.rx_vel_x, L1.rx_vel_y, L1.rx_vel_z)

# Quality Flags
quality_flag_calculations(
    L0,
    L1,
    L1.rx_roll,
    L1.rx_pitch,
    L1.rx_yaw,
    L1.postCal["ant_temp_nadir"],
    L1.postCal["add_range_to_sp"],
    L1.rx_pos_lla,
    L1.rx_vel_x,
    L1.rx_vel_y,
    L1.rx_vel_z,
)

definition_file = "./dat/L1_Dict/L1_Dict_v2_1m.xlsx"
L1.add_to_postcal(L0)
# to netcdf
write_netcdf(L1.postCal, definition_file, output_file)
