# ----------- Input L0 dir ---------------
# If not defined then default to ./dat/raw/
#L1_L0_INPUT="/PATH/TO/L0_FILES"

# ----------- Output L1 dir --------------
# If not defined then default to ./out/
#L1_L1_OUTPUT="/PATH/TO/L1_FILES"

# ------------ L0 handling ---------------
# Boolean flag to control whether to delete
# L0 files once they are used to create a L1 file
DELETE_LO_FILE=0

# ----------- input dat files ------------
# ./dat/A_phy_LUT/
# A_phy_LUT files are hardocded for now
# ./dat/cst/
L1_LANDMASK="dist_to_coast_nz_v1.dat"
# ./dat/dem/
L1_DEM="dem_200m.dat"
# ./dat/dtu/
L1_DTU="dtu10_v1.dat"
# ./dat/gps/
L1_SV_PRN="PRN_SV_LUT_v1.dat"
L1_SV_eirp="GPS_SV_EIRP_Params_v7.dat"
# ./dat/L1_Dict/
L1_DICT="L1_Dict_v2_4.xlsx"
# ./dat/L1a_cal/
L1a_CAL_COUNTS="L1A_cal_ddm_counts_dB_v3.dat"
L1a_CAL_POWER="L1A_cal_ddm_power_dBm_v3.dat"
# ./dat/lcv/
L1_LANDCOVER="lcv.png"
# ./dat/orbits/
## this will be automatically populated
# ./dat/pek/
## water_mask_paths are hardcoded for now
# ./dat/rng/
L1_LHCP_L="GNSS_LHCP_L_gain_db_i_v2.dat"
L1_LHCP_R="GNSS_LHCP_R_gain_db_i_v2.dat"
L1_RHCP_L="GNSS_RHCP_L_gain_db_i_v2.dat"
L1_RHCP_R="GNSS_RHCP_R_gain_db_i_v2.dat"

# Credentials for automated orbital retrieval via cddis.nasa.gov
L1_CDDIS_USERNAME="USERNAME"
L1_CDDIS_PASSWORD="PASSWORD"

# ---------- L1 Metadata info ------------
# default value
AIRCRAFT_REG="ZK-NFA"
# 1=GPS signal simulator, 2=aircraft
DDM_SOURCE=2
# 1 = middle of DDM sampling period
DDM_TIME_TYPE_SELECTOR=1
DEM_SOURCE="SRTM30-200m"
# write algorithm and LUT versions
L1_ALGORITHM_VERSION="2.4.4"
L1_DATA_VERSION="2.4"
L1A_SIG_LUT_VERSION="1"
L1A_NOISE_LUT_VERSION="1"
A_LUT_VERSION="1.1"
NGRX_PORT_MAPPING_VERSION="1"
NADIR_ANT_DATA_VERSION="2"
ZENITH_ANT_DATA_VERSION="1"
PRN_SV_MAPS_VERSION="1"
GPS_EIRP_PARAM_VERSION="7"
LAND_MASK_VERSION="1"
SURFACE_TYPE_VERSION="1"
MEAN_SEA_SURFACE_VERSION="1"
PER_BIN_ANT_VERSION="1"

# global variables - update compliance results 30 June in MATLAB
CONVENTIONS="CF-1.9, ACDD-1.3, ISO-8601"
TITLE="Rongowai Level 1 Science Data Record Version 1.0"
HISTORY="TBD"
STANDARD_NAME_VOCABULARY="CF Standard Name Table v30"
COMMENT="DDMs are calibrated into Power (Watts) and Bistatic Radar Cross Section (m^2)"
PROCESSING_LEVEL="1"
CREATOR_TYPE="institution"
INSTITUTION="University of Auckland (UoA)"
CREATOR_NAME="Rongowai Science Payloads Operations Centre"
PUBLISHER_NAME="PO.DAAC"
PUBLISHER_EMAIL="rongowai.auckland.ac.nz"
PUBLISHER_URL="spoc.auckland.ac.nz"
GEOSPATIAL_LAT_MIN="-48.034"
GEOSPATIAL_LAT_MAX="-34.374"
GEOSPATIAL_LON_MIN="165.319"
GEOSPATIAL_LON_MAX="179.767"
