import netCDF4 as nc
import numpy as np

from pathlib import Path
from termcolor import colored
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("Qt5Agg")

mat_data_path = Path().absolute().joinpath(Path("./out/"))
mat_L1_filename = Path("20221103-121416_NZNV-NZCH_L1.nc")
# L0_filenames = glob.glob("*.nc")
mat_L1 = nc.Dataset(mat_data_path.joinpath(mat_L1_filename))

py_data_path = Path().absolute().joinpath(Path("./out/"))
py_L1_filename = Path("mike_test.nc")
# L0_filenames = glob.glob("*.nc")
py_L1 = nc.Dataset(py_data_path.joinpath(py_L1_filename))

print(py_L1)

skip_list = [
    "delay_resolution",
    "dopp_resolution",
    "add_range_to_sp_pvt",
    "pvt_timestamp_gps_week",
    "pvt_timestamp_gps_sec",
    "pvt_timestamp_utc",
    "ddm_timestamp_gps_week",
    "ddm_timestamp_gps_sec",
    "ddm_timestamp_utc",
    "ddm_pvt_bias",
    "sample",
    "ddm",
    "sp_fsw_delay",
    "sp_ngrx_dopp",
    "add_range_to_sp",
    "add_range_to_sp_pvt",
    "ac_pos_x_pvt",
    "ac_pos_y_pvt",
    "ac_pos_z_pvt",
    "ac_pos_x",
    "ac_pos_y",
    "ac_pos_z",
    "ac_vel_x_pvt",
    "ac_vel_y_pvt",
    "ac_vel_z_pvt",
    "ac_roll_pvt",
    "ac_pitch_pvt",
    "ac_yaw_pvt",
    "rx_clk_bias_pvt",
    "rx_clk_drift_pvt",
    "rx_clk_bias",
    "rx_clk_drift",
    "ant_temp_nadir",
    "ant_temp_zenith",
    "zenith_sig_i2q2",
    "prn_code",
    "sv_num",
    "track_id",
    "brcs_ddm_peak_bin_delay_row",
    "brcs_ddm_peak_bin_dopp_col",
    "lna_noise_figure",
    "ddm_ant",
    "tx_clk_bias",
    "inst_gain",
    "LOS_flag",
    "gps_tx_power_db_w",
]

for var in py_L1.variables:
    print(var)
    if var in skip_list:
        continue
    # print("should be fine...")
    if var == "sample":
        var2 = "sample_index"
    else:
        var2 = var
    if var == "norm_refl_waveform":
        continue  # shape differences
    print(
        var,
        colored(mat_L1[var2].dtype, "green"),
        colored(py_L1[var].dtype, "blue"),
    )
    # print(colored(mat_L1[var2][:], "green"))
    # print(colored(py_L1[var][:], "blue"))
    print(colored(py_L1[var][:] - mat_L1[var2][:], "yellow"))
    """mean = np.nanmean(
        np.divide(np.abs(mat_L1[var2][:] - py_L1[var][:]), mat_L1[var2][:])
    )
    mmin = np.nanmin(
        np.divide(np.abs(mat_L1[var2][:] - py_L1[var][:]), mat_L1[var2][:])
    )
    mmax = np.nanmax(
        np.divide(np.abs(mat_L1[var2][:] - py_L1[var][:]), mat_L1[var2][:])
    )

    print(f"min={mmin}, mean={mean}, max={mmax}")"""

    if np.array_equal(
        np.array(mat_L1[var2][:]), np.array(py_L1[var][:]), equal_nan=True
    ):
        print(colored("yay", "yellow"))
    else:
        print(colored("################", "red"))
    # try:
    #    plt.imshow(py_L1[var][:] - mat_L1[var2][:], aspect="auto")
    #    plt.colorbar()
    #    plt.show()
    # except TypeError:
    #    print(f"invalid shape {(py_L1[var][:] - mat_L1[var2][:]).shape}")
    #    pass
    print()
    # plt.clf()
    # print()
    # print()

# print(len(mat_L1.variables))
# print(len(py_L1.variables))
