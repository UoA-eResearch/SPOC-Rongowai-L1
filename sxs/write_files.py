import numpy as np
import pandas as pd
from pathlib import Path
import netCDF4 as nc

from utils import expand_to_RHCP


class L1_file:
    """
    Basic class to hold variables from the input L0 file
    """

    def __init__(self, filename, config_file, L0):
        self.postCal = {}
        self.filename = filename
        self.load_from_config(config_file)
        self.data_from_L0(L0)
        self.initialise_empties(L0)

    def load_from_config(self, config_file):
        # for now just hardcode stuff here
        # TODO read from config
        self.postCal["aircraft_reg"] = "ZK-NFA"  # default value
        self.postCal["ddm_source"] = 2  # 1 = GPS signal simulator, 2 = aircraft
        self.postCal["ddm_time_type_selector"] = 1  # 1 = middle of DDM sampling period
        self.postCal["dem_source"] = "SRTM30"

        # write algorithm and LUT versions
        self.postCal["l1_algorithm_version"] = "2.0"
        self.postCal["l1_data_version"] = "2.0"
        self.postCal["l1a_sig_LUT_version"] = "1"
        self.postCal["l1a_noise_LUT_version"] = "1"
        self.postCal["A_LUT_version"] = "1"
        self.postCal["ngrx_port_mapping_version"] = "1"
        self.postCal["nadir_ant_data_version"] = "1"
        self.postCal["zenith_ant_data_version"] = "1"
        self.postCal["prn_sv_maps_version"] = "1"
        self.postCal["gps_eirp_param_version"] = "7"
        self.postCal["land_mask_version"] = "1"
        self.postCal["surface_type_version"] = "1"
        self.postCal["mean_sea_surface_version"] = "1"
        self.postCal["per_bin_ant_version"] = "1"

    def data_from_L0(self, L0):
        # initialise variables that are diretly taken from L0 file
        self.postCal["delay_resolution"] = L0.delay_bin_res  # unit in chips
        self.postCal["dopp_resolution"] = L0.doppler_bin_res  # unit in Hz

        # write timestamps and ac-related variables
        self.postCal["pvt_timestamp_gps_week"] = L0.pvt_gps_week
        self.postCal["pvt_timestamp_gps_sec"] = L0.pvt_gps_sec

        self.postCal["sp_fsw_delay"] = L0.delay_center_chips
        self.postCal["sp_ngrx_dopp"] = L0.doppler_center_hz

        self.postCal["add_range_to_sp_pvt"] = L0.add_range_to_sp_pvt

        self.postCal["ac_pos_x_pvt"] = L0.rx_pos_x_pvt
        self.postCal["ac_pos_y_pvt"] = L0.rx_pos_y_pvt
        self.postCal["ac_pos_z_pvt"] = L0.rx_pos_z_pvt

        self.postCal["ac_vel_x_pvt"] = L0.rx_vel_x_pvt
        self.postCal["ac_vel_y_pvt"] = L0.rx_vel_y_pvt
        self.postCal["ac_vel_z_pvt"] = L0.rx_vel_z_pvt

        self.postCal["ac_roll_pvt"] = L0.rx_roll_pvt
        self.postCal["ac_pitch_pvt"] = L0.rx_pitch_pvt
        self.postCal["ac_yaw_pvt"] = L0.rx_yaw_pvt

        self.postCal["rx_clk_bias_pvt"] = L0.rx_clk_bias_m_pvt
        self.postCal["rx_clk_drift_pvt"] = L0.rx_clk_drift_mps_pvt

        self.postCal["zenith_sig_i2q2"] = L0.zenith_i2q2

    def initialise_empties(self, L0):
        self.postCal["tx_pos_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_pos_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_pos_z"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_vel_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_vel_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_vel_z"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["tx_clk_bias"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["prn_code"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sv_num"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["track_id"] = np.full([*L0.shape_2d], np.nan)

        # create data arrays to hold DDM power/count arrays
        self.ddm_power_counts = np.full([*L0.shape_4d], np.nan)
        self.power_analog = np.full([*L0.shape_4d], np.nan)

        self.postCal["ddm_ant"] = np.full([*L0.shape_2d], np.nan)  # 0-based
        self.postCal["inst_gain"] = np.full([*L0.shape_2d], np.nan)

        self.postCal["LOS_flag"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_pos_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_pos_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_pos_z"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_lat"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_lon"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_alt"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_vel_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_vel_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_vel_z"] = np.full([*L0.shape_2d], np.nan)

        self.dist_to_coast_km = np.full([*L0.shape_2d], np.nan)
        self.surface_type = np.full([*L0.shape_2d], np.nan)

        self.postCal["sx_inc_angle"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_d_snell_angle"] = np.full([*L0.shape_2d], np.nan)

        self.postCal["sx_theta_body"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_az_body"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_theta_enu"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sx_az_enu"] = np.full([*L0.shape_2d], np.nan)

        self.gps_boresight = np.full([*L0.shape_2d], np.nan)

        self.postCal["tx_to_sp_range"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["rx_to_sp_range"] = np.full([*L0.shape_2d], np.nan)

        self.postCal["gps_tx_power_db_w"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["gps_ant_gain_db_i"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["static_gps_eirp"] = np.full([*L0.shape_2d], np.nan)

        self.sx_rx_gain_copol = np.full([*L0.shape_2d], np.nan)
        self.sx_rx_gain_xpol = np.full([*L0.shape_2d], np.nan)

    def add_to_postcal(self):
        # quick hack for code variables that are saved with different dict names
        self.postCal["raw_counts"] = self.ddm_power_counts
        self.postCal["l1a_power_ddm"] = self.power_analog
        self.postCal["sp_surface_type"] = self.surface_type
        self.postCal["sp_dist_to_coast_km"] = self.dist_to_coast_km
        self.postCal["gps_off_boresight_angle_deg"] = self.gps_boresight
        self.postCal["sp_rx_gain_copol"] = self.sx_rx_gain_copol
        self.postCal["sp_rx_gain_xpol"] = self.sx_rx_gain_xpol

    def expand_sp_arrays(self):
        for key in [
            "sx_pos_x",
            "sx_pos_y",
            "sx_pos_z",
            "sx_lat",
            "sx_lon",
            "sx_alt",
            "sx_vel_x",
            "sx_vel_y",
            "sx_vel_z",
            "LOS_flag",
            "sx_inc_angle",
            "sx_d_snell_angle",
            "sx_theta_body",
            "sx_az_body",
            "sx_theta_enu",
            "sx_az_enu",
            "rx_to_sp_range",
            "tx_to_sp_range",
            "static_gps_eirp",
            "gps_tx_power_db_w",
            "gps_ant_gain_db_i",
        ]:
            self.postCal[key] = expand_to_RHCP(self.postCal[key])
        self.surface_type = expand_to_RHCP(self.surface_type)
        self.dist_to_coast_km = expand_to_RHCP(self.dist_to_coast_km)
        self.gps_boresight = expand_to_RHCP(self.gps_boresight)


def get_datatype(data_series, value=None):
    datatype = data_series["Data_type"].values[0]
    if datatype == "single":
        return np.single
    elif datatype == "double":
        return np.double
    elif datatype == "int8":
        return np.int8
    elif datatype == "int16":
        return np.int16
    elif datatype == "int32":
        return np.int32
    elif datatype == "int64":
        return np.int64
    elif datatype == "uint8":
        return np.uint8
    elif datatype == "uint16":
        return np.uint16
    elif datatype == "uint32":
        return np.uint32
    elif datatype == "uint64":
        return np.uint64
    elif datatype == "string":
        if isinstance(value, str):
            return "S" + str(len(value))
    else:
        raise Exception(f"datatype '{datatype}' not supported")


def get_dimensions(data_series):
    dim = data_series["Dimensions"].values[0].split(",")
    return tuple([x.strip() for x in dim])


def write_netcdf(dict_in, definition_file, output_file):
    assert isinstance(dict_in, dict), "input must be a dictionary"
    assert (
        Path(definition_file).suffix == ".xlsx"
    ), "definition file must be a .xlsx file"

    # read definition file
    df = pd.read_excel(definition_file)

    # open netcdf file
    with nc.Dataset(output_file, mode="w") as ncfile:
        # create dimensions
        ncfile.createDimension("sample", None)
        ncfile.createDimension("ddm", None)
        ncfile.createDimension("delay", None)
        ncfile.createDimension("doppler", None)

        for k, v in dict_in.items():
            print("writing: ", k)
            ds_k = df[df["Name"] == k]

            if ds_k.empty:
                print(
                    f"Warning: variable {k} not found in definition file, skip this variable."
                )
                continue
            elif len(ds_k) > 1:
                print(
                    f"Warning: find multiple variable {k} definition in definition file, skip this variable."
                )
                continue

            # if ds_k["Data_type"].str.contains("attribute").any():  # attribute
            if ds_k["Dimensions"].item() == "<none>":
                if ds_k["Units"].item() == "<none>":  # scalar
                    ncfile.setncattr(k, str(v))
                else:
                    var_k = ncfile.createVariable(
                        k, get_datatype(ds_k, v), (), zlib=True
                    )
                    var_k.units = ds_k["Units"].values[0]
                    var_k.long_name = ds_k["Long_name"].values[0]
                    var_k.comment = ds_k["Comment"].values[0]
                    var_k[()] = v
            else:  # variable
                var_k = ncfile.createVariable(
                    k, get_datatype(ds_k), get_dimensions(ds_k), zlib=True
                )
                var_k.units = ds_k["Units"].values[0]
                var_k.long_name = ds_k["Long_name"].values[0]
                var_k.comment = ds_k["Comment"].values[0]
                if len(get_dimensions(ds_k)) == len(v.shape) == 1:
                    var_k[:] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 2:
                    var_k[:, :] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 3:
                    var_k[:, :, :] = v
                elif len(get_dimensions(ds_k)) == len(v.shape) == 4:
                    var_k[:, :, :, :] = v
                elif (
                    len(get_dimensions(ds_k)) == 3
                    and len(v.shape) == 4
                    and v.shape[3] == 1
                ):  # norm_refl_waveform
                    var_k[:, :, :] = np.squeeze(v, axis=3)
                else:
                    raise Exception(f"variable {k} has unsupported dimensions")

        # print the Dataset object to see what we've got
        print(ncfile)
