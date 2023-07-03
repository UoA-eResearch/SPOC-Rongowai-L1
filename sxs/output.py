from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import netCDF4 as nc

from gps import gps2utc, utc2gps
from projections import ecef2lla
from utils import expand_to_RHCP, interp_ddm


class L1_file:
    """
    Basic class to hold variables from the input L0 file
    """

    def __init__(self, filename, config_file, L0, inp):
        self.postCal = {}
        self.filename = filename
        # load L1 file info from config file: versions etc
        self.load_from_config(config_file)
        # prep L0 variables that will be written to L1
        self.data_from_L0(L0)
        # initialise lots of empty arrays for later population
        self.initialise_empties(L0)
        # interpolate several aircraft variables onto fixed grid
        self.interpolate_L0(L0, inp)

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
        self.postCal["sp_pos_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_pos_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_pos_z"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_lat"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_lon"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_alt"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_vel_x"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_vel_y"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_vel_z"] = np.full([*L0.shape_2d], np.nan)

        self.dist_to_coast_km = np.full([*L0.shape_2d], np.nan)
        self.surface_type = np.full([*L0.shape_2d], np.nan)

        self.postCal["sp_inc_angle"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_d_snell_angle"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_theta_body"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_az_body"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_theta_enu"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["sp_az_enu"] = np.full([*L0.shape_2d], np.nan)

        self.gps_boresight = np.full([*L0.shape_2d], np.nan)

        self.postCal["tx_to_sp_range"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["rx_to_sp_range"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["gps_tx_power_db_w"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["gps_ant_gain_db_i"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["static_gps_eirp"] = np.full([*L0.shape_2d], np.nan)

        self.sx_rx_gain_copol = np.full([*L0.shape_2d], np.nan)
        self.sx_rx_gain_xpol = np.full([*L0.shape_2d], np.nan)

        self.peak_delay_row = np.full([*L0.shape_2d], np.nan)
        self.peak_doppler_col = np.full([*L0.shape_2d], np.nan)

        self.sp_delay_row = np.full([*L0.shape_2d], np.nan)
        self.sp_doppler_col = np.full([*L0.shape_2d], np.nan)
        self.sp_delay_error = np.full([*L0.shape_2d], np.nan)
        self.sp_doppler_error = np.full([*L0.shape_2d], np.nan)

        self.noise_floor_all_LHCP = np.full([L0.I, L0.J_2], np.nan)
        self.noise_floor_all_RHCP = np.full([L0.I, L0.J_2], np.nan)

        self.postCal["zenith_code_phase"] = np.full([*L0.shape_2d], np.nan)

        self.confidence_flag = np.full([*L0.shape_2d], np.nan)
        self.snr_flag = np.full([*L0.shape_2d], np.nan)

        self.postCal["ddm_snr"] = np.full([*L0.shape_2d], np.nan)

        self.postCal["brcs"] = np.full([*L0.shape_4d], np.nan)
        self.postCal["surface_reflectivity"] = np.full([*L0.shape_4d], np.nan)
        self.postCal["norm_refl_waveform"] = np.full([*L0.shape_2d, 40, 1], np.nan)

        self.sp_refl = np.full([*L0.shape_2d], np.nan)

        self.A_eff = np.full([*L0.shape_4d], np.nan)
        self.postCal["nbrcs_scatter_area"] = np.full([*L0.shape_2d], np.nan)

        self.nbrcs = np.full([*L0.shape_2d], np.nan)
        self.coherency_ratio = np.full([*L0.shape_2d], np.nan)
        self.coherency_state = np.full([*L0.shape_2d], np.nan)

        self.postCal["fresnel_coeff"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["fresnel_minor"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["fresnel_major"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["fresnel_orientation"] = np.full([*L0.shape_2d], np.nan)
        self.postCal["nbrcs_cross_pol"] = np.full([*L0.shape_2d], np.nan)

        self.postCal["quality_flags1"] = np.full([*L0.shape_2d], np.nan)

    def interpolate_L0(self, L0, inp):
        """interpolates several sets of variables onto fixed per second shape"""

        # make array (ddm_pvt_bias) of non_coherent_integrations divided by 2
        self.postCal["ddm_pvt_bias"] = L0.non_coherent_integrations / 2

        # make array (pvt_utc) of gps to unix time (see above)
        pvt_utc = np.array(
            [gps2utc(week, L0.pvt_gps_sec[i]) for i, week in enumerate(L0.pvt_gps_week)]
        )
        # make array (ddm_utc) of ddm_pvt_bias + pvt_utc
        ddm_utc = pvt_utc + self.postCal["ddm_pvt_bias"]
        # make arrays (gps_week, gps_tow) of ddm_utc to gps week/sec (inc. 1/2*integration time)
        self.gps_week, self.gps_tow = utc2gps(ddm_utc)

        # interpolate rx positions onto new time grid
        self.rx_pos_x = interp_ddm(pvt_utc, L0.rx_pos_x_pvt, ddm_utc)
        self.rx_pos_y = interp_ddm(pvt_utc, L0.rx_pos_y_pvt, ddm_utc)
        self.rx_pos_z = interp_ddm(pvt_utc, L0.rx_pos_z_pvt, ddm_utc)
        self.rx_pos_xyz = [self.rx_pos_x, self.rx_pos_y, self.rx_pos_z]
        # interpolate rx velocities onto new time grid
        self.rx_vel_x = interp_ddm(pvt_utc, L0.rx_vel_x_pvt, ddm_utc)
        self.rx_vel_y = interp_ddm(pvt_utc, L0.rx_vel_y_pvt, ddm_utc)
        self.rx_vel_z = interp_ddm(pvt_utc, L0.rx_vel_z_pvt, ddm_utc)
        # rx_vel_xyz = [self.rx_vel_x, self.rx_vel_y, self.rx_vel_z]
        # interpolate rx roll/pitch/yaw onto new time grid
        self.rx_roll = interp_ddm(pvt_utc, L0.rx_roll_pvt, ddm_utc)
        self.rx_pitch = interp_ddm(pvt_utc, L0.rx_pitch_pvt, ddm_utc)
        self.rx_yaw = interp_ddm(pvt_utc, L0.rx_yaw_pvt, ddm_utc)
        self.rx_attitude = [self.rx_roll, self.rx_pitch, self.rx_yaw]
        # interpolate bias+drift onto new time grid
        self.rx_clk_bias_m = interp_ddm(pvt_utc, L0.rx_clk_bias_m_pvt, ddm_utc)
        self.rx_clk_drift_mps = interp_ddm(pvt_utc, L0.rx_clk_drift_mps_pvt, ddm_utc)
        # rx_clk = [rx_clk_bias_m, rx_clk_drift_mps]

        # interpolate "additional_range_to_SP" to new time grid
        self.postCal["add_range_to_sp"] = np.full(
            [*L0.add_range_to_sp_pvt.shape], np.nan
        )
        for ngrx_channel in range(L0.J):
            self.postCal["add_range_to_sp"][:, ngrx_channel] = interp_ddm(
                pvt_utc, L0.add_range_to_sp_pvt[:, ngrx_channel], ddm_utc
            )
        # interpolate temperatures onto new time grid
        self.postCal["ant_temp_zenith"] = interp_ddm(
            L0.eng_timestamp, L0.zenith_ant_temp_eng, ddm_utc
        )
        self.postCal["ant_temp_nadir"] = interp_ddm(
            L0.eng_timestamp, L0.nadir_ant_temp_eng, ddm_utc
        )

        # ecef2ella
        lon, lat, alt = ecef2lla.transform(*self.rx_pos_xyz, radians=False)
        self.rx_pos_lla = [lat, lon, alt]

        # determine specular point "over land" flag from landmask
        # replaces get_map_value function
        self.postCal["status_flags_one_hz"] = inp.landmask_nz((lon, lat))
        self.postCal["status_flags_one_hz"][self.postCal["status_flags_one_hz"] > 0] = 5
        self.postCal["status_flags_one_hz"][
            self.postCal["status_flags_one_hz"] <= 0
        ] = 4

        self.time_coverage_start_obj = datetime.utcfromtimestamp(ddm_utc[0])
        self.postCal["time_coverage_start"] = self.time_coverage_start_obj.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.time_coverage_end_obj = datetime.utcfromtimestamp(ddm_utc[-1])
        self.postCal["time_coverage_end"] = self.time_coverage_end_obj.strftime(
            "%d-%m-%Y %H:%M:%S"
        )
        self.postCal["time_coverage_resolution"] = ddm_utc[1] - ddm_utc[0]

        # time coverage
        hours, remainder = divmod((ddm_utc[-1] - ddm_utc[0] + 1), 3600)
        minutes, seconds = divmod(remainder, 60)

        # below is new for algorithm version 1.1
        ref_timestamp_utc = ddm_utc[0]

        self.postCal["pvt_timestamp_utc"] = pvt_utc - ref_timestamp_utc
        self.postCal["ddm_timestamp_utc"] = ddm_utc - ref_timestamp_utc

        self.postCal[
            "time_coverage_duration"
        ] = f"P0DT{int(hours)}H{int(minutes)}M{int(seconds)}S"

        # 0-indexed sample and DDM
        self.postCal["sample"] = np.arange(0, len(L0.pvt_gps_sec))
        self.postCal["ddm"] = np.arange(0, L0.J)

        # determine unique satellite transponder IDs
        self.trans_id_unique = np.unique(L0.transmitter_id)
        self.trans_id_unique = self.trans_id_unique[self.trans_id_unique > 0]

    def add_to_postcal(self, L0):
        # quick hack for code variables that are saved with different dict names
        self.postCal["raw_counts"] = self.ddm_power_counts
        self.postCal["l1a_power_ddm"] = self.power_analog
        self.postCal["sp_surface_type"] = self.surface_type
        self.postCal["sp_dist_to_coast_km"] = self.dist_to_coast_km
        self.postCal["gps_off_boresight_angle_deg"] = self.gps_boresight
        self.postCal["sp_rx_gain_copol"] = self.sx_rx_gain_copol
        self.postCal["sp_rx_gain_xpol"] = self.sx_rx_gain_xpol
        self.postCal["brcs_ddm_peak_bin_delay_row"] = self.peak_delay_row
        self.postCal["brcs_ddm_peak_bin_dopp_col"] = self.peak_doppler_col
        self.postCal["brcs_ddm_sp_bin_delay_row"] = self.sp_delay_row
        self.postCal["brcs_ddm_sp_bin_dopp_col"] = self.sp_doppler_col
        self.postCal["sp_delay_error"] = self.sp_delay_error
        self.postCal["sp_dopp_error"] = self.sp_doppler_error
        self.postCal["sp_ngrx_delay_correction"] = self.sp_delay_error
        self.postCal["sp_ngrx_dopp_correction"] = self.sp_doppler_error
        self.postCal["ddm_snr_flag"] = self.snr_flag
        self.postCal["sp_confidence_flag"] = self.confidence_flag
        self.postCal["surface_reflectivity_peak"] = self.sp_refl
        self.postCal["eff_scatter"] = self.A_eff
        self.postCal["ddm_nbrcs"] = self.nbrcs
        self.postCal["coherence_metric"] = self.coherency_ratio
        self.postCal["coherence_state"] = self.coherency_state
        self.postCal["ddm_timestamp_gps_week"] = self.gps_week
        self.postCal["ddm_timestamp_gps_sec"] = self.gps_tow
        self.postCal["ac_pos_x"] = self.rx_pos_x
        self.postCal["ac_pos_y"] = self.rx_pos_y
        self.postCal["ac_pos_z"] = self.rx_pos_z
        self.postCal["ac_vel_x"] = self.rx_vel_x
        self.postCal["ac_vel_y"] = self.rx_vel_y
        self.postCal["ac_vel_z"] = self.rx_vel_z
        self.postCal["ac_roll"] = self.rx_attitude[0]
        self.postCal["ac_pitch"] = self.rx_attitude[1]
        self.postCal["ac_yaw"] = self.rx_attitude[2]
        self.postCal["rx_clk_bias"] = self.rx_clk_bias_m
        self.postCal["rx_clk_drift"] = self.rx_clk_drift_mps
        self.postCal["ac_lat"] = self.rx_pos_lla[0]
        self.postCal["ac_lon"] = self.rx_pos_lla[1]
        self.postCal["ac_alt"] = self.rx_pos_lla[2]
        # LNA noise figure is 3 dB according to the specification
        self.postCal["lna_noise_figure"] = np.full([*L0.shape_2d], 3)

    def expand_sp_arrays(self, J_2, J):
        """Expands a number of specular-point (sp) arrays from the lefthand channel
        to the righthand channels as well, i.e. (X,10) to (X,20)"""
        for key in [
            "sp_pos_x",
            "sp_pos_y",
            "sp_pos_z",
            "sp_lat",
            "sp_lon",
            "sp_alt",
            "sp_vel_x",
            "sp_vel_y",
            "sp_vel_z",
            "LOS_flag",
            "sp_inc_angle",
            "sp_d_snell_angle",
            "sp_theta_body",
            "sp_az_body",
            "sp_theta_enu",
            "sp_az_enu",
            "rx_to_sp_range",
            "tx_to_sp_range",
            "static_gps_eirp",
            "gps_tx_power_db_w",
            "gps_ant_gain_db_i",
        ]:
            self.postCal[key] = expand_to_RHCP(self.postCal[key], J_2, J)
        self.surface_type = expand_to_RHCP(self.surface_type, J_2, J)
        self.dist_to_coast_km = expand_to_RHCP(self.dist_to_coast_km, J_2, J)
        self.gps_boresight = expand_to_RHCP(self.gps_boresight, J_2, J)

    def expand_noise_arrays(self, J_2, J):
        """Expands a number of noise-related arrays from the lefthand channel
        to the righthand channels as well, i.e. (X,10) to (X,20)"""
        self.peak_delay_row = expand_to_RHCP(self.peak_delay_row, J_2, J)
        self.peak_doppler_col = expand_to_RHCP(self.peak_doppler_col, J_2, J)
        self.sp_delay_row = expand_to_RHCP(self.sp_delay_row, J_2, J)
        self.sp_doppler_col = expand_to_RHCP(self.sp_doppler_col, J_2, J)
        self.sp_delay_error = expand_to_RHCP(self.sp_delay_error, J_2, J)
        self.sp_doppler_error = expand_to_RHCP(self.sp_doppler_error, J_2, J)
        self.postCal["zenith_code_phase"] = expand_to_RHCP(
            self.postCal["zenith_code_phase"], J_2, J
        )


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
