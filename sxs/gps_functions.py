from astropy.time import Time as astro_time
from ctypes import CDLL, c_double, c_uint, c_char_p, byref
import numpy as np
from pathlib import Path
from scipy import constants


c_path = Path().absolute().joinpath(Path("./sxs/lib/"))
GPS_GetSVInfo_filename = Path("GPS_GetSVInfo.so")
GPS_GetSVInfo = CDLL(str(c_path.joinpath(GPS_GetSVInfo_filename)))
double_array_8 = c_double * 8


def gps2utc(gpsweek, gpsseconds):
    """GPS time to unix timestamp.

    Parameters
    ----------
    gpsweek : int
        GPS week number, i.e. 1866.
    gpsseconds : int
        Number of seconds since the beginning of week.

    Returns
    -------
    numpy.float64
        Unix timestamp (UTC time).
    """
    secs_in_week = 604800
    secs = gpsweek * secs_in_week + gpsseconds

    t_gps = astro_time(secs, format="gps")
    t_utc = astro_time(t_gps, format="iso", scale="utc")

    return t_utc.unix


def utc2gps(timestamp):
    """unix timestamp to GPS.

    Parameters
    ----------
    numpy.float64
        Unix timestamp (UTC time).

    Returns
    -------
    gpsweek : int
        GPS week number, i.e. 1866.
    gpsseconds : int
        Number of seconds since the beginning of week.
    """
    secs_in_week = 604800
    t_utc = astro_time(timestamp, format="unix", scale="utc")
    t_gps = astro_time(t_utc, format="gps")
    gpsweek, gpsseconds = divmod(t_gps.value, secs_in_week)
    return gpsweek, gpsseconds


def satellite_orbits(
    J_2,
    gps_week,
    gps_tow,
    transmitter_id,
    SV_PRN_LUT,
    orbit_file,
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
    end=0,
    start=0,
):
    # Default range of seconds to iterate over
    iter_range = range(len(gps_tow))
    # truncate at end first, then at start for idx consistency
    if end:
        iter_range = iter_range[:end]
    if start:
        iter_range = iter_range[start:]

    for sec in iter_range:
        # ngrx_channel=j, 20 channels, 10 satellites
        for ngrx_channel in range(J_2):
            prn1 = transmitter_id[sec][ngrx_channel]
            if prn1:
                # change this to later in code?
                sv_num1 = SV_PRN_LUT[np.where(SV_PRN_LUT == prn1)[0][0]][1]
                sat_pos = double_array_8()
                # print(bytes(str(orbit_file), "utf-8"))
                GPS_GetSVInfo.main(
                    c_uint(prn1),
                    c_uint(int(gps_week[sec])),
                    c_double(gps_tow[sec]),
                    byref(sat_pos),
                    c_char_p(bytes(str(orbit_file), "utf-8")),
                )
                posx, posy, posz, bias, velx, vely, velz, _ = sat_pos
                tx_pos_x[sec][ngrx_channel] = tx_pos_x[sec][ngrx_channel + J_2] = posx
                tx_pos_y[sec][ngrx_channel] = tx_pos_y[sec][ngrx_channel + J_2] = posy
                tx_pos_z[sec][ngrx_channel] = tx_pos_z[sec][ngrx_channel + J_2] = posz
                tx_vel_x[sec][ngrx_channel] = tx_vel_x[sec][ngrx_channel + J_2] = velx
                tx_vel_y[sec][ngrx_channel] = tx_vel_y[sec][ngrx_channel + J_2] = vely
                tx_vel_z[sec][ngrx_channel] = tx_vel_z[sec][ngrx_channel + J_2] = velz
                tx_clk_bias[sec][ngrx_channel] = tx_clk_bias[sec][
                    ngrx_channel + J_2
                ] = (bias * constants.c)
                prn_code[sec][ngrx_channel] = prn_code[sec][ngrx_channel + J_2] = prn1
                sv_num[sec][ngrx_channel] = sv_num[sec][ngrx_channel + J_2] = sv_num1
                track_id[sec][ngrx_channel] = track_id[sec][ngrx_channel + J_2] = (
                    np.where(trans_id_unique == prn1)[0][0] + 1
                )
