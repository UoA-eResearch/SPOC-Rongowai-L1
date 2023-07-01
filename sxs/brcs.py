import cmath
import math
import numpy as np
import pymap3d as pm
from scipy import constants
from timeit import default_timer as timer

from calibration import db2power
from projections import ecef2lla


def ddm_brcs2(power_analog_LHCP, power_analog_RHCP, eirp_watt, rx_gain_db_i, TSx, RSx):
    """
    This version has copol xpol antenna gain implemented
    This function computes bistatic radar cross section (BRCS) according to
    Bistatic radar equation based on the inputs as below
    inputs:
    1) power_analog: L1a product in watts
    2) eirp_watt, rx_gain_db_i: gps eirp in watts and rx antenna gain in dBi
    3) TSx, RSx: Tx to Sx and Rx to Sx ranges
    outputs:
    1) brcs: bistatic RCS
    """
    # define constants
    f = 1575.42e6  # GPS L1 band, Hz
    _lambda = constants.c / f  # wavelength, m
    _lambda2 = _lambda * _lambda

    # derive BRCS
    rx_gain = db2power(rx_gain_db_i)  # linear rx gain
    rx_gain2 = rx_gain.reshape(2, 2)
    term1 = 4 * math.pi * np.power(4 * math.pi * TSx * RSx, 2)
    term2 = eirp_watt * _lambda2
    term3 = term1 / term2
    # term4 = term3 * np.power(rx_gain2, -1)
    term4 = term3 * np.linalg.matrix_power(rx_gain2, -1)

    brcs_copol = (term4[0, 0] * power_analog_LHCP) + (term4[0, 1] * power_analog_RHCP)
    brcs_xpol = (term4[1, 0] * power_analog_LHCP) + (term4[1, 1] * power_analog_RHCP)

    return brcs_copol, brcs_xpol


def ddm_refl2(
    power_analog_LHCP, power_analog_RHCP, eirp_watt, rx_gain_db_i, R_tsx, R_rsx
):
    """
    This function computes the land reflectivity by implementing the xpol
    antenna gain
    1)power_analog: L1a product, DDM power in watt
    2)eirp_watt: transmitter eirp in watt
    3)rx_gain_db_i: receiver antenna gain in the direction of SP, in dBi
    4)R_tsx, R_rsx: tx to sp range and rx to sp range, in meters
    outputs
    1) copol and xpol reflectivity
    """
    # define constants
    freq = 1575.42e6  # GPS L1 operating frequency, Hz
    _lambda = constants.c / freq  # wavelength, meter
    _lambda2 = _lambda * _lambda

    rx_gain = db2power(rx_gain_db_i)  # convert antenna gain to linear form
    rx_gain2 = rx_gain.reshape(2, 2)

    term1 = np.power(4 * math.pi * (R_tsx + R_rsx), 2)
    term2 = eirp_watt * _lambda2
    term3 = term1 / term2

    # term4 = term3 * np.power(rx_gain, -1)
    term4 = term3 * np.linalg.matrix_power(rx_gain2, -1)

    refl_copol = term4[0, 0] * power_analog_LHCP + term4[0, 1] * power_analog_RHCP
    refl_xpol = term4[1, 0] * power_analog_LHCP + term4[1, 1] * power_analog_RHCP
    return refl_copol, refl_xpol


def get_fresnel(tx_pos_xyz, rx_pos_xyz, sx_pos_xyz, dist_to_coast, inc_angle, ddm_ant):
    """
    this function derives Fresnel dimensions based on the Tx, Rx and Sx positions.
    Fresnel dimension is computed only the DDM is classified as coherent reflection.
    """
    # define constants
    eps_ocean = 74.62 + 51.92j  # complex permittivity of ocean
    fc = 1575.42e6  # operating frequency
    c = 299792458  # speed of light
    _lambda = c / fc  # wavelength

    # compute dimensions
    R_tsp = np.linalg.norm(np.array(tx_pos_xyz) - np.array(sx_pos_xyz), 2)
    R_rsp = np.linalg.norm(np.array(rx_pos_xyz) - np.array(sx_pos_xyz), 2)

    term1 = R_tsp * R_rsp
    term2 = R_tsp + R_rsp

    # semi axis
    a = math.sqrt(_lambda * term1 / term2)  # minor semi
    b = a / math.cos(math.radians(inc_angle))  # major semi

    # compute orientation relative to North
    lon, lat, alt = ecef2lla.transform(*sx_pos_xyz, radians=False)
    sx_lla = [lat, lon, alt]

    tx_e, tx_n, _ = pm.ecef2enu(*tx_pos_xyz, *sx_lla, deg=True)
    rx_e, rx_n, _ = pm.ecef2enu(*rx_pos_xyz, *sx_lla, deg=True)

    tx_en = np.array([tx_e, tx_n])
    rx_en = np.array([rx_e, rx_n])

    vector_tr = rx_en - tx_en
    unit_north = [0, 1]

    term3 = np.dot(vector_tr, unit_north)
    term4 = np.linalg.norm(vector_tr, 2) * np.linalg.norm(unit_north, 2)

    theta = math.degrees(math.acos(term3 / term4))

    fresnel_axis = [2 * b, 2 * a]
    fresnel_orientation = theta

    # fresenel coefficient only compute for ocean SPs
    fresnel_coeff = np.nan

    if dist_to_coast <= 0:
        sint = math.degrees(math.sin(math.radians(inc_angle)))
        cost = math.degrees(math.cos(math.radians(inc_angle)))

        temp1 = cmath.sqrt(eps_ocean - sint * sint)

        R_vv = (eps_ocean * cost - temp1) / (eps_ocean * cost + temp1)
        R_hh = (cost - temp1) / (cost + temp1)

        R_rl = (R_vv - R_hh) / 2
        R_rr = (R_vv + R_hh) / 2

        # -1 offset due to Matlab/Python indexing difference
        if ddm_ant == 1:
            fresnel_coeff = abs(R_rl) * abs(R_rl)
        # -1 offset due to Matlab/Python indexing difference
        elif ddm_ant == 2:
            fresnel_coeff = abs(R_rr) * abs(R_rr)

    return fresnel_coeff, fresnel_axis, fresnel_orientation


def brcs_calculations(L0, L1):
    # separate copol and xpol gain for using later
    rx_gain_copol_LL = L1.sx_rx_gain_copol[:, :10]
    rx_gain_copol_RR = L1.sx_rx_gain_copol[:, 10:20]

    rx_gain_xpol_RL = L1.sx_rx_gain_xpol[:, :10]
    rx_gain_xpol_LR = L1.sx_rx_gain_xpol[:, 10:20]

    # Part 4B: BRCS/NBRCS, reflectivity, coherent status and fresnel zone
    # BRCS, reflectivity

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
                sp_delay_row1 = np.floor(L1.sp_delay_row_LHCP[sec][ngrx_channel])  # + 1
                sp_doppler_col1 = np.floor(L1.sp_doppler_col[sec][ngrx_channel])  # + 1

                if (0 < sp_delay_row1 < 40) and (0 < sp_doppler_col1 < 5):
                    sp_refl_copol1 = refl_copol1[
                        int(sp_delay_row1), int(sp_doppler_col1)
                    ]
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

                L1.postCal["brcs"][sec][ngrx_channel] = brcs_copol1
                L1.postCal["brcs"][sec][ngrx_channel + L0.J_2] = brcs_xpol1

                L1.postCal["surface_reflectivity"][sec][ngrx_channel] = refl_copol1
                L1.postCal["surface_reflectivity"][sec][
                    ngrx_channel + L0.J_2
                ] = refl_xpol1

                L1.sp_refl[sec][ngrx_channel] = sp_refl_copol1
                L1.sp_refl[sec][ngrx_channel + L0.J_2] = sp_refl_xpol1

                L1.postCal["norm_refl_waveform"][sec][
                    ngrx_channel
                ] = norm_refl_waveform_copol1
                L1.postCal["norm_refl_waveform"][sec][
                    ngrx_channel + L0.J_2
                ] = norm_refl_waveform_xpol1
    print(f"******** finish processing part 5 data with {timer() - t0}********")
