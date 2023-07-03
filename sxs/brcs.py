import math
import numpy as np
from scipy import constants
from timeit import default_timer as timer

from calibration import db2power


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

    # single noise floor from valid DDMs
    sp_delay_row_LHCP = L1.sp_delay_row[:, :10]  # reference to LHCP delay row

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
