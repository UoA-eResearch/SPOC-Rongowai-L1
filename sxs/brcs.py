import cmath
import math
import numpy as np
import pymap3d as pm
from scipy import constants

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


def coh_det(raw_counts, snr_db):
    """
    this function computes the coherency of an input raw-count ddm
    Inputs
    1)raw ddm measured in counts
    2)SNR measured in decibels
    Outputs
    1)coherency ratio (CR)
    2)coherency state (CS)
    """
    peak_counts = np.amax(raw_counts)
    delay_peak, dopp_peak = np.unravel_index(raw_counts.argmax(), raw_counts.shape)

    # thermal noise exclusion
    # TODO: the threshold may need to be redefined
    if not np.isnan(snr_db):
        thre_coeff = 1.055 * math.exp(-0.193 * snr_db)
        thre = thre_coeff * peak_counts  # noise exclusion threshold

        raw_counts[raw_counts < thre] = 0

    # deterimine DDMA range
    delay_range = list(range(delay_peak - 1, delay_peak + 2))
    delay_min = min(delay_range)
    delay_max = max(delay_range)
    dopp_range = list(range(dopp_peak - 1, dopp_peak + 2))
    dopp_min = min(dopp_range)
    dopp_max = max(dopp_range)

    # determine if DDMA is within DDM, refine if needed
    if delay_min < 1:
        delay_range = [0, 1, 2]
    elif delay_max > 38:
        delay_range = [37, 38, 39]

    if dopp_min < 1:
        dopp_range = [0, 1, 2]
    elif dopp_max > 3:
        dopp_range = [2, 3, 4]

    C_in = np.sum(raw_counts[delay_range, :][:, dopp_range])  # summation of DDMA
    C_out = np.sum(raw_counts) - C_in  # summation of DDM excluding DDMA

    CR = C_in / C_out  # coherency ratio

    if CR >= 2:
        CS = 1
    else:  # CR < 2
        CS = 0

    return CR, CS
