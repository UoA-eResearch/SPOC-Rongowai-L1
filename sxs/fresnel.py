import cmath
import math
import numpy as np
import pymap3d as pm
from scipy import constants

from calibration import power2db
from projections import ecef2lla


def get_fresnel(tx_pos_xyz, rx_pos_xyz, sx_pos_xyz, dist_to_coast, inc_angle, ddm_ant):
    """
    this function derives Fresnel dimensions based on the Tx, Rx and Sx positions.
    Fresnel dimension is computed only the DDM is classified as coherent reflection.
    """
    # define constants
    eps_ocean = 74.62 + 51.92j  # complex permittivity of ocean
    fc = 1575.42e6  # operating frequency
    _lambda = constants.c / fc  # wavelength

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


def fresnel_calculations(L0, L1):
    rx_pos_x = L1.rx_pos_x
    rx_pos_y = L1.rx_pos_y
    rx_pos_z = L1.rx_pos_z

    for sec in range(L0.I):
        for ngrx_channel in range(L0.J):
            tx_pos_xyz1 = [
                L1.postCal["tx_pos_x"][sec][ngrx_channel],
                L1.postCal["tx_pos_y"][sec][ngrx_channel],
                L1.postCal["tx_pos_z"][sec][ngrx_channel],
            ]
            rx_pos_xyz1 = [rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]]
            sx_pos_xyz1 = [
                L1.postCal["sp_pos_x"][sec][ngrx_channel],
                L1.postCal["sp_pos_y"][sec][ngrx_channel],
                L1.postCal["sp_pos_z"][sec][ngrx_channel],
            ]

            inc_angle1 = L1.postCal["sp_inc_angle"][sec][ngrx_channel]
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

                L1.postCal["fresnel_coeff"][sec][ngrx_channel] = fresnel_coeff1
                L1.postCal["fresnel_major"][sec][ngrx_channel] = fresnel_axis1[0]
                L1.postCal["fresnel_minor"][sec][ngrx_channel] = fresnel_axis1[1]
                L1.postCal["fresnel_orientation"][sec][ngrx_channel] = fresnel_orientation1

            # Do this once here rather than another loop over Sec and L0.J_2
            # Cross Pol
            if ngrx_channel < L0.J_2:
                nbrcs_LHCP1 = L1.postCal['ddm_nbrcs_v1'][sec][ngrx_channel]
                nbrcs_RHCP1 = L1.postCal['ddm_nbrcs_v1'][sec][ngrx_channel + L0.J_2]
                CP1 = nbrcs_LHCP1 / nbrcs_RHCP1
                if CP1 > 0:
                    CP_db1 = power2db(CP1)
                    L1.postCal["nbrcs_cross_pol_v1"][sec][ngrx_channel] = CP_db1

    L1.postCal["nbrcs_cross_pol_v1"][:, L0.J_2: L0.J] = -1 * L1.postCal["nbrcs_cross_pol_v1"][:, 0: L0.J_2]
