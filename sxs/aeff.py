import math

import numpy as np
from scipy.signal import convolve2d

from projections import ecef2lla


"""def get_ddm_Aeff4(
    rx_alt,
    inc_angle,
    az_angle,
    sp_delay_bin,
    sp_doppler_bin,
    chi2,
    A_phy_LUT_interp,
):
    # this is to construct effective scattering area from LUTs

    # center delay and doppler bin for full DDM
    # not to be used elsewhere
    center_delay_bin = 40
    center_doppler_bin = 5

    # derive full scattering area from LUT
    A_phy_full_1 = A_phy_LUT_interp((rx_alt, inc_angle, az_angle))
    A_phy_full_2 = np.vstack((np.zeros((1, 41)), A_phy_full_1, np.zeros((1, 41))))
    A_phy_full = np.hstack((A_phy_full_2, np.zeros((9, 39))))

    # shift A_phy_full according to the floating SP bin

    # integer and fractional part of the SP bin
    sp_delay_intg = np.floor(sp_delay_bin)
    sp_delay_frac = sp_delay_bin - sp_delay_intg

    sp_doppler_intg = np.floor(sp_doppler_bin)
    sp_doppler_frac = sp_doppler_bin - sp_doppler_intg

    # shift 1: shift along delay direction
    A_phy_shift = np.roll(A_phy_full, 1)
    A_phy_shift = (1 - sp_delay_frac) * A_phy_full + (sp_delay_frac * A_phy_shift)

    # shift 2: shift along doppler direction
    A_phy_shift2 = np.roll(A_phy_shift, 1, axis=0)
    A_phy_shift = (1 - sp_doppler_frac) * A_phy_shift + (sp_doppler_frac * A_phy_shift2)

    # crop the A_phy_full to Rongowai 5*40 DDM size
    delay_shift_bin = center_delay_bin - sp_delay_intg
    doppler_shift_bin = center_doppler_bin - sp_doppler_intg

    # Change from Matlab offsets to account for Matlab/Python indexing differences
    A_phy = A_phy_shift[
        int(doppler_shift_bin - 1) : int(doppler_shift_bin + 4),
        int(delay_shift_bin - 1) : int(delay_shift_bin + 39),
    ]

    # convolution to A_eff
    A_eff1 = convolve2d(A_phy, chi2.T)
    A_eff = A_eff1[2:7, 20:60]  # cut suitable size for A_eff, 0-based index

    return A_eff"""


def get_ddm_Aeff5(
    rx_alt,
    inc_angle,
    az_angle,
    sp_delay_bin,
    sp_doppler_bin,
    chi2,
    A_phy_LUT_all,
):
    # this is to construct effective scattering area from LUTs

    # center delay and doppler bin for full DDM
    # note: not to be used elsewhere
    center_delay_bin = 40
    center_doppler_bin = 5

    # integer and fractional part of the SP bin
    sp_delay_intg = np.floor(sp_delay_bin)
    sp_delay_frac = sp_delay_bin - sp_delay_intg

    sp_doppler_intg = np.round(sp_doppler_bin)
    sp_doppler_frac = sp_doppler_bin - sp_doppler_intg

    # index for the floating Doppler bin
    # which LUT should be used for interploation
    # no +1 due to 0/1-base index differences of Python/Matlab
    k = int(np.floor((sp_doppler_frac + 0.5) / 0.1))  # +1

    if sp_doppler_frac >= 0:
        A_phy_LUT1 = A_phy_LUT_all[k]["LUT"]
        A_phy_LUT2 = A_phy_LUT_all[k + 1]["LUT"]

    else:
        A_phy_LUT2 = A_phy_LUT_all[k]["LUT"]
        A_phy_LUT1 = A_phy_LUT_all[k + 1]["LUT"]

    # derive full scattering area from LUT
    A_phy_full_1_1 = A_phy_LUT1((rx_alt, inc_angle, az_angle))
    A_phy_full_2_1 = np.vstack((np.zeros((1, 41)), A_phy_full_1_1, np.zeros((1, 41))))
    A_phy_full_1 = np.hstack((A_phy_full_2_1, np.zeros((9, 39))))

    A_phy_full_1_2 = A_phy_LUT2((rx_alt, inc_angle, az_angle))
    A_phy_full_2_2 = np.vstack((np.zeros((1, 41)), A_phy_full_1_2, np.zeros((1, 41))))
    A_phy_full_2 = np.hstack((A_phy_full_2_2, np.zeros((9, 39))))

    A_phy_full = (1 - sp_doppler_frac) * A_phy_full_1 + (sp_doppler_frac * A_phy_full_2)

    # shift A_phy_full according to the floating SP bin
    # shift 1: shift along delay direction
    A_phy_shift = np.roll(A_phy_full, 1)
    A_phy_shift = (1 - sp_delay_frac) * A_phy_full + (sp_delay_frac * A_phy_shift)

    # shift 2: shift along doppler direction
    # A_phy_shift2 = np.roll(A_phy_shift, 1, axis=0)
    # A_phy_shift = (1 - sp_doppler_frac) * A_phy_shift + (sp_doppler_frac * A_phy_shift2)

    # crop the A_phy_full to Rongowai 5*40 DDM size
    delay_shift_bin = center_delay_bin - sp_delay_intg
    doppler_shift_bin = center_doppler_bin - sp_doppler_intg

    # Change from Matlab offsets to account for Matlab/Python indexing differences
    A_phy = A_phy_shift[
        int(doppler_shift_bin - 1) : int(doppler_shift_bin + 4),
        int(delay_shift_bin - 1) : int(delay_shift_bin + 39),
    ]

    # convolution to A_eff
    A_eff1 = convolve2d(A_phy, chi2.T)
    A_eff = A_eff1[2:7, 20:60]  # cut suitable size for A_eff, 0-based index

    return A_eff


def get_ddma_v1(brcs_copol, brcs_xpol, A_eff, sp_delay_row, sp_doppler_col):
    """
    this function gets the brcs and A_eff within ddma region - SP bin

    Parameters
    ----------
    brcs_copol
    brcs_xpol
    A_eff
    sp_delay_row
    sp_doppler_col

    Returns
    -------
    brcs_copol_ddma
    brcs_xpol_ddma
    A_eff_ddma
    """
    delay_intg = math.floor(sp_delay_row)
    delay_frac = sp_delay_row - math.floor(sp_delay_row)
    doppler_intg = round(sp_doppler_col)
    doppler_frac = sp_doppler_col - round(sp_doppler_col)

    # NBRCS SP bin
    if doppler_frac >= 0:
        if doppler_intg <= 3:
            brcs_copol_ddma = (
                (1 - doppler_frac)
                * (1 - delay_frac)
                * brcs_copol[delay_intg, doppler_intg]
                + (1 - doppler_frac)
                * delay_frac
                * brcs_copol[delay_intg + 1, doppler_intg]
                + doppler_frac
                * (1 - delay_frac)
                * brcs_copol[delay_intg, doppler_intg + 1]
                + doppler_frac
                * delay_frac
                * brcs_copol[delay_intg + 1, doppler_intg + 1]
            )
            brcs_xpol_ddma = (
                (1 - doppler_frac)
                * (1 - delay_frac)
                * brcs_xpol[delay_intg, doppler_intg]
                + (1 - doppler_frac)
                * delay_frac
                * brcs_xpol[delay_intg + 1, doppler_intg]
                + doppler_frac
                * (1 - delay_frac)
                * brcs_xpol[delay_intg, doppler_intg + 1]
                + doppler_frac
                * delay_frac
                * brcs_xpol[delay_intg + 1, doppler_intg + 1]
            )
            A_eff_ddma = (
                (1 - doppler_frac) * (1 - delay_frac) * A_eff[delay_intg, doppler_intg]
                + (1 - doppler_frac) * delay_frac * A_eff[delay_intg + 1, doppler_intg]
                + doppler_frac * (1 - delay_frac) * A_eff[delay_intg, doppler_intg + 1]
                + doppler_frac * delay_frac * A_eff[delay_intg + 1, doppler_intg + 1]
            )

        elif doppler_intg > 3:
            brcs_copol_ddma = (1 - delay_frac) * brcs_copol[
                delay_intg, doppler_intg
            ] + delay_frac * brcs_copol[delay_intg + 1, doppler_intg]

            brcs_xpol_ddma = (1 - delay_frac) * brcs_xpol[
                delay_intg, doppler_intg
            ] + delay_frac * brcs_xpol[delay_intg + 1, doppler_intg]

            A_eff_ddma = (1 - delay_frac) * A_eff[
                delay_intg, doppler_intg
            ] + delay_frac * A_eff[delay_intg + 1, doppler_intg]

    elif doppler_frac < 0:
        if doppler_intg >= 1:
            brcs_copol_ddma = (
                (1 - abs(doppler_frac))
                * (1 - delay_frac)
                * brcs_copol[delay_intg, doppler_intg]
                + (1 - abs(doppler_frac))
                * delay_frac
                * brcs_copol[delay_intg + 1, doppler_intg]
                + abs(doppler_frac)
                * (1 - delay_frac)
                * brcs_copol[delay_intg, doppler_intg - 1]
                + abs(doppler_frac)
                * delay_frac
                * brcs_copol[delay_intg + 1, doppler_intg - 1]
            )

            brcs_xpol_ddma = (
                (1 - abs(doppler_frac))
                * (1 - delay_frac)
                * brcs_xpol[delay_intg, doppler_intg]
                + (1 - abs(doppler_frac))
                * delay_frac
                * brcs_xpol[delay_intg + 1, doppler_intg]
                + abs(doppler_frac)
                * (1 - delay_frac)
                * brcs_xpol[delay_intg, doppler_intg - 1]
                + abs(doppler_frac)
                * delay_frac
                * brcs_xpol[delay_intg + 1, doppler_intg - 1]
            )

            A_eff_ddma = (
                (1 - abs(doppler_frac))
                * (1 - delay_frac)
                * A_eff[delay_intg, doppler_intg]
                + (1 - abs(doppler_frac))
                * delay_frac
                * A_eff[delay_intg + 1, doppler_intg]
                + abs(doppler_frac)
                * (1 - delay_frac)
                * A_eff[delay_intg, doppler_intg - 1]
                + abs(doppler_frac)
                * delay_frac
                * A_eff[delay_intg + 1, doppler_intg - 1]
            )

        elif doppler_intg < 1:
            brcs_copol_ddma = (1 - delay_frac) * brcs_copol[
                delay_intg, doppler_intg
            ] + delay_frac * brcs_copol[delay_intg + 1, doppler_intg]

            brcs_xpol_ddma = (1 - delay_frac) * brcs_xpol[
                delay_intg, doppler_intg
            ] + delay_frac * brcs_xpol[delay_intg + 1, doppler_intg]

            A_eff_ddma = (1 - delay_frac) * A_eff[
                delay_intg, doppler_intg
            ] + delay_frac * A_eff[delay_intg + 1, doppler_intg]

    return brcs_copol_ddma, brcs_xpol_ddma, A_eff_ddma


def get_ddma_v2(brcs_copol, brcs_xpol, A_eff, sp_delay_row, sp_doppler_col):
    """
    this function gets the brcs and A_eff within ddma region - 3*3 bin

    Parameters
    ----------
    brcs_copol
    brcs_xpol
    A_eff
    sp_delay_row
    sp_doppler_col

    Returns
    -------
    brcs_copol_ddma
    brcs_xpol_ddma
    A_eff_ddma
    """

    delay_intg = math.floor(sp_delay_row)
    delay_frac = sp_delay_row - math.floor(sp_delay_row)

    doppler_intg = round(sp_doppler_col)
    doppler_frac = sp_doppler_col - round(sp_doppler_col)

    if delay_intg >= 2:
        delay_range = range(delay_intg - 2, delay_intg + 2)
    elif delay_intg < 2:
        delay_range = range(1, 5)

    if doppler_frac >= 0:
        if doppler_intg < 3 and doppler_intg >= 1:
            doppler_range = range(doppler_intg - 1, doppler_intg + 3)

        elif doppler_intg >= 3:
            doppler_range = range(2, 6)

        elif doppler_intg < 1:
            doppler_range = range(1, 5)

        brcs_copol_ddma = (
            (1 - delay_frac)
            * (1 - doppler_frac)
            * brcs_copol[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * brcs_copol[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * brcs_copol[delay_range[0], doppler_range[2]]
            + (1 - delay_frac)
            * doppler_frac
            * brcs_copol[delay_range[0], doppler_range[3]]
            + (1 - doppler_frac) * brcs_copol[delay_range[1], doppler_range[0]]
            + brcs_copol[delay_range[1], doppler_range[1]]
            + brcs_copol[delay_range[1], doppler_range[2]]
            + doppler_frac * brcs_copol[delay_range[1], doppler_range[3]]
            + (1 - doppler_frac) * brcs_copol[delay_range[2], doppler_range[0]]
            + brcs_copol[delay_range[2], doppler_range[1]]
            + brcs_copol[delay_range[2], doppler_range[2]]
            + doppler_frac * brcs_copol[delay_range[2], doppler_range[3]]
            + (1 - doppler_frac)
            * delay_frac
            * brcs_copol[delay_range[3], doppler_range[0]]
            + delay_frac * brcs_copol[delay_range[3], doppler_range[1]]
            + delay_frac * brcs_copol[delay_range[3], doppler_range[2]]
            + delay_frac * doppler_frac * brcs_copol[delay_range[3], doppler_range[3]]
        )

        brcs_xpol_ddma = (
            (1 - delay_frac)
            * (1 - doppler_frac)
            * brcs_xpol[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * brcs_xpol[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * brcs_xpol[delay_range[0], doppler_range[2]]
            + (1 - delay_frac)
            * doppler_frac
            * brcs_xpol[delay_range[0], doppler_range[3]]
            + (1 - doppler_frac) * brcs_xpol[delay_range[1], doppler_range[0]]
            + brcs_xpol[delay_range[1], doppler_range[1]]
            + brcs_xpol[delay_range[1], doppler_range[2]]
            + doppler_frac * brcs_xpol[delay_range[1], doppler_range[3]]
            + (1 - doppler_frac) * brcs_xpol[delay_range[2], doppler_range[0]]
            + brcs_xpol[delay_range[2], doppler_range[1]]
            + brcs_xpol[delay_range[2], doppler_range[2]]
            + doppler_frac * brcs_xpol[delay_range[2], doppler_range[3]]
            + (1 - doppler_frac)
            * delay_frac
            * brcs_xpol[delay_range[3], doppler_range[0]]
            + delay_frac * brcs_xpol[delay_range[3], doppler_range[1]]
            + delay_frac * brcs_xpol[delay_range[3], doppler_range[2]]
            + delay_frac * doppler_frac * brcs_xpol[delay_range[3], doppler_range[3]]
        )

        A_eff_ddma = (
            (1 - delay_frac)
            * (1 - doppler_frac)
            * A_eff[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * A_eff[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * A_eff[delay_range[0], doppler_range[2]]
            + (1 - delay_frac) * doppler_frac * A_eff[delay_range[0], doppler_range[3]]
            + (1 - doppler_frac) * A_eff[delay_range[1], doppler_range[0]]
            + A_eff[delay_range[1], doppler_range[1]]
            + A_eff[delay_range[1], doppler_range[2]]
            + doppler_frac * A_eff[delay_range[1], doppler_range[3]]
            + (1 - doppler_frac) * A_eff[delay_range[2], doppler_range[0]]
            + A_eff[delay_range[2], doppler_range[1]]
            + A_eff[delay_range[2], doppler_range[2]]
            + doppler_frac * A_eff[delay_range[2], doppler_range[3]]
            + (1 - doppler_frac) * delay_frac * A_eff[delay_range[3], doppler_range[0]]
            + delay_frac * A_eff[delay_range[3], doppler_range[1]]
            + delay_frac * A_eff[delay_range[3], doppler_range[2]]
            + delay_frac * doppler_frac * A_eff[delay_range[3], doppler_range[3]]
        )

    elif doppler_frac < 0:
        if doppler_intg <= 3 and doppler_intg > 1:
            doppler_range = range(doppler_intg - 2, doppler_intg + 2)

        elif doppler_intg > 3:
            doppler_range = range(2, 6)

        elif doppler_intg <= 1:
            doppler_range = range(1, 5)

        brcs_copol_ddma = (
            (1 - delay_frac)
            * abs(doppler_frac)
            * brcs_copol[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * brcs_copol[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * brcs_copol[delay_range[0], doppler_range[2]]
            + (1 - delay_frac)
            * (1 - abs(doppler_frac))
            * brcs_copol[delay_range[0], doppler_range[3]]
            + abs(doppler_frac) * brcs_copol[delay_range[1], doppler_range[0]]
            + brcs_copol[delay_range[1], doppler_range[1]]
            + brcs_copol[delay_range[1], doppler_range[2]]
            + (1 - abs(doppler_frac)) * brcs_copol[delay_range[1], doppler_range[3]]
            + abs(doppler_frac) * brcs_copol[delay_range[2], doppler_range[0]]
            + brcs_copol[delay_range[2], doppler_range[1]]
            + brcs_copol[delay_range[2], doppler_range[2]]
            + (1 - abs(doppler_frac)) * brcs_copol[delay_range[2], doppler_range[3]]
            + abs(doppler_frac)
            * delay_frac
            * brcs_copol[delay_range[3], doppler_range[0]]
            + delay_frac * brcs_copol[delay_range[3], doppler_range[1]]
            + delay_frac * brcs_copol[delay_range[3], doppler_range[2]]
            + delay_frac
            * (1 - abs(doppler_frac))
            * brcs_copol[delay_range[3], doppler_range[3]]
        )

        brcs_xpol_ddma = (
            (1 - delay_frac)
            * abs(doppler_frac)
            * brcs_xpol[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * brcs_xpol[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * brcs_xpol[delay_range[0], doppler_range[2]]
            + (1 - delay_frac)
            * (1 - abs(doppler_frac))
            * brcs_xpol[delay_range[0], doppler_range[3]]
            + abs(doppler_frac) * brcs_copol[delay_range[1], doppler_range[0]]
            + brcs_xpol[delay_range[1], doppler_range[1]]
            + brcs_xpol[delay_range[1], doppler_range[2]]
            + (1 - abs(doppler_frac)) * brcs_xpol[delay_range[1], doppler_range[3]]
            + abs(doppler_frac) * brcs_xpol[delay_range[2], doppler_range[0]]
            + brcs_xpol[delay_range[2], doppler_range[1]]
            + brcs_xpol[delay_range[2], doppler_range[2]]
            + (1 - abs(doppler_frac)) * brcs_xpol[delay_range[2], doppler_range[3]]
            + abs(doppler_frac)
            * delay_frac
            * brcs_xpol[delay_range[3], doppler_range[0]]
            + delay_frac * brcs_xpol[delay_range[3], doppler_range[1]]
            + delay_frac * brcs_xpol[delay_range[3], doppler_range[2]]
            + delay_frac
            * (1 - abs(doppler_frac))
            * brcs_xpol[delay_range[3], doppler_range[3]]
        )

        A_eff_ddma = (
            (1 - delay_frac)
            * abs(doppler_frac)
            * A_eff[delay_range[0], doppler_range[0]]
            + (1 - delay_frac) * A_eff[delay_range[0], doppler_range[1]]
            + (1 - delay_frac) * A_eff[delay_range[0], doppler_range[2]]
            + (1 - delay_frac)
            * (1 - abs(doppler_frac))
            * A_eff[delay_range[0], doppler_range[3]]
            + abs(doppler_frac) * A_eff[delay_range[1], doppler_range[0]]
            + A_eff[delay_range[1], doppler_range[1]]
            + A_eff[delay_range[1], doppler_range[2]]
            + (1 - abs(doppler_frac)) * A_eff[delay_range[1], doppler_range[3]]
            + abs(doppler_frac) * A_eff[delay_range[2], doppler_range[0]]
            + A_eff[delay_range[2], doppler_range[1]]
            + A_eff[delay_range[2], doppler_range[2]]
            + (1 - abs(doppler_frac)) * A_eff[delay_range[2], doppler_range[3]]
            + abs(doppler_frac) * delay_frac * A_eff[delay_range[3], doppler_range[0]]
            + delay_frac * A_eff[delay_range[3], doppler_range[1]]
            + delay_frac * A_eff[delay_range[3], doppler_range[2]]
            + delay_frac
            * (1 - abs(doppler_frac))
            * A_eff[delay_range[3], doppler_range[3]]
        )

    return brcs_copol_ddma, brcs_xpol_ddma, A_eff_ddma


def get_amb_fun(dtau_s, dfreq_Hz, tau_c, Ti):
    """
    this function computes the ambiguity function
    inputs
    1) tau_s: delay in seconds
    2) freq_Hz: Doppler in Hz
    3) tau_c: chipping period in second, 1/chip_rate
    4) Ti: coherent integration time in seconds
    output
    1) chi: ambiguity function, product of Lambda and S
    """
    det = tau_c * (1 + tau_c / Ti)  # discriminant for computing Lambda

    Lambda = np.full_like(dtau_s, np.nan)

    # compute Lambda - delay
    Lambda[np.abs(dtau_s) <= det] = (1 - np.abs(dtau_s) / tau_c)[np.abs(dtau_s) <= det]
    Lambda[np.abs(dtau_s) > det] = -tau_c / Ti

    # compute S - Doppler
    S1 = math.pi * dfreq_Hz * Ti

    S = np.full_like(S1, np.nan, dtype=complex)

    S[S1 == 0] = 1

    term1 = np.sin(S1[S1 != 0]) / S1[S1 != 0]
    term2 = np.exp(-1j * S1[S1 != 0])
    S[S1 != 0] = term1 * term2

    # compute complex chi
    chi = Lambda * S
    return chi


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


def get_chi2(
    num_delay_bins,
    num_doppler_bins,
    delay_center_bin,
    doppler_center_bin,
    delay_res,
    doppler_res,
):
    # this function gets 2D AF

    chip_rate = 1.023e6
    tau_c = 1 / chip_rate
    T_coh = 1 / 1000

    # delay_res = 0.25
    # doppler_res = 500
    # delay_center_bin = 20  # 0-based index
    # doppler_center_bin = 2  # 0-based index
    # chi = np.zeros([num_delay_bins, num_doppler_bins])

    def ix_func(i, j):
        dtau = (i - delay_center_bin) * delay_res * tau_c
        dfreq = (j - doppler_center_bin) * doppler_res
        # compute complex AF value at each delay-doppler bin
        return get_amb_fun(dtau, dfreq, tau_c, T_coh)

    chi = np.fromfunction(
        ix_func, (num_delay_bins, num_doppler_bins)
    )  # 10 times faster than for loop

    chi_mag = np.abs(chi)  # magnitude
    chi2 = np.square(chi_mag)  # chi_square

    return chi2


def aeff_and_nbrcs(L0, L1, inp, rx_vel_x, rx_vel_y, rx_vel_z, rx_pos_lla):
    # derive amb-function (chi2) to be used in computing A_eff
    # % Matlab corrects delay/Doppler index by adding +1, Python doesn't
    chi2 = get_chi2(
        40,
        5,
        L0.center_delay_bin,
        L0.center_doppler_bin,
        L0.delay_bin_res,
        L0.doppler_bin_res,
    )  # 0-based
    nbrcs_copol1_v1 = np.full([L0.I, L0.J_2], np.nan)
    nbrcs_xpol1_v1 = np.full([L0.I, L0.J_2], np.nan)
    nbrcs_scatter_area_v1 = np.full([L0.I, L0.J], np.nan)
    # nbrcs_scatter_area_v1 = np.full([L0.I, L0.J], np.nan)
    # iterate over each second of flight
    for sec in range(L0.I):
        # retrieve velocities and altitdues
        # bundle up craft vel data into per sec
        rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
        rx_alt1 = rx_pos_lla[2][sec]

        # variables are solved only for LHCP channels
        for ngrx_channel in range(L0.J_2):
            # retrieve tx velocities
            # bundle up velocity data into per sec
            tx_vel_xyz1 = np.array(
                [
                    L1.postCal["tx_vel_x"][sec][ngrx_channel],
                    L1.postCal["tx_vel_y"][sec][ngrx_channel],
                    L1.postCal["tx_vel_z"][sec][ngrx_channel],
                ]
            )

            # azimuth angle between TX and RX velocity
            unit_rx_vel1 = rx_vel_xyz1 / np.linalg.norm(rx_vel_xyz1, 2)
            unit_tx_vel1 = tx_vel_xyz1 / np.linalg.norm(tx_vel_xyz1, 2)

            # 1st input of A_eff
            az_angle1 = math.degrees(math.acos(-1 * np.dot(unit_rx_vel1, unit_tx_vel1)))

            sx_pos_xyz1 = [
                L1.postCal["sp_pos_x"][sec][ngrx_channel],
                L1.postCal["sp_pos_y"][sec][ngrx_channel],
                L1.postCal["sp_pos_z"][sec][ngrx_channel],
            ]
            sx_lla1 = ecef2lla.transform(*sx_pos_xyz1, radians=False)
            # 2nd input of A_eff
            rx_alt_corrected1 = rx_alt1 - sx_lla1[2]

            # % 3rd input of A_eff
            inc_angle1 = L1.postCal["sp_inc_angle"][sec][ngrx_channel]

            brcs_copol1 = L1.postCal["brcs"][sec][ngrx_channel]
            brcs_xpol1 = L1.postCal["brcs"][sec][ngrx_channel + L0.J_2]
            # counts_LHCP1 = L1.ddm_power_counts[sec][ngrx_channel]
            # snr_LHCP1 = L1.postCal["ddm_snr"][sec][ngrx_channel]

            # evaluate delay and Doppler bin location at SP
            # Matlab uses +1, not required in Python 0-based indexing
            sp_delay_row1 = L1.sp_delay_row[sec][ngrx_channel]  # +1;
            sp_doppler_col1 = L1.sp_doppler_col[sec][ngrx_channel]  # +1;

            # ensure the SP is within DDM range (account for python vs Matlab indexing)
            SP_cond = (0 <= sp_delay_row1 <= 38) and (0 <= sp_doppler_col1 <= 4)
            # ensure interpolate within reasonable range
            interp_cond = inp.rx_alt_bins[0] <= rx_alt_corrected1 <= inp.rx_alt_bins[-1]
            angle_cond = 0 <= inc_angle1 <= 80 and not np.isnan(az_angle1)

            if SP_cond and interp_cond and angle_cond:
                # note that the A_eff1 is transposed due to shape inconsistencies
                #  (40,5) vs (5,40)
                A_eff1 = get_ddm_Aeff5(
                    rx_alt_corrected1,
                    inc_angle1,
                    az_angle1,
                    sp_delay_row1,
                    sp_doppler_col1,
                    chi2,
                    inp.A_phy_LUT_all,
                ).T

                L1.A_eff[sec][ngrx_channel] = A_eff1
                # nbrcs for SP bin
                brcs_copol_ddma1, brcs_xpol_ddma1, A_eff_ddma1 = get_ddma_v1(
                    brcs_copol1, brcs_xpol1, A_eff1, sp_delay_row1, sp_doppler_col1
                )

                nbrcs_copol1_v1[sec, ngrx_channel] = brcs_copol_ddma1 / A_eff_ddma1
                nbrcs_xpol1_v1[sec, ngrx_channel] = brcs_xpol_ddma1 / A_eff_ddma1
                nbrcs_scatter_area_v1[sec, ngrx_channel] = A_eff_ddma1

                # nbrcs for 3*3bin
                # brcs_copol_ddma2, brcs_xpol_ddma2, A_eff_ddma2 = get_ddma_v2(
                #     brcs_copol1,
                #     brcs_xpol1,
                #     A_eff1,
                #     sp_delay_row1,
                #     sp_doppler_col1
                # )

                # nbrcs_copol1_v2[sec, ngrx_channel] = brcs_copol_ddma2 / A_eff_ddma2
                # nbrcs_xpol1_v2[sec, ngrx_channel] = brcs_xpol_ddma2 / A_eff_ddma2
                # nbrcs_scatter_area_v2[sec, ngrx_channel] = A_eff_ddma2

    L1.A_eff[:, L0.J_2 : L0.J] = L1.A_eff[:, 0 : L0.J_2]
    nbrcs_scatter_area_v1[:, L0.J_2 : L0.J] = nbrcs_scatter_area_v1[:, 0 : L0.J_2]
    ddm_nbrcs_v1 = np.concatenate((nbrcs_copol1_v1, nbrcs_xpol1_v1), axis=1)
    # ddm_nbrcs_v2 = np.concatenate((nbrcs_copol1_v2, nbrcs_xpol1_v2), axis=1)
    L1.postCal["nbrcs_scatter"] = L1.A_eff
    L1.postCal["nbrcs_scatter_area_v1"] = nbrcs_scatter_area_v1
    # L1.postCal["nbrcs_scatter_area_v2"] = nbrcs_scatter_area_v2

    L1.postCal["ddm_nbrcs_v1"] = ddm_nbrcs_v1
    # L1.postCal["ddm_nbrcs_v2"] = ddm_nbrcs_v2
