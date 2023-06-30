import math
import numpy as np
from scipy.signal import convolve2d


def meter2chips(x):
    """
    this function converts from meters to chips
    input: x - distance in meters
    output: y - distance in chips
    """
    # define constants
    c = 299792458  # light speed metre per second
    chip_rate = 1.023e6  # L1 GPS chip-per-second, code modulation frequency
    tau_c = 1 / chip_rate  # C/A code chiping period
    l_chip = c * tau_c  # chip length
    y = x / l_chip
    return y


def delay_correction(delay_chips_in, P):
    # this function correct the input code phase to a value between 0 and
    # a defined value P, P = 1023 for GPS L1 and P = 4092 for GAL E1
    temp = delay_chips_in

    if temp < 0:
        while temp < 0:
            temp = temp + P
    elif temp > 1023:
        while temp > 1023:
            temp = temp - P

    delay_chips_out = temp

    return delay_chips_out


def deldop(tx_pos_xyz, rx_pos_xyz, tx_vel_xyz, rx_vel_xyz, p_xyz):
    """
    # This function computes absolute delay and doppler values for a given
    # pixel whose coordinate is <lat,lon,ele>
    # The ECEF position and velocity vectors of tx and rx are also required
    # Inputs:
    # 1) tx_xyz, rx_xyz: ecef position of tx, rx
    # 2) tx_vel, rx_vel: ecef velocity of tx, rx
    # 3) p_xyz of the pixel under computation
    # Outputs:
    # 1) delay_chips: delay measured in chips
    # 2) doppler_Hz: doppler measured in Hz
    # 3) add_delay_chips: additional delay measured in chips
    """
    # common parameters
    c = 299792458  # light speed metre per second
    fc = 1575.42e6  # L1 carrier frequency in Hz
    _lambda = c / fc  # wavelength

    V_tp = tx_pos_xyz - p_xyz
    R_tp = np.linalg.norm(V_tp, 2)
    V_tp_unit = V_tp / R_tp
    V_rp = rx_pos_xyz - p_xyz
    R_rp = np.linalg.norm(V_rp, 2)
    V_rp_unit = V_rp / R_rp
    V_tr = tx_pos_xyz - rx_pos_xyz
    R_tr = np.linalg.norm(V_tr, 2)

    delay = R_tp + R_rp
    delay_chips = meter2chips(delay)
    add_delay_chips = meter2chips(R_tp + R_rp - R_tr)

    # absolute Doppler frequency in Hz
    term1 = np.dot(tx_vel_xyz, V_tp_unit)
    term2 = np.dot(rx_vel_xyz, V_rp_unit)

    doppler_hz = -1 * (term1 + term2) / _lambda  # Doppler in Hz

    return delay_chips, doppler_hz, add_delay_chips


def get_ddm_Aeff4(
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

    return A_eff


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
