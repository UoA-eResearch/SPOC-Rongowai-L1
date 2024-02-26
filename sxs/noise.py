import numpy as np
from scipy import constants

from calibration import power2db
from utils import expand_to_RHCP


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
    fc = 1575.42e6  # L1 carrier frequency in Hz
    _lambda = constants.c / fc  # wavelength
    fclk_drift = 150

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

    doppler_hz = -1 * (term1 + term2) / _lambda - fclk_drift  # Doppler in Hz

    return delay_chips, doppler_hz, add_delay_chips


def meter2chips(x):
    """
    this function converts from meters to chips
    input: x - distance in meters
    output: y - distance in chips
    """
    # define constants
    chip_rate = 1.023e6  # L1 GPS chip-per-second, code modulation frequency
    tau_c = 1 / chip_rate  # C/A code chiping period
    l_chip = constants.c * tau_c  # chip length
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


def noise_floor_prep(
    L0,
    L1,
    add_range_to_sp,
    rx_pos_x,
    rx_pos_y,
    rx_pos_z,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
):
    delay_offset = 4

    for ngrx_channel in range(L0.J):
        for sec in range(L0.I):
            first_scale_factor1 = L0.first_scale_factor[sec][ngrx_channel]
            raw_counts1 = L0.raw_counts[sec][ngrx_channel]
            L1.ddm_power_counts[sec][ngrx_channel] = first_scale_factor1 * raw_counts1

    L1.postCal["raw_counts"] = L1.ddm_power_counts

    # derive floating SP bin location and effective scattering area A_eff
    for sec in range(L0.I):
        # retrieve rx positions and velocities
        rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
        rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])

        for ngrx_channel in range(L0.J_2):
            # retrieve tx positions and velocities
            tx_pos_xyz1 = np.array(
                [
                    L1.postCal["tx_pos_x"][sec][ngrx_channel],
                    L1.postCal["tx_pos_y"][sec][ngrx_channel],
                    L1.postCal["tx_pos_z"][sec][ngrx_channel],
                ]
            )
            tx_vel_xyz1 = np.array(
                [
                    L1.postCal["tx_vel_x"][sec][ngrx_channel],
                    L1.postCal["tx_vel_y"][sec][ngrx_channel],
                    L1.postCal["tx_vel_z"][sec][ngrx_channel],
                ]
            )

            # retrieve sx-related parameters
            sx_pos_xyz1 = np.array(
                [
                    L1.postCal["sp_pos_x"][sec][ngrx_channel],
                    L1.postCal["sp_pos_y"][sec][ngrx_channel],
                    L1.postCal["sp_pos_z"][sec][ngrx_channel],
                ]
            )

            counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
            # from onboard tracker
            add_range_to_sp1 = add_range_to_sp[sec][ngrx_channel]
            delay_center_chips1 = L0.delay_center_chips[sec][ngrx_channel]

            # zenith code phase
            add_range_to_sp_chips1 = meter2chips(add_range_to_sp1)
            zenith_code_phase1 = delay_center_chips1 + add_range_to_sp_chips1
            zenith_code_phase1 = delay_correction(zenith_code_phase1, 1023)

            # Part 3B: noise floor here to avoid another interation over [sec,L0.J_2]
            nf_counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
            nf_counts_RHCP1 = L1.ddm_power_counts[sec, ngrx_channel + L0.J_2, :, :]

            # delay_offset+1 due to difference between Matlab and Python indexing
            noise_floor_bins_LHCP1 = nf_counts_LHCP1[-(delay_offset + 1) :, :]
            noise_floor_bins_RHCP1 = nf_counts_RHCP1[-(delay_offset + 1) :, :]

            if (not np.isnan(L1.postCal["tx_pos_x"][sec][ngrx_channel])) and (
                not np.isnan(counts_LHCP1).all()
            ):
                # peak delay and doppler location
                # assume LHCP and RHCP DDMs have the same peak location
                peak_counts1 = np.max(counts_LHCP1)
                # invert order compared to Matlab
                [peak_delay_row1, peak_doppler_col1] = np.where(
                    counts_LHCP1 == peak_counts1
                )

                # tx to rx range
                r_trx1 = np.linalg.norm(
                    np.array(tx_pos_xyz1) - np.array(rx_pos_xyz1), 2
                )

                # SOC derived more accurate additional range to SP
                r_tsx1 = np.linalg.norm(
                    np.array(tx_pos_xyz1) - np.array(sx_pos_xyz1), 2
                )
                r_rsx1 = np.linalg.norm(
                    np.array(rx_pos_xyz1) - np.array(sx_pos_xyz1), 2
                )

                add_range_to_sp_soc1 = r_tsx1 + r_rsx1 - r_trx1
                d_add_range1 = add_range_to_sp_soc1 - add_range_to_sp1
                d_delay_chips1 = meter2chips(d_add_range1)
                d_delay_bin1 = d_delay_chips1 / L0.delay_bin_res

                sp_delay_row1 = L0.center_delay_bin - d_delay_bin1 -1

                # SP doppler value
                _, sp_doppler_hz1, _ = deldop(
                    tx_pos_xyz1, rx_pos_xyz1, tx_vel_xyz1, rx_vel_xyz1, sx_pos_xyz1
                )

                doppler_center_hz1 = L0.doppler_center_hz[sec][ngrx_channel]

                d_doppler_hz1 = doppler_center_hz1 - sp_doppler_hz1
                d_doppler_bin1 = d_doppler_hz1 / L0.doppler_bin_res

                sp_doppler_col1 = L0.center_doppler_bin - d_doppler_bin1

                # SP delay and doppler location
                L1.peak_delay_row[sec][ngrx_channel] = peak_delay_row1[
                    0
                ]  # 0-based index
                L1.peak_doppler_col[sec][ngrx_channel] = peak_doppler_col1[0]

                L1.sp_delay_row[sec][ngrx_channel] = sp_delay_row1
                L1.sp_delay_error[sec][ngrx_channel] = d_delay_chips1
                L1.sp_doppler_col[sec][ngrx_channel] = sp_doppler_col1
                L1.sp_doppler_error[sec][ngrx_channel] = d_doppler_hz1

                L1.noise_floor_all_LHCP[sec][ngrx_channel] = np.nanmean(
                    noise_floor_bins_LHCP1
                )
                L1.noise_floor_all_RHCP[sec][ngrx_channel] = np.nanmean(
                    noise_floor_bins_RHCP1
                )

            L1.postCal["zenith_code_phase"][sec][ngrx_channel] = zenith_code_phase1

    # extend to RHCP channels
    L1.expand_noise_arrays(L0.J_2, L0.J)


def noise_floor(L0, L1):
    sp_safe_margin = 9  # safe space between SP and DDM end

    # single noise floor from valid DDMs
    sp_delay_row_LHCP = L1.sp_delay_row[:, :10]  # reference to LHCP delay row

    valid_idx = np.where(
        (sp_delay_row_LHCP > 0)
        & (sp_delay_row_LHCP < (39 - sp_safe_margin))
        & ~np.isnan(L1.noise_floor_all_LHCP)
    )

    # noise floor is the median of the average counts
    noise_floor_LHCP = np.nanmedian(L1.noise_floor_all_LHCP[valid_idx])
    noise_floor_RHCP = np.nanmedian(L1.noise_floor_all_RHCP[valid_idx])

    # SNR of SP
    # flag 0 for signal < 0
    for ngrx_channel in range(L0.J_2):
        for sec in range(L0.I):
            counts_LHCP1 = L1.ddm_power_counts[sec, ngrx_channel, :, :]
            counts_RHCP1 = L1.ddm_power_counts[sec, ngrx_channel + L0.J_2, :, :]

            # Removed +1 due to Python 0-based indexing
            sp_delay_row1 = np.floor(sp_delay_row_LHCP[sec][ngrx_channel])  # + 1
            sp_doppler_col1 = np.floor(L1.sp_doppler_col[sec][ngrx_channel])  # + 1

            if (0 < sp_delay_row1 < 40) and (0 < sp_doppler_col1 < 5):
                sp_counts_LHCP1 = counts_LHCP1[int(sp_delay_row1), int(sp_doppler_col1)]
                sp_counts_RHCP1 = counts_RHCP1[int(sp_delay_row1), int(sp_doppler_col1)]

                signal_counts_LHCP1 = sp_counts_LHCP1 - noise_floor_LHCP
                snr_LHCP1 = signal_counts_LHCP1 / noise_floor_LHCP
                signal_counts_RHCP1 = sp_counts_RHCP1 - noise_floor_RHCP
                snr_RHCP1 = signal_counts_RHCP1 / noise_floor_RHCP

                if signal_counts_LHCP1 > 0:
                    snr_LHCP_db1 = power2db(snr_LHCP1)
                    snr_flag_LHCP1 = 1
                else:
                    snr_LHCP_db1 = np.nan
                    snr_flag_LHCP1 = 0
                L1.postCal["ddm_snr"][sec][ngrx_channel] = snr_LHCP_db1
                L1.snr_flag[sec][ngrx_channel] = snr_flag_LHCP1

                if signal_counts_RHCP1 > 0:
                    snr_RHCP_db1 = power2db(snr_RHCP1)
                    snr_flag_RHCP1 = 1
                else:
                    snr_RHCP_db1 = np.nan
                    snr_flag_RHCP1 = 0
                L1.postCal["ddm_snr"][sec][ngrx_channel + L0.J_2] = snr_RHCP_db1
                L1.snr_flag[sec][ngrx_channel + L0.J_2] = snr_flag_RHCP1

    noise_floor = np.hstack(
        (
            np.full([L0.shape_4d[0], int(L0.shape_4d[1] / 2)], noise_floor_LHCP),
            np.full([L0.shape_4d[0], int(L0.shape_4d[1] / 2)], noise_floor_RHCP),
        )
    )
    L1.postCal["ddm_noise_floor"] = noise_floor
    L1.postCal["ddm_snr_flag"] = L1.snr_flag


def confidence_flag(L0, L1):
    for ngrx_channel in range(L0.J_2):
        for sec in range(L0.I):
            sx_delay_error = abs(L1.sp_delay_error[sec][ngrx_channel])
            sx_doppler_error = abs(L1.sp_doppler_error[sec][ngrx_channel])
            sx_d_snell_angle = abs(L1.postCal["sp_d_snell_angle"][sec][ngrx_channel])

            if not np.isnan(L1.postCal["tx_pos_x"][sec][ngrx_channel]):
                # criteria may change at a later stage
                delay_doppler_snell = (
                    (sx_delay_error < 1.25)
                    and (abs(sx_doppler_error) < 250)
                    and (sx_d_snell_angle < 2)
                )

                if (
                    L1.postCal["ddm_snr"][sec][ngrx_channel] >= 2.0
                ) and not delay_doppler_snell:
                    confidence_flag = 0
                elif (
                    L1.postCal["ddm_snr"][sec][ngrx_channel] < 2.0
                ) and not delay_doppler_snell:
                    confidence_flag = 1
                elif (
                    L1.postCal["ddm_snr"][sec][ngrx_channel] < 2.0
                ) and delay_doppler_snell:
                    confidence_flag = 2
                elif (
                    L1.postCal["ddm_snr"][sec][ngrx_channel] >= 2.0
                ) and delay_doppler_snell:
                    confidence_flag = 3
                else:
                    confidence_flag = np.nan

                L1.confidence_flag[sec][ngrx_channel] = confidence_flag

    L1.confidence_flag = expand_to_RHCP(L1.confidence_flag, L0.J_2, L0.J)
