import math
import numpy as np

from calibration import db2power
from noise import meter2chips, delay_correction


def get_quality_flag(quality_flag1):
    quality_flag = (
        2**25 * quality_flag1[0]
        + 2**24 * quality_flag1[1]
        + 2**23 * quality_flag1[2]
        + 2**22 * quality_flag1[3]
        + 2**21 * quality_flag1[4]
        + 2**20 * quality_flag1[5]
        + 2**19 * quality_flag1[6]
        + 2**18 * quality_flag1[7]
        + 2**17 * quality_flag1[8]
        + 2**16 * quality_flag1[9]
        + 2**15 * quality_flag1[10]
        + 2**14 * quality_flag1[11]
        + 2**13 * quality_flag1[12]
        + 2**12 * quality_flag1[13]
        + 2**11 * quality_flag1[14]
        + 2**10 * quality_flag1[15]
        + 2**9 * quality_flag1[16]
        + 2**8 * quality_flag1[17]
        + 2**7 * quality_flag1[18]
        + 2**6 * quality_flag1[19]
        + 2**5 * quality_flag1[20]
        + 2**4 * quality_flag1[21]
        + 2**3 * quality_flag1[22]
        + 2**2 * quality_flag1[23]
        + 2**1 * quality_flag1[24]
        + 2**0 * quality_flag1[25]
    )
    return quality_flag


def quality_flag_calculations(
    L0,
    L1,
    rx_roll,
    rx_pitch,
    rx_heading,
    ant_temp_nadir,
    add_range_to_sp,
    rx_pos_lla,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
):
    for sec in range(L0.I):
        for ngrx_channel in range(L0.J):
            quality_flag1_1 = np.full([26, 1], 0)

            # flag 1, 2 and 22  0-based indexing
            rx_roll1 = rx_roll[sec]
            rx_pitch1 = rx_pitch[sec]
            rx_heading1 = rx_heading[sec]

            if (
                (rx_roll1 >= 29) or (rx_pitch1 >= 9) or (rx_heading1 >= 4)
            ):  # 0-based indexing
                quality_flag1_1[2] = 1
            else:
                quality_flag1_1[1] = 1

            if rx_roll1 > 1:
                quality_flag1_1[22] = 1

            # flag 3   0-based indexing
            quality_flag1_1[3] = 0

            # flag 4 and 5
            trans_id1 = L0.transmitter_id[sec][ngrx_channel]
            if trans_id1 == 0:
                quality_flag1_1[4] = 1

            if trans_id1 == 28:
                quality_flag1_1[5] = 1

            # flag 6 and 9
            snr_db1 = L1.postCal["ddm_snr"][sec][ngrx_channel]

            if sec > 0:  # 0-based indexing
                snr_db2 = L1.postCal["ddm_snr"][sec - 1][ngrx_channel]
                diff1 = (db2power(snr_db1) - db2power(snr_db2)) / db2power(snr_db1)
                diff2 = snr_db1 - snr_db2

                if abs(diff1) > 0.1:
                    quality_flag1_1[6] = 1

                if abs(diff2) > 0.24:
                    quality_flag1_1[9] = 1

            # flag 7 and 8
            dist_to_coast1 = L1.dist_to_coast_km[sec][ngrx_channel]

            if dist_to_coast1 > 0:
                quality_flag1_1[7] = 1

            if dist_to_coast1 > -25:
                quality_flag1_1[8] = 1

            # flag 10
            ant_temp1 = ant_temp_nadir[sec]
            if sec > 0:
                ant_temp2 = ant_temp_nadir[sec - 1]
                rate = (ant_temp2 - ant_temp1) * 60

                if rate > 1:
                    quality_flag1_1[10] = 1

            # flag 11
            zenith_code_phase1 = L1.postCal["zenith_code_phase"][sec][ngrx_channel]
            signal_code_phase1 = delay_correction(
                meter2chips(add_range_to_sp[sec][ngrx_channel]), 1023
            )
            diff1 = zenith_code_phase1 - signal_code_phase1
            if diff1 >= 10:
                quality_flag1_1[11] = 1

            # flag 14 and 15
            sp_delay_row1 = L1.sp_delay_row[sec][ngrx_channel]
            sp_dopp_col = L1.sp_doppler_col[sec][ngrx_channel]

            if not np.isnan(sp_delay_row1):
                if (sp_delay_row1 < 15) or (sp_delay_row1 > 35):
                    quality_flag1_1[14] = 1

            if not np.isnan(sp_dopp_col):
                if (sp_dopp_col < 2) or (sp_dopp_col > 4):
                    quality_flag1_1[15] = 1

            # flag 16
            if not np.isnan(sp_delay_row1) and not np.isnan(sp_dopp_col):
                if (
                    (math.floor(sp_delay_row1) < 38)
                    and (math.floor(sp_delay_row1) > 0)
                    and (math.floor(sp_dopp_col) < 5)
                    and (math.floor(sp_dopp_col) > 1)
                ):
                    sp_dopp_col_range = list(
                        range(math.floor(sp_dopp_col) - 1, math.floor(sp_dopp_col) + 2)
                    )
                    sp_delay_raw_range = list(
                        range(math.floor(sp_delay_row1), math.floor(sp_dopp_col) + 4)
                    )  # TODO: sp_dopp_col, again?
                    brcs_ddma = L1.postCal["brcs"][sp_delay_raw_range, :][
                        :, sp_dopp_col_range
                    ]
                    det = brcs_ddma[brcs_ddma < 0]
                    if len(det) > 0:
                        quality_flag1_1[16] = 1

            # flag 17
            tx_pos_x1 = L1.postCal["tx_pos_x"][sec][ngrx_channel]
            prn_code1 = L1.postCal["prn_code"][sec][ngrx_channel]
            if (tx_pos_x1 == 0) and (not np.isnan(prn_code1)):
                quality_flag1_1[17] = 1

            # flag 18
            sx_pos_x1 = L1.postCal["sp_pos_x"][sec][ngrx_channel]
            if np.isnan(sx_pos_x1) and (not np.isnan(prn_code1)):
                quality_flag1_1[18] = 1

            # flag 19
            rx_gain1 = L1.sx_rx_gain_copol[sec][ngrx_channel]
            if np.isnan(rx_gain1) and (not np.isnan(prn_code1)):
                quality_flag1_1[19] = 1

            quality_flag1_1[20] = 1

            # flag 21 and 25
            rx_alt = rx_pos_lla[2][sec]
            if rx_alt > 15000:
                quality_flag1_1[21] = 1
            if rx_alt < 700:
                quality_flag1_1[25] = 1

            # flag 23
            if prn_code1 == 28:
                quality_flag1_1[23] = 1

            # flag 24
            # rx_vel_xyz1 = rx_vel_xyz[sec][ngrx_channel]
            rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
            rx_speed1 = np.linalg.norm(rx_vel_xyz1, 2)
            if rx_speed1 > 150:
                quality_flag1_1[24] = 1

            # TODO flag 12 and flag 13 missing from matlab?

            # flag 1
            if (
                quality_flag1_1[2] == 1
                or quality_flag1_1[3] == 1
                or quality_flag1_1[4] == 1
                or quality_flag1_1[5] == 1
                or quality_flag1_1[6] == 1
                or quality_flag1_1[9] == 1
                or quality_flag1_1[10] == 1
                or quality_flag1_1[11] == 1
                or quality_flag1_1[12] == 1
                or quality_flag1_1[13] == 1
                or quality_flag1_1[14] == 1
                or quality_flag1_1[15] == 1
                or quality_flag1_1[16] == 1
                or quality_flag1_1[17] == 1
                or quality_flag1_1[19] == 1
                or quality_flag1_1[21] == 1
                or quality_flag1_1[22] == 1
            ):
                quality_flag1_1[0] = 1

            L1.postCal["quality_flags1"][sec][ngrx_channel] = get_quality_flag(
                quality_flag1_1
            )
