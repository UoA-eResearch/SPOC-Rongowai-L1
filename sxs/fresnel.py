import numpy as np
from timeit import default_timer as timer

from brcs import get_fresnel
from calibration import power2db


def fresnel_calculations(L0, L1, rx_pos_x, rx_pos_y, rx_pos_z):
    t0 = timer()
    # TODO can probably condense this loop into thre above loop
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
                L1.postCal["fresnel_orientation"][sec][
                    ngrx_channel
                ] = fresnel_orientation1

            # Do this once here rather than another loop over Sec and L0.J_2
            if ngrx_channel < L0.J_2:
                nbrcs_LHCP1 = L1.nbrcs[sec][ngrx_channel]
                nbrcs_RHCP1 = L1.nbrcs[sec][ngrx_channel + L0.J_2]
                CP1 = nbrcs_LHCP1 / nbrcs_RHCP1
                if CP1 > 0:
                    CP_db1 = power2db(CP1)
                    L1.postCal["nbrcs_cross_pol"][sec][ngrx_channel] = CP_db1

    print(f"******** finish processing part 7 data with {timer() - t0}********")

    L1.postCal["nbrcs_cross_pol"][:, L0.J_2 : L0.J] = L1.postCal["nbrcs_cross_pol"][
        :, 0 : L0.J_2
    ]
