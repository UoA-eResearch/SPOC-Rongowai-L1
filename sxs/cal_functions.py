import numpy as np
import math
from pathlib import Path
from scipy.interpolate import interp1d


# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path("./dat/L1a_cal/"))
L1a_cal_ddm_counts_db_filename = Path("L1A_cal_ddm_counts_dB.dat")
L1a_cal_ddm_power_dbm_filename = Path("L1A_cal_ddm_power_dBm.dat")
L1a_cal_ddm_counts_db = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_counts_db_filename))
L1a_cal_ddm_power_dbm = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_power_dbm_filename))

# offset delay rows to derive noise floor
offset = 4
# map rf_source to ANZ_port
ANZ_port_dict = {0: 0, 4: 1, 8: 2}
binning_thres_db = [50.5, 49.6, 50.4]

# create the interpolation functions for the 3 ports
L1a_cal_1dinterp = {}
for i in range(3):
    L1a_cal_1dinterp[i] = interp1d(
        L1a_cal_ddm_counts_db[i, :],
        L1a_cal_ddm_power_dbm[i, :],
        kind="cubic",
        fill_value="extrapolate",
    )


def power2db(power):
    return np.multiply(math.log10(power), 10)


def db2power(db):
    return np.power(10, np.divide(db, 10))


def L1a_counts2watts(ddm_counts, ANZ_port, std_dev):
    """Converts raw DDM counts to DDM power in watts

    Parameters
    ----------
    ddm_counts : np.array()
        Scaled counts of a given DDM
    std_dev : List[float,float,float]
        List of binning standard deviation values per sec per RF1/RF2/RF3
    rf_source : int
        Radio frequency source value (RF1/RF2/RF3) to determine std to use

    Returns
    -------
    ddm_power_dbm : numpy.array(numpy.float64)
        Returns DDM as calibrated power in Watts
    """
    binning_thres_db = [50.5, 49.6, 50.4]
    # cable_loss_db = [1.8051, 0.6600, 0.5840]

    # select approiate calibration constants based on the input ANZ port channel
    # ddm_counts_db_ch = ddm_counts_cal_db[ANZ_port]
    # ddm_power_dbm_ch = ddm_power_cal_dbm[ANZ_port]

    std_dev_ch = std_dev[ANZ_port]

    # binning_thres_db_ch = binning_thres_db[ANZ_port]
    # cable_loss_db_ch = cable_loss_db[ANZ_port]

    # convert to dB scale
    ddm_counts_db = 10 * np.log10(ddm_counts)
    std_dev_db_ch = 20 * np.log10(std_dev_ch)

    # evaluate ddm power in dBm
    # Scipy doesn't like masked arrays, so undo here and reply after
    ddm_power_dbm = L1a_cal_1dinterp[ANZ_port](np.ma.getdata(ddm_counts_db))
    ddm_power_dbm = (
        ddm_power_dbm
        + std_dev_db_ch
        - binning_thres_db[ANZ_port]  # cable loss to be compensated when computing BRCS
    )
    # ? reapply mask to array to hide nonsense interp.
    # ? ddm_power_dbm = np.ma.masked_where(np.ma.getmask(ddm_counts_db), ddm_power_dbm)
    # convert to watts, 9 dB difference from L1a
    return 10 ** ((ddm_power_dbm - 30 - 9) / 10)


def ddm_calibration(
    std_dev_rf1,
    std_dev_rf2,
    std_dev_rf3,
    J,
    prn_code,
    raw_counts,
    rf_source,
    first_scale_factor,
    ddm_power_counts,
    power_analog,
    ddm_ant,
    inst_gain,
):
    """Calibrates raw DDMs into power DDMs in Watts

    Parameters
    ----------
    std_dev_rf1 : numpy.array()
        Binning standard deviation of RF1 channel
    std_dev_rf2 : numpy.array()
        Binning standard deviation of RF2 channel
    std_dev_rf3 : numpy.array()
        Binning standard deviation of RF3 channel
    J : int
        Number of NGRX_channels to iterate over
    prn_code : numpy.array()
        Array of PRNs of satellites
    raw_counts : 4D numpy.array()
        Array of DDMs per ngrx_channel (J) per second of flight
    rf_source : numpy.array()
        Array of RF sources (RF1/RF2/RF3) per ngrx_channel per second of flight
    first_scale_factor : numpy.array()
        Scale factor to calibrate DDM raw counts
    ddm_power_counts : numpy.array()
        Empty 4D array to receive calibrated counts of DDMs
    power_analog : numpy.array()
        Empty 4D array to recieve calibrated powers of DDMs in Watts
    ddm_ant : numpy.array()
        Empty array to receive ANZ_port of each DDM
    inst_gain : numpy.array()
        Empty array to receive inst_gain
    """
    # derive signal power
    # iterate over seconds of flight
    for sec in range(len(std_dev_rf1)):
        # retrieve noise standard deviation in counts for all three channels
        # bundle std_X[i] values for ease
        std_dev1 = [std_dev_rf1[sec], std_dev_rf2[sec], std_dev_rf3[sec]]
        # iterate over the 20 NGRX_channels
        for ngrx_channel in range(J):
            # assign local variables for PRN and DDM counts
            prn_code1 = prn_code[sec, ngrx_channel]
            rf_source1 = rf_source[sec, ngrx_channel]
            first_scale_factor1 = first_scale_factor[sec, ngrx_channel]
            raw_counts1 = raw_counts[sec, ngrx_channel, :, :]
            # solve only when presenting a valid PRN and DDM counts
            if any(
                [
                    np.isnan(prn_code1),
                    (raw_counts1[1, 1] == 0),
                    raw_counts1[0, 0] == raw_counts1[20, 2],
                ]
            ):
                continue

            # if (
            #    (not np.isnan(prn_code1))
            #    and (raw_counts1[0, 0] != raw_counts1[20, 2])
            #    and (raw_counts1[1, 1] != 0)
            # ):
            # scale raw counts and convert from counts to watts
            ANZ_port1 = ANZ_port_dict[rf_source1]
            ddm_power_counts1 = raw_counts1 * first_scale_factor1

            # perform L1a calibration from Counts to Watts
            ddm_power_watts1 = L1a_counts2watts(
                ddm_power_counts1,
                ANZ_port1,
                std_dev1,
            )

            # peak ddm location
            # find peak counts/watts/delay from DDM data
            peak_counts1 = np.max(ddm_power_counts1)
            # 0-based index
            peak_delay_bin1 = np.where(ddm_power_counts1 == peak_counts1)[0][0]
            peak_power1 = np.max(ddm_power_watts1)

            inst_gain1 = peak_counts1 / peak_power1

            # save variables
            ddm_power_counts[sec][ngrx_channel] = ddm_power_counts1
            power_analog[sec][ngrx_channel] = ddm_power_watts1
            # 0-based index
            ddm_ant[sec][ngrx_channel] = ANZ_port1 + 1
            inst_gain[sec][ngrx_channel] = inst_gain1


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
