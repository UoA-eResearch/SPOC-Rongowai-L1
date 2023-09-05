import math

import numpy as np
from scipy.interpolate import interp1d

# ignore divide by zero in log10
np.seterr(divide="ignore")

# offset delay rows to derive noise floor
offset = 4
# map rf_source to ANZ_port
ANZ_port_dict = {0: 0, 4: 1, 8: 2}
binning_thres_db = [50.5, 49.6, 50.4]


def power2db(power):
    return np.multiply(math.log10(power), 10)


def db2power(db):
    return np.power(10, np.divide(db, 10))


def mag2db(magnitude):
    return np.multiply(np.log10(magnitude), 20)


def L1a_counts2watts(inp, ddm_counts, ANZ_port, std_dev):
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
    ddm_power_dbm = inp.L1a_cal_1dinterp[ANZ_port](np.ma.getdata(ddm_counts_db))
    ddm_power_dbm = (
        ddm_power_dbm
        + std_dev_db_ch
        - binning_thres_db[ANZ_port]  # cable loss to be compensated when computing BRCS
    )
    # reapply mask to array to hide nonsense interp.
    # ddm_power_dbm = np.ma.masked_where(np.ma.getmask(ddm_counts_db), ddm_power_dbm)
    # convert to watts
    return 10 ** ((ddm_power_dbm - 30) / 10)


def L1a_counts2watts2(inp, ddm_counts, ANZ_port, std_dev, noise_floor):
    """Converts raw DDM counts to DDM power in watts

    Parameters
    ----------
    inp : class
    ddm_counts : np.array()
        Scaled counts of a given DDM
    ANZ_port : int
    std_dev : List[float,float,float]
        List of binning standard deviation values per sec per RF1/RF2/RF3
    noise_floor : np.array()

    Returns
    -------
    ddm_power_dbm : numpy.array(numpy.float64)
        Returns DDM as calibrated power in Watts
    """
    binning_thres_db = [50.5, 49.6, 50.4]
    binning_thres = db2power(binning_thres_db)

    noise_floor_LHCP = noise_floor[0][0]
    noise_floor_RHCP = noise_floor[0][-1]

    # select approiate calibration constants based on the input ANZ port channel
    ddm_counts_ch = inp.L1a_cal_ddm_counts[ANZ_port]
    ddm_power_ch = inp.L1a_cal_ddm_power[ANZ_port]
    if ANZ_port == 1:
        ddm_counts = ddm_counts - noise_floor_LHCP
    elif ANZ_port == 2:
        ddm_counts = ddm_counts - noise_floor_RHCP
    else:
        ddm_counts = None
    mag_std_dev_ch = std_dev[ANZ_port]
    binning_thres_ch = binning_thres[ANZ_port]
    std_dev_ch = mag_std_dev_ch**2

    # evaluate ddm power in watts
    # ddm_power = inp.L1a_cal_1dinterp[ANZ_port](np.ma.getdata(ddm_counts))
    f = interp1d(ddm_counts_ch, ddm_power_ch, kind="cubic", fill_value="extrapolate")
    ddm_power = f(np.ma.getdata(ddm_counts))
    ddm_power_watts = ddm_power * std_dev_ch / binning_thres_ch
    return ddm_power_watts


def ddm_calibration(
    inp,
    L0,
    L1
    # std_dev_rf1,
    # std_dev_rf2,
    # std_dev_rf3,
    # J,
    # prn_code,
    # raw_counts,
    # rf_source,
    # first_scale_factor,
    # ddm_power_counts,
    # power_analog,
    # ddm_ant,
    # inst_gain,
    # noise_floor,
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
    std_dev_rf1 = L0.std_dev_rf1
    std_dev_rf2 = L0.std_dev_rf2
    std_dev_rf3 = L0.std_dev_rf3
    J = L0.J
    prn_code = L1.postCal["prn_code"]
    raw_counts = L0.raw_counts
    rf_source = L0.rf_source
    first_scale_factor = L0.first_scale_factor
    ddm_power_counts = L1.ddm_power_counts
    power_analog = L1.power_analog
    ddm_ant = L1.postCal["ddm_ant"]
    inst_gain = L1.postCal["inst_gain"]
    noise_floor = L1.postCal["ddm_noise_floor"]
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
                    (raw_counts1[0, 0] == 0),
                    raw_counts1[0, 0] == raw_counts1[20, 2],
                ]
            ):
                continue

            # if (
            #    (not np.isnan(prn_code1))
            #    and (raw_counts1[0, 0] != raw_counts1[20, 2])
            #    and (raw_counts1[0, 0] != 0)
            # ):
            # scale raw counts and convert from counts to watts
            ANZ_port1 = ANZ_port_dict[rf_source1]
            ddm_power_counts1 = raw_counts1 * first_scale_factor1

            # perform L1a calibration from Counts to Watts
            ddm_power_watts1 = L1a_counts2watts2(
                inp, ddm_power_counts1, ANZ_port1, std_dev1, noise_floor
            )

            # peak ddm location
            # find peak counts/watts/delay from DDM data
            peak_counts1 = np.max(ddm_power_counts1)
            peak_power1 = np.max(ddm_power_watts1)

            inst_gain1 = peak_counts1 / peak_power1

            # save variables
            ddm_power_counts[sec][ngrx_channel] = ddm_power_counts1
            power_analog[sec][ngrx_channel] = ddm_power_watts1
            # 0-based index
            ddm_ant[sec][ngrx_channel] = ANZ_port1 + 1
            inst_gain[sec][ngrx_channel] = inst_gain1

    L1a_power_calibration_flag_LHCP = np.zeros((len(std_dev_rf1), int(J / 2)))
    L1a_power_calibration_flag_RHCP = np.zeros((len(std_dev_rf1), int(J / 2)))

    std_dev_rf2_db = np.tile(mag2db(std_dev_rf2), (int(J / 2), 1)).T
    std_dev_rf3_db = np.tile(mag2db(std_dev_rf3), (int(J / 2), 1)).T

    flag_idx_rf2 = std_dev_rf2_db > 65
    flag_idx_rf3 = std_dev_rf3_db > 65

    L1a_power_calibration_flag_LHCP[flag_idx_rf2] = 1
    L1a_power_calibration_flag_RHCP[flag_idx_rf3] = 1

    L1a_power_calibration_flag = np.concatenate(
        (L1a_power_calibration_flag_LHCP, L1a_power_calibration_flag_RHCP), axis=1
    )

    L1.postCal["L1a_power_calibration_flag"] = L1a_power_calibration_flag
    L1.postCal["L1a_power_ddm"] = power_analog
    L1.postCal["inst_gain"] = inst_gain
    L1.postCal["ddm_ant"] = ddm_ant
