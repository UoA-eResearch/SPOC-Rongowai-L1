import numpy as np


def L1a_counts2watts(
    ddm_counts, std_dev, ANZ_port, rf_source, L1a_cal_1dinterp, binning_thres_db
):
    ddm_counts_db = 10 * np.log10(ddm_counts)
    std_dev_db_ch = 20 * np.log10(std_dev[ANZ_port[rf_source]])

    # evaluate ddm power in dBm
    # Scipy doesn't like masked arrays, so undo here and reply after
    ddm_power_dbm = L1a_cal_1dinterp[ANZ_port[rf_source]](np.ma.getdata(ddm_counts_db))
    ddm_power_dbm = (
        ddm_power_dbm + std_dev_db_ch - binning_thres_db[ANZ_port[rf_source]]
    )
    # reapply mask to array to hide nonsense interp.
    ddm_power_dbm = np.ma.masked_where(np.ma.getmask(ddm_counts_db), ddm_power_dbm)
    # convert to watts (TODO - why 30?)
    return 10 ** ((ddm_power_dbm - 30) / 10)
