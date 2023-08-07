import math
import numpy as np
from scipy.interpolate import interp1d


def rongowaiWAF(doppler_center,delay_center,dim,delay_axis_chips):
    """

    Parameters
    ----------
    doppler_center
    delay_center
    dim
    delay_axis_chips

    Returns
    -------
    ddm
    """
    # dim = [DelayBins, DopplerBins]

    cohT = 1e-3
    tauChip_s = 1 / 1.023e6
    delaybins = dim[1] - 1
    dopplerbins = dim[0] - 1
    centerDoppler = doppler_center - 1
    centerDelay = delay_center - 1
    resChips = 0.2552 / 2
    resDoppler = 250  #500  Rongowai Doppler Res should be 250
    ddm = np.zeros((dopplerbins, delaybins))
    for l in range(delaybins):
        dtau_s = delay_axis_chips(l + 1) * tauChip_s
        for k in range(dopplerbins):
            dfreq_Hz = (k - centerDoppler) * resDoppler
            if abs(dtau_s) <= (tauChip_s * (1 + tauChip_s / cohT)):
                _lambda = 1 - abs(dtau_s) / tauChip_s
            else:
                _lambda = -tauChip_s / cohT
            x = dfreq_Hz * math.pi * cohT
            rads = -math.pi * dfreq_Hz * cohT
            if x == 0:
                s = 1
            else:
                s = (math.sin(x) / x) * math.exp(1j * rads)
            ddm[l + 1, k + 1] = _lambda * s
    ddm = abs(ddm).T
    ddm = ddm * ddm
    return ddm


def get_RongowaiWAFRMSD(ddm,
                        delay_res_chips,
                        rmsd_delay_span_chips,
                        fft_interpolation_factor,
                        power_vs_delay_noise_floor_rows):
    """
    
    Parameters
    ----------
    ddm
    delay_res_chips
    rmsd_delay_span_chips
    fft_interpolation_factor
    power_vs_delay_noise_floor_rows

    Returns
    -------
    outputs
    """
    s = ddm.shape # [doppler by delay; 40 by 5]
    power_vs_delay = np.sum(ddm,1)
    f = interp1d(s(0), power_vs_delay)
    power_vs_delay = f(fft_interpolation_factor * s(0))
    delay_peak_index = np.argmax(power_vs_delay)
    downsample_axis = list(range(delay_peak_index - math.floor(fft_interpolation_factor * s[0] / 2),
                            delay_peak_index + math.floor(fft_interpolation_factor * s[0] / 2) + 1,
                            fft_interpolation_factor))
    downsample_axis = [x for x in downsample_axis if 1 <= x <= len(power_vs_delay)]
    power_vs_delay = [power_vs_delay[x] for x in downsample_axis]
    power_vs_delay = np.fliplr(power_vs_delay)
    noise_floor_counts = np.mean(power_vs_delay[0: power_vs_delay_noise_floor_rows])
    power_vs_delay = power_vs_delay - noise_floor_counts
    snr_db = 10 * math.log10(max(power_vs_delay) / noise_floor_counts)
    power_vs_delay = power_vs_delay / max(power_vs_delay)
    delay_peak_index = np.argmax(power_vs_delay)
    delay_axis_chips = list(range(len(power_vs_delay)))
    delay_axis_chips[0: delay_peak_index] = [x - delay_axis_chips[delay_peak_index]
                                             for x in delay_axis_chips[0: delay_peak_index]]
    delay_axis_chips[delay_peak_index + 1: -1] = [x - delay_axis_chips[delay_peak_index]
                                                  for x in delay_axis_chips[delay_peak_index + 1: -1]]
    delay_axis_chips[delay_peak_index] = 0
    delay_axis_chips = delay_axis_chips * delay_res_chips

    rwaf = rongowaiWAF(3, delay_peak_index, [len(delay_axis_chips), s(1)], delay_axis_chips)
    rwaf = np.sum(rwaf, 1)
    rwaf = rwaf / np.max(rwaf)

    outputs = {}
    outputs.delay_axis_chips = delay_axis_chips
    outputs.waf = rwaf
    outputs.power_vs_delay = power_vs_delay
    outputs.power_vs_delay_snr_db = snr_db

    span_over_which_to_compute_rmsd = list(
        range(delay_peak_index - round(rmsd_delay_span_chips / delay_res_chips),
              delay_peak_index + round(rmsd_delay_span_chips / delay_res_chips) + 1)
    )
    span_over_which_to_compute_rmsd = [x for x in span_over_which_to_compute_rmsd if 1 <= x <= len(power_vs_delay)]
    rwaf = [rwaf[x] for x in span_over_which_to_compute_rmsd]
    power_vs_delay = [power_vs_delay[x] for x in span_over_which_to_compute_rmsd]
    rmsd = math.sqrt(np.sum((rwaf - power_vs_delay) ** 2))
    outputs.rmsd = rmsd
    return outputs


def coherence_detection(L0, L1, rx_pos_lla):
    """
    coherence detection

    Parameters
    ----------
    L0
    L1
    rx_pos_lla

    Returns
    -------
    coherence_metric
    coherence_state
    """
    rmsd_delay_span_chips = 1.5  # delay span over which to compute error relative to WAF
    fft_interpolation_factor = 10  # used in getRongowaiWAFRMSD for interpolation
    power_vs_delay_noise_floor_rows = 5  # use in getRongowaiWAFRMSD power vs delay noise floor estimation
    power_vs_delay_snr_min = -10  # minimum power vs delay snr threshold
    aircraft_alt_min = 2e3  # minimum height of aircraft, otherwise coherence state is 'uncertain'
    max_peak_delay_row = 30  # only use power vs delay waveforms where the peak is less than this

    ac_alt = rx_pos_lla[2]
    coherence_metric = np.full([len(ac_alt), 10], np.nan)
    coherence_state = np.full([len(ac_alt), 10], np.nan)
    power_vs_delay_snr_db = np.full([len(ac_alt), 10], np.nan)
    peak_delay_row = np.full([len(ac_alt), 10], np.nan)

    for channel in range(10):  # channel-by-channel processing
        altitude = ac_alt
        ddm = np.squeeze(L1.ddm_power_counts[:, channel, :, :])  # select one channel at a time
        index = list(range(len(ac_alt)))

        # select valid DDMs
        zeros_check = np.squeeze(np.sum(np.sum(ddm, 0), 1))  # select valid DDMs
        idx = np.argwhere(zeros_check > 0)
        altitude = [altitude[x] for x in idx]
        ddm = [ddm[:, :, x] for x in idx]
        index = [index[x] for x in idx]

        nan_check = np.squeeze(np.sum(np.sum(ddm, 0), 1))
        idx = np.argwhere(np.isnotnan(nan_check))
        altitude = [altitude[x] for x in idx]
        ddm = [ddm[:, :, x] for x in idx]
        index = [index[x] for x in idx]

        peak_delay_row_index = np.full(len(altitude), np.nan)
        for z in range(len(altitude)):
            i = index[z]
            hold_p_vs_delay = np.squeeze(ddm[:, :, z])
            hold_p_vs_delay = np.sum(hold_p_vs_delay, 1)
            m = np.argmax(hold_p_vs_delay)
            peak_delay_row[channel, i] = m
            peak_delay_row_index[z] = m

        # detection not attempted if peak exceeds this limit
        idx = np.argwhere(peak_delay_row_index <= max_peak_delay_row)
        ddm = ddm[:, :, idx]
        index = index[idx]
        altitude = altitude[idx]

        for z in range(len(altitude)):
            i = index[z]
            outputs = get_RongowaiWAFRMSD(np.squeeze(ddm[:, :, z]), L0.delay_bin_res,
                                          rmsd_delay_span_chips, fft_interpolation_factor,
                                          power_vs_delay_noise_floor_rows)

            coherence_metric[channel, i] = outputs.rmsd
            power_vs_delay_snr_db[channel, i] = outputs.power_vs_delay_snr_db

            if power_vs_delay_snr_db[channel, i] < power_vs_delay_snr_min or altitude[z] < aircraft_alt_min:
                coherence_state[channel, i] = 5  # Uncertain coherence state
            elif coherence_metric[channel, i] <= 0.25:
                coherence_state[channel, i] = 1  # With high confidence, state is dominantly coherent
            elif 0.25 < coherence_metric[channel, i] <= 0.50:
                coherence_state[channel, i] = 2  # State is likely coherent
            elif 0.50 < coherence_metric[channel, i] < 0.75:
                coherence_state[channel, i] = 3  # State is likely mixed/weakly diffuse
            elif coherence_metric[channel, i] >= 0.75:
                coherence_state[channel, i] = 4  # With high confidence, state is dominantly incoherent

    coherence_metric[channel + 1:channel + 10, :] = coherence_metric
    coherence_state[channel + 1:channel + 10, :] = coherence_state

    L1.coherence_metric = coherence_metric
    L1.coherence_state = coherence_state
