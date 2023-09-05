import math
import cmath

import numpy as np


def rongowaiWAF(doppler_center, delay_center, dim, delay_axis_chips):
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
    delaybins = dim[0]
    dopplerbins = dim[1]
    centerDoppler = doppler_center - 1
    # centerDelay = delay_center - 1
    # resChips = 0.2552 / 2
    resDoppler = 250  # 500  Rongowai Doppler Res should be 250
    ddm = np.empty((delaybins, dopplerbins), dtype=complex)
    for l in range(delaybins):  # MATLAB is 0-based here
        dtau_s = delay_axis_chips[l] * tauChip_s
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
                s = (math.sin(x) / x) * cmath.exp(1j * rads)
            ddm[l, k] = _lambda * s
    ddm = abs(ddm).T
    ddm = ddm * ddm
    return ddm


def interpft(d_in, n):
    """
    exactly the same as interpft in MATLAB

    Parameters
    ----------
    d_in: input data
    n: length of the target interpolated data

    Returns
    -------

    """
    # Compute the FFT of the original signal
    d_fft = np.fft.fft(d_in)
    # Insert zeros to increase the length of the FFT result
    d_interp = np.zeros(n, dtype=complex)
    d_interp[: len(d_fft) // 2] = d_fft[: len(d_fft) // 2]
    d_interp[-len(d_fft) // 2 :] = d_fft[-len(d_fft) // 2 :]
    # Compute the inverse FFT to get the interpolated signal
    d_out = np.fft.ifft(d_interp)
    return d_out.real


def get_RongowaiWAFRMSD(
    ddm,
    delay_res_chips,
    rmsd_delay_span_chips,
    fft_interpolation_factor,
    power_vs_delay_noise_floor_rows,
):
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
    s = ddm.shape  # [doppler by delay; 40 by 5]
    power_vs_delay = np.sum(ddm, 1)
    """ 1d interpolation as alternative to fft interpolation
    n_interp = np.arange(fft_interpolation_factor * s[0])
    n = n_interp[::fft_interpolation_factor]
    f = interp1d(n, power_vs_delay, kind='linear', fill_value='extrapolate')
    power_vs_delay = f(n_interp)
    """
    power_vs_delay = interpft(power_vs_delay, fft_interpolation_factor * s[0])
    delay_peak_index = np.argmax(power_vs_delay)
    downsample_axis = list(
        range(
            delay_peak_index - math.floor(fft_interpolation_factor * s[0] / 2),
            delay_peak_index + math.floor(fft_interpolation_factor * s[0] / 2) + 1,
            fft_interpolation_factor,
        )
    )
    downsample_axis = [x for x in downsample_axis if 0 <= x < len(power_vs_delay)]
    power_vs_delay = [power_vs_delay[x] for x in downsample_axis]
    power_vs_delay = np.flip(power_vs_delay)
    noise_floor_counts = np.mean(power_vs_delay[0:power_vs_delay_noise_floor_rows])
    power_vs_delay = power_vs_delay - noise_floor_counts
    snr_db = 10 * math.log10(max(power_vs_delay) / noise_floor_counts)
    power_vs_delay = power_vs_delay / max(power_vs_delay)
    delay_peak_index = np.argmax(power_vs_delay)
    delay_axis_chips = np.arange(len(power_vs_delay))
    delay_axis_chips[:delay_peak_index] = [
        x - delay_axis_chips[delay_peak_index]
        for x in delay_axis_chips[:delay_peak_index]
    ]
    delay_axis_chips[delay_peak_index + 1 :] = [
        x - delay_axis_chips[delay_peak_index]
        for x in delay_axis_chips[delay_peak_index + 1 :]
    ]
    delay_axis_chips[delay_peak_index] = 0
    delay_axis_chips = delay_axis_chips * delay_res_chips

    rwaf = rongowaiWAF(
        3, delay_peak_index, [len(delay_axis_chips), s[1]], delay_axis_chips
    )
    rwaf = np.sum(rwaf, 0)
    rwaf = rwaf / np.max(rwaf)

    outputs = {}
    outputs["delay_axis_chips"] = delay_axis_chips
    outputs["waf"] = rwaf
    outputs["power_vs_delay"] = power_vs_delay
    outputs["power_vs_delay_snr_db"] = snr_db

    span_over_which_to_compute_rmsd = list(
        range(
            delay_peak_index - round(rmsd_delay_span_chips / delay_res_chips),
            delay_peak_index + round(rmsd_delay_span_chips / delay_res_chips) + 1,
        )
    )
    span_over_which_to_compute_rmsd = [
        x for x in span_over_which_to_compute_rmsd if 0 <= x < len(power_vs_delay)
    ]
    rwaf = [rwaf[x] for x in span_over_which_to_compute_rmsd]
    power_vs_delay = [power_vs_delay[x] for x in span_over_which_to_compute_rmsd]
    rmsd = np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(rwaf, power_vs_delay)]))
    outputs["rmsd"] = rmsd
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
    rmsd_delay_span_chips = (
        1.5  # delay span over which to compute error relative to WAF
    )
    fft_interpolation_factor = 10  # used in getRongowaiWAFRMSD for interpolation
    power_vs_delay_noise_floor_rows = (
        5  # use in getRongowaiWAFRMSD power vs delay noise floor estimation
    )
    power_vs_delay_snr_min = -10  # minimum power vs delay snr threshold
    aircraft_alt_min = (
        2e3  # minimum height of aircraft, otherwise coherence state is 'uncertain'
    )
    max_peak_delay_row = 29  # 0-base  # only use power vs delay waveforms where the peak is less than this

    ac_alt = rx_pos_lla[2]
    coherence_metric = np.full([len(ac_alt), 10], np.nan)
    coherence_state = np.full([len(ac_alt), 10], np.nan)
    power_vs_delay_snr_db = np.full([len(ac_alt), 10], np.nan)
    peak_delay_row = np.full([len(ac_alt), 10], np.nan)

    for channel in range(L0.J_2):  # channel-by-channel processing
        altitude = ac_alt
        ddm = np.squeeze(
            L1.ddm_power_counts[:, channel, :, :]
        )  # select one channel at a time
        index = list(range(len(ac_alt)))

        # select valid DDMs
        zeros_check = np.squeeze(np.sum(np.sum(ddm, 2), 1))  # select valid DDMs
        idx = np.squeeze(np.argwhere(zeros_check > 0))
        altitude = [altitude[i] for i in idx]
        ddm = ddm[idx, :, :]
        index = [index[i] for i in idx]

        nan_check = np.squeeze(np.sum(np.sum(ddm, 2), 1))
        idx = np.squeeze(np.argwhere(~np.isnan(nan_check)))
        altitude = [altitude[i] for i in idx]
        ddm = ddm[idx, :, :]
        index = [index[i] for i in idx]

        peak_delay_row_index = np.full(len(altitude), np.nan)
        for z in range(len(altitude)):
            i = index[z]
            hold_p_vs_delay = np.squeeze(ddm[z, :, :])
            hold_p_vs_delay = np.sum(hold_p_vs_delay, 1)
            m = np.argmax(hold_p_vs_delay)
            peak_delay_row[i, channel] = m
            peak_delay_row_index[z] = m

        # detection not attempted if peak exceeds this limit
        idx = np.squeeze(np.argwhere(peak_delay_row_index <= max_peak_delay_row))
        ddm = ddm[idx, :, :]
        index = [index[i] for i in idx]
        altitude = [altitude[i] for i in idx]

        for z in range(len(altitude)):
            i = index[z]
            outputs = get_RongowaiWAFRMSD(
                np.squeeze(ddm[z, :, :]),
                L0.delay_bin_res,
                rmsd_delay_span_chips,
                fft_interpolation_factor,
                power_vs_delay_noise_floor_rows,
            )

            coherence_metric[i, channel] = outputs["rmsd"]
            power_vs_delay_snr_db[i, channel] = outputs["power_vs_delay_snr_db"]

            if (
                power_vs_delay_snr_db[i, channel] < power_vs_delay_snr_min
                or altitude[z] < aircraft_alt_min
            ):
                coherence_state[i, channel] = 5  # Uncertain coherence state
            elif coherence_metric[i, channel] <= 0.25:
                coherence_state[
                    i, channel
                ] = 1  # With high confidence, state is dominantly coherent
            elif 0.25 < coherence_metric[i, channel] <= 0.50:
                coherence_state[i, channel] = 2  # State is likely coherent
            elif 0.50 < coherence_metric[i, channel] < 0.75:
                coherence_state[i, channel] = 3  # State is likely mixed/weakly diffuse
            elif coherence_metric[i, channel] >= 0.75:
                coherence_state[
                    i, channel
                ] = 4  # With high confidence, state is dominantly incoherent

    L1.postCal["coherence_metric"] = np.concatenate(
        (coherence_metric, coherence_metric), axis=1
    )
    L1.postCal["coherence_state"] = np.concatenate(
        (coherence_state, coherence_state), axis=1
    )
