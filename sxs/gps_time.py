from astropy.time import Time as astro_time


def gps2utc(gpsweek, gpsseconds):
    """GPS time to unix timestamp.

    Parameters
    ----------
    gpsweek : int
        GPS week number, i.e. 1866.
    gpsseconds : int
        Number of seconds since the beginning of week.

    Returns
    -------
    numpy.float64
        Unix timestamp (UTC time).
    """
    secs_in_week = 604800
    secs = gpsweek * secs_in_week + gpsseconds

    t_gps = astro_time(secs, format="gps")
    t_utc = astro_time(t_gps, format="iso", scale="utc")

    return t_utc.unix


def utc2gps(timestamp):
    """unix timestamp to GPS.

    Parameters
    ----------
    numpy.float64
        Unix timestamp (UTC time).

    Returns
    -------
    gpsweek : int
        GPS week number, i.e. 1866.
    gpsseconds : int
        Number of seconds since the beginning of week.
    """
    secs_in_week = 604800
    t_utc = astro_time(timestamp, format="unix", scale="utc")
    t_gps = astro_time(t_utc, format="gps")
    gpsweek, gpsseconds = divmod(t_gps.value, secs_in_week)
    return gpsweek, gpsseconds
