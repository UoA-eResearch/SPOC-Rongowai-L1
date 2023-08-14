# mike.laverick@auckland.ac.nz
# Specular point related functions
import geopy.distance as geo_dist
import math
from numba import njit
from numba.typed import List as numba_list
import numpy as np
import pymap3d as pm
import pyproj
from scipy import constants

from utils import get_local_dem, get_surf_type2
from projections import ecef2lla, lla2ecef


# define WGS84
wgs84 = pyproj.Geod(ellps="WGS84")
wgs84_a = float(wgs84.a)
wgs84_b = float(wgs84.b)
wgs84_es = float(wgs84.es)
abc = np.array([wgs84_a, wgs84_a, wgs84_b])


# define Fibonacci sequence here, once
k_fib = range(60)
term1_fib = np.ones(60) * ((1 + np.sqrt(5)) / 2)
term1_fib = term1_fib ** (np.array(k_fib) + 1)
term2_fib = np.ones(60) * ((1 - np.sqrt(5)) / 2)
term2_fib = term2_fib ** (np.array(k_fib) + 1)
fib_seq = (term1_fib - term2_fib) / np.sqrt(5)

# define chip length
# L1 GPS cps
chip_rate = 1.023e6
# chip length
l_chip = constants.c / chip_rate
# constant grid matrix 11*11
num_grid = 11


@njit
def nadir(m_ecef):
    """% This function, based on the WGS84 model, computes the ECEF coordinate
    of the nadir point (n) of a in-space point (m) on a WGS84 model"""
    # calculate nadir point. latitude degrees
    thetaD = math.asin(m_ecef[2] / np.linalg.norm(m_ecef, 2))
    cost2 = math.cos(thetaD) * math.cos(thetaD)
    # lat-dependent Earth radius
    r = wgs84_a * np.sqrt((1 - wgs84_es) / (1 - wgs84_es * cost2))
    # return nadir of m on WGS84
    return r * m_ecef / np.linalg.norm(m_ecef, 2)


@njit
def ite(tx_pos_xyz, rx_pos_xyz):
    """% This function iteratively solve the positions of specular points
    % based on the WGS84 model
    % Inputs:
    % 1) Tx and Rx coordiates in ECEF XYZ
    % Ouputputs
    % 1) Sx coordinate in ECEF XYZ
    """
    s_t2r = np.linalg.norm(rx_pos_xyz - tx_pos_xyz, 2)

    # determine iteration
    N = np.nanmin([x[0] for x in enumerate(fib_seq) if x[1] > s_t2r])

    # first iteration parameters
    a = rx_pos_xyz
    b = tx_pos_xyz

    for k in range(int(N) - 1):
        term1 = fib_seq[N - k - 1] / fib_seq[N - k + 1]
        term2 = fib_seq[N - k] / fib_seq[N - k + 1]
        m_lambda = a + term1 * (b - a)
        m_mu = a + term2 * (b - a)

        # nadir points
        s_lambda = nadir(m_lambda)
        s_mu = nadir(m_mu)
        # propagation distance
        f_lambda = np.linalg.norm(s_lambda - tx_pos_xyz, 2) + np.linalg.norm(
            rx_pos_xyz - s_lambda, 2
        )
        f_mu = np.linalg.norm(s_mu - tx_pos_xyz, 2) + np.linalg.norm(
            rx_pos_xyz - s_mu, 2
        )

        if f_lambda > f_mu:
            a = m_lambda
        else:
            b = m_mu
    return nadir(m_lambda)


def coarsetune(tx_pos_xyz, rx_pos_xyz):
    """% this function computes the SP on a pure WGS84 datum based on
    % Inputs:
    % 1) tx_pos_xyz: ECEF coordinate of the TX
    % 2) rx_pos_xyz: ECEF coordinate of the RX
    % Outputs:
    % 1) SP_xyz, SP_lla: ECEF and LLA coordinate of a SP on a pure WGS84 datum"""

    # find coarse SP using Fibonacci sequence
    SP_xyz_coarse = ite(tx_pos_xyz, rx_pos_xyz)
    SP_lla_coarse = ecef2lla.transform(*SP_xyz_coarse, radians=False)
    # longitude adjustment
    if SP_lla_coarse[0] < 0:
        SP_lla_coarse[0] += 360
    elif SP_lla_coarse[0] > 360:
        SP_lla_coarse[0] -= 360
    # change order to lat, lon, alt
    SP_lla_coarse = SP_lla_coarse[1], SP_lla_coarse[0], SP_lla_coarse[2]
    return SP_xyz_coarse, SP_lla_coarse


@njit
def los_status(tx_pos_xyz, rx_pos_xyz):
    """% This function determines if the RT vector has intersections
    % with the WGS84 ellipsoid (LOS existence)
    % input: tx and rx locations in ECEF frame
    % output: flag to indicate if LOS exists between tx and rx"""

    # rx for NGRx, tx for satellite, given in ECEF-XYZ, pos vectors
    T_ecef = np.divide(tx_pos_xyz, abc)
    R_ecef = np.divide(rx_pos_xyz, abc)

    # unit vector of RT
    RT_unit = (T_ecef - R_ecef) / np.linalg.norm((T_ecef - R_ecef), 2)

    # determine if LOS exists (flag = 1)
    A = np.linalg.norm(RT_unit, 2) * np.linalg.norm(RT_unit, 2)
    B = 2 * np.dot(R_ecef, RT_unit)
    C = np.linalg.norm(R_ecef, 2) * np.linalg.norm(R_ecef, 2) - 1

    t1 = (B * B) - 4 * A * C
    if t1 < 0:
        return True
    t2 = (B * -1) + np.sqrt(t1) / (2 * A)
    t3 = (B * -1) - np.sqrt(t1) / (2 * A)
    if (t2 < 0) and (t3 < 0):
        return True
    return False


@njit
def finetune_p1(sx_lla, L):
    # find the pixel location
    # in Python sx_lla is (lon, lat, alt) not (lat, lon, alt)
    min_lat, max_lat = (sx_lla[0] - L / 2, sx_lla[0] + L / 2)
    min_lon, max_lon = (sx_lla[1] - L / 2, sx_lla[1] + L / 2)

    lat_bin = np.linspace(min_lat, max_lat, num_grid)
    lon_bin = np.linspace(min_lon, max_lon, num_grid)

    # Vectorise the 11*11 nested loop
    lat_bin_v = np.repeat(lat_bin, 11)
    # lon_bin_v = np.tile(lon_bin, 11)
    lon_bin_v = np.repeat(lon_bin, 11).reshape(-1, 11).T.flatten()
    return lat_bin, lon_bin, lat_bin_v, lon_bin_v


@njit
def finetune_p2(p_x, p_xyz, tx_xyz, rx_xyz, ele):
    p_xyz_t = p_xyz - tx_xyz.reshape(-1, 1)
    p_xyz_r = np.repeat(rx_xyz, len(p_x))
    p_xyz_r_new1 = p_xyz_r.reshape(3, -1)
    p_xyz_r_new2 = p_xyz_r_new1 - p_xyz

    for i in range(p_xyz_t.shape[1]):
        nrm = np.linalg.norm(p_xyz_t[:, i], 2)
        # assign first element as nrm
        p_xyz_t[0, i] = nrm

    for i in range(p_xyz_r_new2.shape[1]):
        tmp = p_xyz_r_new2[:, i]
        nrm = np.linalg.norm(p_xyz_r_new2[:, i], 2)
        # assign first element as nrm
        p_xyz_r_new2[0, i] = nrm

    delay_chip = p_xyz_t[0, :] + p_xyz_r_new2[0, :]
    # delay_chip = np.linalg.norm(p_xyz_t, 2, axis=0) + np.linalg.norm(p_xyz_r, 2, axis=0)
    ele = ele.reshape(11, -1)
    delay_chip = (delay_chip / l_chip).reshape(11, -1)

    # index of the pixel with minimal reflection path
    min_delay = np.nanmin(delay_chip)
    m_i, n_i = np.where(delay_chip == (np.min(delay_chip)))

    return min_delay, m_i, n_i, ele


def finetune(tx_xyz, rx_xyz, sx_lla, L, model):
    """% This code fine-tunes the coordinate of the initial SP based on the DTU10
    % datum thorugh a number of iterative steps."""
    # numba optimisation
    sp_temp = numba_list()
    for sp_val in sx_lla:
        sp_temp.append(sp_val)
    lat_bin, lon_bin, lat_bin_v, lon_bin_v = finetune_p1(sp_temp, L)
    ele = model((lon_bin_v, lat_bin_v))
    p_x, p_y, p_z = lla2ecef.transform(*[lon_bin_v, lat_bin_v, ele], radians=False)
    p_xyz = np.array([p_x, p_y, p_z])
    min_delay, m_i, n_i, ele = finetune_p2(p_x, p_xyz, tx_xyz, rx_xyz, ele)
    # unpack arrays with [0] else they keep nesting
    sx_temp = [lat_bin[m_i][0], lon_bin[n_i][0], ele[m_i, n_i][0]]
    # this is between Matlab idx = 5,6 so extra "-1" due to python 0-indexing (Mat5,6 -> Py4,5)
    NN = int((num_grid - 1) / 2) - 1
    # We calculate geodesic distance between points in metres - replaces m_lldist.m
    res = geo_dist.geodesic(
        (lat_bin[NN], lon_bin[NN]), (lat_bin[NN + 1], lon_bin[NN + 1])
    ).m
    return res, min_delay, sx_temp


def finetune_ocean(tx_pos_xyz, rx_pos_xyz, sp_lla_coarse, model, L, res_grid):
    """% This function fine tunes the SP coordiantes using a DTU10 datum
    % Inputs:
    % 1) TX and 2) RX coordinates in the form of ECEF-XYZ and
    % 3) SP coordinate in the form of LLA
    % 4) model: earth model - currently DTU10
    % 5) L: inital searching area in deg
    % 6) res_grid: targeted resolution of each grid when quitting the iteration
    % in metres
    % Output: return
    % 1) fine-tuned SP coordiantes in ECEF-XYZ, and
    % 2) local incidence angle"""

    # derive SP on the ocean surface
    res = 1000
    sp_temp = sp_lla_coarse
    while res > res_grid:
        res, _, sp_temp = finetune(tx_pos_xyz, rx_pos_xyz, sp_temp, L, model)
        # parameters for the next iteration - new searching area, new SP coordinate
        L = L * 2.0 / 11.0
    sp_temp1 = [sp_temp[1], sp_temp[0], sp_temp[2]]
    sx_xyz = lla2ecef.transform(*sp_temp1, radians=False)
    return sx_xyz, sp_temp


def angles(local_dem, tx_pos_xyz, rx_pos_xyz):
    """% This function computes the local incidence and reflection angles of
    % the middle pixel in a 3 by 3 DEM pixel matrix
    % Inputs:
    % 1) lat,lon, and ele matrices of the 3*3 pixel matrix
    % 2) Tx and Rx coordinates ECEF(x,y,z)
    % Outputs:
    % 1) theta_i, phi_i: local incidence angle along elevation and azimuth
    % angles in degree
    % 2) theta_s, phi_s: local scattering (reflection) angles along elevation
    % and azimuth angles in degree"""

    # origin of the local enu frame
    s0 = [local_dem["lat"][1], local_dem["lon"][1], local_dem["ele"][1, 1]]

    # convert tx and rx to local ENU centred at s0
    ts = np.array([0, 0, 0]) - np.array(
        pm.ecef2enu(*tx_pos_xyz, *s0, deg=True)
    )  # default = wgs84
    sr = pm.ecef2enu(*rx_pos_xyz, *s0, deg=True)  # -[0,0,0]  == same...

    # convert s1-s4 to the same local ENU
    s1 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][0], local_dem["lon"][1], local_dem["ele"][0, 1], *s0
        )
    )  # north
    s2 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][2], local_dem["lon"][1], local_dem["ele"][2, 1], *s0
        )
    )  # south
    s3 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][1], local_dem["lon"][2], local_dem["ele"][1, 0], *s0
        )
    )  # east
    s4 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][1], local_dem["lon"][0], local_dem["ele"][1, 2], *s0
        )
    )  # west

    # local unit North, East and Up vectors
    unit_e = (s3 - s4) / np.linalg.norm(s3 - s4, 2)
    unit_n = (s1 - s2) / np.linalg.norm(s1 - s2, 2)
    unit_u = np.cross(unit_e, unit_n)

    p_1e, p_1n, p_1u = np.dot(ts, unit_e), np.dot(ts, unit_n), np.dot(ts, unit_u)
    p_2e, p_2n, p_2u = np.dot(sr, unit_e), np.dot(sr, unit_n), np.dot(sr, unit_u)

    term1, term2 = p_1e * p_1e + p_1n * p_1n, p_2e * p_2e + p_2n * p_2n
    theta_i = np.rad2deg(np.arctan(math.sqrt(term1) / abs(p_1u)))
    theta_s = np.rad2deg(np.arctan(math.sqrt(term2) / p_2u))
    phi_i = np.rad2deg(np.arctan(p_1n / p_1e))
    phi_s = np.rad2deg(np.arctan(p_2n / p_2e))
    return theta_i, theta_s, phi_i, phi_s


def sp_solver(tx_pos_xyz, rx_pos_xyz, dem, dtu10, dist_to_coast_nz):
    """% SP solver derives the coordinate(s) of the specular reflection (sx)
    % SP solver also reports the local incidence angle and the distance to coast in km where the SP occur
    % All variables are reported in ECEF
    % Inputs:
    % 1) tx and rx positions
    % 2) DEM models: dtu10, NZSRTM30, and land-ocean mask
    % Outputs:
    % 1) sx_pos_xyz: sx positions in ECEF
    % 2) in_angle_deg: local incidence angle at the specular reflection
    % 3) distance to coast in kilometer
    % 4) LOS flag"""

    # derive SP coordinate on WGS84 and DTU10
    sx_xyz_coarse, sx_lla_coarse = coarsetune(tx_pos_xyz, rx_pos_xyz)

    # initial searching region in degrees
    L_ocean_deg = 1.0
    # converge criteria 0.01 meter
    res_ocean_meter = 0.01

    # derive local angles
    sx_pos_xyz, sx_pos_lla = finetune_ocean(
        tx_pos_xyz, rx_pos_xyz, sx_lla_coarse, dtu10, L_ocean_deg, res_ocean_meter
    )
    # sx_pos_xyz = lla2ecef.transform(*sx_pos_lla, radians=False)
    # replaces get_map_value function
    dist = dist_to_coast_nz((sx_pos_lla[1], sx_pos_lla[0]))
    # dist = get_map_value(sx_pos_lla[0], sx_pos_lla[1], dist_to_coast_nz)

    local_dem = get_local_dem(sx_pos_lla, dem, dtu10, dist)
    theta_i, theta_s, phi_i, phi_s = angles(local_dem, tx_pos_xyz, rx_pos_xyz)

    if dist > 0:
        # local height of the SP = local_dem["ele"][1,1]
        # projection to local dem
        # sx_pos_xyz += (sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)) * local_dem["ele"][
        #     1, 1
        # ]
        local_height = local_dem["ele"]
        local_height = local_height[1, 1]  # local height of the SP

        # projection to local dem
        term1 = np.array(sx_xyz_coarse) / np.linalg.norm(sx_xyz_coarse, 2)
        term2 = term1.dot(local_height)
        sx_pos_xyz = np.array(sx_xyz_coarse) + term2

    v_tsx = tx_pos_xyz - sx_pos_xyz
    unit_tsx = v_tsx / np.linalg.norm(v_tsx, 2)
    unit_sx = sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)
    inc_angle_deg = np.rad2deg(np.arccos(np.dot(unit_tsx, unit_sx)))

    d_phi1 = np.sin(np.deg2rad(phi_s - phi_i + 180)) / np.cos(
        np.deg2rad(phi_s - phi_i + 180)
    )
    d_phi = np.rad2deg(np.arctan(d_phi1))
    d_snell_deg = abs(theta_i - theta_s) + abs(d_phi)

    return sx_pos_xyz, inc_angle_deg, d_snell_deg, dist  # , LOS_flag


def ecef2orf(P, V, S_ecef):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a point
    in the object's orbit reference frame (orf)
    Input (all vectors are row vectors):
    1) P & V: object's ECEF position (P) and velocity (V) vectors
    2) S_ecef: ECEF coordinate of the point to be computed (S_ecef)
    Output:
    1) theta_orf & phi_orf: polar and azimuth angles of S in SV's orf in degree
    2) S_orf: coordinate of S in orf S_orf
    """
    P = P.T
    V = V.T
    S_ecef = np.array(S_ecef).T
    u_ecef = S_ecef - P  # vector from P to S

    theta_e = 7.2921158553e-5  # earth rotation rate, rad/s
    W_e = np.array([0, 0, theta_e]).T  # earth rotation vector
    Vi = V + np.cross(W_e, P)  # SC ECEF inertial velocity vector

    # define orbit reference frame - unit vectors
    y_orf = np.cross(-1 * P, Vi) / np.linalg.norm(np.cross(-1 * P, Vi), 2)
    z_orf = -1 * P / np.linalg.norm(P, 2)
    x_orf = np.cross(y_orf, z_orf)

    # transformation matrix
    T_orf = np.array([x_orf.T, y_orf.T, z_orf.T])
    S_orf = np.dot(T_orf, u_ecef)

    # elevation and azimuth angles
    theta_orf = np.rad2deg(np.arccos(S_orf[2] / (np.linalg.norm(S_orf, 2))))
    phi_orf = math.degrees(math.atan2(S_orf[1], S_orf[0]))

    if phi_orf < 0:
        phi_orf = 360 + phi_orf

    return theta_orf, phi_orf


def get_sx_rx_gain(sp_angle_ant, nadir_pattern):
    """
    define azimuth and elevation angle in the antenna frame
    Parameters
    ----------
    sp_angle_ant
    nadir_pattern
    Returns
    -------
    """
    res = 0.1  # resolution in degrees
    az_deg = np.arange(0, 360, res)
    el_deg = np.arange(120, 0, -1 * res)

    lhcp_gain_pattern = nadir_pattern["LHCP"]
    rhcp_gain_pattern = nadir_pattern["RHCP"]

    sp_theta_ant = sp_angle_ant[0]
    sp_az_ant = sp_angle_ant[1]

    az_index = np.argmin(np.abs(sp_az_ant - az_deg))
    el_index = np.argmin(np.abs(sp_theta_ant - el_deg))

    lhcp_gain_dbi = lhcp_gain_pattern[el_index, az_index]
    rhcp_gain_dbi = rhcp_gain_pattern[el_index, az_index]

    sx_rx_gain = [lhcp_gain_dbi, rhcp_gain_dbi]

    return sx_rx_gain


def ecef2brf(P, V, S_ecef, SC_att):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a
    ecef vector in the objects's body reference frame (brf)
    Input:
    1) P, V: object's ecef position vector
    2) SC_att: object's attitude (Euler angle) in the sequence of
    roll, pitch, yaw, in degrees
    3) S_ecef: ecef coordinate of the point to be computed
    Output:
    1) theta_brf: elevation angle of S in the SC's brf in degree
    2) phi_brf: azimuth angle of S in the SC's brf in degree
    """
    P = P.T
    V = V.T
    S_ecef = np.array(S_ecef).T

    phi = SC_att[0]  # roll
    theta = SC_att[1]  # pitch
    psi = SC_att[2]  # yaw

    u_ecef = S_ecef - P  # vector from P to S

    # define heading frame - unit vectors
    y_hrf = np.cross(-1 * P, V) / np.linalg.norm(np.cross(-1 * P, V), 2).T
    z_hrf = -1 * P / np.linalg.norm(-1 * P, 2).T
    x_hrf = np.cross(y_hrf, z_hrf).T

    T_hrf = np.array([x_hrf.T, y_hrf.T, z_hrf.T])

    # S in hrf
    S_hrf = np.dot(T_hrf, u_ecef)

    # construct aircraft's attitude matrix
    Rx_phi = np.array(
        [
            [1, 0, 0],
            [0, math.cos(phi), math.sin(phi)],
            [0, -1 * math.sin(phi), math.cos(phi)],
        ]
    )

    Ry_theta = np.array(
        [
            [math.cos(theta), 0, -1 * math.sin(theta)],
            [0, 1, 0],
            [math.sin(theta), 0, math.cos(theta)],
        ]
    )

    Rz_psi = np.array(
        [
            [math.cos(psi), math.sin(psi), 0],
            [-1 * math.sin(psi), math.cos(psi), 0],
            [0, 0, 1],
        ]
    )

    R = Ry_theta.dot(Rx_phi).dot(Rz_psi)  # transformation matrix

    S_brf = np.dot(R, S_hrf.T)

    theta_brf = np.rad2deg(np.arccos(S_brf[2] / (np.linalg.norm(S_brf, 2))))
    phi_brf = math.degrees(math.atan2(S_brf[1], S_brf[0]))

    if phi_brf < 0:
        phi_brf = 360 + phi_brf

    return theta_brf, phi_brf


@njit
def cart2sph(x, y, z):
    xy = x**2 + y**2
    r = math.sqrt(xy + z**2)
    theta = math.atan2(z, math.sqrt(xy))
    phi = math.atan2(y, x)
    return phi, theta, r  # for consistency with MATLAB


def ecef2enuf(P, S_ecef):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a point
    in the object's ENU frame (enuf)
    input:
    1) P: object's ECEF position vector
    2) S_ecef: ECEF coordinate of the point to be computed
    output:
    1) theta_enuf & phi_enuf: elevation and azimuth angles of S in enuf in degree
    """
    # P = [-4593021.50000000,	608280.500000000,	-4370184.50000000]
    # S_ecef = [-4590047.30433596,	610685.547457113,	-4371634.83935421]

    lon, lat, alt = ecef2lla.transform(*P, radians=False)
    S_east, S_north, S_up = pm.ecef2enu(*S_ecef, lat, lon, alt, deg=True)
    phi_enuf, theta_enuf1, _ = cart2sph(S_east, S_north, S_up)

    phi_enuf = np.rad2deg(phi_enuf)
    theta_enuf1 = np.rad2deg(theta_enuf1)

    theta_enuf = 90 - theta_enuf1

    return theta_enuf, phi_enuf


def sp_related(tx, rx, sx_pos_xyz, SV_eirp_LUT):
    """
    this function computes the sp-related variables, including angles in
    various coordinate frames, ranges, EIRP, nadir antenna gain etc
    Inputs:
    1) tx, rx: tx and rx structures
    2) sx_pos_xyz: sx ECEF position vector
    3) SV_PRN_LUT,SV_eirp_LUT: look-up table between SV number and PRN
    Outputs:
    1) sp_angle_body: sp angle in body frame, az and theta
    2) sp_angle_enu: sp angle in ENU frame, az and theta
    3) theta_gps: GPS off boresight angle
    4) range: tx to sx range, and rx to sx range
    5) gps_rad: EIRP, tx power
    """
    # sparse structres
    tx_pos_xyz = tx["tx_pos_xyz"]
    tx_vel_xyz = tx["tx_vel_xyz"]
    sv_num = tx["sv_num"]

    rx_pos_xyz = rx["rx_pos_xyz"]
    rx_vel_xyz = rx["rx_vel_xyz"]
    rx_att = rx["rx_attitude"]

    # compute angles
    theta_gps, _ = ecef2orf(tx_pos_xyz, tx_vel_xyz, sx_pos_xyz)

    sp_theta_body, sp_az_body = ecef2brf(rx_pos_xyz, rx_vel_xyz, sx_pos_xyz, rx_att)
    sp_theta_enu, sp_az_enu = ecef2enuf(rx_pos_xyz, sx_pos_xyz)

    sp_angle_body = [sp_theta_body, sp_az_body]
    sp_angle_enu = [sp_theta_enu, sp_az_enu]

    # compute ranges
    R_tsx = np.linalg.norm(sx_pos_xyz - tx_pos_xyz, 2)  # range from tx to sx
    R_rsx = np.linalg.norm(sx_pos_xyz - rx_pos_xyz, 2)  # range from rx to sx

    range = [R_tsx, R_rsx]

    # 0-based index
    # compute gps radiation properties
    j = SV_eirp_LUT[:, 0] == sv_num  # index of SV number in eirp LUT

    gps_pow_dbw = SV_eirp_LUT[j, 2]  # gps power in dBw

    # coefficients to compute gps antenna gain
    a = SV_eirp_LUT[j, 3]
    b = SV_eirp_LUT[j, 4]
    c = SV_eirp_LUT[j, 5]
    d = SV_eirp_LUT[j, 6]
    e = SV_eirp_LUT[j, 7]
    f = SV_eirp_LUT[j, 8]

    # fitting antenna gain
    gps_gain_dbi = (
        a * theta_gps**5
        + b * theta_gps**4
        + c * theta_gps**3
        + d * theta_gps**2
        + e * theta_gps
        + f
    )

    # compute static gps eirp
    stat_eirp_dbw = gps_pow_dbw + gps_gain_dbi  # static eirp in dbw
    stat_eirp_watt = 10 ** (stat_eirp_dbw / 10)  # static eirp in linear watts

    gps_rad = [gps_pow_dbw[0], gps_gain_dbi[0], stat_eirp_watt[0]]

    # compute angles in nadir antenna frame and rx gain
    sp_theta_ant = sp_theta_body
    # % no need the 180 compensation as it has been done in the gain LUT - 28 June
    sp_az_ant = sp_az_body  # + 180

    if sp_az_ant > 360:
        sp_az_ant = sp_az_ant - 360

    sp_angle_ant = [sp_theta_ant, sp_az_ant]

    return sp_angle_body, sp_angle_enu, sp_angle_ant, theta_gps, range, gps_rad


def specular_calculations(
    L0,
    L1,
    inp,
    rx_pos_x,
    rx_pos_y,
    rx_pos_z,
    rx_vel_x,
    rx_vel_y,
    rx_vel_z,
    rx_roll,
    rx_pitch,
):
    # iterate over each second of flight
    for sec in range(L0.I):
        # retrieve rx positions, velocities and attitdues
        # bundle up craft pos/vel/attitude data into per sec, and rx1
        rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
        rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
        # Euler angels are now in radians and yaw is resp. North
        # Hard-code of 0 due to alignment of antenna and craft
        rx_attitude1 = np.array([rx_roll[sec], rx_pitch[sec], 0])  # rx_heading[sec]])
        rx1 = {
            "rx_pos_xyz": rx_pos_xyz1,
            "rx_vel_xyz": rx_vel_xyz1,
            "rx_attitude": rx_attitude1,
        }

        # variables are solved only for LHCP channels
        # RHCP channels share the same vales except RX gain solved for each channel
        for ngrx_channel in range(L0.J_2):
            # retrieve tx positions and velocities
            # bundle up satellite position and velocity data into per sec, and tx1
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

            # trans_id1 = L1.postCal["prn_code"][sec][ngrx_channel]
            sv_num1 = L1.postCal["sv_num"][sec][ngrx_channel]
            # ddm_ant1 = L1.postCal["ddm_ant"][sec][ngrx_channel]

            tx1 = {
                "tx_pos_xyz": tx_pos_xyz1,
                "tx_vel_xyz": tx_vel_xyz1,
                "sv_num": sv_num1,
            }

            # only process these with valid TX positions
            if not np.isnan(L1.postCal["tx_pos_x"][sec][ngrx_channel]):
                LOS_flag1 = los_status(tx_pos_xyz1, rx_pos_xyz1)

                L1.postCal["LOS_flag"][sec][ngrx_channel] = int(LOS_flag1)

                # only process samples with valid sx positions, i.e., LOS = True
                if LOS_flag1:
                    # Part 4.1: SP solver
                    # derive SP positions, angle of incidence and distance to coast
                    # returning sx_pos_lla1 in Py version to avoid needless coord conversions
                    # derive sx velocity
                    # time step in second
                    dt = 1
                    tx_pos_xyz_dt = tx_pos_xyz1 + (dt * tx_vel_xyz1)
                    rx_pos_xyz_dt = rx_pos_xyz1 + (dt * rx_vel_xyz1)
                    (
                        sx_pos_xyz1,
                        inc_angle_deg1,
                        d_snell_deg1,
                        dist_to_coast_km1,
                    ) = sp_solver(
                        tx_pos_xyz1, rx_pos_xyz1, inp.dem, inp.dtu10, inp.landmask_nz
                    )

                    (
                        sx_pos_xyz_dt,
                        _,
                        _,
                        _,
                    ) = sp_solver(
                        tx_pos_xyz_dt,
                        rx_pos_xyz_dt,
                        inp.dem,
                        inp.dtu10,
                        inp.landmask_nz,
                    )
                    lon, lat, alt = ecef2lla.transform(*sx_pos_xyz1, radians=False)
                    sx_pos_lla1 = [lat, lon, alt]
                    # <lon,lat,alt> of the specular reflection
                    # algorithm version 1.11
                    surface_type1 = get_surf_type2(
                        sx_pos_lla1, inp.landmask_nz, inp.lcv_mask, inp.water_mask
                    )

                    sx_vel_xyz1 = np.array(sx_pos_xyz_dt) - np.array(sx_pos_xyz1)

                    # save sx values to variables
                    L1.postCal["sp_pos_x"][sec][ngrx_channel] = sx_pos_xyz1[0]
                    L1.postCal["sp_pos_y"][sec][ngrx_channel] = sx_pos_xyz1[1]
                    L1.postCal["sp_pos_z"][sec][ngrx_channel] = sx_pos_xyz1[2]

                    L1.postCal["sp_lat"][sec][ngrx_channel] = sx_pos_lla1[0]
                    L1.postCal["sp_lon"][sec][ngrx_channel] = sx_pos_lla1[1]
                    L1.postCal["sp_alt"][sec][ngrx_channel] = sx_pos_lla1[2]

                    L1.postCal["sp_vel_x"][sec][ngrx_channel] = sx_vel_xyz1[0]
                    L1.postCal["sp_vel_y"][sec][ngrx_channel] = sx_vel_xyz1[1]
                    L1.postCal["sp_vel_z"][sec][ngrx_channel] = sx_vel_xyz1[2]
                    L1.surface_type[sec][ngrx_channel] = surface_type1
                    L1.dist_to_coast_km[sec][ngrx_channel] = dist_to_coast_km1

                    L1.postCal["sp_inc_angle"][sec][ngrx_channel] = inc_angle_deg1
                    L1.postCal["sp_d_snell_angle"][sec][ngrx_channel] = d_snell_deg1

                    # Part 4.2: SP-related variables - 1
                    # this part derives tx/rx gains, ranges and other related variables
                    # derive SP related geo-parameters, including angles in various frames, ranges and antenna gain/GPS EIRP
                    (
                        sx_angle_body1,
                        sx_angle_enu1,
                        sx_angle_ant1,
                        theta_gps1,
                        ranges1,
                        gps_rad1,
                    ) = sp_related(tx1, rx1, sx_pos_xyz1, inp.SV_eirp_LUT)

                    # get values for deriving BRCS and reflectivity
                    # R_tsx1 = ranges1[0]
                    # R_rsx1 = ranges1[1]
                    # gps_eirp_watt1 = gps_rad1[2]

                    # get active antenna gain for LHCP and RHCP channels
                    sx_rx_gain_LHCP1 = get_sx_rx_gain(sx_angle_ant1, inp.LHCP_pattern)
                    sx_rx_gain_RHCP1 = get_sx_rx_gain(sx_angle_ant1, inp.RHCP_pattern)

                    # determine L1a xpol calibration flag - 28 June
                    sx_theta_body1 = sx_angle_body1[0]         # off-boresight angle

                    # antenna x-pol gain ratio
                    copol_ratio1 = sx_rx_gain_LHCP1[0]-sx_rx_gain_LHCP1[1]
                    xpol_ratio1 = sx_rx_gain_RHCP1[1]-sx_rx_gain_RHCP1[0]

                    if sx_theta_body1 <= 60 and copol_ratio1 >= 14:
                        L1a_xpol_calibration_flag_copol1 = 0  # consistent with L1 dictionary
                    else:
                        L1a_xpol_calibration_flag_copol1 = 1

                    if sx_theta_body1 <= 60 and xpol_ratio1 >= 14:
                        L1a_xpol_confidence_flag_xpol1 = 0
                    else:
                        L1a_xpol_confidence_flag_xpol1 = 1

                    # save to variables
                    L1.postCal["sp_theta_body"][sec, ngrx_channel] = sx_angle_body1[0]
                    L1.postCal["sp_az_body"][sec, ngrx_channel] = sx_angle_body1[1]
                    L1.postCal["sp_theta_enu"][sec, ngrx_channel] = sx_angle_enu1[0]
                    L1.postCal["sp_az_enu"][sec, ngrx_channel] = sx_angle_enu1[1]

                    L1.gps_boresight[sec, ngrx_channel] = theta_gps1

                    L1.postCal["tx_to_sp_range"][sec, ngrx_channel] = ranges1[0]
                    L1.postCal["rx_to_sp_range"][sec, ngrx_channel] = ranges1[1]

                    L1.postCal["gps_tx_power_db_w"][sec, ngrx_channel] = gps_rad1[0]
                    L1.postCal["gps_ant_gain_db_i"][sec, ngrx_channel] = gps_rad1[1]
                    L1.postCal["static_gps_eirp"][sec, ngrx_channel] = gps_rad1[2]

                    # copol gain
                    # LHCP channel rx gain
                    L1.sx_rx_gain_copol[sec, ngrx_channel] = sx_rx_gain_LHCP1[0]
                    # RHCP channel rx gain
                    L1.sx_rx_gain_copol[sec, ngrx_channel + L0.J_2] = sx_rx_gain_RHCP1[
                        1
                    ]
                    # xpol gain gain
                    # LHCP channel rx gain
                    L1.sx_rx_gain_xpol[sec, ngrx_channel] = sx_rx_gain_LHCP1[1]
                    # RHCP channel rx gain
                    L1.sx_rx_gain_xpol[sec, ngrx_channel + L0.J_2] = sx_rx_gain_RHCP1[0]

                    # L1a xpol calibration flag - rename 22 July
                    L1.postCal["L1a_xpol_calibration_flag"][sec, ngrx_channel] = L1a_xpol_calibration_flag_copol1
                    L1.postCal["L1a_xpol_calibration_flag"][sec, ngrx_channel + L0.J_2] = L1a_xpol_confidence_flag_xpol1

    # expand to RHCP channels
    L1.expand_sp_arrays(L0.J_2, L0.J)
