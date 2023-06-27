# mike.laverick@auckland.ac.nz
# Specular point related functions
import math
import cmath
import numpy as np
import pyproj
from scipy import constants
from scipy.interpolate import interp2d
from scipy.signal import convolve2d
import geopy.distance as geo_dist
import pymap3d as pm
from numba import njit
from numba.typed import List as numba_list

from load_files import get_local_dem
from cal_functions import db2power
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

    # check if LOS exists
    # LOS_flag = los_status(tx_pos_xyz, rx_pos_xyz)

    # if not LOS_flag:
    #    # no sx if no LOS between rx and tx
    #    return [np.nan, np.nan, np.nan], np.nan, np.nan, np.nan, LOS_flag

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
    sp_az_ant = sp_az_body + 180

    if sp_az_ant > 360:
        sp_az_ant = sp_az_ant - 360

    sp_angle_ant = [sp_theta_ant, sp_az_ant]

    return sp_angle_body, sp_angle_enu, sp_angle_ant, theta_gps, range, gps_rad


def get_amb_fun(dtau_s, dfreq_Hz, tau_c, Ti):
    """
    this function computes the ambiguity function
    inputs
    1) tau_s: delay in seconds
    2) freq_Hz: Doppler in Hz
    3) tau_c: chipping period in second, 1/chip_rate
    4) Ti: coherent integration time in seconds
    output
    1) chi: ambiguity function, product of Lambda and S
    """
    det = tau_c * (1 + tau_c / Ti)  # discriminant for computing Lambda

    Lambda = np.full_like(dtau_s, np.nan)

    # compute Lambda - delay
    Lambda[np.abs(dtau_s) <= det] = (1 - np.abs(dtau_s) / tau_c)[np.abs(dtau_s) <= det]
    Lambda[np.abs(dtau_s) > det] = -tau_c / Ti

    # compute S - Doppler
    S1 = math.pi * dfreq_Hz * Ti

    S = np.full_like(S1, np.nan, dtype=complex)

    S[S1 == 0] = 1

    term1 = np.sin(S1[S1 != 0]) / S1[S1 != 0]
    term2 = np.exp(-1j * S1[S1 != 0])
    S[S1 != 0] = term1 * term2

    # compute complex chi
    chi = Lambda * S
    return chi


def get_chi2(
    num_delay_bins,
    num_doppler_bins,
    delay_center_bin,
    doppler_center_bin,
    delay_res,
    doppler_res,
):
    # this function gets 2D AF

    chip_rate = 1.023e6
    tau_c = 1 / chip_rate
    T_coh = 1 / 1000

    # delay_res = 0.25
    # doppler_res = 500
    # delay_center_bin = 20  # 0-based index
    # doppler_center_bin = 2  # 0-based index
    # chi = np.zeros([num_delay_bins, num_doppler_bins])

    def ix_func(i, j):
        dtau = (i - delay_center_bin) * delay_res * tau_c
        dfreq = (j - doppler_center_bin) * doppler_res
        # compute complex AF value at each delay-doppler bin
        return get_amb_fun(dtau, dfreq, tau_c, T_coh)

    chi = np.fromfunction(
        ix_func, (num_delay_bins, num_doppler_bins)
    )  # 10 times faster than for loop

    chi_mag = np.abs(chi)  # magnitude
    chi2 = np.square(chi_mag)  # chi_square

    return chi2


def meter2chips(x):
    """
    this function converts from meters to chips
    input: x - distance in meters
    output: y - distance in chips
    """
    # define constants
    c = 299792458  # light speed metre per second
    chip_rate = 1.023e6  # L1 GPS chip-per-second, code modulation frequency
    tau_c = 1 / chip_rate  # C/A code chiping period
    l_chip = c * tau_c  # chip length
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
    c = 299792458  # light speed metre per second
    fc = 1575.42e6  # L1 carrier frequency in Hz
    _lambda = c / fc  # wavelength

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

    doppler_hz = -1 * (term1 + term2) / _lambda  # Doppler in Hz

    return delay_chips, doppler_hz, add_delay_chips


def get_ddm_Aeff4(
    rx_alt,
    inc_angle,
    az_angle,
    sp_delay_bin,
    sp_doppler_bin,
    chi2,
    A_phy_LUT_interp,
):
    # this is to construct effective scattering area from LUTs

    # center delay and doppler bin for full DDM
    # not to be used elsewhere
    center_delay_bin = 40
    center_doppler_bin = 5

    # derive full scattering area from LUT
    A_phy_full_1 = A_phy_LUT_interp((rx_alt, inc_angle, az_angle))
    A_phy_full_2 = np.vstack((np.zeros((1, 41)), A_phy_full_1, np.zeros((1, 41))))
    A_phy_full = np.hstack((A_phy_full_2, np.zeros((9, 39))))

    # shift A_phy_full according to the floating SP bin

    # integer and fractional part of the SP bin
    sp_delay_intg = np.floor(sp_delay_bin)
    sp_delay_frac = sp_delay_bin - sp_delay_intg

    sp_doppler_intg = np.floor(sp_doppler_bin)
    sp_doppler_frac = sp_doppler_bin - sp_doppler_intg

    # shift 1: shift along delay direction
    A_phy_shift = np.roll(A_phy_full, 1)
    A_phy_shift = (1 - sp_delay_frac) * A_phy_full + (sp_delay_frac * A_phy_shift)

    # shift 2: shift along doppler direction
    A_phy_shift2 = np.roll(A_phy_shift, 1, axis=0)
    A_phy_shift = (1 - sp_doppler_frac) * A_phy_shift + (sp_doppler_frac * A_phy_shift2)

    # crop the A_phy_full to Rongowai 5*40 DDM size
    delay_shift_bin = center_delay_bin - sp_delay_intg
    doppler_shift_bin = center_doppler_bin - sp_doppler_intg

    # Change from Matlab offsets to account for Matlab/Python indexing differences
    A_phy = A_phy_shift[
        int(doppler_shift_bin - 1) : int(doppler_shift_bin + 4),
        int(delay_shift_bin - 1) : int(delay_shift_bin + 39),
    ]

    # convolution to A_eff
    A_eff1 = convolve2d(A_phy, chi2.T)
    A_eff = A_eff1[2:7, 20:60]  # cut suitable size for A_eff, 0-based index

    return A_eff


def ddm_brcs2(power_analog_LHCP, power_analog_RHCP, eirp_watt, rx_gain_db_i, TSx, RSx):
    """
    This version has copol xpol antenna gain implemented
    This function computes bistatic radar cross section (BRCS) according to
    Bistatic radar equation based on the inputs as below
    inputs:
    1) power_analog: L1a product in watts
    2) eirp_watt, rx_gain_db_i: gps eirp in watts and rx antenna gain in dBi
    3) TSx, RSx: Tx to Sx and Rx to Sx ranges
    outputs:
    1) brcs: bistatic RCS
    """
    # define constants
    f = 1575.42e6  # GPS L1 band, Hz
    _lambda = constants.c / f  # wavelength, m
    _lambda2 = _lambda * _lambda

    # derive BRCS
    rx_gain = db2power(rx_gain_db_i)  # linear rx gain
    rx_gain2 = rx_gain.reshape(2, 2)
    term1 = 4 * math.pi * np.power(4 * math.pi * TSx * RSx, 2)
    term2 = eirp_watt * _lambda2
    term3 = term1 / term2
    # term4 = term3 * np.power(rx_gain2, -1)
    term4 = term3 * np.linalg.matrix_power(rx_gain2, -1)

    brcs_copol = (term4[0, 0] * power_analog_LHCP) + (term4[0, 1] * power_analog_RHCP)
    brcs_xpol = (term4[1, 0] * power_analog_LHCP) + (term4[1, 1] * power_analog_RHCP)

    return brcs_copol, brcs_xpol


def ddm_refl2(
    power_analog_LHCP, power_analog_RHCP, eirp_watt, rx_gain_db_i, R_tsx, R_rsx
):
    """
    This function computes the land reflectivity by implementing the xpol
    antenna gain
    1)power_analog: L1a product, DDM power in watt
    2)eirp_watt: transmitter eirp in watt
    3)rx_gain_db_i: receiver antenna gain in the direction of SP, in dBi
    4)R_tsx, R_rsx: tx to sp range and rx to sp range, in meters
    outputs
    1) copol and xpol reflectivity
    """
    # define constants
    freq = 1575.42e6  # GPS L1 operating frequency, Hz
    _lambda = constants.c / freq  # wavelength, meter
    _lambda2 = _lambda * _lambda

    rx_gain = db2power(rx_gain_db_i)  # convert antenna gain to linear form
    rx_gain2 = rx_gain.reshape(2, 2)

    term1 = np.power(4 * math.pi * (R_tsx + R_rsx), 2)
    term2 = eirp_watt * _lambda2
    term3 = term1 / term2

    # term4 = term3 * np.power(rx_gain, -1)
    term4 = term3 * np.linalg.matrix_power(rx_gain2, -1)

    refl_copol = term4[0, 0] * power_analog_LHCP + term4[0, 1] * power_analog_RHCP
    refl_xpol = term4[1, 0] * power_analog_LHCP + term4[1, 1] * power_analog_RHCP
    return refl_copol, refl_xpol


def get_fresnel(tx_pos_xyz, rx_pos_xyz, sx_pos_xyz, dist_to_coast, inc_angle, ddm_ant):
    """
    this function derives Fresnel dimensions based on the Tx, Rx and Sx positions.
    Fresnel dimension is computed only the DDM is classified as coherent reflection.
    """
    # define constants
    eps_ocean = 74.62 + 51.92j  # complex permittivity of ocean
    fc = 1575.42e6  # operating frequency
    c = 299792458  # speed of light
    _lambda = c / fc  # wavelength

    # compute dimensions
    R_tsp = np.linalg.norm(np.array(tx_pos_xyz) - np.array(sx_pos_xyz), 2)
    R_rsp = np.linalg.norm(np.array(rx_pos_xyz) - np.array(sx_pos_xyz), 2)

    term1 = R_tsp * R_rsp
    term2 = R_tsp + R_rsp

    # semi axis
    a = math.sqrt(_lambda * term1 / term2)  # major semi
    b = a / math.cos(math.radians(inc_angle))  # minor semi

    # compute orientation relative to North
    lon, lat, alt = ecef2lla.transform(*sx_pos_xyz, radians=False)
    sx_lla = [lat, lon, alt]

    tx_e, tx_n, _ = pm.ecef2enu(*tx_pos_xyz, *sx_lla, deg=True)
    rx_e, rx_n, _ = pm.ecef2enu(*rx_pos_xyz, *sx_lla, deg=True)

    tx_en = np.array([tx_e, tx_n])
    rx_en = np.array([rx_e, rx_n])

    vector_tr = rx_en - tx_en
    unit_north = [0, 1]

    term3 = np.dot(vector_tr, unit_north)
    term4 = np.linalg.norm(vector_tr, 2) * np.linalg.norm(unit_north, 2)

    theta = math.degrees(math.acos(term3 / term4))

    fresnel_axis = [2 * a, 2 * b]
    fresnel_orientation = theta

    # fresenel coefficient only compute for ocean SPs
    fresnel_coeff = np.nan

    if dist_to_coast <= 0:
        sint = math.sin(math.radians(inc_angle))
        cost = math.cos(math.radians(inc_angle))

        temp1 = cmath.sqrt(eps_ocean - sint * sint)

        R_vv = (eps_ocean * cost - temp1) / (eps_ocean * cost + temp1)
        R_hh = (cost - temp1) / (cost + temp1)

        R_rl = (R_vv - R_hh) / 2
        R_rr = (R_vv + R_hh) / 2

        if ddm_ant == 1:
            fresnel_coeff = abs(R_rl) * abs(R_rl)

        elif ddm_ant == 2:
            fresnel_coeff = abs(R_rr) * abs(R_rr)

    return fresnel_coeff, fresnel_axis, fresnel_orientation


def coh_det(raw_counts, snr_db):
    """
    this function computes the coherency of an input raw-count ddm
    Inputs
    1)raw ddm measured in counts
    2)SNR measured in decibels
    Outputs
    1)coherency ratio (CR)
    2)coherency state (CS)
    """
    peak_counts = np.amax(raw_counts)
    delay_peak, dopp_peak = np.unravel_index(raw_counts.argmax(), raw_counts.shape)

    # thermal noise exclusion
    # TODO: the threshold may need to be redefined
    if not np.isnan(snr_db):
        thre_coeff = 1.055 * math.exp(-0.193 * snr_db)
        thre = thre_coeff * peak_counts  # noise exclusion threshold

        raw_counts[raw_counts < thre] = 0

    # deterimine DDMA range
    delay_range = list(range(delay_peak - 1, delay_peak + 2))
    delay_min = min(delay_range)
    delay_max = max(delay_range)
    dopp_range = list(range(dopp_peak - 1, dopp_peak + 2))
    dopp_min = min(dopp_range)
    dopp_max = max(dopp_range)

    # determine if DDMA is within DDM, refine if needed
    if delay_min < 1:
        delay_range = [0, 1, 2]
    elif delay_max > 38:
        delay_range = [37, 38, 39]

    if dopp_min < 1:
        dopp_range = [0, 1, 2]
    elif dopp_max > 3:
        dopp_range = [2, 3, 4]

    C_in = np.sum(raw_counts[delay_range, :][:, dopp_range])  # summation of DDMA
    C_out = np.sum(raw_counts) - C_in  # summation of DDM excluding DDMA

    CR = C_in / C_out  # coherency ratio

    if CR >= 2:
        CS = 1
    else:  # CR < 2
        CS = 0

    return CR, CS
