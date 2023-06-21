import pyproj


# ecef2lla Matlab function
# define projections and transform
# Pyproj 1 version
# ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
# lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
# lon, lat, alt = pyproj.transform(ecef, lla, *rx_pos_xyz, radians=False)

ecef = pyproj.CRS(proj="geocent", ellps="WGS84", datum="WGS84")
lla = pyproj.CRS(proj="latlon", ellps="WGS84", datum="WGS84")

ecef2lla = pyproj.Transformer.from_crs(ecef, lla)
lla2ecef = pyproj.Transformer.from_crs(lla, ecef)
