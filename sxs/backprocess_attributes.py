import netCDF4 as nc

from pathlib import Path


L1_path = Path("/home/ubuntu/data/L1s")
L1_files = [filepath for filepath in L1_path.glob("*.nc")]

for L1_file in L1_files:
    print(L1_file)
    py_L1 = nc.Dataset(L1_file, "r+", clobber=True)

    # TODO
    #  1) remove global "standard_name" attribute
    #  2) add standard names for lat/lon fields

    if py_L1.standard_name:
        del py_L1.standard_name

    for var in py_L1.variables:
        if var in ["ac_lon", "sp_lon"]:
            py_L1[var].setncattr("standard_name", "longitude")
        if var in ["ac_lat", "sp_lat"]:
            py_L1[var].setncattr("standard_name", "latitude")
    py_L1.close()
