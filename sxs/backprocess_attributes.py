import netCDF4 as nc

from pathlib import Path


L1_path = Path("/srv/data/L1/")
L1_files = [filepath for filepath in L1_path.glob("*.nc")]

for L1_file in L1_files:
    py_L1 = nc.Dataset(L1_file, "r+", clobber=True)

    # TODO 1) update Compliance line (so first add it, then try to update it)
    #      2) add standard names for lat/lon fields
    #      3) unsigned values solved by point 1)

    py_L1.setncattr("Conventions", "CF-1.9, ACDD-1.3, ISO-8601")

    for var in py_L1.variables:
        if var in ["ac_lon", "sp_lon"]:
            py_L1[var].setncattr("standard_name", "longitude")
        if var in ["ac_lat", "sp_lat"]:
            py_L1[var].setncattr("standard_name", "latitude")
    py_L1.close()
    with open("/home/ubuntu/backlog-done.txt", "a") as f:
        f.write(str(L1_file) + " \n")
