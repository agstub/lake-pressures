import numpy as np
from paterson import paterson
from netCDF4 import Dataset
from ll2xy import ll2xy
from xy2ll import xy2ll
from InterpFromGridToMesh import InterpFromGridToMesh
from BamgTriangulate import BamgTriangulate
from InterpFromMeshToMesh2d import InterpFromMeshToMesh2d
from SetIceSheetBC import SetIceSheetBC
import os
ISSM_DIR = os.getenv('ISSM_DIR')


#Name and Coordinate system
md.miscellaneous.name = 'SquareLakes'
md.mesh.epsg = 3031


# THICKNESS
bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineAntarctica-v3.nc')
x_bm = bedmachine['x'][:].data.astype(np.float64)
y_bm = np.flipud(bedmachine['y'][:].data.astype(np.float64))
H_bm = np.flipud(bedmachine['thickness'][:].data.astype(np.float64))
bed_bm = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
surf_bm = np.flipud(bedmachine['surface'][:].data.astype(np.float64))
bedmachine.close()


md.geometry.base = InterpFromGridToMesh(x_bm, y_bm, bed_bm, md.mesh.x, md.mesh.y, 0)
md.geometry.surface = InterpFromGridToMesh(x_bm, y_bm, surf_bm, md.mesh.x, md.mesh.y, 0)
md.geometry.thickness = md.geometry.surface - md.geometry.base

#Set min thickness to 1 meter
pos0 = np.nonzero(md.geometry.thickness <= 0)
md.geometry.thickness[pos0] = 1
md.geometry.surface = md.geometry.thickness + md.geometry.base

# VELOCITY
measures = Dataset('/Users/agstubbl/Desktop/measures/antarctic_ice_vel_phase_map_v01.nc')
vx_m = np.flipud(measures['VX'][:].data)
vy_m = np.flipud(measures['VY'][:].data)
x_m = measures['x'][:].data
y_m = np.flipud(measures['y'][:].data)
measures.close()

vx = InterpFromGridToMesh(x_m, y_m, vx_m, md.mesh.x, md.mesh.y, 0)
vy = InterpFromGridToMesh(x_m, y_m, vy_m, md.mesh.x, md.mesh.y, 0)
speed = np.sqrt(vx**2 + vy**2)

md.initialization.vx = vx
md.initialization.vy = vy
md.initialization.vz = np.zeros((md.mesh.numberofvertices))
md.initialization.vel = speed

md.inversion.vx_obs = vx
md.inversion.vy_obs = vy
md.inversion.vel_obs = speed


# interpolate temperature
dir_str = '/Users/agstubbl/Desktop/merra2/'
directory = os.fsencode(dir_str)
filelist = []    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.nc'): 
        filelist.append(filename)
        # print(os.path.join(directory, filename))
        continue
    else:
        continue

num_months = len(filelist)

ds = Dataset(dir_str+filelist[0])
T_surf = np.zeros(np.shape(ds['TS'][:].data[0]))

for i in range(num_months):
    ds = Dataset(dir_str+filelist[i])
    T_surf += ds['TS'][:].data[0].astype(np.float64)/num_months

lat = ds['lat'][:].data.astype(np.float64)
lon = ds['lon'][:].astype(np.float64)
llo,lla = np.meshgrid(lon,lat)
[xi, yi] = ll2xy(lla, llo, - 1, 0, 71)

x1 = xi.flatten()
y1 = yi.flatten()
T_ = T_surf.flatten()

T_ = T_[x1>0]
y1 = y1[x1>0]
x1 = x1[x1>0]

index = BamgTriangulate(x1, y1)
T_ = InterpFromMeshToMesh2d(index, x1, y1, T_, md.mesh.x, md.mesh.y)

md.initialization.temperature = T_

# initialize temperature
md.initialization.temperature = InterpFromGridToMesh(x_m, y_m, 260+0*vx_m, md.mesh.x, md.mesh.y, 0) 

# impose observed temperature on surface
md.thermal.spctemperature = md.initialization.temperature
md.masstransport.spcthickness = np.nan * np.ones((md.mesh.numberofvertices))

# initialize basal friction
md.friction.coefficient = 30 * np.ones((md.mesh.numberofvertices))
pos = np.nonzero(md.mask.ocean_levelset < 0)
md.friction.coefficient[pos] = 0  #no friction applied on floating ice
md.friction.p = np.ones((md.mesh.numberofelements))
md.friction.q = np.ones((md.mesh.numberofelements))

# initialize ice rheology
md.materials.rheology_n = 3 * np.ones((md.mesh.numberofelements))
md.materials.rheology_B = paterson(md.initialization.temperature)

# set other boundary conditions
md.mask.ice_levelset[np.nonzero(md.mesh.vertexonboundary == 1)] = 0

# initialize pressure
md.initialization.pressure = md.materials.rho_ice * md.constants.g * md.geometry.thickness

# initialize single point constraints
md.stressbalance.referential = np.nan * np.ones((md.mesh.numberofvertices, 6))
md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices))
md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices))
md.stressbalance.spcvz = np.nan * np.ones((md.mesh.numberofvertices))


md = SetIceSheetBC(md)