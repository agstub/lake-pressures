#-------------------------------------------------------------------------------
# This script preprocesses the ICESat-2 data for use in the inversion
#
# OVERVIEW
# The main choices are specifying:
# (I) lake_name (str): name of lake from the inventory:
#
#   Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial lake
#   activity in Antarctica from multi-mission satellite altimetry. Annals of
#   Glaciology, 59(76pt1), 42-55.
#
# (II) L0 (float):  half-length(/half-width) of horizontal domain (a box) surrounding
#                   the subglacial lake that was selected in the first step
#
# *There is also a function below ("localize") that removes the off-lake component
#  of the elevation change at each timestep
#
# DATA REQUIREMENTS: see README
#-------------------------------------------------------------------------------
import sys
sys.path.insert(0, '../source')

from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from load_lakes import gdf


def proc_data(lake_name,paths):

    outline = gdf.loc[gdf['name']==lake_name]
    data_name = 'data_'+lake_name
    if os.path.isdir('../data/'+data_name)==False:
        os.mkdir('../data/'+data_name)
    x0 = float(outline.centroid.x.iloc[0])*1e3
    y0 = float(outline.centroid.y.iloc[0])*1e3

    # STEP (II): Select half-width L0 of box surrounding lake
    L0 = 30*1000
    x_min = x0-L0
    x_max = x0+L0
    y_min = y0-L0
    y_max = y0+L0

    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    # load the ATL15 data
    fn = paths['icesat']
    ds = nc.Dataset(fn)
    dsh = ds['delta_h']

    dh = dsh['delta_h'][:]        # elevation change (m)
    x = dsh['x'][:]               # x coordinate array (m)
    y = dsh['y'][:]               # y coordinate array (m)
    t = dsh['time'][:]            # t coordinate array (d)

    nt = np.size(t)

    ind_x = np.arange(0,np.size(x),1)
    ind_y = np.arange(0,np.size(y),1)

    # extract the data that is inside the bounding box
    x_sub = x[(x>=x_min)&(x<=x_max)]
    y_sub = y[(y>=y_min)&(y<=y_max)]
    inds_x = ind_x[(x>=x_min)&(x<=x_max)]
    inds_y = ind_y[(y>=y_min)&(y<=y_max)]

    nx = np.size(inds_x)
    ny = np.size(inds_y)

    inds_xy = np.ix_(inds_y,inds_x)
    dh_sub = np.zeros((nt,ny,nx))

    # put elevation change maps into 3D array with time being the first index
    for i in range(nt):
        dh0 = dh[i,:,:]
        dh_sub[i,:,:] = dh0[inds_xy]

    #--------------------------PLOTTING-------------------------------

    levels=np.arange(-1,1.1,0.1)*np.max(np.abs(dh_sub))

    # plot png at each time step

    if os.path.isdir('../data/'+data_name+'/data_pngs')==False:
        os.mkdir('../data/'+data_name+'/data_pngs')

    X_sub,Y_sub = np.meshgrid(x_sub,y_sub)


    # PLOT elevation change anomaly
    for i in range(np.size(t)):
        
        plt.close()
        plt.figure(figsize=(6,6))
        plt.title(r'$t=$ '+'{:.2f}'.format(t[i])+' d',fontsize=24)
        p = plt.contourf(X_sub/1e3,Y_sub/1e3,dh_sub[i,:,:],levels=levels,cmap='coolwarm',extend='both')
        outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)
        plt.xlabel(r'$x$ (km)',fontsize=20)
        plt.ylabel(r'$y$ (km)',fontsize=20)
        cbar = plt.colorbar(p)
        cbar.set_label(r'$dh$ (m)',fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig('../data/'+data_name+'/data_pngs/dh_'+str(i))
        plt.close()
    print('saved some images of elevation change maps in the data directory!')

    ##------------------------------------------------------------------------------
    ## INTERPOLATE DATA

    def interp_tyx(f,t,y,x):
        Nx_f = 101            # fine Nx
        Ny_f = 101            # fine Ny
        Nt_f = 100            # fine Nt

        t0_f = np.linspace(t.min(),t.max(),num=Nt_f)  # fine time array
        x0_f = np.linspace(x.min(),x.max(),num=Nx_f)  # fine x coordinate array
        y0_f = np.linspace(y.min(),y.max(),num=Ny_f)  # fine y coordinate array
        t_f,y_f,x_f = np.meshgrid(t0_f,y0_f,x0_f,indexing='ij')

        points = (t_f,y_f,x_f)

        f_fine = griddata((t.ravel(),y.ravel(),x.ravel()),f.ravel(),points)

        return f_fine,t0_f,y0_f,x0_f



    t_g,y_g,x_g = np.meshgrid(t,y_sub,x_sub,indexing='ij')

    dh_f,t_f,y_f,x_f = interp_tyx(dh_sub,t_g,y_g,x_g)

    t,y,x = np.meshgrid(t_f,y_f,x_f,indexing='ij')

    def localize(f):
        f_far = np.copy(f)
        f_far[np.sqrt((x-x.mean())**2+(y-y.mean())**2)<0.8*np.sqrt((x-x.mean())**2+(y-y.mean())**2).max()] = 0
        F = (f_far != 0).sum(axis=(1,2))+1e-10
        f_far = f_far.sum(axis=(1,2))/F
        f_loc = f- np.multiply.outer(f_far,np.ones(np.shape(f[0,:,:])))
        return f_loc

    dh_loc = localize(dh_f)

    off_lake = dh_f-dh_loc
    off_lake = off_lake[:,0,0]


    # ----------------------------- SAVE DATA --------------------------------------
    np.save('../data/'+data_name+'/h_obs.npy',dh_loc)          # elevation anomaly: (m)
    np.save('../data/'+data_name+'/off_lake.npy',off_lake)     # off-lake timeseries: (m)
    np.save('../data/'+data_name+'/t.npy',(t_f-t_f[0])/365.0)  # time: (yr)
    np.save('../data/'+data_name+'/x_d.npy',x_f/1e3)           # x coord. (km)
    np.save('../data/'+data_name+'/y_d.npy',y_f/1e3)           # y coord. (km)




