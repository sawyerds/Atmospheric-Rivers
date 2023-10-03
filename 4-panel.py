#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Notebook to compare basic meteorological parameters between WRF simulations and ERA5 data
# Some of the following imports are not used right now, but will retain for future flexibility
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.cbook as cbook
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import pandas as pd
import xarray as xr
import numpy as np
import math
from numpy import *
from pylab import *
import pygrib
import pyproj

from siphon.catalog import TDSCatalog
from siphon.http_util import session_manager
from datetime import datetime, timedelta
from xarray.backends import NetCDF4DataStore
from netCDF4 import Dataset
import metpy as metpy
import metpy.calc as mpcalc
from metpy.plots import ctables
from metpy.units import units
from metpy.plots import add_metpy_logo, add_timestamp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter

import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from cartopy import config
import wrf
from wrf import (to_np, interplevel, geo_bounds, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)
# Download and add the states and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m",
                          facecolor="none", name="admin_1_states_provinces_shp")
import glob


# In[2]:


"""
Data user settings, constants, and level ranges
"""
################## wrfout files path ###################################################

Thompson = "/scratch/sawyer/wwrf/2017-01-09/ctrl/wrfout_d01*"
WSM6 = "/scratch/sawyer/wwrf/2017-01-09/wsm6/wrfout_d01*"
P3 = "/scratch/sawyer/wwrf/2017-01-09/p3_1-cat/wrfout_d01*"
P3_3mom = "/scratch/sawyer/wwrf/2017-01-09/p3_3mom/wrfout_d01*"

################### Saving plots option ##################

save = True

########### If you want to save plots, specify the subdirectory you want them in ###########

plotsdir = "/home/jupyter-sdsmit12@ncsu.edu/WRF_project_1/wwrf/2017-01-09/plots/"
print(plotsdir)
############ Gravity ###########
g0 = 9.80665

############### Select a pressure level #################
p_level = 850

################ Set bounds for plotting #################

wlon = -165
elon = -115
slat = 20
nlat = 50

############################# Do we only want to plot IVT? ###############################

compute_ivt = True

############################# Set ranges and intervals for plotting ###############################

if p_level == 1000:
    levels = np.arange(-180,180,30)
    q_levels = np.arange(0, 22, 1)
    wnd_levels = np.arange(0, 30, 1)
elif p_level == 850:
    levels = np.arange(1000, 2000, 50)
    q_levels = np.arange(0, 16, .5)
    wnd_levels = np.arange(20, 70, 5)    
elif p_level == 700:
    levels = np.arange(2350, 3150, 30)
    q_levels = np.arange(0, 12, .5)
    wnd_levels = np.arange(20, 65, 1) 
elif p_level == 500:
    levels = np.arange(4000, 6000, 60)
    q_levels = np.arange(0, 16, .5)
    wnd_levels = np.arange(20, 120, 1) 
elif p_level == 300:
    levels = np.arange(8200, 9600, 120)
    wnd_levels = np.arange(20, 180, 1) 
elif p_level == 200:
    levels = np.arange(10800,12300, 120)
    wnd_levels = np.arange(20, 160, 10) 

ivt_levels = np.arange(250,1150,1)
pv_levels = np.arange(0,1,.1)
pmsl_levels = np.arange(960, 1060, 4)
########### Define the colors in the desired sequence for IVT: ###########

colors = ['#FFFF00', '#FFEE00','#FFDC00', '#FFB700',
          '#FFA300', '#FF9000', '#FF7D00', '#FF6800',
          '#FF5200', '#C70039','#900C3F', (.88,.24,.69)]

############## Create a colormap using ListedColormap #################

cmap = mcolors.ListedColormap(colors)

########## Create a list of Forecast Hours for plotting ##############

fcst = list(arange(0, 171,3))


# In[3]:


"""
This function computes Integrated Vapor Transport (IVT) from wrfout files
"""
def calculate_IVT(ua,va,p,mr):
   
    
    uflux_l = []
    vflux_l = []
    for m in range(0,len(mr)-2):
        layer_diff = p[m,:,:]-p[m+1,:,:]
        ql = (mr[m+1,:,:]+mr[m,:,:])/2
        ul = (ua[m+1,:,:]+ua[m,:,:])/2
        vl = (va[m+1,:,:]+va[m,:,:])/2
        qfl = (ql/9.80665)*layer_diff
        uflux= ul * qfl
        vflux = vl * qfl
        uflux_l.append(uflux)
        vflux_l.append(vflux)

    uflux_l=np.asarray(uflux_l)
    vflux_l=np.asarray(vflux_l)
    uIVT = np.sum(uflux_l, axis = 0)
    vIVT = np.sum(vflux_l, axis = 0)
    IVT_tot=np.sqrt(uIVT**2+vIVT**2)
    #IVT_tot=xr.DataArray(IVT_tot,dims=['lat','lon'])
    
    return IVT_tot


# In[4]:


# Prepare files for processing
thompson_datafiles = (glob.glob(Thompson))
thompson_datafiles.sort()

wsm6_datafiles = (glob.glob(WSM6))
wsm6_datafiles.sort()

p3_datafiles = (glob.glob(P3))
p3_datafiles.sort()

p3mom_datafiles = (glob.glob(P3_3mom))
p3mom_datafiles.sort()

numfiles=len(p3_datafiles)
print(numfiles)


# In[ ]:


for i in range(0,numfiles):

    ncfile = Dataset(thompson_datafiles[i])
    wsm6_file = Dataset(wsm6_datafiles[i])
    p3_file = Dataset(p3_datafiles[i])
    p3mom_file = Dataset(p3mom_datafiles[i])
    Time=wrf.extract_times(ncfile, timeidx=0, method='cat', squeeze=True, cache=None, meta=False, do_xtime=False)
    timestr=(str(Time))
    
    # Set up one time string for plot titles, another for file names
    titletime=(timestr[0:10]+' '+timestr[11:16])
    filetime=(timestr[0:10]+'_'+timestr[11:13])
    print('WRF valid time: ',filetime)
    plot_filetime = (timestr[0:10]+' '+timestr[11:13])
    dates = int(timestr[0:10].replace('-', ''))
    
    # Get all the variables we need from wrf
    
    thom_p = getvar(ncfile, "pressure")
    wsm6_p = getvar(wsm6_file, "pressure")
    p3_p = getvar(p3_file, "pressure")
    p3mom_p = getvar(p3mom_file, "pressure")
    

    
    # Get the lat/lon coordinates 
    lats, lons = latlon_coords(thom_p)

    # fix lons around dateline from wrf output
    new_lons = np.where(lons > 0, lons - 360, lons)

    # Set up the plot
    cart_proj = get_cartopy(thom_p)

    if compute_ivt == True:
        
        thom_slp = getvar(ncfile, "slp")
        wsm6_slp = getvar(wsm6_file, "slp")
        p3_slp = getvar(p3_file, "slp")
        p3mom_slp = getvar(p3mom_file, "slp")
        # Smooth the sea level pressure since it tends to be noisy near complex terrain
        smooth_slp_thom = smooth2d(thom_slp, 3, cenweight=4)
        smooth_slp_wsm6 = smooth2d(wsm6_slp, 3, cenweight=4)
        smooth_slp_p3 = smooth2d(p3_slp, 3, cenweight=4)
        smooth_slp_p3mom = smooth2d(p3mom_slp, 3, cenweight=4)

        thom_ua = getvar(ncfile, "ua")
        wsm6_ua = getvar(wsm6_file, "ua")
        p3_ua = getvar(p3_file, "ua")
        p3mom_ua = getvar(p3mom_file, "ua")

        thom_va = getvar(ncfile, "va")
        wsm6_va = getvar(wsm6_file, "va")
        p3_va = getvar(p3_file, "va")
        p3mom_va = getvar(p3mom_file, "va")

        thom_p = thom_p * 100
        wsm6_p = wsm6_p * 100
        p3_p = p3_p * 100
        p3mom_p = p3mom_p * 100

        thom_mr = ncfile['QVAPOR'][0,:,:,:]
        wsm6_mr = wsm6_file['QVAPOR'][0,:,:,:]
        p3_mr = p3_file['QVAPOR'][0,:,:,:]
        p3mom_mr = p3mom_file['QVAPOR'][0,:,:,:]

        thom_ivt = calculate_IVT(thom_ua,thom_va,thom_p,thom_mr)
        wsm6_ivt = calculate_IVT(wsm6_ua,wsm6_va,wsm6_p,wsm6_mr)
        p3_ivt = calculate_IVT(p3_ua,p3_va,p3_p,p3_mr)
        p3mom_ivt = calculate_IVT(p3mom_ua,p3mom_va,p3mom_p,p3mom_mr)

        fig, axes = plt.subplots(2,2,subplot_kw={'projection': cart_proj},figsize=(24.5, 17.))

        # dark brown for state/coastlines
        dark_brown = (0.4, 0.2, 0)

        ################################################################################
        # Upper-Left Panel

        axes[0,0].set_extent([wlon,elon,slat,nlat])
        axes[0,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[0,0].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[0,0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        cs = axes[0,0].contour(to_np(new_lons), to_np(lats), to_np(smooth_slp_thom), levels=pmsl_levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[0,0].clabel(cs, fmt='%1.0f', inline=True,levels=pmsl_levels)

        cs1 = axes[0,0].contourf(to_np(new_lons), to_np(lats), to_np(thom_ivt), cmap = cmap, levels = ivt_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        ivt_cbar = plt.colorbar(cs1, orientation = 'horizontal', shrink = 0.8)
        ivt_cbar.set_label("IVT (kg $m^{-1}$$s^{-1}$)", fontsize = 12)

        # Set the plot title
        axes[0,0].set_title(f'Thompson MP Scheme \nSea Level Pressure (hPa) \nIVT (kg $m^{-1}$$s^{-1}$)', fontsize=12,loc='left')

        axes[0,0].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Upper-Right Panel
        axes[0,1].set_extent([wlon,elon,slat,nlat])
        axes[0,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[0,1].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[0,1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        wsm6_cs = axes[0,1].contour(to_np(new_lons), to_np(lats), to_np(smooth_slp_wsm6), levels=pmsl_levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[0,1].clabel(wsm6_cs, fmt='%1.0f', inline=True,levels=pmsl_levels)

        wsm6_cs2 = axes[0,1].contourf(to_np(new_lons), to_np(lats), to_np(wsm6_ivt), cmap = cmap, levels = ivt_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        ivt_cbar = plt.colorbar(wsm6_cs2, orientation = 'horizontal', shrink = 0.8)
        ivt_cbar.set_label("IVT (kg $m^{-1}$$s^{-1}$)", fontsize = 12)

        # Set the plot title
        axes[0,1].set_title(f'WSM6 MP Scheme \nSea Level Pressure (hPa) \nIVT (kg $m^{-1}$$s^{-1}$) ',fontsize=12,loc='left')

        axes[0,1].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Bottom Left Panel
        axes[1,0].set_extent([wlon,elon,slat,nlat])
        axes[1,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[1,0].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[1,0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        p3_cs = axes[1,0].contour(to_np(new_lons), to_np(lats), to_np(smooth_slp_p3), levels=pmsl_levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[1,0].clabel(p3_cs, fmt='%1.0f', inline=True,levels=pmsl_levels)

        p3_cs2 = axes[1,0].contourf(to_np(new_lons), to_np(lats), to_np(p3_ivt), cmap = cmap, levels = ivt_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        ivt_cbar = plt.colorbar(p3_cs2, orientation = 'horizontal', shrink = 0.8)
        ivt_cbar.set_label("IVT (kg $m^{-1}$$s^{-1}$)", fontsize = 12)

        # Set the plot title
        axes[1,0].set_title(f'P3 MP Scheme \nSea Level Pressure (hPa) \nIVT (kg $m^{-1}$$s^{-1}$)',fontsize=12,loc='left')

        axes[1,0].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Bottom-Right Panel
        axes[1,1].set_extent([wlon,elon,slat,nlat])
        axes[1,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[1,1].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[1,1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        p3mom_cs = axes[1,1].contour(to_np(new_lons), to_np(lats), to_np(smooth_slp_p3mom), levels=pmsl_levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[1,1].clabel(p3mom_cs, fmt='%1.0f', inline=True,levels=pmsl_levels)

        p3mom_cs2 = axes[1,1].contourf(to_np(new_lons), to_np(lats), to_np(p3mom_ivt), cmap = cmap, levels = ivt_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        ivt_cbar = plt.colorbar(p3mom_cs2, orientation = 'horizontal', shrink = 0.8)
        ivt_cbar.set_label("IVT (kg $m^{-1}$$s^{-1}$)", fontsize = 12)

        # Set the plot title
        axes[1,1].set_title(f'P3 3-moment MP Scheme \nSea Level Pressure (hPa) \nIVT (kg $m^{-1}$$s^{-1}$)',fontsize=12,loc='left')

        axes[1,1].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')
        #plt.show()
        if save == True:
            # Create separate plot file and save as .png, then show and close
            outTPlotName= str(plot_filetime)+'.png'
            fig.savefig(plotsdir+'/'+outTPlotName)
        plt.close(fig)
    else:

        thom_wspd = getvar(ncfile, "wspd_wdir",units='kts')[0,:]
        thom_windsp = interplevel(thom_wspd, thom_p, p_level)

        wsm6_wspd = getvar(wsm6_file, "wspd_wdir",units='kts')[0,:]
        wsm6_windsp = interplevel(wsm6_wspd, wsm6_p, p_level)

        p3_wspd = getvar(p3_file, "wspd_wdir",units='kts')[0,:]
        p3_windsp = interplevel(p3_wspd, p3_p, p_level)

        p3mom_wspd = getvar(p3mom_file, "wspd_wdir",units='kts')[0,:]
        p3mom_windsp = interplevel(p3mom_wspd, p3mom_p, p_level)
        
        thom_height = getvar(ncfile, "z")
        thom_heights = interplevel(thom_height, thom_p, p_level)    

        wsm6_height = getvar(wsm6_file, "z")
        wsm6_heights = interplevel(wsm6_height, wsm6_p, p_level)   

        p3_height = getvar(p3_file, "z")
        p3_heights = interplevel(p3_height, p3_p, p_level)

        p3mom_height = getvar(p3mom_file, "z")
        p3mom_heights = interplevel(p3mom_height, p3mom_p, p_level)


        fig, axes = plt.subplots(2,2,subplot_kw={'projection': cart_proj},figsize=(24.5, 17.))

        # dark brown for state/coastlines
        dark_brown = (0.4, 0.2, 0)

        ################################################################################
        # Upper-Left Panel

        axes[0,0].set_extent([wlon,elon,slat,nlat])
        axes[0,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[0,0].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[0,0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        cs = axes[0,0].contour(to_np(new_lons), to_np(lats), to_np(thom_heights), levels=levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[0,0].clabel(cs, fmt='%1.0f', inline=True,levels=levels)

        cs1 = axes[0,0].contourf(to_np(new_lons), to_np(lats), to_np(thom_windsp), cmap = 'jet', levels = wnd_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        wind_cbar = plt.colorbar(cs1, orientation = 'horizontal', shrink = 0.8)
        wind_cbar.set_label("wind speed (kt)", fontsize = 12)

        # Set the plot title
        axes[0,0].set_title(f'Thompson MP Scheme \n{p_level} hPa Geopotential Heights (m) \nWinds (kt), \n ',fontsize=12,loc='left')

        axes[0,0].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Upper-Right Panel
        axes[0,1].set_extent([wlon,elon,slat,nlat])
        axes[0,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[0,1].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[0,1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        wsm6_cs = axes[0,1].contour(to_np(new_lons), to_np(lats), to_np(wsm6_heights), levels=levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[0,1].clabel(wsm6_cs, fmt='%1.0f', inline=True,levels=levels)

        wsm6_cs2 = axes[0,1].contourf(to_np(new_lons), to_np(lats), to_np(wsm6_windsp), cmap = 'jet', levels = wnd_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        wind_cbar = plt.colorbar(wsm6_cs2, orientation = 'horizontal', shrink = 0.8)
        wind_cbar.set_label("wind speed (kt)", fontsize = 12)

        # Set the plot title
        axes[0,1].set_title(f'WSM6 MP Scheme \n{p_level} hPa Geopotential Heights (m) \nWinds (kt), \n ',fontsize=12,loc='left')

        axes[0,1].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Bottom Left Panel
        axes[1,0].set_extent([wlon,elon,slat,nlat])
        axes[1,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[1,0].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[1,0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        p3_cs = axes[1,0].contour(to_np(new_lons), to_np(lats), to_np(p3_heights), levels=levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[1,0].clabel(p3_cs, fmt='%1.0f', inline=True,levels=levels)

        p3_cs2 = axes[1,0].contourf(to_np(new_lons), to_np(lats), to_np(p3_windsp), cmap = 'jet', levels = wnd_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        wind_cbar = plt.colorbar(p3_cs2, orientation = 'horizontal', shrink = 0.8)
        wind_cbar.set_label("wind speed (kt)", fontsize = 12)

        # Set the plot title
        axes[1,0].set_title(f'P3 MP Scheme \n{p_level} hPa Geopotential Heights (m) \nWinds (kt), \n ',fontsize=12,loc='left')

        axes[1,0].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')

        ################################################################################
        # Bottom-Right Panel
        axes[1,1].set_extent([wlon,elon,slat,nlat])
        axes[1,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=dark_brown)#Add coastlines)
        axes[1,1].add_feature(cfeature.STATES, linewidth=0.5,edgecolor=dark_brown) #Add US states
        gl = axes[1,1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels=False   # suppress top labels
        gl.right_labels=False 

        # Plot the contours
        p3mom_cs = axes[1,1].contour(to_np(new_lons), to_np(lats), to_np(p3mom_heights), levels=levels, colors='black',linewidths=0.9,transform=ccrs.PlateCarree())
        axes[1,1].clabel(p3mom_cs, fmt='%1.0f', inline=True,levels=levels)

        p3mom_cs2 = axes[1,1].contourf(to_np(new_lons), to_np(lats), to_np(p3mom_windsp), cmap = 'jet', levels = wnd_levels, transform=ccrs.PlateCarree())

        #Add a colorbar
        wind_cbar = plt.colorbar(p3mom_cs2, orientation = 'horizontal', shrink = 0.8)
        wind_cbar.set_label("wind speed (kt)", fontsize = 12)

        # Set the plot title
        axes[1,1].set_title(f'P3 3-moment MP Scheme \n{p_level} hPa Geopotential Heights (m) \nWinds (kt), \n ',fontsize=12,loc='left')

        axes[1,1].set_title(f'VALID: {plot_filetime} UTC \n Forecast Hour: {fcst[i]}',fontsize=12,loc='right')
        #plt.show()
        if save == True:
            # Create separate plot file and save as .png, then show and close
            outTPlotName= str(plot_filetime)+'.png'
            fig.savefig(plotsdir+'/'+outTPlotName)
        plt.close(fig)


# In[ ]:




