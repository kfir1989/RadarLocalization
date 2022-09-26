import matplotlib.pyplot as plt
import numpy as np
from tools import *
import cv2
import re
from map_utils import getRoadBorders, getCombinedMap, getLayer
import nuscenes.map_expansion.arcline_path_utils as path_utils
from matplotlib.ticker import MaxNLocator
from metrics import calcTrackPosition

def isLateralPolynom(polynom, heading):
    R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
    polynom_rotated = np.dot(R, polynom.T).T
    dx = np.max(polynom_rotated[:, 0])-np.min(polynom_rotated[:, 0])
    dy = np.max(polynom_rotated[:, 1])-np.min(polynom_rotated[:, 1])
    if abs(dy) > abs(dx) * 4:
        return True
    
    return False

def isVehicle(trk, x_offset=0, y_offset=0, velThr=2, n_last_frames=1000):
    if trk.confirmed:
        hstate, hego, hspeed = trk.getHistory()
        n_last_frames = min(n_last_frames,hstate.shape[0])
        history_len = hstate.shape[0]
        if trk.hits > 10:
        #rotate translate each state according to hego
            tstate = np.zeros((hstate.shape[0], 2, 1))
            tspeed = np.zeros((hstate.shape[0], 2, 1))
            for i, (state, ego, speed) in enumerate(zip(hstate, hego, hspeed)):
                R = np.array([[np.cos(ego["heading"]), -np.sin(ego["heading"])], [np.sin(ego["heading"]), np.cos(ego["heading"])]])
                tstate[i, :, :] = np.dot(R, state[0:2]) + ego["T"][0:2].reshape(-1,1)
                tspeed[i, :, :] = np.dot(R, state[2:4]) + np.dot(R, speed[0:2].reshape(-1,1))
            abs_vel = np.mean(np.linalg.norm(tspeed,axis=1), axis=0)
            if abs_vel > velThr:
                return True

    return False

def drawEllipses(measurements, key1, key2, ax, n=10, edgecolor='firebrick'):
        ellipses = range(0,measurements[key1].shape[0])
        ellipses = random.sample(ellipses, n)
        for i in ellipses:
            cov = measurements[key2][i]
            ax = confidence_ellipse(measurements[key1][i,1], measurements[key1][i,0], cov, ax, edgecolor=edgecolor)

        return ax
    
def drawPrior(ax, priors, xlim, **kwargs):
    for idx,prior in enumerate(priors):
        x,y = createPolynom(prior["c"][0],prior["c"][1],prior["c"][2],xstart=prior["xmin"],xend=prior["xmax"])
        label = kwargs.pop('label', f"Object {idx+1}")
        if prior["fx"]:
            ax.plot(y,x,label=label,**kwargs)
        else:
            ax.plot(x,y,label=label,**kwargs)
            
    return ax

def generateGraphSimulationMsr(data, frames, fig, ax, xlimits=[], ylimits=[]):
    colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                  "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]

    x_lim_min = -10
    x_lim_max = 10
    y_lim_min = -10
    y_lim_max = 50
        
    
    heading = 0
    for idx in frames:
        prior, measurements, points, polynoms, debug_info, pos = data.load(idx)
    
        x_lim_min = min(min(x_lim_min, np.min(measurements["polynom"][:,1])), np.min(measurements["other"][:,1])) -5
        x_lim_max = max(max(x_lim_max, np.max(measurements["polynom"][:,1])), np.max(measurements["other"][:,1])) + 5
        y_lim_min = -10
        y_lim_max = max(max(y_lim_max, np.max(measurements["polynom"][:,0])), np.max(measurements["other"][:,0])) + 5
        
        xlim = xlimits if xlimits else [x_lim_min,x_lim_max]
        ylim = ylimits if ylimits else [y_lim_min,y_lim_max]
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.scatter(measurements["polynom"][:,1],measurements["polynom"][:,0], label="ovservations")
        ax.scatter(measurements["other"][:,1],measurements["other"][:,0], label="noise")
        ax = drawEllipses(measurements=measurements,key1="polynom",key2="dpolynom",n=20,ax=ax,edgecolor='firebrick')
        ax = drawEllipses(measurements=measurements,key1="other",key2="dother",n=10,ax=ax,edgecolor='blue')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax = drawPrior(ax=ax,priors=prior,xlim=ylim,linewidth=5,label="prior")
        ax = drawEgo(x0=pos[1],y0=pos[0]-5,angle=heading,ax=ax,edgecolor='red',width=2,height=5)
        
    return fig, ax

def generateGraphSimulationComparison(data1, data2, frames, observation, fig, ax, xlimits=[], ylimits=[]):
    colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                  "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]

    x_lim_min = -10
    x_lim_max = 10
    y_lim_min = -10
    y_lim_max = 20
        
    
    heading = 0
    for idx in frames:
        prior, measurements, points, polynoms, debug_info, pos = data1.load(idx)
        _, _, _, polynoms_adv, _, _ = data2.load(idx)
    
        x_lim_min = min(min(x_lim_min, np.min(measurements["polynom"][:,1])), np.min(measurements["other"][:,1])) -5
        x_lim_max = max(max(x_lim_max, np.max(measurements["polynom"][:,1])), np.max(measurements["other"][:,1])) + 5
        y_lim_min = -10
        y_lim_max = max(max(y_lim_max, np.max(measurements["polynom"][:,0])), np.max(measurements["other"][:,0])) + 5
        
        xlim = xlimits if xlimits else [x_lim_min,x_lim_max]
        ylim = ylimits if ylimits else [y_lim_min,y_lim_max]
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        #ax = drawPrior(ax=ax,priors=prior,xlim=ylim,linewidth=2,linestyle='--',label="prior") 
        ax = drawPrior(ax=ax,priors=observation,xlim=ylim,linewidth=2,linestyle='--',label="observation")        
        ax = drawEgo(x0=pos[1],y0=pos[0]-5,angle=heading,ax=ax,edgecolor='red',width=2,height=5)
        for polynom in polynoms:
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax.plot(y_plot,x_plot,linewidth=3, label="improved detector")
            else:
                ax.plot(x_plot,y_plot,linewidth=3, label="improved detector")
                
        for polynom in polynoms_adv:
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax.plot(y_plot,x_plot,linewidth=2, label="original detector")
            else:
                ax.plot(x_plot,y_plot,linewidth=2, label="original detector")
        
    return fig, ax

def generateGraphPerformance(data, frames, fig, ax, xlimits=[], ylimits=[]):
    colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                  "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]

    x_lim_min = -10
    x_lim_max = 10
    y_lim_min = -10
    y_lim_max = 20
        
    
    heading = 0
    for idx in frames:
        prior, measurements, points, polynoms, debug_info, pos = data.load(idx)
    
        x_lim_min = min(min(x_lim_min, np.min(measurements["polynom"][:,1])), np.min(measurements["other"][:,1])) -5
        x_lim_max = max(max(x_lim_max, np.max(measurements["polynom"][:,1])), np.max(measurements["other"][:,1])) + 5
        y_lim_min = -10
        y_lim_max = max(max(y_lim_max, np.max(measurements["polynom"][:,0])), np.max(measurements["other"][:,0])) + 5
        
        xlim = xlimits if xlimits else [x_lim_min,x_lim_max]
        ylim = ylimits if ylimits else [y_lim_min,y_lim_max]
        
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        ax[0].scatter(points[:,1], points[:,0], label="Point tracks")
        ax[0] = drawPrior(ax=ax[0],priors=prior,xlim=ylim,linewidth=2,linestyle='--')        
        ax[0] = drawEgo(x0=pos[1],y0=pos[0]-5,angle=heading,ax=ax[0],edgecolor='red',width=2,height=5)
        
        
        #ax[1].set_xlim(xlim)
        #ax[1].set_ylim(ylim)
        #ax = drawPrior(ax=ax,priors=prior,xlim=ylim,linewidth=2,linestyle='--',label="prior") 
        ax[1] = drawPrior(ax=ax[1],priors=prior,xlim=ylim,linewidth=2,linestyle='--')        
        ax[1] = drawEgo(x0=pos[1],y0=pos[0]-5,angle=heading,ax=ax[1],edgecolor='red',width=2,height=5)
        for ipol, polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax[1].plot(y_plot,x_plot,linewidth=3, label=f"Ext track {ipol+1}")
            else:
                ax[1].plot(x_plot,y_plot,linewidth=3, label=f"Ext track {ipol+1}")
        
    return fig, ax

class PolynomsOnMapGraph():
    def __init__(self):
        self.counter = 0
        
    def run(self, t, gt_pos, ego_path, polynoms, nusc_map, fig, ax, xlimits=[], ylimits=[]):
        #Polynoms and GT on map
        if self.counter == 0:
            x_min = np.min(ego_path[:,0])
            x_max = np.max(ego_path[:,0])
            x_mean = 0.5*(x_min+x_max)
            y_min = np.min(ego_path[:,1])
            y_max = np.max(ego_path[:,1])
            y_mean = 0.5*(y_min+y_max)
            patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100

            self.first_pos = [x_mean, y_mean]
            self.patch_size = patch_size
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size)

            ax.imshow(edges, origin='lower')
            ax.grid(False)
            
            xlim = xlimits if xlimits else [self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50]
            ylim = ylimits if ylimits else [self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50]
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        for c,polynom in enumerate(polynoms):
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_plot = xx if polynom["fxFlag"] else polynom["f"](xx)
            y_plot = polynom["f"](xx) if polynom["fxFlag"] else xx 
            ax.plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=1,label="polynoms") 
        
        ax.scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=2,color="green",label="GT")
            
        
        return ax
    
    
    
def generateGraphCurve(video_data, polynoms, mm_results, ax, draw_polynoms=True, xlimits=[], ylimits=[], rotate=0, rotate_around=0):
    colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                  "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]
    pc = video_data['pc']
    img = video_data['img']
    prior = video_data['prior']
    pos = video_data['pos']
    heading = video_data['heading']
    pf_mean_pos = np.array(mm_results['pf_mean_pos'])
    imu_pos = np.array(video_data["pos_imu"][0:2])
        
    ax[0].imshow(img)
    ax[0].grid(None)
    ax[0].axis('off')
    
    R = np.array([[np.cos(rotate),-np.sin(rotate)], [np.sin(rotate),np.cos(rotate)]])
    tx = pos[0] - imu_pos[0]
    ty = pos[1] - imu_pos[1]
    if xlimits:
        ax[1].set_xlim(xlimits)
        ax[1].set_ylim(ylimits)
    
    
    ego_pos = pos[0:2].reshape([2,1])
    ego_pos = ego_pos - rotate_around
    ego_pos = np.dot(R, ego_pos) + rotate_around
    if draw_polynoms:
        for c,polynom in enumerate(polynoms):
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_plot = xx - tx if polynom["fxFlag"] else polynom["f"](xx) - ty
            y_plot = polynom["f"](xx) - ty if polynom["fxFlag"] else xx - tx 
            if rotate != 0:
                X = np.array([x_plot, y_plot])
                X -= rotate_around
                T = np.dot(R, X) + rotate_around
                x_plot = T[0, :]
                y_plot = T[1, :]

            ax[1].plot(x_plot,y_plot,linewidth=3,color='gray')
            
        ax[1] = drawEgo(x0=ego_pos[0],y0=ego_pos[1],angle=heading+np.rad2deg(rotate),ax=ax[1],edgecolor='red', width=1.5, height=4)
        
    
    if rotate != 0:        
        X = np.array([pc[:, 0], pc[:, 1]])
        print("X.shape", X.shape)
        X -= rotate_around
        T = np.dot(R, X) + rotate_around
        x_plot = T[0, :]
        y_plot = T[1, :]
        
    
    ax[1].scatter(x_plot,y_plot,color='blue',s=1)
    
    
    return ax

def generateGraphCrossAlong(data, frames, ax):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    pf_cross_track_errors_arr = np.zeros(N)
    pf_along_track_errors_arr = np.zeros(N)
    imu_cross_track_errors_arr = np.zeros(N)
    imu_along_track_errors_arr = np.zeros(N)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        
        gt_track_pos, pf_track_pos, imu_track_pos = calcTrackPosition(ego_path, ego_trns, gt_pos[0:2], pf_mean_pos, imu_pos)
        pf_track_errors = pf_track_pos - gt_track_pos
        imu_track_errors = imu_track_pos - gt_track_pos
        
        pf_cross_track_errors_arr[idx] = pf_track_errors[0]
        pf_along_track_errors_arr[idx] = pf_track_errors[1]
        imu_cross_track_errors_arr[idx] = imu_track_errors[0]
        imu_along_track_errors_arr[idx] = imu_track_errors[1]
        gt_track_pos_arr[idx, :] = gt_track_pos
        pf_track_pos_arr[idx, :] = pf_track_pos
        imu_track_pos_arr[idx, :] = imu_track_pos
    
    #Cross-Track Position(t)
    ax[0,0].scatter(timestamp_arr, gt_track_pos_arr[:,0],color='green',alpha=0.6,label='GT')
    ax[0,0].scatter(timestamp_arr, pf_track_pos_arr[:,0],color='blue',alpha=0.6,label='RadLoc')
    ax[0,0].scatter(timestamp_arr, imu_track_pos_arr[:,0],color='red',alpha=0.6,label='INS')

    #Cross-Track Err(t)
    ax[1,0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,0].plot(timestamp_arr, pf_cross_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,0].plot(timestamp_arr, imu_cross_track_errors_arr, color='red', label='GT-INS', linewidth=3)

    #Along-Track Position(t)
    ax[0,1].scatter(timestamp_arr, gt_track_pos_arr[:,1],color='green',alpha=0.6,label='GT')
    ax[0,1].scatter(timestamp_arr, pf_track_pos_arr[:,1],color='blue',alpha=0.6,label='RadLoc')
    ax[0,1].scatter(timestamp_arr, imu_track_pos_arr[:,1],color='red',alpha=0.6,label='INS')

    #Along-Track Err(t)
    ax[1,1].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,1].plot(timestamp_arr, pf_along_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,1].plot(timestamp_arr, imu_along_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    
    return ax

def generateGraphXY(data, frames, ax):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    pf_x_errors_arr = np.zeros(N)
    pf_y_errors_arr = np.zeros(N)
    imu_x_errors_arr = np.zeros(N)
    imu_y_errors_arr = np.zeros(N)
    gt_pos_arr = np.zeros((N,2))
    pf_pos_arr = np.zeros((N,2))
    imu_pos_arr = np.zeros((N,2))
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])[0:2]
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        
        pf_errors = gt_pos - pf_mean_pos
        imu_errors = gt_pos - imu_pos
        
        pf_x_errors_arr[idx] = pf_errors[0]
        pf_y_errors_arr[idx] = pf_errors[1]
        imu_x_errors_arr[idx] = imu_errors[0]
        imu_y_errors_arr[idx] = imu_errors[1]
        gt_pos_arr[idx, :] = gt_pos
        pf_pos_arr[idx, :] = pf_mean_pos
        imu_pos_arr[idx, :] = imu_pos
    
    #X(t)
    ax[0,0].scatter(timestamp_arr, gt_pos_arr[:,0],color='green',alpha=0.6,label='GT')
    ax[0,0].scatter(timestamp_arr, pf_pos_arr[:,0],color='blue',alpha=0.6,label='RadLoc')
    ax[0,0].scatter(timestamp_arr, imu_pos_arr[:,0],color='red',alpha=0.6,label='INS')

    #X Err(t)
    ax[1,0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,0].plot(timestamp_arr, pf_x_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,0].plot(timestamp_arr, imu_x_errors_arr, color='red', label='GT-INS', linewidth=3)

    #Along-Track Position(t)
    ax[0,1].scatter(timestamp_arr, gt_pos_arr[:,1],color='green',alpha=0.6,label='GT')
    ax[0,1].scatter(timestamp_arr, pf_pos_arr[:,1],color='blue',alpha=0.6,label='RadLoc')
    ax[0,1].scatter(timestamp_arr, imu_pos_arr[:,1],color='red',alpha=0.6,label='INS')

    #Along-Track Err(t)
    ax[1,1].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,1].plot(timestamp_arr, pf_y_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,1].plot(timestamp_arr, imu_y_errors_arr, color='red', label='GT-INS', linewidth=3)
    
    return ax

def generateGraphCrossNumPolynoms(data, frames, ax):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    pf_cross_track_errors_arr = np.zeros(N)
    pf_along_track_errors_arr = np.zeros(N)
    imu_cross_track_errors_arr = np.zeros(N)
    imu_along_track_errors_arr = np.zeros(N)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    n_polynoms_arr = np.zeros(N)
    n_dyn_tracks_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        
        gt_track_pos, pf_track_pos, imu_track_pos = calcTrackPosition(ego_path, ego_trns, gt_pos[0:2], pf_mean_pos, imu_pos)
        pf_track_errors = pf_track_pos - gt_track_pos
        imu_track_errors = imu_track_pos - gt_track_pos
        
        pf_cross_track_errors_arr[idx] = pf_track_errors[0]
        pf_along_track_errors_arr[idx] = pf_track_errors[1]
        imu_cross_track_errors_arr[idx] = imu_track_errors[0]
        imu_along_track_errors_arr[idx] = imu_track_errors[1]
        gt_track_pos_arr[idx, :] = gt_track_pos
        pf_track_pos_arr[idx, :] = pf_track_pos
        imu_track_pos_arr[idx, :] = imu_track_pos
        n_polynoms_arr[idx] = len(polynoms)
        for trk in dynamic_tracks:
            if isVehicle(trk):
                n_dyn_tracks_arr[idx] += 1
    
    #Cross-Track Err(t)
    ax[0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[0].plot(timestamp_arr, pf_cross_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[0].plot(timestamp_arr, imu_cross_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    
    #sttaic tracks
    ax[1].plot(timestamp_arr, n_polynoms_arr,label='static tracks', linewidth=3)
    
    #Cross-Track Err(t)
    ax[2].plot(timestamp_arr, n_dyn_tracks_arr,label='dynamic tracks', linewidth=3)

    
    return ax

def generateGraphAlongNumPolynoms(data, frames, ax):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    pf_cross_track_errors_arr = np.zeros(N)
    pf_along_track_errors_arr = np.zeros(N)
    imu_cross_track_errors_arr = np.zeros(N)
    imu_along_track_errors_arr = np.zeros(N)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    n_lateral_polynoms_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        
        gt_track_pos, pf_track_pos, imu_track_pos = calcTrackPosition(ego_path, ego_trns, gt_pos[0:2], pf_mean_pos, imu_pos)
        pf_track_errors = pf_track_pos - gt_track_pos
        imu_track_errors = imu_track_pos - gt_track_pos
        
        pf_cross_track_errors_arr[idx] = pf_track_errors[0]
        pf_along_track_errors_arr[idx] = pf_track_errors[1]
        imu_cross_track_errors_arr[idx] = imu_track_errors[0]
        imu_along_track_errors_arr[idx] = imu_track_errors[1]
        gt_track_pos_arr[idx, :] = gt_track_pos
        pf_track_pos_arr[idx, :] = pf_track_pos
        imu_track_pos_arr[idx, :] = imu_track_pos
        
        for polynom in polynoms:
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_pol = xx if polynom["fxFlag"] else polynom["f"](xx)
            y_pol = polynom["f"](xx) if polynom["fxFlag"] else xx 
            if isLateralPolynom(np.array([x_pol, y_pol]).T, video_data["heading"]):
                n_lateral_polynoms_arr[idx] += 1
    
    #Along-Track Err(t)
    ax[0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[0].plot(timestamp_arr, pf_along_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[0].plot(timestamp_arr, imu_along_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    
    #static tracks
    ax[1].plot(timestamp_arr, n_lateral_polynoms_arr,label='static tracks', linewidth=3)

    
    return ax

def generateGraphPath(data, frames, nusc_map, ax, xlimits=[], ylimits=[]):
        
    N = len(frames)
    gt_pos_arr = np.zeros((N,2))
    pf_pos_arr = np.zeros((N,2))
    imu_pos_arr = np.zeros((N,2))

    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)

        gt_pos = np.array(video_data['pos'])[0:2]
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]

        gt_pos_arr[idx, :] = gt_pos
        pf_pos_arr[idx, :] = pf_mean_pos
        imu_pos_arr[idx, :] = imu_pos


    x_min = np.min(ego_path[:,0])
    x_max = np.max(ego_path[:,0])
    x_mean = 0.5*(x_min+x_max)
    y_min = np.min(ego_path[:,1])
    y_max = np.max(ego_path[:,1])
    y_mean = 0.5*(y_min+y_max)
    patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100

    first_pos = [x_mean, y_mean]
    patch_size = patch_size
    #edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)
    edges = getLayer(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)
    edges[edges==0] = 255
    edges[edges==1] = 200
    print(edges)

    ax.imshow(edges, origin='lower', cmap='gray', vmin=0, vmax=255)
    ax.grid(False)

    xlim = xlimits if xlimits else [patch_size/2 - (x_mean-x_min) - 50,patch_size/2 + (x_max-x_mean) + 50]
    ylim = ylimits if ylimits else [patch_size/2 - (y_mean-y_min) - 50,patch_size/2 + (y_max-y_mean) + 50]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(gt_pos_arr[:,0]-first_pos[0]+patch_size*0.5,gt_pos_arr[:,1]-first_pos[1]+patch_size*0.5,s=2,color="green",label="GT")
    ax.scatter(pf_pos_arr[:,0]-first_pos[0]+patch_size*0.5,pf_pos_arr[:,1]-first_pos[1]+patch_size*0.5,s=2,color="blue",label="RadLoc")
    ax.scatter(imu_pos_arr[:,0]-first_pos[0]+patch_size*0.5,imu_pos_arr[:,1]-first_pos[1]+patch_size*0.5,s=2,color="red",label="INS")
            
        
    return ax

def generateGraphCurve(video_data, polynoms, mm_results, ax, draw_polynoms=True, xlimits=[], ylimits=[]):
    colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                  "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]
    pc = video_data['pc']
    img = video_data['img']
    prior = video_data['prior']
    pos = video_data['pos']
    heading = video_data['heading']
    pf_mean_pos = np.array(mm_results['pf_mean_pos'])
    imu_pos = np.array(video_data["pos_imu"][0:2])
        
    ax[0].imshow(img)
    ax[0].grid(None)
    ax[0].axis('off')
    
    if xlimits:
        ax[1].set_xlim(xlimits)
        ax[1].set_ylim(ylimits)
    
    
    ego_pos = pos[0:2].reshape([2,1])
    if draw_polynoms:
        for c,polynom in enumerate(polynoms):
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_plot = xx - tx if polynom["fxFlag"] else polynom["f"](xx) - ty
            y_plot = polynom["f"](xx) - ty if polynom["fxFlag"] else xx - tx 
            ax[1].plot(x_plot,y_plot,linewidth=3,color='gray')
            
        ax[1] = drawEgo(x0=ego_pos[0],y0=ego_pos[1],angle=heading+np.rad2deg(rotate),ax=ax[1],edgecolor='red', width=1.5, height=4)
        
    
    ax[1].scatter(x_plot,y_plot,color='blue',s=1)
    
    
    return ax

class PolynomsOnMapWithCameraGraph():
    def __init__(self):
        self.counter = 0
        
    def run(self, t, gt_pos, ego_path, polynoms, nusc_map, img, fig, ax, xlimits=[], ylimits=[]):
        
        ax[0].imshow(img)
        ax[0].grid(None)
        ax[0].axis('off')
        
        #Polynoms and GT on map
        if self.counter == 0:
            x_min = np.min(ego_path[:,0])
            x_max = np.max(ego_path[:,0])
            x_mean = 0.5*(x_min+x_max)
            y_min = np.min(ego_path[:,1])
            y_max = np.max(ego_path[:,1])
            y_mean = 0.5*(y_min+y_max)
            patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100

            self.first_pos = [x_mean, y_mean]
            self.patch_size = patch_size
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size)

            ax[1].imshow(edges, origin='lower')
            ax[1].grid(False)
            
            xlim = xlimits if xlimits else [self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50]
            ylim = ylimits if ylimits else [self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50]
            
            ax[1].set_xlim(xlim)
            ax[1].set_ylim(ylim)

        for c,polynom in enumerate(polynoms):
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_plot = xx if polynom["fxFlag"] else polynom["f"](xx)
            y_plot = polynom["f"](xx) if polynom["fxFlag"] else xx 
            ax[1].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=1,label="polynoms") 
        
        ax[1].scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=2,color="green",label="GT")    
        
        return ax
    """
def something():
        
        
        ax2[0].axis('scaled')
        ax2[0].set_title("Measurements frame={}".format(idx), fontsize=30)
        ax2[0].scatter(measurements["polynom"][:,1],measurements["polynom"][:,0],label='polynom measurements')
        ax2[0].scatter(measurements["other"][:,1],measurements["other"][:,0], label='random noise')
        ax2[0] = drawEllipses(measurements=measurements,key1="polynom",key2="dpolynom",n=20,ax=ax2[0],edgecolor='firebrick')
        ax2[0] = drawEllipses(measurements=measurements,key1="other",n=10,key2="dother",ax=ax2[0],edgecolor='blue')
        ax2[0].set_xlim(xlim)
        ax2[0].set_ylim(ylim)
        drawPrior(ax=ax2[0],priors=prior,xlim=ylim,linewidth=5)
        ax2[0] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax2[0],edgecolor='red')
        ax2[0].set_xlabel('x [m]', fontsize=20)
        ax2[0].set_ylabel('y [m]', fontsize=20)
        ax2[0].legend(loc="upper left")
        
        
        ax[0,1].set_title("Point tracks frame={}".format(idx), fontsize=30)
        ax[0,1].scatter(points[:,1], points[:,0])
        ax[0,1].set_xlim(xlim)
        ax[0,1].set_ylim(ylim)
        ax[0,1] = drawPrior(ax=ax[0,1],priors=prior,xlim=ylim,linewidth=3,linestyle='--')
        ax[0,1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[0,1])
        
        ax2[1].axis('scaled')
        ax2[1].set_title("Point tracks frame={}".format(idx), fontsize=30)
        ax2[1].scatter(points[:,1], points[:,0])
        ax2[1].set_xlim(xlim)
        ax2[1].set_ylim(ylim)
        ax2[1] = drawPrior(ax=ax2[1],priors=prior,xlim=ylim,linewidth=3,linestyle='--')
        ax2[1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax2[1])
        ax2[1].set_xlabel('x [m]', fontsize=20)
        ax2[1].set_ylabel('y [m]', fontsize=20)
        ax2[1].legend(loc="upper left")
        
        
        ax[0,2].set_title("Extended tracks frame={}".format(idx), fontsize=30)
        ax[0,2].set_xlim(xlim)
        ax[0,2].set_ylim(ylim)
        ax[0,2] = drawPrior(ax=ax[0,2],priors=prior,xlim=ylim,linewidth=3,linestyle='--') 
        ax[0,2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[0,2],edgecolor='red')
        for polynom in polynoms:
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax[0,2].plot(y_plot,x_plot,linewidth=10)
            else:
                ax[0,2].plot(x_plot,y_plot,linewidth=10)
                
        
        ax2[2].set_title("Extended tracks frame={}".format(idx), fontsize=30)
        ax2[2].axis('scaled')
        ax2[2].set_xlim(xlim)
        ax2[2].set_ylim(ylim)
        ax2[2] = drawPrior(ax=ax2[2],priors=prior,xlim=ylim,linewidth=3,linestyle='--') 
        ax2[2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax2[2],edgecolor='red')
        for ipol,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax2[2].plot(y_plot,x_plot,linewidth=10,label=f"track {ipol}")
            else:
                ax2[2].plot(x_plot,y_plot,linewidth=10,label=f"track {ipol}")
                
        ax2[2].set_xlabel('x [m]', fontsize=20)
        ax2[2].set_ylabel('y [m]', fontsize=20)
        ax2[2].legend(loc="upper left")
        
            
        ax[1,0].set_title("Points that generated a new polynom frame={}".format(idx), fontsize=30)
        ax[1,0].set_xlim(xlim)
        ax[1,0].set_ylim(ylim)
        drawPrior(ax=ax[1,0],priors=prior,xlim=ylim,linewidth=3,linestyle='--') 
        ax[1,0] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[1,0])
        for c,pair in enumerate(debug["pgpol"]):
            
            x_plot = np.linspace(pair["polynom"]["x_start"], pair["polynom"]["x_end"], 100)
            y_plot = pair["polynom"]["f"](x_plot)
            if pair["polynom"]["fxFlag"]:
                ax[1,0].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=colors[c])
                ax[1,0].scatter(pair["points"][1,:], pair["points"][0,:],c=[colors[c]]*pair["points"].shape[1])
            else:
                ax[1,0].plot(x_plot,y_plot,linewidth=3,linestyle='--',color=colors[c])
                ax[1,0].scatter(pair["points"][0,:], pair["points"][1,:],c=[colors[c]]*pair["points"].shape[1])
            
        ax[1,1].set_title("Points that updated point tracks frame={}".format(idx), fontsize=30)
        ax[1,1].set_xlim(xlim)
        ax[1,1].set_ylim(ylim)
        ax[1,1] = drawPrior(ax=ax[1,1],priors=prior,xlim=ylim,linewidth=3,linestyle='--') 
        ax[1,1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[1,1])
        for pair in debug["mupoi"]:
            ax[1,1].scatter(pair["measurements"][1], pair["measurements"][0],color='blue')
            ax[1,1].scatter(pair["points"][1], pair["points"][0],color='orange')
            
        ax[1,2].set_title("Points that updated extended tracks frame={}".format(idx), fontsize=30)
        ax[1,2].set_xlim(xlim)
        ax[1,2].set_ylim(ylim)
        drawPrior(ax=ax[1,2],priors=prior,xlim=ylim,linewidth=3,linestyle='--') 
        ax[1,2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[1,2])
        unique_polynoms = set(d['id'] for d in debug["mupol"])
        print("len(unique_polynoms)", len(unique_polynoms))
        for c,upol in enumerate(unique_polynoms):
            first = True
            xy = []
            for pair in debug["mupol"]:
                if pair["id"] == upol:
                    if first:
                        if pair["fxFlag"]:
                            x_plot = np.linspace(pair["polynom"][3], pair["polynom"][4], 100)
                            y_plot = pair["polynom"][0] + pair["polynom"][1]*x_plot + pair["polynom"][2]*x_plot**2
                        if pair["fxFlag"]:
                            ax[1,2].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=colors[c])
                        else:
                            ax[1,2].plot(x_plot,y_plot,linewidth=3,linestyle='--',color=colors[c])
                        first = False
                    xy.append(np.array([pair["measurements"][0], pair["measurements"][1]]))
            xy = np.array(xy).T
            if pair["fxFlag"]:
                ax[1,2].scatter(xy[1,:], xy[0,:],c=colors[c])
            else:
                ax[1,2].scatter(xy[0,:], xy[1,:],c=colors[c])
      
  
        
        ax[0,1].clear()
        ax[0,2].clear()
        ax[1,0].clear()
        ax[1,1].clear()
        ax[1,2].clear()
        ax2[0].clear()
        ax2[1].clear()
        ax2[2].clear()
"""
