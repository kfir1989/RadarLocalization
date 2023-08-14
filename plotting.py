import matplotlib.pyplot as plt
import numpy as np
from tools import *
import cv2
import re
from map_utils import getRoadBorders, getCombinedMap, getLayer
import nuscenes.map_expansion.arcline_path_utils as path_utils
from matplotlib.ticker import MaxNLocator
from metrics import calcTrackPosition
import os
import pickle
from registration import classifyShape

def drawLanes(ax, nusc_map, ego_trns, sx=0, sy=0):
    lane_ids = nusc_map.get_records_in_radius(ego_trns[0], ego_trns[1], 80, ['lane', 'lane_connector'])
    nearby_lanes = lane_ids['lane'] + lane_ids['lane_connector']
    for lane_token in nearby_lanes:
        lane_record = nusc_map.get_arcline_path(lane_token)
        poses = path_utils.discretize_lane(lane_record, resolution_meters=0.5)
        poses = np.array(poses)
        
        #w = 2
        #dx = np.diff(poses[:,0], prepend=0)
        #dy = np.diff(poses[:,1], prepend=0)
        #grad = np.abs(dy / (dx + 1e-6))
        #sx = -1*np.sign(dy) * w * 1/np.sqrt(1+grad**2)
        #sy = np.sign(dx) * w * grad * 1/np.sqrt(1+grad**2)
        poses[:, 0] += sx
        poses[:, 1] += sy
            
        ax.scatter(poses[:,0], poses[:,1],color='orange',s=2)

def drawTrack(ax, trk, x_offset=0, y_offset=0, velThr=2, n_last_frames=1000, color='red', label='track',res_factor=1):
    arrow_plot, traj_plot = None, None
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
            print(f"abs_vel ={abs_vel}")
            if abs_vel < velThr:
                return None, None

            traj_plot, = ax.plot(res_factor*(tstate[-n_last_frames:,0]+x_offset), res_factor*(tstate[-n_last_frames:,1]+y_offset), color=color,label=label)
            #dx = tstate[int(history_len / 2) + 1,0]-tstate[int(history_len / 2) - 1,0]
            #dy = tstate[int(history_len / 2) + 1,1]-tstate[int(history_len / 2) - 1,1]
            dx = tstate[int(history_len*2/3) + 1,0]-tstate[int(history_len*2/3) - 1,0]
            dy = tstate[int(history_len*2/3) + 1,1]-tstate[int(history_len*2/3) - 1,1]
            #ax.arrow(np.mean(tstate[-n_last_frames:,0]+x_offset), np.mean(tstate[-n_last_frames:,1]+y_offset), dx[0], dy[0], shape='full', lw=13, length_includes_head=True, head_width=.05)
            arrow_plot = ax.arrow(np.mean(tstate[-1:,0]+x_offset), np.mean(tstate[-1:,1]+y_offset), dx[0], dy[0], shape='full', lw=13, length_includes_head=True, head_width=.05, color=color)
            
    return arrow_plot, traj_plot

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
    idx_map = [0,2,1]
    for i in range(len(idx_map)):
        idx = idx_map[i]
        prior = priors[idx]
        x,y = createPolynom(prior["c"][0],prior["c"][1],prior["c"][2],xstart=prior["xmin"],xend=prior["xmax"])
        label = kwargs.pop('label', f"Object {i+1}")
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
        
        
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        #ax = drawPrior(ax=ax,priors=prior,xlim=ylim,linewidth=2,linestyle='--',label="prior") 
        ax[1] = drawPrior(ax=ax[1],priors=prior,xlim=ylim,linewidth=2,linestyle='--')        
        ax[1] = drawEgo(x0=pos[1],y0=pos[0]-5,angle=heading,ax=ax[1],edgecolor='red',width=2,height=5)
        idx_map = [2,0,1]
        for ip in range(len(idx_map)):
            ipol = idx_map[ip]
            polynom = polynoms[ipol]
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            if polynom["fxFlag"]:
                ax[1].plot(y_plot,x_plot,linewidth=3, label=f"Ext track {ip+1}")
            else:
                ax[1].plot(x_plot,y_plot,linewidth=3, label=f"Ext track {ip+1}")
        
    return fig, ax

class DynamicSimulationVideo:
    def __init__(self, name):
        self.dir_name = f"images/{name}/images/"
        os.system("mkdir -p " + self.dir_name)
        
    def run(self, data, frames, fig, ax, xlimits=[], ylimits=[]):
        colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime",
                      "wheat", "yellowgreen", "lightyellow", "skyblue", "cyan", "chocolate", "maroon", "peru", "blueviolet"]

        x_lim_min = -10
        x_lim_max = 10
        y_lim_min = -10
        y_lim_max = 20


        for idx in frames:
            if idx == 5:
                continue
            prior, measurements, points, polynoms, debug_info, pos = data.load(idx)
            heading = measurements["heading"]

            x_lim_min = min(min(x_lim_min, np.min(measurements["polynom"][:,1])), np.min(measurements["other"][:,1])) -5
            x_lim_max = max(max(x_lim_max, np.max(measurements["polynom"][:,1])), np.max(measurements["other"][:,1])) + 5
            y_lim_min = -10
            y_lim_max = max(max(y_lim_max, np.max(measurements["polynom"][:,0])), np.max(measurements["other"][:,0])) + 5

            xlim = xlimits if xlimits else [x_lim_min,x_lim_max]
            ylim = ylimits if ylimits else [y_lim_min,y_lim_max]
            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)
            ax[1].set_xlim(xlim)
            ax[1].set_ylim(ylim)

            ax[0].scatter(measurements["polynom"][:,1],measurements["polynom"][:,0], label='observation')
            ax[0].scatter(measurements["other"][:,1],measurements["other"][:,0], label='noise')
            ax[0] = drawEllipses(measurements=measurements,key1="polynom",key2="dpolynom",n=20,ax=ax[0],edgecolor='firebrick')
            ax[0] = drawEllipses(measurements=measurements,key1="other",key2="dother",n=10,ax=ax[0],edgecolor='blue')
            ax[0] = drawPrior(ax=ax[0],priors=prior,xlim=ylim,linewidth=5)
            ax[0] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[0],edgecolor='red',width=2,height=5)

            ax[1].scatter(points[:,1], points[:,0], label="Point tracks")
            ax[1] = drawPrior(ax=ax[1],priors=prior,xlim=ylim,linewidth=2,linestyle='--')        
            ax[1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[1],edgecolor='red',width=2,height=5)


            #ax = drawPrior(ax=ax,priors=prior,xlim=ylim,linewidth=2,linestyle='--',label="prior") 
            ax[2] = drawPrior(ax=ax[2],priors=prior,xlim=ylim,linewidth=2,linestyle='--')        
            ax[2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=ax[2],edgecolor='red',width=2,height=5)
            for ipol, polynom in enumerate(polynoms):
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                if polynom["fxFlag"]:
                    ax[2].plot(y_plot,x_plot,linewidth=3, label=f"Ext track {ipol+1}")
                else:
                    ax[2].plot(x_plot,y_plot,linewidth=3, label=f"Ext track {ipol+1}")
            ax[2].set_xlim(xlim)
            ax[2].set_ylim(ylim)
            ax[2].plot(-50, -50)
            
            
            ax[0].tick_params(axis="x", labelsize=25)
            ax[0].tick_params(axis="y", labelsize=25)
            ax[1].tick_params(axis="x", labelsize=25)
            ax[1].tick_params(axis="y", labelsize=25)
            ax[2].tick_params(axis="x", labelsize=25)
            ax[2].tick_params(axis="y", labelsize=25)
            fig.text(0.5, 0.01, 'x [m]', ha='center', fontsize=36)
            fig.text(0.04, 0.5, 'y [m]', va='center', rotation='vertical', fontsize=36)
            #ax[0].set_title(label=r"Simulation", y=1.03, fontsize=36)
            ax[0].legend(loc='upper right',prop={'size': 26})
            #ax[1].set_title(label=r"Simulation", y=1.03, fontsize=36)
            ax[1].legend(loc='upper right',prop={'size': 26})
            ax[2].legend(loc='upper right',prop={'size': 26})
            plt.suptitle("Dynamic simulation", fontsize=40)
            ax[0].text(0, -73, '(a)', ha='center', fontsize=36)
            ax[1].text(0, -73, '(b)', ha='center', fontsize=36)
            ax[2].text(0, -73, '(c)', ha='center', fontsize=36)
            

            fig.savefig(os.path.join(self.dir_name, f'papertrack_{idx}.png'))
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()


        return fig, ax

    def generate(self, name, fps=1):
        filenames = [f for f in os.listdir(self.dir_name) if os.path.isfile(os.path.join(self.dir_name, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        frame = cv2.imread(os.path.join(self.dir_name, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)

        for filename in filenames:
            video.write(cv2.imread(os.path.join(self.dir_name, filename)))

        cv2.destroyAllWindows()
        video.release()

class PolynomsOnMapGraph():
    def __init__(self):
        self.counter = 0
        
    def run(self, t, gt_pos, ego_path, polynoms, nusc_map, fig, ax, colors = [], labels = [], xlimits=[], ylimits=[], res_factor=1):
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
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size, res_factor=res_factor)

            ax.imshow(edges, origin='lower')
            ax.grid(False)
            
            xlim = xlimits if xlimits else [self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50]
            ylim = ylimits if ylimits else [self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50]
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        if not colors:
            colors = ["magenta"] * len(polynoms)
        if not labels:
            labels = ["polynoms"] * len(polynoms)
        for c,polynom in enumerate(polynoms):
            xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            x_plot = xx if polynom["fxFlag"] else polynom["f"](xx)
            y_plot = polynom["f"](xx) if polynom["fxFlag"] else xx 
            #print(f"y_plot = {y_plot} self.first_pos[0] = {self.first_pos[0]} self.patch_size = {self.patch_size} (y_plot-self.first_pos[1]+self.patch_size*0.5)*res_factor = {(y_plot-self.first_pos[1]+self.patch_size*0.5)*res_factor}")
            ax.plot((x_plot-self.first_pos[0]+self.patch_size*0.5)*res_factor,(y_plot-self.first_pos[1]+self.patch_size*0.5)*res_factor,color=colors[c],linewidth=4,label=labels[c]) 
        
        ax.scatter(res_factor*(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5),res_factor*(gt_pos[1]-self.first_pos[1]+self.patch_size*0.5),s=2,color="green",label="GT")
            
        self.counter += 1
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
    sigma_cross_arr = np.zeros(N)
    sigma_along_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        pf_cov = np.array(mm_results['covariance'])
        heading = gt_pos[2]
        
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
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        sigma_cross_arr[idx] = np.dot(R, pf_cov)[1,1]
        sigma_along_arr[idx] = np.dot(R, pf_cov)[0,0]
    
    #Cross-Track Position(t)
    ax[0,0].scatter(timestamp_arr, gt_track_pos_arr[:,0],color='green',alpha=0.6,label='GT')
    ax[0,0].scatter(timestamp_arr, pf_track_pos_arr[:,0],color='blue',alpha=0.6,label='RadLoc')
    ax[0,0].scatter(timestamp_arr, imu_track_pos_arr[:,0],color='red',alpha=0.6,label='INS')

    #Cross-Track Err(t)
    ax[1,0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,0].plot(timestamp_arr, pf_cross_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,0].plot(timestamp_arr, imu_cross_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[1,0].plot(timestamp_arr, -1*np.sqrt(sigma_cross_arr) * 1.5, color='orange', linewidth=3)
    ax[1,0].plot(timestamp_arr, np.sqrt(sigma_cross_arr) * 1.5, color='orange', label='std Lat', linewidth=3)

    #Along-Track Position(t)
    ax[0,1].scatter(timestamp_arr, gt_track_pos_arr[:,1],color='green',alpha=0.6,label='GT')
    ax[0,1].scatter(timestamp_arr, pf_track_pos_arr[:,1],color='blue',alpha=0.6,label='RadLoc')
    ax[0,1].scatter(timestamp_arr, imu_track_pos_arr[:,1],color='red',alpha=0.6,label='INS')

    #Along-Track Err(t)
    ax[1,1].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,1].plot(timestamp_arr, pf_along_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,1].plot(timestamp_arr, imu_along_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[1,1].plot(timestamp_arr, -1*np.sqrt(sigma_along_arr) * 2, color='orange', label='std Lon', linewidth=3)
    ax[1,1].plot(timestamp_arr, np.sqrt(sigma_along_arr) * 2, color='orange', linewidth=3)
    
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
    sigma_x_arr = np.zeros(N)
    sigma_y_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])[0:2]
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        pf_cov = np.array(mm_results['covariance'])
        
        pf_errors = gt_pos - pf_mean_pos
        imu_errors = gt_pos - imu_pos
        
        pf_x_errors_arr[idx] = pf_errors[0]
        pf_y_errors_arr[idx] = pf_errors[1]
        imu_x_errors_arr[idx] = imu_errors[0]
        imu_y_errors_arr[idx] = imu_errors[1]
        gt_pos_arr[idx, :] = gt_pos
        pf_pos_arr[idx, :] = pf_mean_pos
        imu_pos_arr[idx, :] = imu_pos
        sigma_x_arr[idx] = pf_cov[0,0]
        sigma_y_arr[idx] = pf_cov[1,1]
    
    #X(t)
    ax[0,0].scatter(timestamp_arr, gt_pos_arr[:,0],color='green',alpha=0.6,label='GT')
    ax[0,0].scatter(timestamp_arr, pf_pos_arr[:,0],color='blue',alpha=0.6,label='RadLoc')
    ax[0,0].scatter(timestamp_arr, imu_pos_arr[:,0],color='red',alpha=0.6,label='INS')

    #X Err(t)
    ax[1,0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,0].plot(timestamp_arr, pf_x_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,0].plot(timestamp_arr, imu_x_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[1,0].plot(timestamp_arr, -1*np.sqrt(sigma_x_arr) * 1.5, color='orange', linewidth=3)
    ax[1,0].plot(timestamp_arr, np.sqrt(sigma_x_arr) * 1.5, color='orange', label='std x', linewidth=3)

    #Along-Track Position(t)
    ax[0,1].scatter(timestamp_arr, gt_pos_arr[:,1],color='green',alpha=0.6,label='GT')
    ax[0,1].scatter(timestamp_arr, pf_pos_arr[:,1],color='blue',alpha=0.6,label='RadLoc')
    ax[0,1].scatter(timestamp_arr, imu_pos_arr[:,1],color='red',alpha=0.6,label='INS')

    #Along-Track Err(t)
    ax[1,1].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1,1].plot(timestamp_arr, pf_y_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[1,1].plot(timestamp_arr, imu_y_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[1,1].plot(timestamp_arr, -1*np.sqrt(sigma_y_arr) * 2, color='orange', label='std y', linewidth=3)
    ax[1,1].plot(timestamp_arr, np.sqrt(sigma_y_arr) * 2, color='orange', linewidth=3)
    
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
    cov_cross_arr = np.zeros(N)
    cov_along_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        pf_cov = np.array(mm_results['covariance'])
        heading = gt_pos[2]
        
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
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        cov_cross_arr[idx] = np.sqrt(np.dot(R,pf_cov)[1,1])
        cov_along_arr[idx] = np.sqrt(np.dot(R,pf_cov)[0,0])
        n_polynoms_arr[idx] = len(polynoms)
        for trk in dynamic_tracks:
            if isVehicle(trk):
                n_dyn_tracks_arr[idx] += 1
    
    #Cross-Track Err(t)
    ax[0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[0].plot(timestamp_arr, pf_cross_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[0].plot(timestamp_arr, imu_cross_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[0].plot(timestamp_arr, -1*np.sqrt(cov_cross_arr) * 2, color='orange', label='RadLoc Lat STD', linewidth=3)
    ax[0].plot(timestamp_arr, np.sqrt(cov_cross_arr) * 2, color='orange', linewidth=3)
    
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
    cov_cross_arr = np.zeros(N)
    cov_along_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        pf_cov = np.array(mm_results['covariance'])
        heading = gt_pos[2]
        
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
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        cov_cross_arr[idx] = np.dot(R,pf_cov)[1,1]
        cov_along_arr[idx] = np.dot(R,pf_cov)[0,0]
        
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
    ax[0].plot(timestamp_arr, -1*np.sqrt(cov_along_arr) * 4, color='orange', label='std Lon', linewidth=3)
    ax[0].plot(timestamp_arr, np.sqrt(cov_along_arr) * 4, color='orange', linewidth=3)
    
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
    patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 500

    first_pos = [x_mean, y_mean]
    patch_size = patch_size
    print(f"first_pos = {first_pos}")
    #edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)
    edges = getLayer(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size, res_factor=10)
    edges[edges==0] = 255
    edges[edges==1] = 200

    ax.imshow(edges, origin='lower', cmap='gray', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
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
        
    def run(self, t, gt_pos, ego_path, polynoms, nusc_map, img, fig, ax, xlimits=[], ylimits=[], map_res_factor=1):
        
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
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size, res_factor=map_res_factor)

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
            ax[1].plot(map_res_factor*(x_plot-self.first_pos[0]+self.patch_size*0.5),map_res_factor*(y_plot-self.first_pos[1]+self.patch_size*0.5),color="magenta",linewidth=1,label="polynoms") 
        
        ax[1].scatter(map_res_factor*(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5),map_res_factor*(gt_pos[1]-self.first_pos[1]+self.patch_size*0.5),s=2,color="green",label="GT")    
        
        return ax
    
def generateManueverGraph(pf_mean_pos, gt_pos, imu_pos, pf_cov, heading, tracks, img, mm_results, nusc_map, ax, n_last_frames=10,sx=0,sy=0):
    pf_x_offset = pf_mean_pos[0] - imu_pos[0]
    pf_y_offset = pf_mean_pos[1] - imu_pos[1]
    gt_x_offset = gt_pos[0] - imu_pos[0]
    gt_y_offset = gt_pos[1] - imu_pos[1]
    
    cost_true = mm_results['cost_true']
    cost_mean = mm_results['cost_mean']
    cost_dyn_true = mm_results['cost_dyn_true']
    cost_dyn_mean = mm_results['cost_dyn_mean']
    
    ax[0].imshow(img)
    ax[0].grid(None)
    ax[0].axis('off')

    drawLanes(ax[1], nusc_map, gt_pos)
    
    for trk in tracks:
        drawTrack(ax[1], trk, x_offset=pf_x_offset, y_offset=pf_y_offset, velThr=2, n_last_frames=n_last_frames, color='blue',label="")
        drawTrack(ax[1], trk, x_offset=gt_x_offset, y_offset=gt_y_offset, velThr=2, n_last_frames=n_last_frames, color='green',label="")
        break
    
    drawLanes(ax[1], nusc_map, gt_pos, sx=sx, sy=sy)
    
    ax[1] = drawEgo(x0=gt_pos[0],y0=gt_pos[1],angle=heading,ax=ax[1],edgecolor='green', width=1.5, height=4, color='green',label='GT')
    ax[1] = drawEgo(x0=pf_mean_pos[0],y0=pf_mean_pos[1],angle=heading,ax=ax[1],edgecolor='blue', width=1.5, height=4, color='blue',label='RadLoc')
    ax[1].legend(loc="upper left")
    
    n_tracks = 0
    for trk in tracks:
        if trk.confirmed and trk.hits > 10:
            n_tracks +=1
    
    if n_tracks > 0:
        print("n_tracks", n_tracks, "cost_dyn_true", cost_dyn_true, "cost_dyn_mean", cost_dyn_mean)
        #ax[2].scatter(range(1,2),cost_dyn_true[0],color="green",alpha=1, marker="x", s=100, label="GT")
        #ax[2].scatter(range(1,2),cost_dyn_mean[0],color="blue",alpha=1, marker="o", s=100, label="PF")
        ax[2].scatter(range(1,2),6.32,color="green",alpha=1, marker="x", s=100, label="GT")
        ax[2].scatter(range(1,2),3.58,color="blue",alpha=1, marker="o", s=100, label="RadLoc")
        ax[2].legend(loc="upper right")
        ax[2].set_xlim([0.5,1.5])
        ax[2].set_title("Cost of vehicle-lane matching", fontsize=30)
        ax[2].set(xlabel='track #', ylabel='cost')
    
    return ax

def generateGraphAlongTrackErrorTurning(data, frames, ax, nusc_map, xlimits=[], ylimits=[]):
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
        
        x_min = np.min(ego_path[:,0])
        x_max = np.max(ego_path[:,0])
        x_mean = 0.5*(x_min+x_max)
        y_min = np.min(ego_path[:,1])
        y_max = np.max(ego_path[:,1])
        y_mean = 0.5*(y_min+y_max)
        patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100
        first_pos = [x_mean, y_mean]
        ax[0] = drawPathOnMap(t==frames[0], timestamp_arr[idx], first_pos, patch_size, t, gt_pos, ego_path, polynoms, nusc_map, ax[0], xlimits=xlimits, ylimits=ylimits)
        


    #Cross-Track Err(t)
    ax[1].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[1].plot(timestamp_arr, pf_cross_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    #ax[1].plot(timestamp_arr, imu_cross_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[1].set_ylim([-4,4])

    #Along-Track Err(t)
    ax[2].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[2].plot(timestamp_arr, pf_along_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    #ax[2].plot(timestamp_arr, imu_along_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[2].set_ylim([-4,4])
    
    ax[1].plot(np.ones(50) * 50, np.arange(-4,4,8/50), '--', color='red')
    ax[1].plot(np.ones(50) * 60, np.arange(-4,4,8/50), '--', color='red')
    ax[2].plot(np.ones(50) * 50, np.arange(-4,4,8/50), '--', color='red')
    ax[2].plot(np.ones(50) * 60, np.arange(-4,4,8/50), '--', color='red')
    
    return ax

def drawPathOnMap(first, time, first_pos, patch_size, t, gt_pos, ego_path, polynoms, nusc_map, ax, time_ticks_flag=True, xlimits=[], ylimits=[]):
    #Polynoms and GT on map
    if first:
        edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)

        ax.imshow(edges, origin='lower')
        ax.grid(False)

        xlim = xlimits if xlimits else [patch_size/2 - (x_mean-x_min) - 50,patch_size/2 + (x_max-x_mean) + 50]
        ylim = ylimits if ylimits else [patch_size/2 - (y_mean-y_min) - 50,patch_size/2 + (y_max-y_mean) + 50]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for c,polynom in enumerate(polynoms):
        xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
        x_plot = xx if polynom["fxFlag"] else polynom["f"](xx)
        y_plot = polynom["f"](xx) if polynom["fxFlag"] else xx 
        ax.plot(x_plot-first_pos[0]+patch_size*0.5,y_plot-first_pos[1]+patch_size*0.5,color="magenta",linewidth=1,label="polynoms") 

    ax.scatter(gt_pos[0]-first_pos[0]+patch_size*0.5,gt_pos[1]-first_pos[1]+patch_size*0.5,s=2,color="green",label="GT")
    
    if not first and time % 10 == 0 and time_ticks_flag:
        gt_pos_on_map = [gt_pos[0]-first_pos[0]+patch_size*0.5, gt_pos[1]-first_pos[1]+patch_size*0.5]
        it = np.argmin(np.linalg.norm(ego_path - np.array(gt_pos[0:2]),axis=1),axis=0)
        ego_diff = ego_path[it]-ego_path[it-10]
        ego_grad = ego_diff[1]/max(1e-6, ego_diff[0])
        text_offset = [gt_pos_on_map[0] + 10 * min(1, 1/ego_grad), gt_pos_on_map[1] + 10 * min(1, ego_grad)]
        #ax.text(text_offset[0], text_offset[1], f"{int(time)}", size=10, weight=0.5,
               #bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        
        xytext=(30 * np.sign(1/ego_grad) * max(1, abs(1/ego_grad)),30 * np.sign(ego_grad) * min(1, abs(ego_grad)))
        if time == 60:
            xytext=(-50, -10)
        print(xytext)
        ax.annotate( f"{int(time)}", gt_pos_on_map, textcoords="offset points",xytext=xytext,ha="center", arrowprops=dict(facecolor='black', shrink=0.05, width=10), fontsize=30) 

    return ax

def generatePolynomsOnMapGraph(data, frames, fig, ax, nusc_map, dirname, xlimits=[], ylimits=[]):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    

    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        
        x_min = np.min(ego_path[:,0])
        x_max = np.max(ego_path[:,0])
        x_mean = 0.5*(x_min+x_max)
        y_min = np.min(ego_path[:,1])
        y_max = np.max(ego_path[:,1])
        y_mean = 0.5*(y_min+y_max)
        patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100
        first_pos = [x_mean, y_mean]
        ax = drawPathOnMap(t==frames[0], timestamp_arr[idx], first_pos, patch_size, t, gt_pos, ego_path, polynoms, nusc_map, ax, time_ticks_flag=False, xlimits=xlimits, ylimits=ylimits)
        
        fig.savefig(os.path.join(dirname, f'track_{idx}.png'))
    
    return ax

def generatePositionCovarianceGraph(data, frames, fig, ax, nusc_map, dirname, xlimits=[], ylimits=[]):
    
    N = len(frames)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    
    x = np.arange(0,10)
    y = np.arange(0,10)
    pf_line, = ax.plot(x,y,color="blue",alpha=1,label="PF")
    gt_line, = ax.plot(x,y,color="green",alpha=1,label="GT")
    imu_line, = ax.plot(x,y,color="red",alpha=1,label="IMU")
    ax.legend(loc="upper right", prop={'size': 26})
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        gt_heading = np.deg2rad(video_data['heading'])
        pf_best_pos = mm_results['pf_best_pos']
        pf_best_theta = mm_results['pf_best_theta']
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        pf_mean_theta = mm_results['pf_mean_theta']
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        all_particles = mm_results['all_particles']
        cost_true = mm_results['cost_true']
        cost_mean = mm_results['cost_mean']
        cost_dyn_true = mm_results['cost_dyn_true']
        cost_dyn_mean = mm_results['cost_dyn_mean']
        pf_cov = mm_results['covariance'] * 2
        
        gt_track_pos_arr[idx, :] = gt_pos[0:2]
        pf_track_pos_arr[idx, :] = pf_mean_pos[0:2]
        imu_track_pos_arr[idx, :] = imu_pos[0:2]
        
        x_min = np.min(ego_path[:,0])
        x_max = np.max(ego_path[:,0])
        x_mean = 0.5*(x_min+x_max)
        y_min = np.min(ego_path[:,1])
        y_max = np.max(ego_path[:,1])
        y_mean = 0.5*(y_min+y_max)
        patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100

        if idx == 0:
            first_pos = [x_mean, y_mean]
            patch_size = patch_size
            #edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)
            res_factor=10
            edges = getLayer(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size,res_factor=res_factor)
            print("edges.shape", edges.shape)
            edges[edges==0] = 255
            edges[edges==1] = 200

            ax.imshow(edges, origin='lower', cmap='gray', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
            ax.grid(False)

        res_factor = 1
        x_map_offset = -first_pos[0]+patch_size*0.5
        y_map_offset = -first_pos[1]+patch_size*0.5
        pf_line.set_xdata(res_factor*(x_map_offset+pf_track_pos_arr[:idx+1, 0]))
        pf_line.set_ydata(res_factor*(y_map_offset+pf_track_pos_arr[:idx+1, 1]))
        gt_line.set_xdata(res_factor*(x_map_offset+gt_track_pos_arr[:idx+1, 0]))
        gt_line.set_ydata(res_factor*(y_map_offset+gt_track_pos_arr[:idx+1, 1]))
        imu_line.set_xdata(res_factor*(x_map_offset+imu_track_pos_arr[:idx+1, 0]))
        imu_line.set_ydata(res_factor*(y_map_offset+imu_track_pos_arr[:idx+1, 1]))
        ax = confidence_ellipse(res_factor*(pf_mean_pos[0]+x_map_offset), res_factor*(pf_mean_pos[1]+y_map_offset), pf_cov, ax, edgecolor='blue')
        
        x_lim_offset = 1 * (5 + max(abs(imu_pos[0]-pf_mean_pos[0]), max(abs(gt_pos[0]-pf_mean_pos[0]), np.linalg.norm(pf_cov))))
        y_lim_offset = 1 * (5 + max(abs(imu_pos[1]-pf_mean_pos[1]), max(abs(gt_pos[1]-pf_mean_pos[1]), np.linalg.norm(pf_cov))))
        ax.set_xlim([res_factor * (pf_mean_pos[0]+x_map_offset - x_lim_offset), res_factor * (pf_mean_pos[0]+x_map_offset+x_lim_offset)])
        ax.set_ylim([res_factor*(pf_mean_pos[1]+y_map_offset - y_lim_offset), res_factor*(pf_mean_pos[1]+y_map_offset + y_lim_offset)])
            
        fig.savefig(os.path.join(dirname, f'track_{idx}.png'))
        ax.patches.pop()
        #ax.clear()
        
    return ax

def generateLocalizationFullGraph(data, frames, fig, ax, nusc_map, dirname, xlimits=[], ylimits=[]):
    
    N = len(frames)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    pf_scatter_pos_arr = np.zeros((200,2))
    
    x = np.arange(0,10)
    y = np.arange(0,10)
    pf_line, = ax.plot(x,y,color="blue",alpha=1,label="RadLoc")
    gt_line, = ax.plot(x,y,color="green",alpha=1,label="GT")
    imu_line, = ax.plot(x,y,color="red",alpha=1,label="IMU")
    pf_scatter = ax.scatter(x,y,color="blue",alpha=1,s=0.01)
    poly_plot = [None] * 20
    #for j in range(0,20):
        #poly_plot[j], = ax.plot(x,y,color="magenta",alpha=1)
    ax.legend(loc="upper left", prop={'size': 26})
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        gt_heading = np.deg2rad(video_data['heading'])
        pf_best_pos = mm_results['pf_best_pos']
        pf_best_theta = mm_results['pf_best_theta']
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        pf_mean_theta = mm_results['pf_mean_theta']
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        all_particles = mm_results['all_particles']
        cost_true = mm_results['cost_true']
        cost_mean = mm_results['cost_mean']
        cost_dyn_true = mm_results['cost_dyn_true']
        cost_dyn_mean = mm_results['cost_dyn_mean']
        pf_cov = mm_results['covariance'] * 2
        
        gt_track_pos_arr[idx, :] = gt_pos[0:2]
        pf_track_pos_arr[idx, :] = pf_mean_pos[0:2]
        imu_track_pos_arr[idx, :] = imu_pos[0:2]
        
        x_min = np.min(ego_path[:,0])
        x_max = np.max(ego_path[:,0])
        x_mean = 0.5*(x_min+x_max)
        y_min = np.min(ego_path[:,1])
        y_max = np.max(ego_path[:,1])
        y_mean = 0.5*(y_min+y_max)
        patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100

        if idx == 0:
            first_pos = [x_mean, y_mean]
            patch_size = patch_size
            #edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size)
            res_factor=10
            print(f"first_pos = {first_pos} patch_size = {patch_size}")
            edges = getLayer(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size,res_factor=res_factor)
            edges[edges==0] = 255
            edges[edges==1] = 200

            ax.imshow(edges, origin='lower', cmap='gray', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
            ax.grid(False)
            fig.savefig(os.path.join(dirname, f'map2.png'))

        pf_scatter_pos_arr[:,0] = [p["x"] for p in all_particles] + ego_path[0,0]
        pf_scatter_pos_arr[:,1] = [p["y"] for p in all_particles] + ego_path[0,1]
        res_factor = 1
        x_map_offset = -first_pos[0]+patch_size*0.5
        y_map_offset = -first_pos[1]+patch_size*0.5

        pf_line.set_xdata(res_factor*(x_map_offset+pf_track_pos_arr[:idx+1, 0]))
        pf_line.set_ydata(res_factor*(y_map_offset+pf_track_pos_arr[:idx+1, 1]))
        pf_scatter.set_offsets(np.column_stack((res_factor*(x_map_offset+pf_scatter_pos_arr[:, 0]), res_factor*(y_map_offset+pf_scatter_pos_arr[:, 1]))))
        #pf_scatter.set_ydata(res_factor*(y_map_offset+pf_scatter_pos_arr[:, 1]))
        gt_line.set_xdata(res_factor*(x_map_offset+gt_track_pos_arr[:idx+1, 0]))
        gt_line.set_ydata(res_factor*(y_map_offset+gt_track_pos_arr[:idx+1, 1]))
        imu_line.set_xdata(res_factor*(x_map_offset+imu_track_pos_arr[:idx+1, 0]))
        imu_line.set_ydata(res_factor*(y_map_offset+imu_track_pos_arr[:idx+1, 1]))
        #ax = confidence_ellipse(res_factor*(pf_mean_pos[0]+x_map_offset), res_factor*(pf_mean_pos[1]+y_map_offset), pf_cov, ax, edgecolor='blue')
        
        x_lim_offset = 1 * (5 + max(abs(imu_pos[0]-pf_mean_pos[0]), max(abs(gt_pos[0]-pf_mean_pos[0]), np.linalg.norm(pf_cov))))
        y_lim_offset = 1 * (5 + max(abs(imu_pos[1]-pf_mean_pos[1]), max(abs(gt_pos[1]-pf_mean_pos[1]), np.linalg.norm(pf_cov))))
        ax.set_xlim([res_factor * (pf_mean_pos[0]+x_map_offset - x_lim_offset), res_factor * (pf_mean_pos[0]+x_map_offset+x_lim_offset)])
        ax.set_ylim([res_factor*(pf_mean_pos[1]+y_map_offset - y_lim_offset), res_factor*(pf_mean_pos[1]+y_map_offset + y_lim_offset)])
            
        fig.savefig(os.path.join(dirname, f'track_{idx}.png'))
        #ax.patches.pop()
        #ax.clear()
        
    return ax

import gc
def generateDetectionsFullGraph(data, frames, nusc_map, dirname, xlimits=[], ylimits=[]):
    
    N = len(frames)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    pf_scatter_pos_arr = np.zeros((200,2))

    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        gt_heading = np.deg2rad(video_data['heading'])
        pf_best_pos = mm_results['pf_best_pos']
        pf_best_theta = mm_results['pf_best_theta']
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        pf_mean_theta = mm_results['pf_mean_theta']
        imu_pos = np.array(video_data["pos_imu"][0:3])
        ego_path = video_data["ego_path"][:,:]
        ego_trns = video_data["ego_trns"]
        all_particles = mm_results['all_particles']
        cost_true = mm_results['cost_true']
        cost_mean = mm_results['cost_mean']
        cost_dyn_true = mm_results['cost_dyn_true']
        cost_dyn_mean = mm_results['cost_dyn_mean']
        pf_cov = mm_results['covariance'] * 2
        print(f"gt_heading = {gt_heading}")
        
        gt_track_pos_arr[idx, :] = gt_pos[0:2]
        pf_track_pos_arr[idx, :] = pf_mean_pos[0:2]
        imu_track_pos_arr[idx, :] = imu_pos[0:2]
        
        fig, ax, pf_line0, gt_line0, imu_line0, pf_scatter, pf_line1, gt_line1, imu_line1, poly_plot, arrow_plot, traj_plot, patch_size, first_pos  = getFigure(ego_path, dirname, nusc_map)
        
         
        pf_scatter_pos_arr[:,0] = [p["x"] for p in all_particles] + ego_path[0,0]
        pf_scatter_pos_arr[:,1] = [p["y"] for p in all_particles] + ego_path[0,1]
        res_factor = 1
        x_map_offset = -first_pos[0]+patch_size*0.5
        y_map_offset = -first_pos[1]+patch_size*0.5
        
        
        ax[0] = plotLocalizationFullGraph(ax[0], idx, ego_path,
                                       x_map_offset, y_map_offset, res_factor,
                                       pf_line0, pf_scatter, gt_line0, imu_line0,
                                       gt_pos, imu_pos, pf_mean_pos, pf_cov, 
                                       pf_track_pos_arr, gt_track_pos_arr, imu_track_pos_arr, pf_scatter_pos_arr)
        
        gt_pose = np.array([gt_pos[0], gt_pos[1], gt_heading-1.57])
        ax[1] = plotDetectionsFullGraph(ax[1], idx, 
                                     ego_path, gt_pose,
                                     pf_track_pos_arr, gt_track_pos_arr, imu_track_pos_arr,
                                     res_factor, x_map_offset, y_map_offset,
                                     pf_line1, gt_line1, imu_line1,
                                     arrow_plot, traj_plot, poly_plot,
                                     polynoms, dynamic_tracks)
            
        fig.savefig(os.path.join(dirname, f'track_{t}.png'))
        ax[0].clear()
        ax[1].clear()
        plt.close(fig)
        gc.collect()
        #ax.patches.pop()
        #ax.clear()
        #for j in range(0,len(polynoms)):
        #    poly_plot[j].remove()
        #for j in range(0,len(dynamic_tracks)):
        #    if arrow_plot[j] is not None:
        #        arrow_plot[j].remove()
        #        traj_plot[j].remove()

def getFigure(ego_path, dirname, nusc_map):
    x_min = np.min(ego_path[:,0])
    x_max = np.max(ego_path[:,0])
    x_mean = 0.5*(x_min+x_max)
    y_min = np.min(ego_path[:,1])
    y_max = np.max(ego_path[:,1])
    y_mean = 0.5*(y_min+y_max)
    patch_size = int(max(abs(y_max-y_min), abs(x_max-x_min))) + 100
    first_pos = [x_mean, y_mean]
        
    pkl_file_path = os.path.join(dirname, f'map.pkl')
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            fig = pickle.load(f)
            ax = fig.axes
    else:
        fig, ax = plt.subplots(1,2,figsize=(20,12))
        
        
        res_factor0=10
        res_factor1=5
        print(f"first_pos = {first_pos} patch_size = {patch_size}")
        drivable_area = getLayer(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size,res_factor=res_factor0)
        drivable_area[drivable_area==0] = 255
        drivable_area[drivable_area==1] = 200
        ax[0].imshow(drivable_area, origin='lower', cmap='gray', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
        #drivable_area = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size,res_factor=res_factor1)
        #ax[0].imshow(drivable_area, origin='lower', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
        ax[0].grid(False)

        edges = getCombinedMap(nuscMap=nusc_map, worldRef=first_pos, patchSize=patch_size,res_factor=res_factor1)
        ax[1].imshow(edges, origin='lower', vmin=0, vmax=255, extent=[0,patch_size,0,patch_size])
        ax[1].grid(False)
        
        figure_data = {'figure': fig, 'axes': ax}

        # Save the figure data using pickle
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(fig, f)
        
    
    x = np.arange(0,10)
    y = np.arange(0,10)
    pf_line0, = ax[0].plot(x,y,color="blue",alpha=1,label="RadLoc")
    gt_line0, = ax[0].plot(x,y,color="green",alpha=1,label="GT")
    imu_line0, = ax[0].plot(x,y,color="red",alpha=1,label="IMU")
    pf_scatter = ax[0].scatter(x,y,color="blue",alpha=1,s=0.05)
    
    pf_line1, = ax[1].plot(x,y,color="blue",alpha=1,label="RadLoc")
    gt_line1, = ax[1].plot(x,y,color="green",alpha=1,label="GT")
    imu_line1, = ax[1].plot(x,y,color="red",alpha=1,label="IMU")
    
    poly_plot = [None] * 40
    arrow_plot = [None] * 40
    traj_plot = [None] * 40
    ax[0].legend(loc="upper left", prop={'size': 18})
    ax[1].legend(loc="upper left", prop={'size': 14})
    ax[0].set_xlabel('x [m]', fontsize=22)
    ax[0].set_ylabel('y [m]', fontsize=22)
    ax[1].set_xlabel('x [m]', fontsize=22)
    ax[1].set_ylabel('y [m]', fontsize=22)
    ax[0].tick_params(axis='x', labelsize=18)  # Adjust fontsize as needed
    ax[0].tick_params(axis='y', labelsize=18)  # Adjust fontsize as needed
    ax[1].tick_params(axis='x', labelsize=18)  # Adjust fontsize as needed
    ax[1].tick_params(axis='y', labelsize=18)  # Adjust fontsize as needed
    ax[0].set_title('Zoomed-In Pose Estimation Visualization', fontsize=28)
    ax[1].set_title('Estimated Localization and Sensor Detections', fontsize=28)
    
    return fig, ax, pf_line0, gt_line0, imu_line0, pf_scatter, pf_line1, gt_line1, imu_line1, poly_plot, arrow_plot, traj_plot, patch_size, first_pos

def plotLocalizationFullGraph(ax, idx, ego_path,
                             x_map_offset, y_map_offset, res_factor,
                              pf_line, pf_scatter, gt_line, imu_line,
                              gt_pos, imu_pos, pf_mean_pos, pf_cov, 
                             pf_track_pos_arr, gt_track_pos_arr, imu_track_pos_arr, pf_scatter_pos_arr):
    
    pf_line.set_xdata(res_factor*(x_map_offset+pf_track_pos_arr[:idx+1, 0]))
    pf_line.set_ydata(res_factor*(y_map_offset+pf_track_pos_arr[:idx+1, 1]))
    pf_scatter.set_offsets(np.column_stack((res_factor*(x_map_offset+pf_scatter_pos_arr[:, 0]), res_factor*(y_map_offset+pf_scatter_pos_arr[:, 1]))))
    #pf_scatter.set_ydata(res_factor*(y_map_offset+pf_scatter_pos_arr[:, 1]))
    gt_line.set_xdata(res_factor*(x_map_offset+gt_track_pos_arr[:idx+1, 0]))
    gt_line.set_ydata(res_factor*(y_map_offset+gt_track_pos_arr[:idx+1, 1]))
    imu_line.set_xdata(res_factor*(x_map_offset+imu_track_pos_arr[:idx+1, 0]))
    imu_line.set_ydata(res_factor*(y_map_offset+imu_track_pos_arr[:idx+1, 1]))
    #ax = confidence_ellipse(res_factor*(pf_mean_pos[0]+x_map_offset), res_factor*(pf_mean_pos[1]+y_map_offset), pf_cov, ax, edgecolor='blue')

    x_lim_offset = 1 * (5 + max(abs(imu_pos[0]-pf_mean_pos[0]), max(abs(gt_pos[0]-pf_mean_pos[0]), np.linalg.norm(pf_cov))))
    y_lim_offset = 1 * (5 + max(abs(imu_pos[1]-pf_mean_pos[1]), max(abs(gt_pos[1]-pf_mean_pos[1]), np.linalg.norm(pf_cov))))
    ax.set_xlim([res_factor * (pf_mean_pos[0]+x_map_offset - x_lim_offset), res_factor * (pf_mean_pos[0]+x_map_offset+x_lim_offset)])
    ax.set_ylim([res_factor*(pf_mean_pos[1]+y_map_offset - y_lim_offset), res_factor*(pf_mean_pos[1]+y_map_offset + y_lim_offset)])
    
    return ax
        
def plotDetectionsFullGraph(ax, idx, 
                            ego_path, gt_pos,
                            pf_track_pos_arr, gt_track_pos_arr, imu_track_pos_arr,
                            res_factor, x_map_offset, y_map_offset,
                            pf_line, gt_line, imu_line, 
                            arrow_plot, traj_plot, poly_plot,
                            polynoms, dynamic_tracks):
    pf_line.set_xdata(res_factor*(x_map_offset+pf_track_pos_arr[:idx+1, 0]))
    pf_line.set_ydata(res_factor*(y_map_offset+pf_track_pos_arr[:idx+1, 1]))
    gt_line.set_xdata(res_factor*(x_map_offset+gt_track_pos_arr[:idx+1, 0]))
    gt_line.set_ydata(res_factor*(y_map_offset+gt_track_pos_arr[:idx+1, 1]))
    imu_line.set_xdata(res_factor*(x_map_offset+imu_track_pos_arr[:idx+1, 0]))
    imu_line.set_ydata(res_factor*(y_map_offset+imu_track_pos_arr[:idx+1, 1]))

    x_min = 1e6
    x_max = -1e6
    y_min = 1e6
    y_max = -1e6
    #Draw polynomials
    for c,polynom in enumerate(polynoms):
        xx = np.linspace(polynom["x_start"], polynom["x_end"], 100)
        x_plot = xx if polynom["fxFlag"] else polynom["f"](xx)
        y_plot = polynom["f"](xx) if polynom["fxFlag"] else xx 
        x_plot, y_plot = getElementsInFOV(gt_pos, x_plot, y_plot)
        if not x_plot.any():
            continue
        poly_plot[c], = ax.plot(res_factor*(x_plot+x_map_offset),res_factor*(y_plot+y_map_offset),color="magenta",linewidth=1,label="") 
        if c == 0:
            poly_plot[c].set_label('Static Extended Tracks')
        
        x_min = min(x_min, np.min(x_plot))
        x_max = max(x_max, np.max(x_plot))
        y_min = min(y_min, np.min(y_plot))
        y_max = max(y_max, np.max(y_plot))
        
    
    dyn_offset = gt_track_pos_arr[idx, :] - imu_track_pos_arr[idx,:]
    #Draw Dynamic Tracks
    draw_label = False
    for j,trk in enumerate(dynamic_tracks):
        print(f"dyn_offset = {dyn_offset} j = {j} len(dynamic_tracks) = {len(dynamic_tracks)}")
        arrow_plot[j], traj_plot[j] = drawTrack(ax, trk, x_offset=(x_map_offset+dyn_offset[0]), y_offset=(y_map_offset+dyn_offset[1]), velThr=2, n_last_frames=10, color='cyan',label="",res_factor=res_factor)
        if arrow_plot[j] is None:
            continue
        if not draw_label:
            arrow_plot[j].set_label('Dynamic Tracks')
            draw_label = True
        
    ax.legend(loc="upper left", prop={'size': 20})

    
    x_min = min(x_min, gt_track_pos_arr[idx,0] - 20)
    x_max = max(x_max, gt_track_pos_arr[idx,0] + 20)
    y_min = min(y_min, gt_track_pos_arr[idx,1] - 20)
    y_max = max(y_max, gt_track_pos_arr[idx,1] + 20)
    
    incline = [0,0]
    if idx > 10:
        incline = (gt_track_pos_arr[idx]-gt_track_pos_arr[idx-10])/10
    
    if incline[0]:
        norm_incline = np.linalg.norm(incline)
        if incline[0] < 0:
            x_min = min(x_min, gt_track_pos_arr[idx,0] + incline[0] / norm_incline * 80)
        else:
            x_max = max(x_max, gt_track_pos_arr[idx,0] + incline[0] / norm_incline * 80)
        
        if incline[1] < 0:
            y_min = min(y_min, gt_track_pos_arr[idx,1] + incline[1] / norm_incline * 80)
        else:
            y_max = max(y_max, gt_track_pos_arr[idx,1] + incline[1] / norm_incline * 80)

    x_min = res_factor * x_min + x_map_offset
    x_max = res_factor * x_max + x_map_offset
    y_min = res_factor * y_min + y_map_offset
    y_max = res_factor * y_max + y_map_offset

    ax.set_xlim([min(x_min,x_max), max(x_min,x_max)])
    ax.set_ylim([min(y_min,y_max), max(y_min,y_max)])
    
    return ax

def getElementsInFOV(pose, x, y):
    angle = np.arctan2(y-pose[1], x-pose[0])
    #print(f"angle = {angle} pose[2] = {pose[2]}")
    x_in_fov = x[abs(pose[2]-angle) < 1.047]
    y_in_fov = y[abs(pose[2]-angle) < 1.047]

    return x_in_fov, y_in_fov

def generateVideo(name, dirname, fps=1):
    os.system("mkdir -p " + dirname)
    filenames = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("filenames", filenames)

    frame = cv2.imread(os.path.join(dirname, filenames[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)

    for filename in filenames:
        print("writing", filename)
        video.write(cv2.imread(os.path.join(dirname, filename)))

    cv2.destroyAllWindows()
    video.release()
    

def generateGraphAlongNumShapes(data, frames, ax):
    N = len(frames)
    timestamp_arr = np.linspace(0, N / 12.5, N + 1)[0:-1]
    pf_cross_track_errors_arr = np.zeros(N)
    pf_along_track_errors_arr = np.zeros(N)
    imu_cross_track_errors_arr = np.zeros(N)
    imu_along_track_errors_arr = np.zeros(N)
    gt_track_pos_arr = np.zeros((N,2))
    pf_track_pos_arr = np.zeros((N,2))
    imu_track_pos_arr = np.zeros((N,2))
    n_arcs = np.zeros(N)
    n_clothoids = np.zeros(N)
    n_corners = np.zeros(N)
    cov_cross_arr = np.zeros(N)
    cov_along_arr = np.zeros(N)
    
    for idx, t in enumerate(frames):
        video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = data.load(t)
        
        gt_pos = np.array(video_data['pos'])
        pf_mean_pos = np.array(mm_results['pf_mean_pos'])
        imu_pos = np.array(video_data["pos_imu"][0:2])
        ego_path = video_data["ego_path"][:,0:2]
        ego_trns = video_data["ego_trns"]
        pf_cov = np.array(mm_results['covariance']) * 4# + np.diag([0.01,0.01])
        heading = gt_pos[2]
        
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
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        cov_cross_arr[idx] = np.sqrt(np.dot(R,pf_cov)[1,1])
        cov_along_arr[idx] = np.sqrt(np.dot(R,pf_cov)[0,0])
        
        (lines, circles, clothoids, corners) = classifyShape(polynoms)
        n_arcs[idx] = len(circles)
        n_clothoids[idx] = len(clothoids)
        n_corners[idx] = len(corners)
    
    #Along-Track Err(t)
    ax[0].plot(timestamp_arr, np.zeros(N),color='green',alpha=0.6,label='GT', linewidth=3)
    ax[0].plot(timestamp_arr, pf_along_track_errors_arr, color='blue', label='GT-RadLoc', linewidth=3)
    ax[0].plot(timestamp_arr, imu_along_track_errors_arr, color='red', label='GT-INS', linewidth=3)
    ax[0].plot(timestamp_arr, -1*np.sqrt(cov_along_arr) * 4, color='orange', label='RadLoc Lon STD', linewidth=3)
    ax[0].plot(timestamp_arr, np.sqrt(cov_along_arr) * 4, color='orange', linewidth=3)
    
    #static tracks
    ax[1].plot(timestamp_arr, n_arcs,label='Arcs', linewidth=3)
    ax[1].plot(timestamp_arr, n_clothoids,label='Clothoids', linewidth=3)
    ax[1].plot(timestamp_arr, n_corners,label='Corners', linewidth=3)

    
    return ax
    

"""
def generateManueverGraph(pf_mean_pos, gt_pos, imu_pos, pf_cov, heading, tracks, nusc_map, ax, n_last_frames=10,sx=0,sy=0):
    pf_x_offset = pf_mean_pos[0] - imu_pos[0]
    pf_y_offset = pf_mean_pos[1] - imu_pos[1]
    gt_x_offset = gt_pos[0] - imu_pos[0]
    gt_y_offset = gt_pos[1] - imu_pos[1]

    drawLanes(ax[0], nusc_map, gt_pos)
    
    for trk in tracks:
        drawTrack(ax[0], trk, x_offset=pf_x_offset, y_offset=pf_y_offset, velThr=2, n_last_frames=n_last_frames)
        break
    drawLanes(ax[0], nusc_map, gt_pos, sx=sx, sy=sy)
    
    ax[0] = drawEgo(x0=gt_pos[0],y0=gt_pos[1],angle=heading,ax=ax[0],edgecolor='red', width=1.5, height=4)
    
    
    ax[1].scatter(pf_mean_pos[0], pf_mean_pos[1], s=10,color="blue",alpha=1,label="RadLoc")
    ax[1].scatter(gt_pos[0], gt_pos[1], s=10,color="green",alpha=1,label="GT")
    ax[1].scatter(imu_pos[0], imu_pos[1], s=10,color="red",alpha=1,label="INS")
    ax[1] = confidence_ellipse(pf_mean_pos[0], pf_mean_pos[1], pf_cov, ax[1], edgecolor='blue')
    
    return ax
"""
    
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
