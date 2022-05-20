import matplotlib.pyplot as plt
import numpy as np
from tools import *
import os
from os import listdir
from os.path import isfile, join
import cv2
import re
from map_utils import getRoadBorders, getCombinedMap
from matplotlib.ticker import MaxNLocator

class SimulationVideo:
    def __init__(self):
        self.fig, self.ax = plt.subplots(2,3,figsize=(40,15))
        self.colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime"]
        self.x_lim_min = -50
        self.x_lim_max = 50
        self.y_lim_min = 0
        self.y_lim_max = 100
        
    def drawEllipses(self, measurements, key1, key2, ax, n=5, edgecolor='firebrick'):
        ellipses = range(0,measurements[key1].shape[0])
        ellipses = random.sample(ellipses, n)
        for i in ellipses:
            cov = measurements[key2][i]
            ax = confidence_ellipse(measurements[key1][i,1], measurements[key1][i,0], cov, ax, edgecolor=edgecolor)

        return ax
    
    def drawPrior(self, ax, prior, xlim, **kwargs):
        x,y = createPolynom(prior[0],prior[1],prior[2],xstart=xlim[0],xend=xlim[1])
        ax.plot(y,x,**kwargs)
        
    def save(self, idx, prior, measurements, points, polynoms, debug, pos=[0,0], heading=0):
        self.x_lim_min = min(min(self.x_lim_min, np.min(measurements["polynom"][:,1])), np.min(measurements["other"][:,1]))
        self.x_lim_max = max(max(self.x_lim_max, np.max(measurements["polynom"][:,1])), np.max(measurements["other"][:,1]))
        self.y_lim_min = 0
        self.y_lim_max = max(max(self.y_lim_max, np.max(measurements["polynom"][:,0])), np.max(measurements["other"][:,0]))
        
        xlim = [self.x_lim_min,self.x_lim_max]
        ylim = [self.y_lim_min,self.y_lim_max]
        self.ax[0,0].set_title("Measurements frame={}".format(idx))
        self.ax[0,0].scatter(measurements["polynom"][:,1],measurements["polynom"][:,0])
        self.ax[0,0].scatter(measurements["other"][:,1],measurements["other"][:,0])
        self.ax[0,0] = self.drawEllipses(measurements=measurements,key1="polynom",key2="dpolynom",ax=self.ax[0,0],edgecolor='firebrick')
        self.ax[0,0] = self.drawEllipses(measurements=measurements,key1="other",key2="dother",ax=self.ax[0,0],edgecolor='blue')
        self.ax[0,0].set_xlim(xlim)
        self.ax[0,0].set_ylim(ylim)
        self.drawPrior(ax=self.ax[0,0],prior=prior,xlim=ylim,label='true',linewidth=5)
        self.ax[0,0] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[0,0],edgecolor='red')
        
        self.ax[0,1].set_title("Point tracks frame={}".format(idx))
        self.ax[0,1].scatter(points[:,1], points[:,0])
        self.ax[0,1].set_xlim(xlim)
        self.ax[0,1].set_ylim(ylim)
        self.drawPrior(ax=self.ax[0,1],prior=prior,xlim=ylim,label='true',linewidth=3,linestyle='--')
        self.ax[0,1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[0,1])
        
        self.ax[0,2].set_title("Extended tracks frame={}".format(idx))
        self.ax[0,2].set_xlim(xlim)
        self.ax[0,2].set_ylim(ylim)
        self.drawPrior(ax=self.ax[0,2],prior=prior,xlim=ylim,label='true',linewidth=3,linestyle='--') 
        self.ax[0,2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[0,2],edgecolor='red')
        for polynom in polynoms:
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[0,2].plot(y_plot,x_plot,linewidth=10)
            
        self.ax[1,0].set_title("Points that generated a new polynom frame={}".format(idx))
        self.ax[1,0].set_xlim(xlim)
        self.ax[1,0].set_ylim(ylim)
        self.drawPrior(ax=self.ax[1,0],prior=prior,xlim=ylim,label='true',linewidth=3,linestyle='--') 
        self.ax[1,0] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[1,0])
        for c,pair in enumerate(debug["pgpol"]):
            x_plot = np.linspace(pair["polynom"]["x_start"], pair["polynom"]["x_end"], 100)
            y_plot = pair["polynom"]["f"](x_plot)
            self.ax[1,0].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=self.colors[c])
            self.ax[1,0].scatter(pair["points"][1,:], pair["points"][0,:],c=[self.colors[c]]*pair["points"].shape[1])
            
        self.ax[1,1].set_title("Points that updated point tracks frame={}".format(idx))
        self.ax[1,1].set_xlim(xlim)
        self.ax[1,1].set_ylim(ylim)
        self.drawPrior(ax=self.ax[1,1],prior=prior,xlim=ylim,label='true',linewidth=3,linestyle='--') 
        self.ax[1,1] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[1,1])
        for pair in debug["mupoi"]:
            self.ax[1,1].scatter(pair["measurements"][1], pair["measurements"][0],color='blue')
            self.ax[1,1].scatter(pair["points"][1], pair["points"][0],color='orange')
            
        self.ax[1,2].set_title("Points that updated extended tracks frame={}".format(idx))
        self.ax[1,2].set_xlim(xlim)
        self.ax[1,2].set_ylim(ylim)
        self.drawPrior(ax=self.ax[1,2],prior=prior,xlim=ylim,label='true',linewidth=3,linestyle='--') 
        self.ax[1,2] = drawEgo(x0=pos[1],y0=pos[0],angle=heading,ax=self.ax[1,2])
        unique_polynoms = set(d['id'] for d in debug["mupol"])
        print("len(unique_polynoms)", len(unique_polynoms))
        for c,upol in enumerate(unique_polynoms):
            first = True
            xy = []
            for pair in debug["mupol"]:
                if pair["id"] == upol:
                    if first:
                        x_plot = np.linspace(pair["polynom"][3], pair["polynom"][4], 100)
                        y_plot = pair["polynom"][0] + pair["polynom"][1]*x_plot + pair["polynom"][2]*x_plot**2
                        self.ax[1,2].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=self.colors[c])
                        first = False
                    xy.append(np.array([pair["measurements"][0], pair["measurements"][1]]))
            xy = np.array(xy).T
            self.ax[1,2].scatter(xy[1,:], xy[0,:],c=self.colors[c])
            
        plt.savefig(f'images/track_{idx}.png')
        self.ax[0,0].clear()
        self.ax[0,1].clear()
        self.ax[0,2].clear()
        self.ax[1,0].clear()
        self.ax[1,1].clear()
        self.ax[1,2].clear()
        
    def generate(self, name, fps=1):
        image_folder = "images"
        filenames = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        frame = cv2.imread(os.path.join(image_folder, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)
        
        for filename in filenames:
            print(filename)
            video.write(cv2.imread(os.path.join(image_folder, filename)))
        
        cv2.destroyAllWindows()
        video.release()
        
class NuscenesVideo:
    def __init__(self, scene=5, history=False):
        self.fig, self.ax = plt.subplots(2,3,figsize=(30,14))
        self.x_lim_min = 1e6
        self.x_lim_max = 1e-6
        self.y_lim_min = 1e6
        self.y_lim_max = 1e-6
        
        self.prior_x_lim_min = 1e6
        self.prior_x_lim_max = -1e6
        self.prior_y_lim_min = 1e6
        self.prior_y_lim_max = -1e6
        
        self.colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime"]
        self.dir_name = "images/nuscenes_images_{}".format(scene)
        os.system("mkdir -p " + self.dir_name)
        self.history = history
        self.counter = 0
        self.scene = scene
        
    def save(self, idx, video_data, polynoms, nusc_map, video_with_priors=False):
        pc = video_data['pc']
        img = video_data['img']
        prior = video_data['prior']
        pos = video_data['pos']
        heading = video_data['heading']
        
        self.x_lim_min = min(self.x_lim_min, np.min(pc[:,0]))
        self.x_lim_max = max(self.x_lim_max, np.max(pc[:,0]))
        self.y_lim_min = min(self.y_lim_min, np.min(pc[:,1]))
        self.y_lim_max = max(self.y_lim_max, np.max(pc[:,1]))
        
        self.ax[0,0].set_title("Measurements frame={}".format(idx), fontsize=20)
        self.ax[0,0].scatter(pc[:,0],pc[:,1],color='blue',s=1)
        self.ax[0,0].set_xlim([self.x_lim_min,self.x_lim_max])
        self.ax[0,0].set_ylim([self.y_lim_min,self.y_lim_max])
        self.ax[0,0] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[0,0],edgecolor='red', width=1.5, height=4)
        if video_with_priors:
            for lane in prior:
                x = lane["x"]
                self.ax[0,0].plot(x, lane["poly"](x)) 
        
        self.ax[0,1].set_title("Camera frame={}".format(idx), fontsize=20)
        self.ax[0,1].imshow(img)
        self.ax[0,1].grid(None)
        self.ax[0,1].axis('off')
        
        if 1:
            self.ax[0,2].set_xlim([self.x_lim_min,self.x_lim_max])
            self.ax[0,2].set_ylim([self.y_lim_min,self.y_lim_max])
            self.ax[0,2] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[0,2],edgecolor='red', width=1.5, height=4)
            for c,polynom in enumerate(polynoms):
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                self.ax[0,2].plot(x_plot,y_plot,linewidth=3,color=self.colors[c])
        else:
            self.ax[0,2].set_title("Prior frame={}".format(idx), fontsize=20)
            #self.ax[2].set_xlim([self.x_lim_min,self.x_lim_max])
            #self.ax[2].set_ylim([self.y_lim_min,self.y_lim_max])
            for lane in prior:
                x = lane["x"]
                self.ax[0,2].plot(x,lane["poly"](x)) 
                self.ax[0,2].scatter(pos[0], pos[1])
                self.prior_x_lim_min = min(self.prior_x_lim_min, np.min(x))
                self.prior_x_lim_max = max(self.prior_x_lim_max, np.max(x))
                self.prior_y_lim_min = min(self.prior_y_lim_min, np.min(lane["poly"](x)))
                self.prior_y_lim_max = max(self.prior_y_lim_max, np.max(lane["poly"](x)))
            self.ax[0,2].set_xlim([self.prior_x_lim_min,self.prior_x_lim_max])
            self.ax[0,2].set_ylim([self.prior_y_lim_min,self.prior_y_lim_max])
            
        if 0:
            #x(t)
            self.ax[1,0].set_title("x(t)", fontsize=20)
            self.ax[1,0].scatter(np.ones(pc.shape[0])*self.counter, pc[:,0],color="blue",s=2)
            self.ax[1,0].set_xlim([0,200])
            self.ax[1,0].set_ylim([self.x_lim_min,self.x_lim_max])

            #y(t)
            self.ax[1,1].set_title("y(t)", fontsize=20)
            self.ax[1,1].scatter(np.ones(pc.shape[0])*self.counter, pc[:,1],color="blue",s=2)
            self.ax[1,1].set_xlim([0,200])
            self.ax[1,1].set_ylim([self.y_lim_min,self.y_lim_max])
        
        #map
        if self.counter == 0:
            self.first_pos = pos
            self.patch_size = 500
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size)
            self.ax[1,2].imshow(edges, origin='lower')
            self.ax[1,2].grid(False)
        for c,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[1,2].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="blue",linewidth=4) 
        self.ax[1,2].scatter(pos[0]-self.first_pos[0]+self.patch_size*0.5,pos[1]-self.first_pos[1]+self.patch_size*0.5,s=10,color="red")
            
        self.counter += 1
        self.fig.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
        if not self.history:
            self.ax[0,0].clear()  
            self.ax[0,2].clear()
        else:
            self.ax[0,0].scatter(pc[:,0],pc[:,1],color='gray',s=1)
            for polynom in polynoms:
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                self.ax[0,2].plot(x_plot,y_plot,linewidth=3,color='gray')
            
        self.ax[0,1].clear()
        
    def generate(self, name, fps=1):
        filenames = [f for f in os.listdir(self.dir_name) if os.path.isfile(os.path.join(self.dir_name, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        frame = cv2.imread(os.path.join(self.dir_name, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)
        
        for filename in filenames:
            #print(filename)
            video.write(cv2.imread(os.path.join(self.dir_name, filename)))
        
        cv2.destroyAllWindows()
        video.release()
        
class NuscenesVideoDebug:
    def __init__(self, scene=5, history=False):
        self.fig, self.ax = plt.subplots(2,3,figsize=(30,14))
        self.x_lim_min = 1e6
        self.x_lim_max = 1e-6
        self.y_lim_min = 1e6
        self.y_lim_max = 1e-6
        
        self.prior_x_lim_min = 1e6
        self.prior_x_lim_max = -1e6
        self.prior_y_lim_min = 1e6
        self.prior_y_lim_max = -1e6
        
        self.colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime"]
        self.dir_name = "images/nuscenes_images_debug_{}".format(scene)
        os.system("mkdir -p " + self.dir_name)
        self.history = history
        self.counter = 0
        self.scene = scene
        self.first = True
        
    def save(self, idx, video_data, polynoms, pointTracks, nusc_map, debug, video_with_priors=False):
        pc = video_data['pc']
        img = video_data['img']
        prior = video_data['prior']
        pos = video_data['pos']
        heading = video_data['heading']
        
        self.x_lim_min = min(self.x_lim_min, np.min(pc[:,0]))
        self.x_lim_max = max(self.x_lim_max, np.max(pc[:,0]))
        self.y_lim_min = min(self.y_lim_min, np.min(pc[:,1]))
        self.y_lim_max = max(self.y_lim_max, np.max(pc[:,1]))
        
        xlim = [self.x_lim_min,self.x_lim_max]
        ylim = [self.y_lim_min,self.y_lim_max]
        
        self.ax[0,0].set_title("Measurements frame={}".format(idx), fontsize=20)
        self.ax[0,0].scatter(pc[:,0],pc[:,1],color='blue',s=1)
        self.ax[0,0].set_xlim(xlim)
        self.ax[0,0].set_ylim(ylim)
        self.ax[0,0] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[0,0],edgecolor='red', width=1.5, height=4)
        if video_with_priors:
            for lane in prior:
                x = lane["x"]
                self.ax[0,0].plot(x, lane["poly"](x)) 
        
        self.ax[0,1].set_title("Point tracks", fontsize=20)
        self.ax[0,1].scatter(pointTracks[:,0], pointTracks[:,1])
        self.ax[0,1].set_xlim(xlim)
        self.ax[0,1].set_ylim(ylim)
        self.ax[0,1] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[0,1])
        
        if 1:
            self.ax[0,2].set_xlim(xlim)
            self.ax[0,2].set_ylim(ylim)
            self.ax[0,2] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[0,2],edgecolor='red', width=1.5, height=4)
            for c,polynom in enumerate(polynoms):
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                self.ax[0,2].plot(x_plot,y_plot,linewidth=3,color=self.colors[c])
        else:
            self.ax[0,2].set_title("Prior frame={}".format(idx), fontsize=20)
            #self.ax[2].set_xlim([self.x_lim_min,self.x_lim_max])
            #self.ax[2].set_ylim([self.y_lim_min,self.y_lim_max])
            for lane in prior:
                x = lane["x"]
                self.ax[0,2].plot(x,lane["poly"](x)) 
                self.ax[0,2].scatter(pos[0], pos[1])
                self.prior_x_lim_min = min(self.prior_x_lim_min, np.min(x))
                self.prior_x_lim_max = max(self.prior_x_lim_max, np.max(x))
                self.prior_y_lim_min = min(self.prior_y_lim_min, np.min(lane["poly"](x)))
                self.prior_y_lim_max = max(self.prior_y_lim_max, np.max(lane["poly"](x)))
            self.ax[0,2].set_xlim([self.prior_x_lim_min,self.prior_x_lim_max])
            self.ax[0,2].set_ylim([self.prior_y_lim_min,self.prior_y_lim_max])
            
        if self.first:
            self.first = False
            self.first_pos = pos
            
        self.ax[1,0].set_title("Points that generated a new polynom", fontsize=20)
        print("xlim", xlim, "self.first_pos[0]", self.first_pos[0])
        self.ax[1,0].set_xlim([x-self.first_pos[0] for x in xlim])
        self.ax[1,0].set_ylim([y-self.first_pos[1] for y in ylim])
        self.ax[1,0] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[1,0])
        for c,pair in enumerate(debug["pgpol"]):
            x_plot = np.linspace(pair["polynom"]["x_start"], pair["polynom"]["x_end"], 100)
            y_plot = pair["polynom"]["f"](x_plot)
            self.ax[1,0].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=self.colors[c])
            self.ax[1,0].scatter(pair["points"][0,:], pair["points"][1,:],c=[self.colors[c]]*pair["points"].shape[1])
            
        self.ax[1,1].set_title("Points that updated point tracks", fontsize=20)
        self.ax[1,1].set_xlim([x-self.first_pos[0] for x in xlim])
        self.ax[1,1].set_ylim([y-self.first_pos[1] for y in ylim])
        self.ax[1,1] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[1,1])
        for pair in debug["mupoi"]:
            self.ax[1,1].scatter(pair["measurements"][0], pair["measurements"][1],color='blue')
            self.ax[1,1].scatter(pair["points"][0], pair["points"][1],color='orange')
            
        self.ax[1,2].set_title("Points that updated extended tracks", fontsize=20)
        self.ax[1,2].set_xlim([x-self.first_pos[0] for x in xlim])
        self.ax[1,2].set_ylim([y-self.first_pos[1] for y in ylim])
        self.ax[1,2] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[1,2])
        unique_polynoms = set(d['id'] for d in debug["mupol"])
        print("len(unique_polynoms)", len(unique_polynoms))
        for c,upol in enumerate(unique_polynoms):
            first = True
            xy = []
            for pair in debug["mupol"]:
                if pair["id"] == upol:
                    if first:
                        x_plot = np.linspace(pair["polynom"][3], pair["polynom"][4], 100)
                        y_plot = pair["polynom"][0] + pair["polynom"][1]*x_plot + pair["polynom"][2]*x_plot**2
                        self.ax[1,2].plot(x_plot,y_plot,linewidth=3,linestyle='--',color=self.colors[c])
                        first = False
                    xy.append(np.array([pair["measurements"][0], pair["measurements"][1]]))
            xy = np.array(xy).T
            self.ax[1,2].scatter(xy[0,:], xy[1,:],c=self.colors[c])
            
        self.counter += 1
        self.fig.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
        if not self.history:
            self.ax[0,0].clear()  
            self.ax[0,2].clear()
        else:
            self.ax[0,0].scatter(pc[:,0],pc[:,1],color='gray',s=1)
            for polynom in polynoms:
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                self.ax[0,2].plot(x_plot,y_plot,linewidth=3,color='gray')
            
        self.ax[0,1].clear()
        self.ax[1,0].clear()
        self.ax[1,1].clear()
        self.ax[1,2].clear()
        
    def generate(self, name, fps=1):
        filenames = [f for f in os.listdir(self.dir_name) if os.path.isfile(os.path.join(self.dir_name, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        frame = cv2.imread(os.path.join(self.dir_name, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)
        
        for filename in filenames:
            #print(filename)
            video.write(cv2.imread(os.path.join(self.dir_name, filename)))
        
        cv2.destroyAllWindows()
        video.release()
        
class PFVideo:
    def __init__(self, scene=5, history=False, N=410):
        self.fig, self.ax = plt.subplots(2,3,figsize=(30,14))
        self.fig2, self.ax2 = plt.subplots(1,1,figsize=(30,14))
        self.fig3, self.ax3 = plt.subplots(1,3,figsize=(30,14))
        self.x_lim_min = 1e6
        self.x_lim_max = 1e-6
        self.y_lim_min = 1e6
        self.y_lim_max = 1e-6
        
        self.prior_x_lim_min = 1e6
        self.prior_x_lim_max = -1e6
        self.prior_y_lim_min = 1e6
        self.prior_y_lim_max = -1e6
        
        self.colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime"]
        self.dir_name = "images/pf_images_{}".format(scene)
        os.system("mkdir -p " + self.dir_name)
        self.history = history
        self.counter = 0
        self.scene = scene
        
        nanArray = np.ones((N,3))
        nanArray[:,:] = np.nan

        #self.cross_track_pos = np.copy(nanArray)
        #self.gt_graph_cross_track, = self.ax[0,0].scatter([], [], color="green",linewidth=3,label='GT')
        #self.pf_graph_cross_track, = self.ax[0,0].scatter([], [], color="blue",linewidth=3,label='PF')
        #self.imu_graph_cross_track, = self.ax[0,0].scatter([], [], color="red",linewidth=3,label='IMU')
        self.ax[0,0].set_title("Cross-Track Position", fontsize=20)
        self.ax[0,0].set_xlim([0,N])
        #self.ax[0,0].set_ylim([0,8])
        self.ax[0,0].set(xlabel='frame #', ylabel='Pos [m]')
        self.ax[0,0].legend(loc="upper left")
        
        self.cross_track_pos_err = np.copy(nanArray[:, 0:2])
        self.pf_graph_cross_track_err, = self.ax[1,0].plot([], [], color="blue",linewidth=3,label='GT-PF')
        self.imu_graph_cross_track_err, = self.ax[1,0].plot([], [], color="red",linewidth=3,label='GT-IMU')
        self.ax[1,0].set_title("Cross-Track Error", fontsize=20)
        self.ax[1,0].set_xlim([0,N])
        self.ax[1,0].set_ylim([-4,4])
        self.ax[1,0].set(xlabel='frame #', ylabel='signed err [m]')
        self.ax[1,0].legend(loc="upper left")
        
        #self.along_track_pos = np.copy(nanArray)
        #self.gt_graph_along_track, = self.ax[0,1].scatter([], [], color="green",linewidth=3,label='GT')
        #self.pf_graph_along_track, = self.ax[0,1].scatter([], [], color="blue",linewidth=3,label='PF')
        #self.imu_graph_along_track, = self.ax[0,1].scatter([], [], color="red",linewidth=3,label='IMU')
        self.ax[0,1].set_title("Along-Track Position", fontsize=20)
        self.ax[0,1].set_xlim([0,N])
        #self.ax[0,1].set_ylim([0,8])
        self.ax[0,1].set(xlabel='frame #', ylabel='Pos [m]')
        self.ax[0,1].legend(loc="upper left")
        
        self.along_track_pos_err = np.copy(nanArray[:, 0:2])
        self.gt_graph_along_track_err, = self.ax[1,1].plot([], [], color="blue",linewidth=3,label='GT-PF')
        self.pf_graph_along_track_err, = self.ax[1,1].plot([], [], color="red",linewidth=3,label='GT-IMU')
        self.ax[1,1].set_title("Along-Track Error", fontsize=20)
        self.ax[1,1].set_xlim([0,N])
        self.ax[1,1].set_ylim([-8,8])
        self.ax[1,1].set(xlabel='frame #', ylabel='signed err [m]')
        self.ax[1,1].legend(loc="upper left")
        
        self.ax[0,2].set_title("GT track and polynoms on Map", fontsize=20)
        self.ax[1,2].set_title("IMU and PF tracks on map", fontsize=20)
        
        self.history_pf_x = None
        self.history_pf_y = None
        
    def calcTrackPosition(self, ego_path, ego_trns, gt_pos, pf_pos, imu_pos):
        #GT position
        gt_cross_track = 0
        it = np.argmin(np.linalg.norm(ego_path - np.array(gt_pos),axis=1),axis=0)
        gt_along_track = np.copy(ego_trns[it])
        #PF position
        it = np.argmin(np.linalg.norm(ego_path - np.array(pf_pos),axis=1),axis=0)
        x,y,x1,y1,x2,y2 = pf_pos[0],pf_pos[1],ego_path[it-1][0], ego_path[it-1][1], ego_path[it+1][0], ego_path[it+1][1]
        d=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        pf_cross_track = np.sign(d) * np.linalg.norm(ego_path[it]-pf_pos) #np.linalg.norm(np.cross(p2-p1, p1-pf_pos))/np.linalg.norm(p2-p1)
        pf_along_track = np.copy(ego_trns[it])
        #IMU position
        it = np.argmin(np.linalg.norm(ego_path - np.array(imu_pos),axis=1),axis=0)
        x,y,x1,y1,x2,y2 = imu_pos[0],imu_pos[1],ego_path[it-1][0], ego_path[it-1][1], ego_path[it+1][0], ego_path[it+1][1]
        d=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        imu_cross_track = np.sign(d) * np.linalg.norm(ego_path[it]-imu_pos)
        imu_along_track = np.copy(ego_trns[it])
        
        return np.array([gt_cross_track, gt_along_track]), np.array([pf_cross_track, pf_along_track]), np.array([imu_cross_track, imu_along_track])
        
    def save(self, idx, video_data, mm_results, polynoms, nusc_map):
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
        
        self.x_lim_min = min(self.x_lim_min, min(gt_pos[0], pf_mean_pos[0]))
        self.x_lim_max = max(self.x_lim_max, max(gt_pos[0], pf_mean_pos[0]))
        self.y_lim_min = min(self.y_lim_min, min(gt_pos[1], pf_mean_pos[1]))
        self.y_lim_max = max(self.y_lim_max, max(gt_pos[1], pf_mean_pos[1]))
        
        xlim = [self.x_lim_min,self.x_lim_max]
        ylim = [self.y_lim_min,self.y_lim_max]
        
        gt_track_pos, pf_track_pos, imu_track_pos = self.calcTrackPosition(ego_path, ego_trns, gt_pos[0:2], pf_mean_pos, imu_pos)
        pf_track_errors = pf_track_pos - gt_track_pos
        imu_track_errors = imu_track_pos - gt_track_pos
        
        #Cross-Track Position(t)
        self.ax[0,0].scatter(self.counter+1, gt_track_pos[0],color='green',alpha=0.6,label='GT')
        self.ax[0,0].scatter(self.counter+1, pf_track_pos[0],color='blue',alpha=0.6,label='PF')
        self.ax[0,0].scatter(self.counter+1, imu_track_pos[0],color='red',alpha=0.6,label='IMU')
        if self.counter == 0:
            self.ax[0,0].legend(loc="upper right")
        
        #Cross-Track Err(t)
        self.cross_track_pos_err[self.counter, :] = np.array([pf_track_errors[0], imu_track_errors[0]])
        self.pf_graph_cross_track_err.set_data(range(self.counter+1), self.cross_track_pos_err[0:self.counter+1, 0])
        self.imu_graph_cross_track_err.set_data(range(self.counter+1), self.cross_track_pos_err[0:self.counter+1, 1])
        
        #Along-Track Position(t)
        self.ax[0,1].scatter(self.counter+1, gt_track_pos[1],color='green',alpha=0.6,label='GT')
        self.ax[0,1].scatter(self.counter+1, pf_track_pos[1],color='blue',alpha=0.6,label='PF')
        self.ax[0,1].scatter(self.counter+1, imu_track_pos[1],color='red',alpha=0.6,label='IMU')
        if self.counter == 0:
            self.ax[0,1].legend(loc="upper right")
        
        #Along-Track Err(t)
        self.along_track_pos_err[self.counter, :] = np.array([pf_track_errors[1], imu_track_errors[1]])
        self.gt_graph_along_track_err.set_data(range(self.counter+1), self.along_track_pos_err[0:self.counter+1, 0])
        self.pf_graph_along_track_err.set_data(range(self.counter+1), self.along_track_pos_err[0:self.counter+1, 1])
        
        #Polynoms and GT on map
        if self.counter == 0:
            x_min = np.min(ego_path[:,0])
            x_max = np.max(ego_path[:,0])
            x_mean = 0.5*(x_min+x_max)
            y_min = np.min(ego_path[:,1])
            y_max = np.max(ego_path[:,1])
            y_mean = 0.5*(y_min+y_max)

            self.first_pos = [x_mean, y_mean]
            self.patch_size = 500
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size)
            self.map = edges

            self.ax[0,2].imshow(edges, origin='lower')
            self.ax[0,2].grid(False)

            self.ax[1,2].imshow(edges, origin='lower')
            self.ax[1,2].legend(loc="upper left")
            self.ax[1,2].grid(False)
            self.ax2.imshow(edges, origin='lower')
            self.ax2.grid(False)
            self.ax[0,2].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax[0,2].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
            self.ax[1,2].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax[1,2].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
            self.ax2.set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax2.set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
            self.ax3[1].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax3[1].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
            self.ax3[2].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax3[2].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
        
        for c,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[0,2].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=2,label="polynoms") 
        self.ax[0,2].scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="green",label="GT")
        if self.counter == 0:
            self.ax[0,2].legend(loc="upper left")
        
        #IMU, PF on map
        self.ax[1,2].scatter(pf_mean_pos[0]-self.first_pos[0]+self.patch_size*0.5,pf_mean_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="blue",label="PF")
        self.ax[1,2].scatter(imu_pos[0]-self.first_pos[0]+self.patch_size*0.5,imu_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="red",label="IMU")
        if self.counter == 0:
            self.ax[1,2].legend(loc="upper left")

        #PF state
        pf_x = [p['x'] for p in all_particles]+ego_path[0,0]-self.first_pos[0]+self.patch_size*0.5
        pf_y = [p['y'] for p in all_particles]+ego_path[0,1]-self.first_pos[1]+self.patch_size*0.5
        if self.history_pf_x is not None:
            self.ax2.scatter(self.history_pf_x, self.history_pf_y,s=1,color="gray",alpha=1)
        self.ax2.scatter(pf_x, pf_y,s=1,color="blue",alpha=0.5)
        self.history_pf_x = pf_x
        self.history_pf_y = pf_y
        
        #Cost function
        n_polynoms = len(polynoms)
        if n_polynoms > 0:
            print("n_polynoms", n_polynoms, "cost_true", cost_true, "cost_mean", cost_mean)
            self.ax3[0].plot(range(1,n_polynoms+1),cost_true,color="green",alpha=1, label="GT")
            self.ax3[0].plot(range(1,n_polynoms+1),cost_mean,color="blue",alpha=1, label="PF")
            self.ax3[0].legend(loc="upper right")
            self.ax3[0].set_xlim([0,n_polynoms+1])
            self.ax3[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            
            self.ax3[1].imshow(self.map, origin='lower')
            self.ax3[1].grid(False)
            self.ax3[2].imshow(self.map, origin='lower')
            self.ax3[2].grid(False)
            for c,polynom in enumerate(polynoms):
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                self.ax3[1].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=2) 
                self.ax3[1].text(x_plot[0]-self.first_pos[0]+self.patch_size*0.5 + 2, y_plot[0]-self.first_pos[1]+self.patch_size*0.5+2, f"{c+1}", size=10)
            self.ax3[1].scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=8,color="green",label="GT")

            self.ax3[1].legend(loc="upper left")

            for c,polynom in enumerate(polynoms):
                x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
                y_plot = polynom["f"](x_plot)
                #translate
                x_plot -= gt_pos[0] - pf_mean_pos[0]
                y_plot -= gt_pos[1] - pf_mean_pos[1]
                self.ax3[2].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=2) 
                self.ax3[2].text(x_plot[0]-self.first_pos[0]+self.patch_size*0.5 + 2, y_plot[0]-self.first_pos[1]+self.patch_size*0.5+2, f"{c+1}", size=10)
            self.ax3[2].scatter(pf_mean_pos[0]-self.first_pos[0]+self.patch_size*0.5,pf_mean_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=8,color="blue",label="PF")
            
            self.ax3[2].legend(loc="upper left")
        
        self.counter += 1

        self.fig.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
        self.fig2.savefig(os.path.join(self.dir_name, f'track_{idx}_pf.png'))
        self.fig3.savefig(os.path.join(self.dir_name, f'track_{idx}_cost.png'))
        
        self.ax3[0].clear()
        self.ax3[1].clear()
        self.ax3[2].clear()
        
    def generate(self, name, fps=1):
        filenames = [f for f in os.listdir(self.dir_name) if os.path.isfile(os.path.join(self.dir_name, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        frame = cv2.imread(os.path.join(self.dir_name, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)
        
        for filename in filenames:
            #print(filename)
            video.write(cv2.imread(os.path.join(self.dir_name, filename)))
        
        cv2.destroyAllWindows()
        video.release()
        
        
class PFXYVideo:
    def __init__(self, scene=5, history=False, N=410):
        self.fig, self.ax = plt.subplots(2,3,figsize=(30,14))
        self.x_lim_min = 1e6
        self.x_lim_max = 1e-6
        self.y_lim_min = 1e6
        self.y_lim_max = 1e-6
        
        self.prior_x_lim_min = 1e6
        self.prior_x_lim_max = -1e6
        self.prior_y_lim_min = 1e6
        self.prior_y_lim_max = -1e6
        
        self.colors = ['blue','orange','green','red','black','pink','yellow','purple',"brown","firebrick","coral","lime"]
        self.dir_name = "images/pf_xy_images_{}".format(scene)
        os.system("mkdir -p " + self.dir_name)
        self.history = history
        self.counter = 0
        self.scene = scene
        
        nanArray = np.ones((N,3))
        nanArray[:,:] = np.nan

        self.ax[0,0].set_title("X Position", fontsize=20)
        self.ax[0,0].set_xlim([0,N])
        self.ax[0,0].set(xlabel='frame #', ylabel='X [m]')
        self.ax[0,0].legend(loc="upper left")
        
        self.x_pos_err = np.copy(nanArray[:, 0:2])
        self.pf_graph_x_err, = self.ax[1,0].plot([], [], color="blue",linewidth=3,label='GT-PF')
        self.imu_graph_x_err, = self.ax[1,0].plot([], [], color="red",linewidth=3,label='GT-IMU')
        self.ax[1,0].set_title("X position Error", fontsize=20)
        self.ax[1,0].set_xlim([0,N])
        self.ax[1,0].set_ylim([-4,4])
        self.ax[1,0].set(xlabel='frame #', ylabel='signed err [m]')
        self.ax[1,0].legend(loc="upper left")

        self.ax[0,1].set_title("Y Position", fontsize=20)
        self.ax[0,1].set_xlim([0,N])
        self.ax[0,1].set(xlabel='frame #', ylabel='Y [m]')
        self.ax[0,1].legend(loc="upper left")
        
        self.y_pos_err = np.copy(nanArray[:, 0:2])
        self.gt_graph_y_err, = self.ax[1,1].plot([], [], color="blue",linewidth=3,label='GT-PF')
        self.pf_graph_y_err, = self.ax[1,1].plot([], [], color="red",linewidth=3,label='GT-IMU')
        self.ax[1,1].set_title("Y position Error", fontsize=20)
        self.ax[1,1].set_xlim([0,N])
        self.ax[1,1].set_ylim([-8,8])
        self.ax[1,1].set(xlabel='frame #', ylabel='signed err [m]')
        self.ax[1,1].legend(loc="upper left")
        
        self.ax[0,2].set_title("GT track and polynoms on Map", fontsize=20)
        self.ax[1,2].set_title("IMU and PF tracks on map", fontsize=20)
        
        self.history_pf_x = None
        self.history_pf_y = None  
        
    def save(self, idx, video_data, mm_results, polynoms, nusc_map):
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
        
        self.x_lim_min = min(self.x_lim_min, min(gt_pos[0], pf_mean_pos[0]))
        self.x_lim_max = max(self.x_lim_max, max(gt_pos[0], pf_mean_pos[0]))
        self.y_lim_min = min(self.y_lim_min, min(gt_pos[1], pf_mean_pos[1]))
        self.y_lim_max = max(self.y_lim_max, max(gt_pos[1], pf_mean_pos[1]))
        
        xlim = [self.x_lim_min,self.x_lim_max]
        ylim = [self.y_lim_min,self.y_lim_max]
        
        pf_xy_errors = pf_mean_pos - gt_pos[0:2]
        imu_xy_errors = imu_pos - gt_pos[0:2]
        
        #X Position(t)
        self.ax[0,0].scatter(self.counter+1, gt_pos[0],color='green',alpha=0.6,label='GT')
        self.ax[0,0].scatter(self.counter+1, pf_mean_pos[0],color='blue',alpha=0.6,label='PF')
        self.ax[0,0].scatter(self.counter+1, imu_pos[0],color='red',alpha=0.6,label='IMU')
        if self.counter == 0:
            self.ax[0,0].legend(loc="upper right")
        
        #X Err(t)
        self.x_pos_err[self.counter, :] = np.array([pf_xy_errors[0], imu_xy_errors[0]])
        self.pf_graph_x_err.set_data(range(self.counter+1), self.x_pos_err[0:self.counter+1, 0])
        self.imu_graph_x_err.set_data(range(self.counter+1), self.x_pos_err[0:self.counter+1, 1])
        
        #Along-Track Position(t)
        self.ax[0,1].scatter(self.counter+1, gt_pos[1],color='green',alpha=0.6,label='GT')
        self.ax[0,1].scatter(self.counter+1, pf_mean_pos[1],color='blue',alpha=0.6,label='PF')
        self.ax[0,1].scatter(self.counter+1, imu_pos[1],color='red',alpha=0.6,label='IMU')
        if self.counter == 0:
            self.ax[0,1].legend(loc="upper right")
        
        #Along-Track Err(t)
        self.y_pos_err[self.counter, :] = np.array([pf_xy_errors[1], imu_xy_errors[1]])
        self.gt_graph_y_err.set_data(range(self.counter+1), self.y_pos_err[0:self.counter+1, 0])
        self.pf_graph_y_err.set_data(range(self.counter+1), self.y_pos_err[0:self.counter+1, 1])
        
        #Polynoms and GT on map
        if self.counter == 0:
            x_min = np.min(ego_path[:,0])
            x_max = np.max(ego_path[:,0])
            x_mean = 0.5*(x_min+x_max)
            y_min = np.min(ego_path[:,1])
            y_max = np.max(ego_path[:,1])
            y_mean = 0.5*(y_min+y_max)

            self.first_pos = [x_mean, y_mean]
            self.patch_size = 500
            edges = getCombinedMap(nuscMap=nusc_map, worldRef=self.first_pos, patchSize=self.patch_size)

            self.ax[0,2].imshow(edges, origin='lower')
            self.ax[0,2].grid(False)
            
            self.ax[1,2].imshow(edges, origin='lower')
            self.ax[1,2].legend(loc="upper left")
            self.ax[1,2].grid(False)
            self.ax[0,2].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax[0,2].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
            self.ax[1,2].set_xlim([self.patch_size/2 - (x_mean-x_min) - 50,self.patch_size/2 + (x_max-x_mean) + 50])
            self.ax[1,2].set_ylim([self.patch_size/2 - (y_mean-y_min) - 50,self.patch_size/2 + (y_max-y_mean) + 50])
        
        for c,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[0,2].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="magenta",linewidth=2,label="polynoms") 
        self.ax[0,2].scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="green",label="GT")
        if self.counter == 0:
            self.ax[0,2].legend(loc="upper left")
        
        #IMU, PF on map
        self.ax[1,2].scatter(pf_mean_pos[0]-self.first_pos[0]+self.patch_size*0.5,pf_mean_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="blue",label="PF")
        self.ax[1,2].scatter(imu_pos[0]-self.first_pos[0]+self.patch_size*0.5,imu_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=3,color="red",label="IMU")
        if self.counter == 0:
            self.ax[1,2].legend(loc="upper left")
        
        self.counter += 1

        self.fig.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
        
    def generate(self, name, fps=1):
        filenames = [f for f in os.listdir(self.dir_name) if os.path.isfile(os.path.join(self.dir_name, f))]
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        frame = cv2.imread(os.path.join(self.dir_name, filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(name, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), frameSize=(width,height),fps=fps)
        
        for filename in filenames:
            #print(filename)
            video.write(cv2.imread(os.path.join(self.dir_name, filename)))
        
        cv2.destroyAllWindows()
        video.release()