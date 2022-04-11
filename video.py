import matplotlib.pyplot as plt
import numpy as np
from tools import *
import os
from os import listdir
from os.path import isfile, join
import cv2
import re

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
            patch_angle = 0
            self.first_pos = pos
            self.patch_size = 500
            patch_box = (pos[0], pos[1], self.patch_size, self.patch_size)
            layer_names = ['walkway']
            canvas_size = (self.patch_size, self.patch_size)
            map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
            #self.ax[1,2].set_title("Map", fontsize=20)
            if self.scene == 5:
                self.ax[1,2].set_xlim([0,0.5*self.patch_size+50])
                self.ax[1,2].set_ylim([0.5*self.patch_size-50,self.patch_size])
            elif self.scene == 1:
                self.ax[1,2].set_xlim([0.5*self.patch_size+-50,self.patch_size])
                self.ax[1,2].set_ylim([100,self.patch_size-100])
            for i in range(len(map_mask)):
                self.ax[1,2].imshow(map_mask[i], origin='lower')
                self.ax[1,2].text(canvas_size[0] * 0.5, canvas_size[1] * 1.1, layer_names[i])
                self.ax[1,2].grid(False)
        for c,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[1,2].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="blue",linewidth=4) 
        self.ax[1,2].scatter(pos[0]-self.first_pos[0]+self.patch_size*0.5,pos[1]-self.first_pos[1]+self.patch_size*0.5,s=10,color="red")
            
        self.counter += 1
        plt.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
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
        self.dir_name = "nuscenes_images_debug_{}".format(scene)
        self.history = history
        self.counter = 0
        self.scene = scene
        
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
            
        self.ax[1,0].set_title("Points that generated a new polynom", fontsize=20)
        self.ax[1,0].set_xlim(xlim)
        self.ax[1,0].set_ylim(ylim)
        self.ax[1,0] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[1,0])
        for c,pair in enumerate(debug["pgpol"]):
            x_plot = np.linspace(pair["polynom"]["x_start"], pair["polynom"]["x_end"], 100)
            y_plot = pair["polynom"]["f"](x_plot)
            self.ax[1,0].plot(y_plot,x_plot,linewidth=3,linestyle='--',color=self.colors[c])
            self.ax[1,0].scatter(pair["points"][0,:], pair["points"][1,:],c=[self.colors[c]]*pair["points"].shape[1])
            
        self.ax[1,1].set_title("Points that updated point tracks", fontsize=20)
        self.ax[1,1].set_xlim(xlim)
        self.ax[1,1].set_ylim(ylim)
        self.ax[1,1] = drawEgo(x0=pos[0],y0=pos[1],angle=heading,ax=self.ax[1,1])
        for pair in debug["mupoi"]:
            self.ax[1,1].scatter(pair["measurements"][0], pair["measurements"][1],color='blue')
            self.ax[1,1].scatter(pair["points"][0], pair["points"][1],color='orange')
            
        self.ax[1,2].set_title("Points that updated extended tracks", fontsize=20)
        self.ax[1,2].set_xlim(xlim)
        self.ax[1,2].set_ylim(ylim)
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
        plt.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
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
        self.fig, self.ax = plt.subplots(2,2,figsize=(30,14))
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
        
        nanArray = np.array(np.ones(250))
        nanArray[:] = np.nan

        self.cross_track_pos = np.copy(nanArray)
        self.graph_cross_track, = self.ax[0,0].plot([], [], color="blue",linewidth=3)
        self.ax[0,0].set_title("Cross-Track Error", fontsize=20)
        self.ax[0,0].set_xlim([0,N])
        self.ax[0,0].set_ylim([0,8])
        
        self.along_track_pos = np.copy(nanArray)
        self.graph_along_track, = self.ax[0,1].plot([], [], color="blue",linewidth=3)
        self.ax[0,1].set_title("Along-Track Error", fontsize=20)
        self.ax[0,1].set_xlim([0,N])
        self.ax[0,1].set_ylim([0,8])
        
    def calcTrackError(self, gt_pos, gt_heading, pf_pos):
        R = np.array([[np.cos(gt_heading), -np.sin(gt_heading)], [np.sin(gt_heading), np.cos(gt_heading)]])
        xy_errors = np.abs(gt_pos - pf_pos)
        track_errors = np.dot(R, xy_errors)
        return abs(track_errors[1]), abs(track_errors[0])
        
    def save(self, idx, video_data, mm_results, polynoms, nusc_map):
        gt_pos = video_data['pos']
        gt_heading = np.deg2rad(video_data['heading']-90)
        pf_best_pos = mm_results['pf_best_pos']
        pf_best_theta = mm_results['pf_best_theta']
        pf_mean_pos = mm_results['pf_mean_pos']
        pf_mean_theta = mm_results['pf_mean_theta']
        
        self.x_lim_min = min(self.x_lim_min, min(gt_pos[0], pf_mean_pos[0]))
        self.x_lim_max = max(self.x_lim_max, max(gt_pos[0], pf_mean_pos[0]))
        self.y_lim_min = min(self.y_lim_min, min(gt_pos[1], pf_mean_pos[1]))
        self.y_lim_max = max(self.y_lim_max, max(gt_pos[1], pf_mean_pos[1]))
        
        xlim = [self.x_lim_min,self.x_lim_max]
        ylim = [self.y_lim_min,self.y_lim_max]
        
        cross_track_error, along_track_error = self.calcTrackError(gt_pos[0:2], gt_heading, pf_mean_pos)
        
        #Cross-Track err(t)
        self.cross_track_pos[self.counter] = cross_track_error
        self.graph_cross_track.set_data(range(self.counter+1), self.cross_track_pos[0:self.counter+1])
        
        #Along-Track err(t)
        self.along_track_pos[self.counter] = along_track_error
        self.graph_along_track.set_data(range(self.counter+1), self.along_track_pos[0:self.counter+1])

        #GT vs. PF position(t)
        self.ax[1,0].set_title("GT vs. PF position frame={}".format(idx), fontsize=20)
        self.ax[1,0].scatter(gt_pos[0], gt_pos[1],color="green",s=2)
        self.ax[1,0].scatter(pf_mean_pos[0], pf_mean_pos[1],color="red",s=2)
        self.ax[1,0].set_xlim(xlim)
        self.ax[1,0].set_ylim(ylim)
        
        #map
        if self.counter == 0:
            patch_angle = 0
            self.first_pos = gt_pos
            self.patch_size = 500
            patch_box = (gt_pos[0], gt_pos[1], self.patch_size, self.patch_size)
            layer_names = ['walkway']
            canvas_size = (self.patch_size, self.patch_size)
            map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
            #self.ax[1,2].set_title("Map", fontsize=20)
            if self.scene == 5:
                self.ax[1,1].set_xlim([0,0.5*self.patch_size+50])
                self.ax[1,1].set_ylim([0.5*self.patch_size-50,self.patch_size])
            elif self.scene == 1:
                self.ax[1,1].set_xlim([0.5*self.patch_size+-50,self.patch_size])
                self.ax[1,1].set_ylim([100,self.patch_size-100])
            for i in range(len(map_mask)):
                self.ax[1,1].imshow(map_mask[i], origin='lower')
                self.ax[1,1].text(canvas_size[0] * 0.5, canvas_size[1] * 1.1, layer_names[i])
                self.ax[1,1].grid(False)
        for c,polynom in enumerate(polynoms):
            x_plot = np.linspace(polynom["x_start"], polynom["x_end"], 100)
            y_plot = polynom["f"](x_plot)
            self.ax[1,1].plot(x_plot-self.first_pos[0]+self.patch_size*0.5,y_plot-self.first_pos[1]+self.patch_size*0.5,color="blue",linewidth=4) 
        self.ax[1,1].scatter(gt_pos[0]-self.first_pos[0]+self.patch_size*0.5,gt_pos[1]-self.first_pos[1]+self.patch_size*0.5,s=10,color="red")

        plt.savefig(os.path.join(self.dir_name, f'track_{idx}.png'))
        self.counter += 1
        
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