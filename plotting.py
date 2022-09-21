import matplotlib.pyplot as plt
import numpy as np
from tools import *
import cv2
import re
from map_utils import getRoadBorders, getCombinedMap
import nuscenes.map_expansion.arcline_path_utils as path_utils
from matplotlib.ticker import MaxNLocator

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
        label = kwargs.pop('label', f"Object {idx}")
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
                ax[1].plot(y_plot,x_plot,linewidth=3, label=f"Ext track {ipol}")
            else:
                ax[1].plot(x_plot,y_plot,linewidth=3, label=f"Ext track {ipol}")
        
    return fig, ax
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
