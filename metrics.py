import numpy as np
from numpy.polynomial import polynomial as P
from tqdm import tqdm
import math

def innerProductPolynoms(f1, f2, xmin, xmax):
        f1mf2 = P.polysub(f1,f2)
        f1mf2 = P.polymul(f1mf2,f1mf2)
        val = P.polyint(f1mf2,lbnd=xmin)
        #print("val", val)
        ip = np.abs(val[0] + val[1]*(xmax-xmin)+val[2]*(xmax-xmin)**2+val[3]*(xmax-xmin)**3+val[4]*(xmax-xmin)**4+val[5]*(xmax-xmin)**5)
        return np.sqrt(ip)/abs(xmax-xmin)
    
def associatePolynomAndPrior(res, priors):
    ass_list = []
    polynoms = res
    
    for iprior,prior in enumerate(priors):
        for ipol,polynom in enumerate(polynoms):
            f1 = (prior["c"][0],prior["c"][1],prior["c"][2])
            f2 = (polynom["f"].c[2],polynom["f"].c[1],polynom["f"].c[0])
            xmin = max(prior["xmin"],polynom["x_start"])
            xmax = min(prior["xmax"],polynom["x_end"])
            dist = innerProductPolynoms(f1,f2,xmin,xmax)
            if dist < 2 and xmax-xmin > 5:
                ass_list.append((iprior,ipol))
                
    return ass_list

def ComputeCoeffErrors(res, prior, ipolynom):
    a0_err = np.zeros(len(res))
    a1_err = np.zeros(len(res))
    a2_err = np.zeros(len(res))
    for iframe,frame in enumerate(res):
        a0_err[iframe] = frame[ipolynom]["f"].c[2] - prior["c"][0]
        a1_err[iframe] = frame[ipolynom]["f"].c[1] - prior["c"][1]
        a2_err[iframe] = frame[ipolynom]["f"].c[0] - prior["c"][2]
        
    return a0_err, a1_err, a2_err
        
def ComputeCoeffMeanErrors(res, prior, ipolynom):
    a0_err, a1_err, a2_err = ComputeCoeffErrors(res, prior, ipolynom)
    a0_mean_err = np.mean(np.abs(a0_err))
    a1_mean_err = np.mean(np.abs(a1_err))
    a2_mean_err = np.mean(np.abs(a2_err))
    print("==================================================================================")
    print("==================================================================================")
    print(f"ComputeCoeffErrors: mean err for prior {prior} = ({a0_mean_err},{a1_mean_err},{a2_mean_err})")
    print("==================================================================================")
    print("==================================================================================")
    
    
def ComputePolynomDistance(res, prior, ipolynom):
    dist = np.zeros(len(res))
    for iframe,frame in enumerate(res):
        
        f1 = (prior["c"][0],prior["c"][1],prior["c"][2])
        f2 = (frame[ipolynom]["f"].c[2],frame[ipolynom]["f"].c[1],frame[ipolynom]["f"].c[0])
        xmin = max(prior["xmin"],frame[ipolynom]["x_start"])
        xmax = min(prior["xmax"],frame[ipolynom]["x_end"])
        dist[iframe] = innerProductPolynoms(f1,f2,xmin,xmax)
        
    return dist
        
def ComputePolynomMeanDistance(res, prior, ipolynom):
    dist = ComputePolynomDistance(res, prior, ipolynom)
    dist_mean_err = np.mean(dist)
    dist_max_err = np.max(dist)
    print("==================================================================================")
    print("==================================================================================")
    print(f"ComputePolynomDistance: dist_mean_err for prior {prior} = {dist_mean_err}")
    print(f"ComputePolynomDistance: dist_max_err for prior {prior} = {dist_max_err}")
    print("==================================================================================")
    print("==================================================================================")
    
def ComputeOverlappingArea(res, prior, ipolynom):
    overlap = np.zeros(len(res))
    for iframe,frame in enumerate(res):
        xmin = max(prior["xmin"],frame[ipolynom]["x_start"])
        xmax = min(prior["xmax"],frame[ipolynom]["x_end"])
        
        xminmin = min(prior["xmin"],frame[ipolynom]["x_start"])
        xmaxmax = max(prior["xmax"],frame[ipolynom]["x_end"])
        overlap[iframe] = np.abs(xmax-xmin)/np.abs(xmaxmax-xminmin)
        
    overlap_mean = np.mean(overlap)
    overlap_min = np.min(overlap)
    print("==================================================================================")
    print("==================================================================================")
    print(f"ComputeOverlappingArea: overlap_mean for prior {prior} = {overlap_mean}")
    print(f"ComputeOverlappingArea: overlap_min for prior {prior} = {overlap_min}")
    print("==================================================================================")
    print("==================================================================================")
    
def ComputeNonOverlappingArea(res, prior, ipolynom):
    nonoverlap = np.zeros(len(res))
    for iframe,frame in enumerate(res):
        dmin = prior["xmin"]-frame[ipolynom]["x_start"]
        dmax = frame[ipolynom]["x_end"] - prior["xmax"]
        
        dmin = dmin if dmin > 0 else 0
        dmax = dmin if dmax > 0 else 0
        
        nonoverlap[iframe] = dmax + dmin
        
    nonoverlap_mean = np.mean(nonoverlap)
    nonoverlap_max = np.max(nonoverlap)
    print("==================================================================================")
    print("==================================================================================")
    print(f"ComputeNonOverlappingArea: onon_verlap_mean for prior {prior} = {nonoverlap_mean}")
    print(f"ComputeNonOverlappingArea: non_overlap_max for prior {prior} = {nonoverlap_max}")
    print("==================================================================================")
    print("==================================================================================")
    
    
    
def calcTrackPosition(ego_path, ego_trns, gt_pos, pf_pos, imu_pos):
    #GT position
    gt_cross_track = 0
    it = np.argmin(np.linalg.norm(ego_path - np.array(gt_pos),axis=1),axis=0)
    it = max(1, min(it, ego_trns.shape[0]-2))
    gt_along_track = np.copy(ego_trns[it])
    #PF position
    it = np.argmin(np.linalg.norm(ego_path - np.array(pf_pos),axis=1),axis=0)
    it = max(1, min(it, ego_path.shape[0]-2))
    x,y,x1,y1,x2,y2 = pf_pos[0],pf_pos[1],ego_path[it-1][0], ego_path[it-1][1], ego_path[it+1][0], ego_path[it+1][1]
    d=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
    pf_cross_track = np.sign(d) * np.linalg.norm(ego_path[it]-pf_pos) #np.linalg.norm(np.cross(p2-p1, p1-pf_pos))/np.linalg.norm(p2-p1)
    pf_along_track = np.copy(ego_trns[it])
    #IMU position
    it = np.argmin(np.linalg.norm(ego_path - np.array(imu_pos),axis=1),axis=0)
    it = max(1, min(it, ego_path.shape[0]-2))
    x,y,x1,y1,x2,y2 = imu_pos[0],imu_pos[1],ego_path[it-1][0], ego_path[it-1][1], ego_path[it+1][0], ego_path[it+1][1]
    d=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
    imu_cross_track = np.sign(d) * np.linalg.norm(ego_path[it]-imu_pos)
    imu_along_track = np.copy(ego_trns[it])

    return np.array([gt_cross_track, gt_along_track]), np.array([pf_cross_track, pf_along_track]), np.array([imu_cross_track, imu_along_track])
    

def calc_rmse(ego_path, ego_trns, gt_pos, pf_mean_pos, imu_pos):
    gt_track_pos, pf_track_pos, imu_track_pos = calcTrackPosition(ego_path, ego_trns, gt_pos[0:2], pf_mean_pos, imu_pos)
    print("gt_track_pos[1]", gt_track_pos[1])
    pf_track_errors = pf_track_pos - gt_track_pos
    imu_track_errors = imu_track_pos - gt_track_pos

    return pf_track_errors, imu_track_errors
    
def calc_acc_rmse(ego_path, ego_trns, pf_mean_pos, imu_pos, N):
    pf_rmse_lateral = np.zeros(N)
    pf_rmse_longitudal = np.zeros(N)
    imu_rmse_lateral = np.zeros(N)
    imu_rmse_longitudal = np.zeros(N)
    for i in tqdm(range(0,N)):
        pf_track_errors, imu_track_errors = calc_rmse(ego_path, ego_trns, ego_path[i,:], pf_mean_pos[i,:], imu_pos[i,:])
        print("pf_track_errors",pf_track_errors,"imu_track_errors",imu_track_errors)
        pf_rmse_lateral[i] = pf_track_errors[0]
        pf_rmse_longitudal[i] = pf_track_errors[1]
        imu_rmse_lateral[i] = imu_track_errors[0]
        imu_rmse_longitudal[i] = imu_track_errors[1]

    print(f"PF RMSE lateral: {math.sqrt((1. / N) * np.sum(pf_rmse_lateral**2))}")
    print(f"PF RMSE longitudal: {math.sqrt((1. / N) * np.sum(pf_rmse_longitudal**2))}")
    print(f"IMU RMSE lateral: {math.sqrt((1. / N) * np.sum(imu_rmse_lateral**2))}")
    print(f"IMU RMSE longitudal: {math.sqrt((1. / N) * np.sum(imu_rmse_longitudal**2))}")
