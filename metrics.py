import numpy as np
from numpy.polynomial import polynomial as P

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
    #print(res[-1][ipolynom]["f"],prior)
    a0_err = np.zeros([len(res), 3])
    a1_err = np.zeros([len(res), 3])
    a2_err = np.zeros([len(res), 3])
    for iframe,frame in enumerate(res):
        a0_err[iframe,0] = abs(frame[ipolynom]["f"].c[2] - prior["c"][0])
        a1_err[iframe,1] = abs(frame[ipolynom]["f"].c[1] - prior["c"][1])
        a2_err[iframe,2] = abs(frame[ipolynom]["f"].c[0] - prior["c"][2])
        
    a0_mean_err = np.mean(a0_err)
    a1_mean_err = np.mean(a1_err)
    a2_mean_err = np.mean(a2_err)
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
        #print("f1",f1,"f2",f2,"xmin",xmin,"xmax",xmax,"dist[iframe]",dist[iframe])
        
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
