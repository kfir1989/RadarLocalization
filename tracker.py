import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
from numpy.polynomial import polynomial as P
from utils import *
import detector
import scipy
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
    
class StaticTracker:
    def __init__(self):
        self.ext_object_list = []
        self.ext_data_associator = ExtObjectDataAssociator(deltaL=10)
        self.pnt_object_list = []
        self.pnt_data_associator = PointObjectDataAssociator(delta=2)
        self.eta = 30
        self.frame_idx = 0
        self.polynom_list = []
        self.k = 8
        self.pnt_max_non_update_iterations = 8
        self.ext_max_non_update_iterations = 10
        self.max_decline_factor = 100
        self.system_rotated_flag = False
        
    def getDebugInfo(self):
        return self.debug
        
    def run(self, z, dz, prior):
        self.frame_idx += 1
        self.debug = {"pgpol": [], "mupoi": [], "mupol": []}
        print("Number of point tracks before run()", len(self.pnt_object_list))
        print("Number of extended tracks before run()", len(self.ext_object_list))
        
        #prediction
        for ext_track in self.ext_object_list:
            ext_track.predict()
        for pnt_track in self.pnt_object_list:
            pnt_track.predict()
        
        Ge, Gp = self.p2t(z)
        z, dz = self.trackUpdate(z, dz, Ge, Gp)
        self.trackInitPnt(z,dz)
            
        if self.frame_idx > 2:
            self.generateExtObject(prior)
        self.trackMaintenance()     
        
        return self.getPoints(), self.getPolynoms()
        
    def p2t(self, measurements):
        Gp = np.zeros((len(measurements), len(self.pnt_object_list)))
        Ge = np.zeros((len(measurements), len(self.ext_object_list)))
        for idx_z, z in enumerate(measurements):
            z = z.reshape(-1,1)
            for idx_track, ext_track in enumerate(self.ext_object_list):
                x = ext_track.getStateVector()
                x_pred = ext_track.getPredictedStateVector()
                zx = z[0,:]if ext_track.getFxFlag() else z[1,:]
                zy = z[1,:]if ext_track.getFxFlag() else z[0,:]
                y_pred = x_pred[0] + x_pred[1]*zx + x_pred[2]*zx**2
                P_pred = ext_track.getPredictedCovarianceMatrix()
                Ha = np.array([1, zx, zx**2])
                S_pred = np.dot(np.dot(Ha, P_pred[0:3,0:3]), Ha.T)
                Ge[idx_z,idx_track] = self.ext_data_associator.calcLikelihood(zx, zy, y_pred, S_pred, x)
                    
            for idx_track, pnt_track in enumerate(self.pnt_object_list):
                x = pnt_track.getStateVector()
                x_pred = pnt_track.getPredictedStateVector()
                #innov_cov = pnt_track.getInnovationCovarianceMatrix()
                innov_cov = pnt_track.getCovarianceMatrix()
                Gp[idx_z,idx_track] = self.pnt_data_associator.calcLikelihood(z, x_pred, innov_cov)
        
        return Ge, Gp
    
    def trackUpdate(self, z, dz, Ge, Gp):
        assigned_meas_list = []
        while(1): #Iterate over GE and GP matrices
            i_meas, lp, le, i_trk_p, i_trk_e = self.getBestAssociation(Ge, Gp)
            if(le == 0 and lp == 0):
                break
            self.zeroOutAssociation(Gp, i_meas, i_trk_p) # Clear from point association matrix
            Ge[i_meas, :] = 0 # Clear from extended association matrix, only measurement!
            x, y = z[i_meas]
            cov = np.diag([dz[i_meas][0,0], dz[i_meas][1,1]])
            assigned_meas_list.append(i_meas)
            ratio = np.sqrt(lp)/le
            #print("lp",lp,"le",le)
            if ratio > self.eta:
                i_trk = i_trk_p
                #print("Updating point track = ", i_trk)
                self.pnt_object_list[i_trk].update(z=np.array([x,y]).T,cov=cov, current_frame_idx=self.frame_idx)
                self.debug["mupoi"].append({"measurements": np.array([x,y]).T, "points":self.pnt_object_list[i_trk].getStateVector()})
            else:
                i_trk = i_trk_e
                trk = self.ext_object_list[i_trk]
                x = x if trk.getFxFlag() else z[i_meas][1]
                y = y if trk.getFxFlag() else z[i_meas][0]
                trk.counter_update += 1
                Hse = np.array([1 if x < trk.getStateVector()[3] else 0, 1 if x > trk.getStateVector()[4] else 0])
                H = np.array([[0, 0, 0, Hse[0], Hse[1]],[1, x, x**2, 0, 0]])
                R = cov
                #print("R",R)
                print("Updating extended object track = ", i_trk)
                self.ext_object_list[i_trk].update(np.array([x,y]).T, current_frame_idx=self.frame_idx, H=H, R=R)
                self.debug["mupol"].append({"measurements": np.array([x,y]).T, "polynom":self.ext_object_list[i_trk].getStateVector(), "id":i_trk, "fxFlag": self.ext_object_list[i_trk].getFxFlag()})
                #print("P after update", self.ext_object_list[i_trk].getStateCovarianceMatrix())
                
        for i_meas in sorted(assigned_meas_list, reverse=True):
            z = np.delete(z, (i_meas), axis=0)
            dz = np.delete(dz, (i_meas), axis=0)
        
        return z, dz
        
    def trackInitPnt(self, z, dz):
        #print("trackInitPnt")
        for x,cov in zip(z,dz):
            new_trk = PointObjectTrack(x=x.reshape(1,-1), P=cov, create_frame_idx=self.frame_idx)
            self.pnt_object_list.append(new_trk)
            
    def trackInitExt(self, polynom, generated_points, fxFlag=True):
        #calc arc length: if too short, don't initiate new track
        integral_expression = np.array([1+polynom["f"].c[1]**2, 4*polynom["f"].c[0]*polynom["f"].c[1], 4*polynom["f"].c[0]**2])
        f = lambda x:integral_expression[0]+integral_expression[1]*x+integral_expression[2]*x**2
        curve_length, _ = scipy.integrate.quad(f, polynom["x_start"], polynom["x_end"])
        #print(f"polynom={polynom['f']} integral_expression={integral_expression} curve_length={curve_length}")
        if curve_length < 5:
            print(f"Extended track is too short curve_length={curve_length}")
            return False
        x = np.array([polynom["f"].c[2], polynom["f"].c[1], polynom["f"].c[0], polynom["x_start"], polynom["x_end"]]).T
        trk_similar_test = True
        if (not trk_similar_test or not self.isTrkSimilar(x)):
            new_trk = ExtendedObjectTrack(x=x,P=polynom["P"],create_frame_idx=self.frame_idx,fxFlag=fxFlag)
            print("created an extended object!", x)
            self.ext_object_list.append(new_trk)
            return True
        else:
            return False
            
    def deleteTrack(self, track_list, indices):
        delete_indices  = np.unique(indices)
        #print("deleteTrack is called. track list length:", len(track_list), "indices", indices)
        for index in sorted(delete_indices, reverse=True):
            #print("delete point track!")
            del track_list[index]
            
    def generateExtObject(self, priors):
        PM = np.zeros((len(self.pnt_object_list),len(self.pnt_object_list),len(priors)))
        for iprior,prior in enumerate(priors):
            PM[:,:,iprior] = self.createProbabilityMatrixExt(self.pnt_object_list, prior)
        #print(P)
        delete_indices = []
        while(1):
            #j = np.argmax(np.sum(P,0), axis=None)
            (j,ip) = np.unravel_index(np.argmax(np.sum(PM,0), axis=None), PM.shape[1:])  # returns a tuple
            P = PM[:,:,ip]
            if(P[j,j] > 0):
                nk = np.count_nonzero(P[:,j])
                selected_pnts = np.where(P[:,j]>0)
                if nk > self.k:
                    fx_flag = priors[ip]["fx"]
                    xy = np.zeros((2,nk))
                    for i_pnt, pnt in enumerate(selected_pnts[0]):
                        #print("i_pnt", pnt)
                        xy[:,i_pnt] = np.squeeze(self.pnt_object_list[pnt].getStateVector(fx_flag),axis=1)
                        delete_indices.append(pnt)
                    
                    w = np.squeeze(P[selected_pnts,j])
                    #print(w.shape)
                    fit, cov = np.polyfit(xy[0,:], xy[1,:], 2, cov=True, w=w)
                    covP = np.zeros((5,5))
                    covP[0,0] = cov[2,2]
                    covP[0,1] = cov[2,1]
                    covP[0,2] = cov[2,0]
                    covP[1,0] = cov[1,2]
                    covP[1,1] = cov[1,1]
                    covP[1,2] = cov[1,0]
                    covP[2,0] = cov[0,2]
                    covP[2,1] = cov[0,1]
                    covP[2,2] = cov[0,0]
                    covP[3,3] = 5
                    covP[4,4] = 5
                    covP = covP * 10
                    f = np.poly1d(fit)
                    if not fx_flag:
                        print("Opening flipped polynom!!!")
                    polynom = {"f":f,"x_start":min(xy[0,:]),"x_end":max(xy[0,:]),"P": covP, "fxFlag": fx_flag}
                    #print("polynom was generated!", polynom)
                    status = self.trackInitExt(polynom, xy.T, fxFlag=fx_flag)
                    if status:
                        self.debug["pgpol"].append({"points": xy, "polynom":polynom})
                    
                PM = self.zeroOutAssociation(PM,selected_pnts[0],j)
            else:
                break
                
        self.deleteTrack(self.pnt_object_list, delete_indices)
        
    def trackMaintenance(self):
        pnt_delete_list = []
        ext_delete_list = []
        for i_trk,trk in enumerate(self.ext_object_list):
            if self.frame_idx - trk.getLastUpdateFrameIdx() > self.ext_max_non_update_iterations:
                #print("i_trk", i_trk, "self.frame_idx", self.frame_idx, "trk.getLastUpdateFrameIdx()", trk.getLastUpdateFrameIdx())
                ext_delete_list.append(i_trk)
            elif abs(trk.getStateVector()[1]) > 1000: ##TODO: Deal with large declines
                ext_delete_list.append(i_trk)
                
        for i_trk,trk in enumerate(self.pnt_object_list):
            if self.frame_idx - trk.getLastUpdateFrameIdx() > self.pnt_max_non_update_iterations:
                pnt_delete_list.append(i_trk)
                
        self.deleteTrack(self.ext_object_list, ext_delete_list)
        self.deleteTrack(self.pnt_object_list, pnt_delete_list)
        
    def translatePolynom(self, polynom, trns, rot):
        #translate polynom
        a0,a1,a2 = polynom[0],polynom[1],polynom[2]
        x = np.array([polynom[3],0.5*(polynom[3]+polynom[4]),polynom[4]])
        y = np.array([a0+a1*x[0]+a2*x[0]**2,a0+a1*x[1]+a2*x[1]**2,a0+a1*x[2]+a2*x[2]**2])
        x -= trns[0]
        y -= trns[1]
        if rot:
            #rotate 
            A = [[1, -y[0],y[0]**2],[1, -y[1], y[1]**2],[1, -y[2], y[2]**2]]
            co = np.linalg.inv(A).dot(x)
            a0,a1,a2 = co[0],co[1],co[2]
        else:
            a2 = (x[0]*(y[2]-y[1])+x[1]*(y[0]-y[2])+x[2]*(y[1]-y[0]))/((x[0]-x[1])*(x[0]-x[2])*(x[1]-x[2]))
            a1 = (y[1]-y[0])/(x[1]-x[0])-a2*(x[0]+x[1])
            a0 = y[0]-a2*x[0]**2-a1*x[0]
        
        return np.array([a0,a1,a2,x[0],x[2]])
        
    def isTrkSimilar(self, x_cand):
        similar = False
        trns = np.array([x_cand[3], x_cand[0]+x_cand[1]*x_cand[3]+x_cand[2]*x_cand[3]**2])
        rot = abs(x_cand[1]) > 2
        #print("before x_cand=", x_cand, trns, rot)
        x_cand = self.translatePolynom(x_cand, trns, rot)
        #print("after x_cand=", x_cand, trns, rot)
        for trk in self.ext_object_list:
            x_trk = trk.getStateVector()
            #print("before x_trk=", x_trk)
            x_trk = self.translatePolynom(x_trk, trns, rot)
            #print("after x_trk=", x_trk)
            c0 = (x_trk[0], x_trk[1], x_trk[2])
            c1 = (x_cand[0], x_cand[1], x_cand[2])
            xright = min(x_trk[4],x_cand[4])
            xleft = max(x_trk[3],x_cand[3])
            lat_distance = abs((c0[0]+c0[1]*xleft+c0[2]*xleft**2)-(c1[0]+c1[1]*xleft+c1[2]*xleft**2))
            overlap = xright-xleft
            #print("overlap", overlap, "trk width", x_trk[4]-x_trk[3], "cand_trk_width", x_cand[4]-x_cand[3], "lat_distance", lat_distance, "c0", c0, "c1", c1)
            if (overlap > 0.5*(x_trk[4]-x_trk[3]) or overlap > 0.5*(x_cand[4]-x_cand[3])):# and lat_distance < 5:
                #print("inside condition", self.innerProductPolynoms(c0,c1,xleft,xright))
                if(self.innerProductPolynoms(c0,c1,xleft,xright) < 10):
                    print("Tracks are similar! do not open a new trk", c0, c1)
                    return True
        return False
     
    @staticmethod
    def getTrkPointsMatrix(pnt_object_list,fx_flag):
        mat = np.zeros([len(pnt_object_list), 2])
        for i, pnt in enumerate(pnt_object_list):
            mat[i, :] = pnt.getStateVector(fx_flag).reshape(-1)
            
        return mat
    @staticmethod
    def createProbabilityMatrixExt_old(pnt_object_list, prior):
        ext_data_associator = Pnts2ExtObjectDataAssociator(deltaL=1)
        #print("createProbabilityMatrixExt for prior", prior)
        fx_flag = prior["fx"]
        xthr = 2
        lat_dist_to_prior_th = 2
        if pnt_object_list:
            M = StaticTracker.getTrkPointsMatrix(pnt_object_list,fx_flag)
            clus = DBSCAN(eps=4, min_samples=2).fit(M)
            labels = clus.labels_ 
            P = np.eye(len(pnt_object_list))
            for i in range(M.shape[0]):
                xi = M[i,:]
                xmin, xmax = prior["xmin"]-xthr, prior["xmax"]+xthr
                c = prior["c"]
                lat_dist_to_prior = xi[1]-c[0]-c[1]*xi[0]-c[2]*xi[0]**2
                if xi[0] >= xmin and xi[0] <= xmax and abs(lat_dist_to_prior) < lat_dist_to_prior_th:
                    xarr = np.linspace(xmin-xthr,xmax+xthr,1000)
                    yarr = c[0] + lat_dist_to_prior + c[1]*xarr + c[2]*xarr**2
                    candidates_indices = np.where((labels[i] >= 0) & (labels==labels[i]))[0]
                    if candidates_indices.size:
                        for j in np.nditer(candidates_indices):
                            if i != j:
                                xpj = M[j,:]
                                if xpj[0] >= xmin and xpj[0] <= xmax:
                                    a = np.argmin(np.sqrt((xarr-xpj[0])**2+(yarr-xpj[1])**2))
                                    xk = [xarr[a],yarr[a]]
                                    Pj = pnt_object_list[j].getCovarianceMatrix()
                                    P[i,j] = ext_data_associator.calcLikelihood(xk, xpj, Pj)
                        
        return P
    
    @staticmethod
    def findClosestPoint(M, prior):
        x = np.linspace(prior["xmin"],prior["xmax"],(round(prior["xmax"]-prior["xmin"])+1)*10)
        c = prior["c"]
        y = c[0] + c[1]*x + c[2]*x**2
        pol = np.array([x,y]).T
        #print("M and pol shape", M.shape, pol.shape)
        dists = cdist(M, pol)
        ind = np.argmin(dists,axis=1)
        return pol[ind,:]
    
    def findPointOnHypotheticalCurve(xpj, Dj, xmin, xmax, lpk, direction):
        if xpj[0] > Dj[0]:
            m = (xpj[1]-Dj[1])/(xpj[0]-Dj[0]+1e-6)
        else:
            m = (Dj[1]-xpj[1])/(Dj[0]-xpj[0]+1e-6)

        x_bool = xpj[0] > Dj[0]
        y_bool = xpj[1] > Dj[1]
        if (direction ^ x_bool ^ y_bool):
            xmin = Dj[0]
        else:
            xmax = Dj[0]
        n = xpj[1]- m*xpj[0]
        xx = np.linspace(xmin, xmax, round(xmax-xmin)*100)
        yy = m*xx+n
        line = np.array([xx,yy]).T
        Dj = Dj.reshape(1,-1)
        #print("Dj",Dj.shape, line.shape)
        dists = np.abs(cdist(Dj, line) - lpk)
        ind = np.argmin(dists, axis=1)
        
        #print("line[ind,:]",line[ind,:])
        
        return line[ind,:].T
        
    
    @staticmethod
    def createProbabilityMatrixExt(pnt_object_list, prior):
        pnt_data_associator = PointObjectDataAssociator(delta=5)
        #print("createProbabilityMatrixExt for prior", prior)
        fx_flag = prior["fx"]
        xthr = 2
        lat_dist_to_prior_th = 5
        if pnt_object_list:
            M = StaticTracker.getTrkPointsMatrix(pnt_object_list,fx_flag)
            D = StaticTracker.findClosestPoint(M, prior)
            clus = DBSCAN(eps=4, min_samples=2).fit(M)
            labels = clus.labels_ 
            P = np.eye(len(pnt_object_list))
            for i in range(M.shape[0]):
                xi = M[i,:]
                xmin, xmax = prior["xmin"]-xthr, prior["xmax"]+xthr
                c = prior["c"]
                lpk = np.linalg.norm(xi-D[i,:])
                direction = xi[1]-D[i,1] > 0
                if xi[0] >= xmin and xi[0] <= xmax and lpk < lat_dist_to_prior_th:
                    candidates_indices = np.where((labels[i] >= 0) & (labels==labels[i]))[0]
                    if candidates_indices.size:
                        squared_dist = np.linalg.norm(M[candidates_indices,:]-M[i,:],axis=1)
                        sort_idx = np.argsort(np.squeeze(squared_dist))
                        group_i = [i]
                        for j in np.nditer(candidates_indices[sort_idx]):
                            if i != j:
                                xpj = M[j,:].reshape(-1,1)
                                kk = np.argmin(np.array([np.abs(xpj[0]-M[k,0]) for k in group_i]))
                                k = group_i[kk]
                                xpk = M[k,:].reshape(-1,1)
                                sqr_dist = np.linalg.norm(xpk-xpj)
                                if sqr_dist < 3:
                                    xij = StaticTracker.findPointOnHypotheticalCurve(xpj, D[j,:], xmin, xmax, lpk, direction)
                                    Pj = pnt_object_list[j].getCovarianceMatrix(fx_flag)
                                    P[i,j] = pnt_data_associator.calcLikelihood(xij, xpj, Pj)
                                    if P[i,j]:
                                        group_i.append(j)
                        
        return P
                    
    @staticmethod
    def getBestAssociation(Ge, Gp):
        i_meas,lp,le,i_trk_p,i_trk_e = None,0,0,None,None
        if Gp.size > 0:
            ind = np.unravel_index(np.argmax(Gp, axis=None), Gp.shape)
            i_meas = ind[0]
            i_trk_p = ind[1]
            lp = Gp[ind]
        if Ge.size > 0:
            if lp > 0:
                i_trk_e = np.argmax(Ge[i_meas,:])
                le = Ge[i_meas,i_trk_e]
        
        return i_meas,lp, le, i_trk_p, i_trk_e
    
    @staticmethod
    def zeroOutAssociation(mat, i, j, k=None):
        if mat is not None:
            if mat.ndim == 2:
                mat[i, :] = 0
                mat[:, j] = 0
            elif mat.ndim == 3:
                mat[i, :, k] = 0
                mat[:, j, k] = 0
        
        return mat
          
    @staticmethod
    def createProbabilityMatrix(pairs):
        if pairs:
            #print("pairs",pairs)
            u_meas = np.unique([item[0] for item in pairs])
            u_trk = np.unique([item[1] for item in pairs])
            #print("u_meas", u_meas, "u_trk", u_trk)
            P = np.zeros((len(u_meas),len(u_trk)))
            for i in range(0, len(u_meas)):
                for j in range(0, len(u_trk)):
                    #print(u_meas[i], u_trk[j], "prob = ", [pair[2] if pair[0]==u_meas[i] and pair[1]==u_trk[j] else 0 for pair in pairs])
                    P[i,j] = max([pair[2] if pair[0]==u_meas[i] and pair[1]==u_trk[j] else 0 for pair in pairs])
        else:
            P = None
            
        #print("createProbabilityMatrix: P = ", P)
 
        return P

    @staticmethod
    def innerProductPolynoms(f1, f2, xmin, xmax):
        f1mf2 = P.polysub(f1,f2)
        f1mf2 = P.polymul(f1mf2,f1mf2)
        val = P.polyint(f1mf2,lbnd=xmin)
        #print(val)
        ip = val[0] + val[1]*(xmax-xmin)+val[2]*(xmax-xmin)**2+val[3]*(xmax-xmin)**3+val[4]*(xmax-xmin)**4+val[5]*(xmax-xmin)**5
        
        return np.sqrt(ip)/(xmax-xmin)

    def getPolynoms(self):
        polynoms = []
        for trk in self.ext_object_list:
            x = trk.getStateVector()
            polynoms.append({"f": np.poly1d([x[2], x[1], x[0]]),"x_start":x[3],"x_end":x[4],"fxFlag":trk.getFxFlag()})
            
        return polynoms
    
    def getPoints(self):
        points = np.zeros((len(self.pnt_object_list), 2))
        for i_trk,trk in enumerate(self.pnt_object_list):
            points[i_trk,:] = np.squeeze(trk.getStateVector(),axis=1)
        
        return points
            
    def getExtendedTracks(self):
        return self.ext_object_list
        