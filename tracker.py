import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
from numpy.polynomial import polynomial as P
from utils import *
import detector
import scipy
    
class StaticTracker:
    def __init__(self):
        self.ext_object_list = []
        self.ext_data_associator = ExtObjectDataAssociator()
        self.pnt_object_list = []
        self.pnt_data_associator = PointObjectDataAssociator()
        self.eta = 30
        self.frame_idx = 0
        self.polynom_list = []
        self.k = 8
        self.pnt_max_non_update_iterations = 12
        self.ext_max_non_update_iterations = 10
        self.max_decline_factor = 100
        self.system_rotated_flag = False
        
    def rotateCoordinateSystem(self,z,dz,prior,rotate=False,rotate_back=False,rotate_all=False):
        if rotate_all:
            for ext_track in self.ext_object_list:
                ext_track.rotate90()
                    
            for pnt_track in self.pnt_object_list:
                pnt_track.rotate90()
                
        if rotate:
            z = z[:, [1,0]]
            dz = z[:, [1,0], [1,0]]
            (a0,a1,a2) = prior
            #rotate by 90 degrees - x ---> y', y---->-x'
            x = np.array([1,2,3])
            y = np.array([a0+a1*x[0]+a2*x[0]**2,a0+a1*x[1]+a2*x[1]**2,a0+a1*x[2]+a2*x[2]**2])
            A = [[1, -y[0],y[0]**2],[1, -y[1], y[1]**2],[1, -y[2], y[2]**2]]
            co = np.linalg.inv(A).dot(x)
            prior = (c0[0],c0[1],c0[2])
            
        if rotate_back:
            for ext_track in self.ext_object_list:
                ext_track.rotateback90()
                    
            for pnt_track in self.pnt_object_list:
                pnt_track.rotateback90()
            
                
    def checkRotation(self,z,dz,prior):
        if prior[1] > self.max_decline_factor:
            z, dz, prior = self.rotateCoordinateSystem(z,dz,prior,rotate=True,rotate_all=(not self.system_rotated_flag))
            if not self.rotate_system_flag:
                self.system_rotated_flag = True
        elif self.system_rotated_flag:
            z, dz, prior = self.rotateCoordinateSystem(z,dz,prior,rotate_back=True)
            self.system_rotated_flag = False
            
        return z,dz,prior
        
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
                y_pred = x_pred[0] + x_pred[1]*z[0] + x_pred[2]*z[0]**2
                P_pred = ext_track.getPredictedCovarianceMatrix()
                Ha = np.array([1, z[0], z[0]**2])
                S_pred = np.dot(np.dot(Ha, P_pred[0:3,0:3]), Ha.T)
                Ge[idx_z,idx_track] = self.ext_data_associator.calcLikelihood(z[0], z[1], y_pred, S_pred, x)
                    
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
                trk.counter_update += 1
                Hse = np.array([1 if x < trk.getStateVector()[3] else 0, 1 if x > trk.getStateVector()[4] else 0])
                H = np.array([[0, 0, 0, Hse[0], Hse[1]],[1, x, x**2, 0, 0]])
                R = cov
                #print("R",R)
                print("Updating extended object track = ", i_trk)
                self.ext_object_list[i_trk].update(np.array([x,y]).T, current_frame_idx=self.frame_idx, H=H, R=R)
                self.debug["mupol"].append({"measurements": np.array([x,y]).T, "polynom":self.ext_object_list[i_trk].getStateVector(), "id":i_trk})
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
            
    def trackInitExt(self, polynom, generated_points):
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
            new_trk = ExtendedObjectTrack(x=x,P=polynom["P"],create_frame_idx=self.frame_idx)
            new_trk.static_cars_flag = detector.detectCarsInARow(generated_points)
            print("created an extended object!", x, "static_cars_flag", new_trk.static_cars_flag)
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
                if nk > self.k:
                    xy = np.zeros((2,nk))
                    selected_pnts = np.where(P[:,j]>0)
                    for i_pnt, pnt in enumerate(selected_pnts[0]):
                        #print("i_pnt", pnt)
                        xy[:,i_pnt] = np.squeeze(self.pnt_object_list[pnt].getStateVector(),axis=1)
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
                    polynom = {"f":f,"x_start":min(xy[0,:]),"x_end":max(xy[0,:]),"P": covP}
                    #print("polynom was generated!", polynom)
                    status = self.trackInitExt(polynom, xy.T)
                    if status:
                        self.debug["pgpol"].append({"points": xy, "polynom":polynom})
                    PM = self.zeroOutAssociation(PM,selected_pnts[0],j)
                PM = self.zeroOutAssociation(PM,None,j)
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
            if (overlap > 0.5*(x_trk[4]-x_trk[3]) or overlap > 0.5*(x_cand[4]-x_cand[3])) and lat_distance < 2:
                #print("inside condition", self.innerProductPolynoms(c0,c1,xleft,xright))
                if(self.innerProductPolynoms(c0,c1,xleft,xright) > 1):
                    print("Tracks are similar! do not open a new trk", c0, c1)
                    return True
        return False
                
    @staticmethod
    def createProbabilityMatrixExt(pnt_object_list, prior):
        ext_data_associator = Pnts2ExtObjectDataAssociator(deltaL=1)
        #print("createProbabilityMatrixExt for prior", prior)
        if pnt_object_list:
            P = np.zeros((len(pnt_object_list),len(pnt_object_list)))
            for i,pnt_track in enumerate(pnt_object_list):
                xi = pnt_track.getStateVector()
                xmin, xmax = prior["xmin"], prior["xmax"]
                c = prior["c"]
                lat_dist_to_prior = abs(xi[1]-c[0]-c[1]*xi[0]-c[2]*xi[0]**2)
                if xi[0] >= xmin-5 and xi[0] <= xmax+5 and lat_dist_to_prior < 12:
                    lat_pi = xi[1]-c[1]*xi[0]-c[2]*xi[0]**2
                    squared_dist = np.array([np.sqrt((xi[0]-pnt.getStateVector()[0])**2+(xi[1]-pnt.getStateVector()[1])**2) for pnt in pnt_object_list])
                    candidates_indices = np.argsort(np.squeeze(squared_dist))
                    group_i = [i]
                    for j in candidates_indices:
                        if squared_dist[j] > 60:
                            break
                        if(i != j):
                            xpj = pnt_object_list[j].getStateVector()
                            kk = np.argmin(np.array([abs(xpj[0]-pnt_object_list[k].getStateVector()[0]) for k in group_i]))
                            k = group_i[kk]
                            xpk = pnt_object_list[k].getStateVector()
                            if np.sqrt((xpk[0]-xpj[0])**2+(xpk[1]-xpj[1])**2) < 4:
                                xpi = [xpk[0], lat_pi + c[1] * xpj[0] + c[2] * xpj[0]**2]
                                Pj = pnt_object_list[j].getCovarianceMatrix()
                                P[i,j] = ext_data_associator.calcLikelihood(xpi, xpj, Pj)
                                if P[i,j]:
                                    group_i.append(j)
                            #else:
                                #print(f"xpk={xpk} xpj={xpj}")
                        else:
                            P[i,j] = 1
                else:
                    P[i,:] = 0 
                        
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
        val = P.polyint(f1mf2,lbnd=-xmin)
        #print(val)
        ip = val[0] + val[1]*xmax+val[2]*xmax**2+val[3]*xmax**3+val[4]*xmax**4+val[5]*xmax**5
        
        return np.sqrt(ip)

    def getPolynoms(self):
        polynoms = []
        for trk in self.ext_object_list:
            x = trk.getStateVector()
            polynoms.append({"f": np.poly1d([x[2], x[1], x[0]]),"x_start":x[3],"x_end":x[4],"static_cars_flag":trk.static_cars_flag})
            
        return polynoms
    
    def getPoints(self):
        points = np.zeros((len(self.pnt_object_list), 2))
        for i_trk,trk in enumerate(self.pnt_object_list):
            points[i_trk,:] = np.squeeze(trk.getStateVector(),axis=1)
        
        return points
            
    def getExtendedTracks(self):
        return self.ext_object_list
        