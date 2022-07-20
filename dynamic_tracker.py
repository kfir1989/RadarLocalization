import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import math
from sklearn.cluster import DBSCAN
from utils import *

def roundAngle(a):
    if a < -np.pi:
        return a + np.pi
    if a > np.pi:
        return a - np.pi

    return a

class ClassicDBSCAN:
    def __init__(self, eps=3, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

    @staticmethod
    def _calcClusterProperiesArray(ci, ego, heading):
        r_bias = 1
        x_com = np.mean(ci[:,0])
        y_com = np.mean(ci[:,1])
        dx = x_com-ego[0]
        dy = y_com-ego[1]
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        transformed_target = np.dot(R, np.array([x_com-ego[0], y_com-ego[1]]))
        vx_com = 1*np.mean(ci[:,6])
        vy_com = 1*np.mean(ci[:,7])
        dx = transformed_target[0]
        dy = transformed_target[1]
        r = np.sqrt(dx**2+dy**2)
        phi = roundAngle(math.atan2(dy, dx))
        vr = np.sign(vx_com) * np.sqrt(vx_com**2+vy_com**2)

        Z = np.array([r, phi, vr])
        X = np.array([x_com,y_com,vx_com,vy_com])
        
        return Z, X

    @staticmethod
    def _calcClusterProperiesScalar(ci, ego, heading):
        x_com = ci[0,0]
        y_com = ci[0,1]
        dx = x_com-ego[0]
        dy = y_com-ego[1]
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        transformed_target = np.dot(R, np.array([x_com-ego[0], y_com-ego[1]]))
        vx_com = ci[0,6]
        vy_com = ci[0,7]
        dx = transformed_target[0]
        dy = transformed_target[1]
        r = np.sqrt(dx**2+dy**2)
        phi = roundAngle(math.atan2(dy, dx))
        vr = np.sign(vx_com) * np.sqrt(vx_com**2+vy_com**2)

        Z = np.array([r, phi, vr])
        X = np.array([x_com,y_com,vx_com,vy_com])
        
        return Z, X


    def run(self, pc, ego, heading):
        Z_list = []
        X_list = []

        if pc.shape[0] < 1:
            return Z_list, X_list
        clus = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pc[:,0:2+6:8])
        c = clus.labels_
        n_clusters = np.unique(c)
        for i in n_clusters:
            ci = pc[c==i, :]
            if i == -1:
                #continue #don't consider 1 point clusters
                for j in range(ci.shape[0]):
                    Z, X = self._calcClusterProperiesScalar(ci, ego, heading)
                    if Z is not None:
                        Z_list.append(Z)
                        X_list.append(X)
            else:
                Z, X = self._calcClusterProperiesArray(ci, ego, heading)
                if Z is not None:
                    Z_list.append(Z)
                    X_list.append(X)
            
        return Z_list, X_list


class DynamicObjectTrack:
    def __init__(self, z, create_frame_idx, ego, ego_speed):
        self.kf = ExtendedKalmanFilter(dim_x=4, dim_z=3)
        self.kf.x = np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1]), z[2]*np.cos(z[1]), z[2]*np.sin(z[1])])
        self.kf.x = self.kf.x.reshape(-1,1)
        self.kf.F = np.eye(4)
        self.kf.P = np.diag([4,4,1,1])
        self.kf.P_post = self.kf.P.copy()
        self.ego = [ego]
        self.ego_speed = [ego_speed]
        
        self.noise_ax = 9.0
        self.noise_ay = 9.0
        dt = 0.125 # [sec]
        dt_2 = dt*dt
        dt_3 = dt_2*dt;
        dt_4 = dt_3*dt;
        self.kf.Q = np.array([[dt_4/4*self.noise_ax, 0, dt_3/2*self.noise_ax, 0],
                              [0, dt_4/4*self.noise_ay, 0, dt_3/2*self.noise_ay],
                              [dt_3/2*self.noise_ax, 0, dt_2*self.noise_ax, 0],
                              [0, dt_3/2*self.noise_ay, 0, dt_2*self.noise_ay]])
        self.kf.R = np.diag([2, 0.02, 0.4])
        self.kf.K = np.zeros((4,3)) # kalman gain
        self.saver = Saver(self.kf)
        self.create_frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.hits = 1
        self.age = 1
        self.confirmed = False
    
    def predict(self, dt, ego, last_ego, speed, last_speed):
        self.age += 1
        self.ego.append(ego)
        self.ego_speed.append(speed)
        dT = (ego["T"]-last_ego["T"])[0:2].reshape(-1,1)
        dV = (speed-last_speed)[0:2].reshape(-1,1)
        dheading = ego["heading"]-last_ego["heading"]
        dR = np.array([[np.cos(-dheading), -np.sin(-dheading)], [np.sin(-dheading), np.cos(-dheading)]])
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt
        dt_2 = dt*dt
        dt_3 = dt_2*dt;
        dt_4 = dt_3*dt;
        self.kf.Q = np.array([[dt_4/4*self.noise_ax, 0, dt_3/2*self.noise_ax, 0],
                              [0, dt_4/4*self.noise_ay, 0, dt_3/2*self.noise_ay],
                              [dt_3/2*self.noise_ax, 0, dt_2*self.noise_ax, 0],
                              [0, dt_3/2*self.noise_ay, 0, dt_2*self.noise_ay]])
        self.kf.predict()
        tmp = np.dot(dR, self.kf.x_prior[0:2])
        dRV = np.dot(dR, dV)
        self.kf.x_prior[0:2] = np.dot(dR, self.kf.x_prior[0:2]) - dRV * dt# - dT
        self.kf.x_prior[2:4] = np.dot(dR, self.kf.x_prior[2:4]) - dRV
        self.kf.x = np.copy(self.kf.x_prior)
        #rotate covariance matrix
        self.kf.P_prior[0:2,0:2] = np.dot(np.dot(dR, self.kf.P_prior[0:2,0:2]), dR.T)
        self.kf.P_prior[2:4,2:4] = np.dot(np.dot(dR, self.kf.P_prior[2:4,2:4]), dR.T)
        self.kf.P = np.copy(self.kf.P_prior)
        
    def calculateJacobian(self, x):
        Hj = np.zeros([3,4])
        #recover state parameters
        px = x[0];
        py = x[1];
        vx = x[2];
        vy = x[3];

        #pre-compute a set of terms to avoid repeated calculation
        c1 = px*px+py*py
        c2 = np.sqrt(c1)
        c3 = c1*c2

        #check division by zero
        if abs(c1) < 0.0001:
            print("CalculateJacobian () - Error - Division by Zero")
            return Hj

        #compute the Jacobian matrix
        Hj = np.array([[float(px/c2), float(py/c2), 0, 0],
                       [-float(py/c1), float(px/c1), 0, 0],
                       [float(py*(vx*py - vy*px)/c3), float(px*(px*vy - py*vx)/c3), float(px/c2), float(py/c2)]])

        return Hj
    
    
    def Hx(self, x):
        px = x[0];
        py = x[1];
        vx = x[2];
        vy = x[3];
        
        r = math.sqrt(px**2+py**2)
        phi = math.atan2(py,px)
        rho = (px*vx+py*vy)/r
        
        hx = np.array([float(r), float(phi), float(rho)])
        hx=hx.reshape(-1,1)
        
        return hx
   
    def update(self, z, current_frame_idx):
        self.hits += 1
    
        self.kf.update(z,HJacobian=self.calculateJacobian,Hx=self.Hx)
        self.last_update_frame_idx = current_frame_idx
        
    def save(self):
        self.saver.save()
        
    def getHistory(self):
        self.saver.to_array()
        return self.saver.x, self.ego, self.ego_speed
    
    def getStateVector(self):
        return self.kf.x
    
    def getPredictedStateVector(self):
        return self.kf.x_prior
        
    def getInnovationCovarianceMatrix(self):
        return self.kf.S
    
    def getCovarianceMatrix(self):
        return self.kf.P_post
    
    def getLastUpdateFrameIdx(self):
        return self.last_update_frame_idx
    
    def getTranslatedState(self):
        hstate, hego, hspeed = self.getHistory()
        history_len = hstate.shape[0]
        #rotate translate each state according to hego
        tstate = np.zeros((hstate.shape[0], 2, 1))
        tspeed = np.zeros((hstate.shape[0], 2, 1))
        for i, (state, ego, speed) in enumerate(zip(hstate, hego, hspeed)):
            R = np.array([[np.cos(ego["heading"]), -np.sin(ego["heading"])], [np.sin(ego["heading"]), np.cos(ego["heading"])]])
            tstate[i, :, :] = np.dot(R, state[0:2]) + ego["T"][0:2].reshape(-1,1)
            tspeed[i, :, :] = np.dot(R, state[2:4]) + np.dot(R, speed[0:2].reshape(-1,1))
        #abs_vel = np.mean(np.linalg.norm(tspeed,axis=1), axis=0)
        
        return tstate, tspeed
    

class DynamicTracker:
    def __init__(self):
        self.pnt_data_associator = PointObjectDataAssociator(dim=3,delta=1)
        self.dyn_object_list = []
        self.dyn_max_non_update_iterations = 5
        self.frame_idx = 0
        self.history_dyn_object_list = []
        self.last_ts = 0

    def run(self, Z, ts, ego, speed):
        self.frame_idx += 1
        print(f"frame_idx = {self.frame_idx}")
        print(f"Number of new plots {len(Z)}")
        print(f"Number of dynamic tracks before run() {len(self.dyn_object_list)}")
        
        #Prediction
        
        if self.last_ts > 0:
            dt = (ts - self.last_ts) / 1e6
            self.last_ts = ts
            for dyn_track in self.dyn_object_list:
                dyn_track.predict(dt, ego, self.last_ego, speed, self.last_speed)
        self.last_ts = ts
        self.last_ego = ego
        self.last_speed = speed
        
        #Association
        Gp = self.p2t(Z)
        
        #Update
        Z = self.trackUpdate(Z, Gp)
        
        #Init
        self.trackInit(Z,ego,speed)
            
        #Maintenance
        self.trackMaintenance()
        
        return self.getTracks()
        
    def p2t(self, Z):
        Gp = np.zeros((len(Z), len(self.dyn_object_list)))
        for idx_z, z in enumerate(Z):
            z = z.reshape(-1,1)
            for idx_track, dyn_track in enumerate(self.dyn_object_list):
                x = dyn_track.getStateVector()
                x_pred = dyn_track.getPredictedStateVector()
                innov_cov = dyn_track.getCovarianceMatrix()
                z_pred = dyn_track.Hx(x_pred)
                z_pred_reshaped = z_pred.reshape(-1,1)
                #print(f"z={z} z_pred={z_pred}")
                H = dyn_track.calculateJacobian(dyn_track.kf.x)
                cov = np.dot(np.dot(H, dyn_track.kf.P), H.T) + dyn_track.kf.R
                Gp[idx_z,idx_track] = self.pnt_data_associator.calcLikelihood(z, z_pred, cov)
        
        #print(f"Gp = {Gp}")
        return Gp
    
    def trackUpdate(self, Z, Gp):
        assigned_meas_list = []
        while(1): #Iterate over GP
            i_meas, lp, i_trk = self.getBestAssociation(Gp)
            if(lp == 0):
                break
            self.zeroOutAssociation(Gp, i_meas, i_trk) # Clear from point association matrix
            zm = Z[i_meas]
            assigned_meas_list.append(i_meas)
            
            zm=zm.reshape(-1,1)

            #print(f"Updating track = {i_trk} z = {zm}")
            self.dyn_object_list[i_trk].update(z=zm, current_frame_idx=self.frame_idx)
                
        for i_meas in sorted(assigned_meas_list, reverse=True):
            Z = np.delete(Z, (i_meas), axis=0)
        
        return Z
        
    def trackInit(self, Z, ego, ego_speed):
        for z in Z:
            new_trk = DynamicObjectTrack(z=z, create_frame_idx=self.frame_idx, ego=ego, ego_speed=ego_speed)
            #print(f"Initiating new track: x={new_trk.kf.x} for measurement z={z}")
            self.dyn_object_list.append(new_trk)

            
    def deleteTrack(self, track_list, indices):
        delete_indices  = np.unique(indices)
        #print("deleteTrack is called. track list length:", len(track_list), "indices", indices)
        for index in sorted(delete_indices, reverse=True):
            self.history_dyn_object_list.append(track_list[index])
            del track_list[index]
        
    def trackMaintenance(self):
        dyn_delete_list = []
        for i_trk,trk in enumerate(self.dyn_object_list):
            trk.save()
            #print(f"trk.kf.x = {trk.kf.x} age = {trk.age} hits = {trk.hits}")
            if trk.age > 10 and (float(trk.hits) / trk.age) > 0.5:
                trk.confirmed = True
            if self.frame_idx - trk.getLastUpdateFrameIdx() > self.dyn_max_non_update_iterations:
                dyn_delete_list.append(i_trk)

        self.deleteTrack(self.dyn_object_list, dyn_delete_list)
        
    def isTrkSimilar(self, x_cand):
        x, y, vx, vy = x_cand[0], x_cand[1], x_cand[2], x_cand[3]
        for trk in self.dyn_object_list:
            state_adv = trk.getStateVector()
            x_adv, y_adv, vx_adv, vy_adv = state_adv[0], state_adv[1], state_adv[2], state_adv[3] 
            
            pos_thr = 4
            vel_thr = 1
            pos_dist = sqrt((x-x_adv)**2+(y-y_adv)**2)
            vel_dist = sqrt((vx-vx_adv)**2+(vy-vy_adv)**2)
            if dist > dist_thr or vel_dist > vel_thr:
                return True
        
        return False
                
                    
    @staticmethod
    def getBestAssociation(Gp):
        i_meas,lp,i_trk_p = None,0,None
        if Gp.size > 0:
            ind = np.unravel_index(np.argmax(Gp, axis=None), Gp.shape)
            i_meas = ind[0]
            i_trk_p = ind[1]
            lp = Gp[ind]
        
        return i_meas,lp, i_trk_p
    
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
            u_meas = np.unique([item[0] for item in pairs])
            u_trk = np.unique([item[1] for item in pairs])
            P = np.zeros((len(u_meas),len(u_trk)))
            for i in range(0, len(u_meas)):
                for j in range(0, len(u_trk)):
                    P[i,j] = max([pair[2] if pair[0]==u_meas[i] and pair[1]==u_trk[j] else 0 for pair in pairs])
        else:
            P = None
            
        #print("createProbabilityMatrix: P = ", P)
 
        return P
    
    def getTracks(self):
        return self.dyn_object_list
    
    def getHistory(self):
        return self.history_dyn_object_list + self.dyn_object_list 