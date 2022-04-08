import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
import random
import math
from numpy.polynomial import polynomial as P

class PointObjectDataAssociator():
    def __init__(self, delta=1):
        self.delta = delta
        self.dim = 2 #rank
        
    def distance(self, x, u, P):
        return np.dot(np.dot((x-u).T,np.linalg.inv(P)),(x-u))
        
    def gating(self, z, z_prior, innov_cov):
        return self.distance(z,z_prior,innov_cov) <= self.delta
        
    def calcLikelihood(self, z, z_prior, innov_cov):
        if self.gating(z, z_prior, innov_cov):
            det = np.linalg.det(innov_cov)
            l = np.power((2*np.pi), -0.5*self.dim) * np.power(det, -0.5) * np.exp(-0.5 * self.distance(z, z_prior, innov_cov)) 
        else:
            l = 0
        
        return l
        
        
class ExtObjectDataAssociator():
    def __init__(self, dim=2,deltaL=1,deltaS=8,deltaE=8):
        self.deltaL = deltaL
        self.deltaS = deltaS # delta x start
        self.deltaE = deltaL # delta x end
        self.dim = dim #rank
        
    def distance(self, x, u, P):
        return (x-u).T*np.linalg.inv(P)*(x-u)
        
    def gating(self, u, y, y_pred, S_pred, x):
        quad_dist = (y-y_pred)**2
        P_inv = 1 / S_pred
        return (x[3]-self.deltaS) < u and (x[4]+self.deltaE) > u and (quad_dist * P_inv <= self.deltaL)
        
    def calcLikelihood(self, u, y, y_pred, S_pred, x):
        if self.gating(u, y, y_pred, S_pred, x):
            det = S_pred
            l = np.power((2*np.pi), -0.5*self.dim) * np.power(det, -0.5) * np.exp(-0.5 * (1/det)*(y-y_pred)**2) 
        else:
            l = 0
        
        return l
    
class Pnts2ExtObjectDataAssociator():
    def __init__(self, dim=1,deltaL=1):
        self.deltaL = deltaL
        self.deltaS = 8 # delta x start
        self.deltaE = 8 # delta x end
        self.dim = dim #rank
        
    def gating(self, xi, xj, Pi):
        quad_dist = (xi[1]-xj[1])**2
        P_inv = 1 / Pi[1,1]
        return (xi[0]-self.deltaS) < xj[0] and (xi[0]+self.deltaE) > xj[0] and (quad_dist * P_inv <= self.deltaL)
        
    def calcLikelihood(self, xi, xj, Pi):
        if(self.gating(xi, xj, Pi)):
            det = Pi[1,1]
            l = np.power((2*np.pi), -0.5*self.dim) * np.power(det, -0.5) * np.exp(-0.5 * (1/det)*(xi[1]-xj[1])**2) 
        else:
            l = 0
        
        return l

class PointObjectTrack:
    def __init__(self, x, P, create_frame_idx):
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.kf.x = x.T # x,y
        self.kf.F = np.diag([1,1])
        self.kf.P = P
        self.kf.Q = np.zeros((2,2))
        self.H = np.eye(2)
        self.saver = Saver(self.kf)
        self.create_frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.hits = [1]
    
    def predict(self):
        self.kf.predict()
        
    def update(self, z, cov, current_frame_idx):
        self.kf.update(z,R=cov)
        self.last_update_frame_idx = current_frame_idx
        
    def save(self):
        self.saver.save()
        
    def getHistory(self):
        self.saver.to_array()
        return self.saver.x
    
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
        
class ExtendedObjectTrack:
    def __init__(self, x=None, P=None, create_frame_idx=0, gamma=0.98):
        self.kf = KalmanFilter(dim_x=5, dim_z=2)
        x0 = np.array([[5, 0, 0, 0, 0]]).T # a1, a2, a3, x_start, x_end
        P0 = np.diag([2, 1, 1, 20, 20]) #Initial state covariance matrix
        if x is None:
            x = x0
        if P is None:
            P = P0  
        self.kf.x = x 
        self.gamma = gamma # Squeezing factor
        self.kf.F = np.array([[1, 0, 0, 0, 0,],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, gamma,1-gamma],[0, 0, 0,1-gamma,gamma]])
        self.kf.P = P
        self.kf.Q = np.zeros((5,5))
        self.kf.Q[3,3] = 5
        self.kf.Q[4,4] = 5
        self.saver = Saver(self.kf)
        self.frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.hits = [1]
        
    def predict(self):
        self.kf.predict()
        
    def update(self, z, current_frame_idx, H=None, R=None):
        if H is not None:
            self.kf.H = H
        if R is not None:
            self.kf.R = R
            
        self.kf.update(z)
        self.last_update_frame_idx = current_frame_idx
        
    def save(self):
        self.saver.save()
        
    def getHistory(self):
        self.saver.to_array()
        return self.saver.x
    
    def getStateVector(self):
        return self.kf.x
    
    def getStateCovarianceMatrix(self):
        return self.kf.P
    
    def getPredictedStateVector(self):
        return self.kf.x_prior
    
    def getPredictedCovarianceMatrix(self):
        return self.kf.P_prior
    
    def getLastUpdateFrameIdx(self):
        return self.last_update_frame_idx