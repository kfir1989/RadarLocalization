import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
import random
import math
from numpy.polynomial import polynomial as P

class PointObjectDataAssociator():
    def __init__(self, dim=2, delta=1):
        self.delta = delta
        self.dim = dim #rank
        
    def distance(self, x, u, P):
        try:
            #print("distance=", np.dot(np.dot((x-u).T,np.linalg.inv(P)),(x-u)))
            return np.dot(np.dot((x-u).T,np.linalg.inv(P)),(x-u))
        except:
            print("OOps! x = ", x, "u = ", u, " P = ", P)
            raise ValueError("Oh NO")
        
    def gating(self, z, z_prior, innov_cov):
        return self.distance(z,z_prior,innov_cov) <= self.delta
        
    def calcLikelihood(self, z, z_prior, innov_cov):
        if self.gating(z, z_prior, innov_cov):
            det = np.linalg.det(innov_cov)
            dist = self.distance(z, z_prior, innov_cov)
            l = np.power((2*np.pi), -0.5*self.dim) * np.power(det, -0.5) * np.exp(-0.5 * dist) 
        else:
            l = 0
        
        return l
        
        
class ExtObjectDataAssociator():
    def __init__(self, dim=2,deltaL=1,deltaS=2,deltaE=2):
        self.deltaL = deltaL
        self.deltaS = deltaS # delta x start
        self.deltaE = deltaE # delta x end
        self.dim = dim #rank
        
    def distance(self, x, u, P):
        return (x-u).T*np.linalg.inv(P)*(x-u)
        
    def gating(self, u, y, y_pred, S_pred, x):
        quad_dist = (y-y_pred)**2
        P_inv = 1 / S_pred
        return (x[3]-self.deltaS) < u and (x[4]+self.deltaE) > u and (quad_dist * P_inv <= self.deltaL)
    
    def getY(self, x, p):
        return x[0]+x[1]*p+x[2]*p**2
    
    def gating_dist(self, u, y, y_pred, S_pred, x):
        quad_dist = (y-y_pred)**2
        P_inv = 1 / S_pred
        
        dist = 0
        if u < x[3]:
            dist = math.sqrt((x[3]-u)**2+(self.getY(x, x[3])-y)**2)
        elif u > x[4]:
            dist = math.sqrt((x[4]-u)**2+(self.getY(x, x[4])-y)**2)
        return dist < self.deltaS and (quad_dist * P_inv <= self.deltaL)
        
    def calcLikelihood(self, u, y, y_pred, S_pred, x):
        if self.gating_dist(u, y, y_pred, S_pred, x):
            det = S_pred
            l = np.power((2*np.pi), -0.5*self.dim) * np.power(det, -0.5) * np.exp(-0.5 * (1/det)*(y-y_pred)**2) 
        else:
            l = 0
        
        return l
    
class Pnts2ExtObjectDataAssociator():
    def __init__(self, dim=1,deltaL=1):
        self.deltaL = deltaL
        self.deltaS = 4 # delta x start
        self.deltaE = 4 # delta x end
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
        self.kf.Q = np.diag([1e-4,1e-4])#np.zeros((2,2))
        self.kf.H = np.eye(2)
        self.saver = Saver(self.kf)
        self.create_frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.last_frame_idx = create_frame_idx
        self.hits = [1]
    
    def predict(self, current_frame_idx):
        self.kf.predict()
        self.saver.save()
        self.last_frame_idx = current_frame_idx
        
    def update(self, z, cov, current_frame_idx):
        self.kf.update(z,R=cov)
        self.last_update_frame_idx = current_frame_idx
        self.saver.save()
        
    def save(self):
        self.saver.save()
        
    def getHistory(self):
        self.saver.to_array()
        return self.saver.x
    
    def getStateVector(self, fxFlag=True):
        if fxFlag:
            return self.kf.x
        else:
            return np.flip(self.kf.x)
    
    def getPredictedStateVector(self, fxFlag=True):
        if fxFlag:
            return self.kf.x_prior
        else:
            return np.flip(self.kf.x_prior)
        
    def getInnovationCovarianceMatrix(self, fxFlag=True):
        if fxFlag:
            return self.kf.S
        else:
            return np.flip(self.kf.S)
    
    def getCovarianceMatrix(self, fxFlag=True):
        if fxFlag:
            return self.kf.P
        else:
            return np.flip(self.kf.P)
    
    def getLastUpdateFrameIdx(self):
        return self.last_update_frame_idx
        
class ExtendedObjectTrack_UKF:
    def __init__(self, x=None, P=None, create_frame_idx=0, gamma=0.995, fxFlag=True):
        # create sigma points to use in the filter. This is standard for Gaussian processes
        points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=0.08, fx=self.fx, hx=self.hx, points=points)
        x0 = np.array([[5, 0, 0, 0]]).T # a1, a2, a3, x_start, x_end
        P0 = np.diag([2, 1, 20, 20]) #Initial state covariance matrix
        if x is None:
            x = x0
        if P is None:
            P = P0  
        self.kf.x = x 
        self.gamma = gamma # Squeezing factor
        self.kf.F = np.array([[1, 0, 0, 0,],[0, 1, 0, 0],[0, 0, gamma,1-gamma],[0, 0,1-gamma,gamma]])
        self.kf.P = P
        #self.kf.Q = np.zeros((5,5))
        #self.kf.Q[3,3] = 5
        #self.kf.Q[4,4] = 5
        self.kf.Q = np.diag([0, 0, 5, 5])
        self.saver = Saver(self.kf)
        self.create_frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.hits = [1]
        self.static_cars_flag = False
        self.counter_update = 0
        self.fx_flag = fxFlag
        
    def hx(self, x):
        return np.matmul(self.kf.H, x)
        
    def fx(self, x, dt):
        return np.matmul(self.kf.F, x)
        
    def predict(self):
        self.kf.predict()
        
    def update(self, z, current_frame_idx, H=None, cov=None):
        if H is not None:
            self.kf.H = H
        if cov is not None:
            ha = np.array([-1 * self.kf.x[1], 1])
            Sigma_a = np.matmul(ha, np.matmul(cov, ha.T))
            self.kf.R = np.diag([cov[0,0], Sigma_a])
            #self.kf.R = np.diag([cov[0,0], cov[1,1,]])
            
        #print(f"updateing z = {z} R = {self.kf.R} before update x = {self.kf.x}")
        self.kf.update(z)
        #print(f"updateing z = {z} R = {self.kf.R} after update x = {self.kf.x} self.kf.K = {self.kf.K}")
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
    
    def getElements(self):
        state = self.getStateVector()
        x_elements = np.linspace(state[3], state[4], int(np.ceil(np.abs(state[4]-state[3]))*10))
        y_elements = state[0] + state[1] * x_elements + state[2] * x_elements**2
        elements = np.array([x_elements, y_elements]).T if self.fx_flag else np.array([y_elements, x_elements]).T
        
        return elements
    
    def getFxFlag(self):
        return self.fx_flag
    
class ExtendedObjectTrack:
    def __init__(self, x=None, P=None, create_frame_idx=0, gamma=0.995, fxFlag=True, prior=None):
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
        #self.kf.Q = np.zeros((5,5))
        #self.kf.Q[3,3] = 5
        #self.kf.Q[4,4] = 5
        self.kf.Q = np.diag([2e-3, 5e-4, 1e-7, 5, 5])
        self.saver = Saver(self.kf)
        self.create_frame_idx = create_frame_idx
        self.last_update_frame_idx = create_frame_idx
        self.last_frame_idx = create_frame_idx
        self.hits = [1]
        self.static_cars_flag = False
        self.counter_update = 0
        self.fx_flag = fxFlag
        self.prior = prior
        
    def predict(self, current_frame_idx):
        self.kf.predict()
        self.last_frame_idx = current_frame_idx
        
    def update(self, z, current_frame_idx, H=None, cov=None):
        if H is not None:
            self.kf.H = H
        if cov is not None:
            ha = np.array([-1 * self.kf.x[1] - 2*self.kf.x[2] * z[0], 1])
            Sigma_a = np.matmul(ha, np.matmul(cov, ha.T))
            self.kf.R = np.diag([cov[0,0], Sigma_a])
            #self.kf.R = np.diag([cov[0,0], cov[1,1,]])
            
        #print(f"updateing z = {z} R = {self.kf.R} before update x = {self.kf.x}")
        self.kf.update(z)
        #print(f"updateing z = {z} R = {self.kf.R} after update x = {self.kf.x} self.kf.K = {self.kf.K}")
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
    
    def isUpdated(self):
        return abs(self.last_update_frame_idx - self.last_frame_idx) < 3
    
    def getElements(self):
        state = self.getStateVector()
        x_elements = np.linspace(state[3], state[4], int(np.ceil(np.abs(state[4]-state[3]))*5))
        y_elements = state[0] + state[1] * x_elements + state[2] * x_elements**2
        elements = np.array([x_elements, y_elements]).T if self.fx_flag else np.array([y_elements, x_elements]).T
        
        return elements
    
    def getElementsInFOV(self, pose):
        elements = self.getElements()
        angle = np.arctan2(elements[:,1]-pose[1], elements[:,0]-pose[0])
        elements_in_fov = elements[abs(pose[2]-angle) < 1.047]
        
        return elements_in_fov
    
    def getFxFlag(self):
        return self.fx_flag
    
    def setPrior(self, prior):
        self.prior = prior
    
    def getPrior(self):
        return self.prior
    
    def setShape(self, shape):
        self.shape = shape
    
    def getShape(self):
        return self.shape