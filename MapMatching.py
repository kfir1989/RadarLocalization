import numpy as np
import math
import copy
import time
import scipy.stats
from scipy.spatial.distance import cdist

from utils import ExtObjectDataAssociator

class PF:
    def __init__(self, N):
        self.N = N
        self.ext_object_associator = ExtObjectDataAssociator(dim=2,deltaL=5,deltaS=1,deltaE=1)
        #Radar measurements estimated errors
        self.sigma_r = 0.2
        self.sigma_theta = 0.01
        #IMU measurements estimated errors
        self.sigma_rot1 = 0.0001
        self.sigma_trans = 0.05
        self.sigma_rot2 = 0.01
        self.current_odom_xy_theta = None
        
    @staticmethod
    def getXYLimits(extTrack):
        xs = extTrack.getStateVector()[3]
        xe = extTrack.getStateVector()[4]
        a0, a1, a2 = extTrack.getStateVector()[0], extTrack.getStateVector()[1], extTrack.getStateVector()[2]
        ys = a0 + a1 * xs + a2 * xs**2
        ye = a0 + a1 * xe + a2 * xe**2
        
        return xs,xe,ys,ye
    
    @staticmethod
    def world2Map(pos, worldRef, mapRef):
        pos[:, 0] -= worldRef[0]
        pos[:, 1] -= worldRef[1]
        dtheta = mapRef[2] - worldRef[2]
        
        transform_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                     [np.sin(dtheta), np.cos(dtheta)]])
        
        transformed_pose = np.dot(transform_matrix, pos)
        transformed_pose[0] += mapRef[0]
        transformed_pose[1] += mapRef[1]
        
        return transformed_pose
    
    @staticmethod
    def map2World(pos, worldRef, mapRef):
        pos[0] -= mapRef[0]
        pos[1] -= mapRef[1]
        dtheta = worldRef[2] - mapRef[2]
        transform_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                     [np.sin(dtheta), np.cos(dtheta)]])
        
        transformed_pose = np.dot(transform_matrix, pos)
        transformed_pose[0] += worldRef[0]
        transformed_pose[1] += worldRef[1]
        
        return transformed_pose
    
    @staticmethod
    def transformMap(roadMap, worldRef, particle):
        height, width = roadMap.shape[:2]
        padded_map = cv2.copyMakeBorder(roadMap, int(height/2), int(height/2), int(width/2), int(width/2), cv2.BORDER_CONSTANT, 0)
        height, width = padded_map.shape[:2]
        dx = worldRef[0]-particle["x"]
        dy = worldRef[1]-particle["y"]
        dtheta = worldRef[2]-particle["theta"]
        center = (width/2-dx, height/2-dy)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=dtheta, scale=1)
        transform_matrix = rotate_matrix
        transform_matrix[0,2] += dx
        transform_matrix[1,2] += dy

        transformed_map = cv2.warpAffine(src=padded_map, M=transform_matrix, dsize=(width, height))
        
        center = np.array([center[0],center[1],1])
        transformed_center = np.dot(transform_matrix,np.array(center))
        transformed_center = np.array([transformed_center[0], transformed_center[1], particle["theta"]])
        
        return transformed_map, transformed_center
   
    def computeDist(self, extTrack, sx, sy):
        state = extTrack.getStateVector()
        P = extTrack.getStateCovarianceMatrix()
        Ha = np.array([1, sx, sx**2])
        P_match = np.dot(np.dot(Ha, P[0:3,0:3]), Ha.T)
        y_match = state[0] + state[1] * sx + state[2] * sx**2
        
        #print(sx, sy, y_match, P_match, state)
        return self.ext_object_associator.calcLikelihood(sx, sy, y_match, P_match, state)
    
    def getClosestPointToDrivableArea(self, pos, drivableArea):
        drivable_area_indices = np.argwhere(drivableArea > 0)
        dists = cdist(pos, drivable_area_indices)
        return drivable_area_indices[np.argmin(dists)]

    def initialize_particles(self, num_particles, worldRef):
        particles = []

        for i in range(num_particles):
            particle = dict()

            #random position inside a unity circle
            t = 2*np.pi*np.random.uniform(0, 1)
            u = np.random.uniform(0, 1)+np.random.uniform(0, 1)
            r = 2-u if u>1 else u

            diameter = 0.1 #Estimated error in the beginning
            xc = worldRef[0]
            yc = worldRef[1]
            #initialize pose: at the beginning, robot is certain it is at [0,0,0]
            particle['x'] = diameter*r*np.cos(t) + xc
            particle['y'] = diameter*r*np.sin(t) + yc
            particle['theta'] = worldRef[2]

            #initial weight
            particle['weight'] = 1.0 / num_particles

            #particle history aka all visited poses
            particle['history'] = []

            #add particle to set
            particles.append(particle)

        self.particles = particles
    
    def sample_motion_model(self, new_odom_xy_theta, drivableArea):
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta is not None:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        for particle in self.particles:
            dr1 = math.atan2(delta[1], delta[0]) - old_odom_xy_theta[2]
            dr1_noisy = dr1 + np.random.normal(0, self.sigma_rot1)
            dt = math.sqrt((delta[0]**2) + (delta[1]**2))
            dt_noisy = dt + np.random.normal(0, self.sigma_trans)
            
            particle["theta"] += dr1
            particle["x"] += dt_noisy * np.cos(particle["theta"] + dr1_noisy)
            particle["y"] += dt_noisy * np.sin(particle["theta"] + dr1_noisy)
            particle["theta"] += (delta[2] - dr1_noisy)
            particle["theta"] = (particle["theta"] + np.pi) % (2 * np.pi) - np.pi
            if 1:
                R = np.array([[np.cos(-particle["theta"]), -np.sin(-particle["theta"])], [np.sin(-particle["theta"]), np.cos(-particle["theta"])]])
                xy = np.array([particle["x"], particle["y"]])
                xy_rotated = np.dot(R, xy)
                sigma_along_track = 0.1 if np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2]) < 0.003 else 0.05
                xy_rotated_plus_errs = xy_rotated + np.array([np.random.normal(0, sigma_along_track), np.random.normal(0, 0.05)])
                xy_rotated_back = np.dot(np.linalg.inv(R), xy_rotated_plus_errs)
                particle["x"] = xy_rotated_back[0]
                particle["y"] = xy_rotated_back[1]
                if 1:
                    x_on_map = round(particle["x"] - new_odom_xy_theta[0] + drivableArea.shape[0] / 2)
                    y_on_map = round(particle["y"] - new_odom_xy_theta[1] + drivableArea.shape[1] / 2)
                    closest_point = self.getClosestPointToDrivableArea([(y_on_map, x_on_map)], drivableArea)
                    dx = closest_point[1] - x_on_map
                    dy = closest_point[0] - y_on_map
                    #print("x_on_map", x_on_map, "y_on_map", y_on_map, "closest_point", closest_point, "dx", dx, "dy", dy, "particle[\"x\"]", particle["x"], "particle[\"y\"]", particle["y"])
                    #print("new_odom_xy_theta", new_odom_xy_theta, "particle[\"x\"]", particle["x"], "particle[\"y\"]", particle["y"], x_on_map, y_on_map, closest_point, dx, dy)
                    particle["x"] += dx
                    particle["y"] += dy
                    #print("after update: ", "particle[\"x\"]", particle["x"], "particle[\"y\"]", particle["y"])

    def eval_sensor_model(self, worldRef, extTracks, roadMap):
        #rate each particle
        for ext_track in extTracks:
            for particle in self.particles:
                P_total = 1e-6 # for accumulating probabilities
                #transform map according to worldRef and particle info
                transformed_map, transformed_center = self.transformMap(roadMap, worldRef, particle)
                world_xs,world_xe,world_ys,world_ye = self.getXYLimits(ext_track)
                map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, transformed_center)
                map_xs,map_xe,map_ys,map_ye = int(min(map_limits[0,0], map_limits[1,0])), int(max(map_limits[0,0], map_limits[1,0])), int(min(map_limits[0,1], map_limits[1,1])), int(max(map_limits[0,1], map_limits[1,1]))
                (row,col) = np.where(transformed_map[map_xs:map_xe,map_ys:map_ye])
                for imap in range(0,row.shape[0]):
                    seg_pos = self.map2World(np.array([row[imap]+map_xs, col[imap]+map_ys]), worldRef, transformed_center)
                    sx = seg_pos[0]
                    sy = seg_pos[1]
                    #print("worldRef", worldRef, "particle", particle, "np.array([row[imap]+map_xs, col[imap]+map_ys])", np.array([row[imap]+map_xs, col[imap]+map_ys]), "map_xe", map_xe, "map_ye", map_ye)
                    #calculate likelihood of matching between polynom and map-segment
                    Pi = self.computeDist(ext_track, sx, sy)
                    #combine (independent) measurements
                    P_total += Pi
                particle['weight'] = P_total

        #normalize weights
        normalizer = sum([p['weight'] for p in self.particles])

        for particle in self.particles:
            particle['weight'] = particle['weight'] / normalizer

    def resample_particles(self):
        # Returns a new set of particles obtained by performing
        # stochastic universal sampling, according to the particle 
        # weights.

        new_particles = []

        '''your code here'''
        # Uniform distribution value
        uni_val = 1.0/len(self.particles)
        # Uniform distribution
        u = np.random.uniform(0,uni_val)
        # Initialize total weight
        w = 0

        #loop over all particle weights
        for particle in self.particles:
            w += particle['weight']
            while u <= w:
                new_particle = copy.deepcopy(particle)
                new_particle['weight'] = uni_val
                new_particles.append(new_particle)
                u += uni_val
        '''***        ***'''
        
        self.particles = new_particles
        
    def getBestParticle(self):
        max_w = 0
        best_particle = []
        for particle in self.particles:
            if particle['weight'] > max_w:
                max_w = particle['weight']
                best_particle = particle
                
        return best_particle
    
    def getMean(self, particles):
        mean_particle = {'x': 0, 'y': 0, 'theta': 0}
        mean_particle['x'] = sum([p['x'] for p in particles]) / len(particles)
        mean_particle['y'] = sum([p['y'] for p in particles]) / len(particles)
        mean_particle['theta'] = sum([p['theta'] for p in particles]) / len(particles)
                
        return mean_particle
    
    def calcDist(self, p1, p2):
        dist = np.linalg.norm(np.array([p1['x'], p1['y']]) - np.array([p2['x'], p2['y']]))
        
        return dist
    
    def getSigmaClipped(self, nSigma=1.05):
        E = self.getMean(self.particles)
        var_dist = sum([self.calcDist(p, E) for p in self.particles]) / len(self.particles)
        particles_within_sigma = [p for p in self.particles if (self.calcDist(p, E) <= var_dist * nSigma)]
        E = self.getMean(particles_within_sigma)
        
        return E      

import cv2
from nuscenes.map_expansion.map_api import NuScenesMap

class MapMatching:
    def __init__(self, N=100):
        self.N = N
        self.pf = PF(N=N)
        
    def getRoadBorders(self, nuscMap, worldRef, layer_names=['walkway'],blur_area=(3,3),threshold1=0.5, threshold2=0.7):
        patch_angle = 0
        patch_size = 200
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        mask_blur = cv2.GaussianBlur(mask, blur_area, 0)
        edges = cv2.Canny(image=mask_blur, threshold1=threshold1, threshold2=threshold2) # Canny Edge Detection
        
        return edges
    
    def getMapReference(self, nuscMap, worldRef):
        edges1 = self.getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, layer_names=['walkway'],blur_area=(3,3),threshold1=0.5, threshold2=0.9)
        edges2 = self.getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0.5, threshold2=0.9)
        edges = edges1 & edges2
        
        return edges2
    
    def getDrivableArea(self, nuscMap, worldRef, layer_names = ['drivable_area'], patch_size=50):
        patch_angle = 0
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        
        return mask

    def run(self, extTracks, nuscMap, origWorldRef, worldRef, R, heading, odometry):
        orig_world_ref = np.array([origWorldRef[0],origWorldRef[1], np.deg2rad(heading-90)])
        world_ref = np.array([worldRef[0],worldRef[1], np.deg2rad(heading-90)])
        if self.pf.current_odom_xy_theta is None:
            self.pf.initialize_particles(self.N, world_ref)
        drivable_area = self.getDrivableArea(nuscMap, orig_world_ref)
        self.pf.sample_motion_model(world_ref, drivable_area)
        road_map = self.getMapReference(nuscMap, orig_world_ref)
        self.pf.eval_sensor_model(world_ref, extTracks, road_map)
        self.best_particle = self.pf.getBestParticle()
        self.pf.resample_particles()
        
    def getResults(self):
        mean_particle = self.pf.getSigmaClipped()
        best_particle = self.best_particle
        mean_pos = np.array([mean_particle['x'], mean_particle['y']])
        best_pos = np.array([best_particle['x'], best_particle['y']])
        results = {"all_particles": self.pf.particles, "pf_best_pos": best_pos, "pf_best_theta": best_particle['theta'], "pf_mean_pos": mean_pos, "pf_mean_theta": mean_particle['theta']}
        
        return results
        