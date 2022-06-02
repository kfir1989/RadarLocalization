import numpy as np
import math
import copy
import time
import scipy.stats
from scipy.spatial.distance import cdist

from utils import ExtObjectDataAssociator
from map_utils import getRoadBorders, getCombinedMap

class PF:
    def __init__(self, N):
        self.N = N
        self.ext_object_associator = ExtObjectDataAssociator(dim=2,deltaL=5,deltaS=1,deltaE=1)
        #Radar measurements estimated errors
        self.sigma_r = 0.2
        self.sigma_theta = 0.01
        #IMU measurements estimated errors
        self.sigma_rot1 = 0.0001
        self.sigma_trans = 0.02
        self.sigma_rot2 = 0.01
        self.current_odom_xy_theta = None
        self.uncertainty_flag = False
        
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
        if 0:
            dtheta = mapRef[2] - worldRef[2]

            transform_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                         [np.sin(dtheta), np.cos(dtheta)]])

            transformed_pose = np.dot(transform_matrix, pos)
            transformed_pose[0] += mapRef[0]
            transformed_pose[1] += mapRef[1]
            
            return transformed_pose
        else:
            pos[:, 0] += mapRef[0]
            pos[:, 1] += mapRef[1]
            
            return pos

    @staticmethod
    def map2World(pos, worldRef, mapRef):
        pos[0] -= mapRef[0]
        pos[1] -= mapRef[1]
        if 0:
            dtheta = worldRef[2] - mapRef[2]
            transform_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                         [np.sin(dtheta), np.cos(dtheta)]])

            transformed_pose = np.dot(transform_matrix, pos)
            transformed_pose[0] += worldRef[0]
            transformed_pose[1] += worldRef[1]
        
            return transformed_pose
        else:
            pos[0] += worldRef[0]
            pos[1] += worldRef[1]
            
            return pos
    
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
        
        #print("dx,dy",dx,dy)

        transformed_map = cv2.warpAffine(src=padded_map, M=transform_matrix, dsize=(width, height))
        
        center = np.array([center[0],center[1],1])
        transformed_center = np.dot(transform_matrix,np.array(center))
        transformed_center = np.array([transformed_center[0], transformed_center[1], particle["theta"]])
        #print("transformed_center", transformed_center)
        
    @staticmethod
    def transformPolynom(polynom, worldRef, particle):
        #print("worldRef[0]", worldRef[0], "worldRef[1]", worldRef[1], "particle[x]", particle["x"], "particle[y]", particle["y"])
        dx = worldRef[0]-particle["x"]
        dy = worldRef[1]-particle["y"]
        dtheta = worldRef[2]-particle["theta"]
        center = (particle["x"], particle["y"])
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=dtheta, scale=1)
        transform_matrix = rotate_matrix
        transform_matrix[0,2] -= dx
        transform_matrix[1,2] -= dy
        
        b = np.ones((polynom.shape[0],1))
        polynom_expanded = np.hstack((polynom,b))
        
        transformed_polynom = np.dot(transform_matrix,polynom_expanded.T)

        return transformed_polynom.T
    
    def calcTrkWeight(self, extTrack):
        if 0:
            trk_age = extTrack.last_update_frame_idx - extTrack.create_frame_idx
            trk_n_updates = extTrack.counter_update
            alpha = 0.5
            weight = (alpha * trk_age + (1-alpha) * trk_n_updates) * 0.01
        else:
            weight = 1
        
        return weight
   
    def computeDist(self, extTrack, sx, sy, heading):
        state = extTrack.getStateVector()
        x_on_polynom = np.linspace(state[3], state[4], 100)
        y_on_polynom = state[0] + state[1] * x_on_polynom + state[2] * x_on_polynom**2
        points_on_polynom = np.array([x_on_polynom, y_on_polynom]).T
        map_point = np.array([sx, sy]).reshape(1,2)
        it = np.argmin(np.linalg.norm(points_on_polynom - map_point,axis=1),axis=0)

        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        dxdy = points_on_polynom[it, :]-map_point
        dxdy_rotated = np.dot(R, dxdy.T)
        sigma_along_track = 0.5
        sigma_cross_track = 1
        P = np.array([[sigma_along_track, 0],[0, sigma_cross_track]])
        det = np.linalg.det(P)
        dist = np.dot(np.dot(dxdy_rotated.T,np.linalg.inv(P)),dxdy_rotated)
        
        #l = np.power((2*np.pi), -0.5*2) * np.power(det, -0.5) * np.exp(-0.5 * dist)

        #print("dxdy", dxdy, "dxdy_rotated", dxdy_rotated, "dist", dist, "l", l)
        return dist
        
    def computeDist(self, polynom, boundaryPoints, heading, debug=False):
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        
        polynom_rotated = np.dot(R, polynom.T).T
        boundary_rotated = np.dot(R, boundaryPoints.T).T
        dx = polynom_rotated[:,0].reshape(-1,1)-boundary_rotated[:,0].reshape(1,-1)
        dy = polynom_rotated[:,1].reshape(-1,1)-boundary_rotated[:,1].reshape(1,-1)
        dxdy_norm = np.sqrt(5*dx**2 + 0.5*dy**2) #dx has higher priority

        it = dxdy_norm.argmin(axis=1)
        dx_min = np.take_along_axis(dx, np.expand_dims(it, axis=-1), axis=1)
        dy_min = np.take_along_axis(dy, np.expand_dims(it, axis=-1), axis=1)
        dxdy_min = np.array([dx_min, dy_min])
        dxdy_min = np.squeeze(dxdy_min, -1) if dxdy_min.ndim == 3 else dxdy_min
        
        sigma_along_track = 0.2
        sigma_cross_track = 4
        P = np.array([[sigma_along_track, 0],[0, sigma_cross_track]])
        
        dist = np.sum(np.dot((dxdy_min.T)**2,np.linalg.inv(P)), axis=1)
        
        #take 20% highest
        n_elements = int(max(1, polynom.shape[0] / 5))
        #print(dist.shape, n_elements)
        top_ind = np.argpartition(dist, -n_elements)[-n_elements:]
        top20_dist = dist[top_ind]
        total_dist = np.mean(top20_dist)
        if debug:
            #print("total_dist", total_dist, "top20_dist", top20_dist)
            print("total_dist", total_dist, "polynom", polynom, "boundaryPoints", boundaryPoints, "dxdy_min", dxdy_min.T, "(dxdy_min.T)**2", (dxdy_min.T)**2, "top20_dist", top20_dist, "n_elements", n_elements, "top_ind", top_ind, "it", it)
            #raise Exception('Found ya!')

        return total_dist
        
    
    def getClosestPointToDrivableArea(self, pos, drivableArea):
        drivable_area_indices = np.argwhere(drivableArea > 0)
        dists = cdist(pos, drivable_area_indices)
        return drivable_area_indices[np.argmin(dists)]
    
    def eval_polynom_map_match(self, ext_track, particle, worldRef, roadMap, debug=False):
        height, width = roadMap.shape[:2]
        map_center = (width/2, height/2)
        weight = self.calcTrkWeight(ext_track)
        state = ext_track.getStateVector()
        x_polynom = np.linspace(state[3], state[4], int(np.ceil(np.abs(state[4]-state[3]))*10))
        y_polynom = state[0] + state[1] * x_polynom + state[2] * x_polynom**2
        polynom = np.array([x_polynom, y_polynom]).T
        transformed_polynom = self.transformPolynom(polynom, worldRef, particle)
        #print(transformed_polynom.shape)
        world_xs,world_xe,world_ys,world_ye = transformed_polynom[0,0], transformed_polynom[-1,0], transformed_polynom[0,1], transformed_polynom[-1,1]
        #print("world_xs", world_xs, "world_xe", world_xe, "world_ys", world_ys, "world_ye", world_ye)
        padding = 5
        map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, map_center)
        map_xs,map_xe,map_ys,map_ye = max(0, int(min(map_limits[0,0], map_limits[1,0])) - padding), min(width-1, int(max(map_limits[0,0], map_limits[1,0])) + padding), max(0, int(min(map_limits[0,1], map_limits[1,1])) - padding), min(height-1, int(max(map_limits[0,1], map_limits[1,1])) + padding)
        #print("map_xs", map_xs, "map_xe", map_xe, "map_ys", map_ys, "map_ye", map_ye)
        (row,col) = np.where(roadMap[map_ys:map_ye,map_xs:map_xe])
        if row.shape[0] > 0:
            boundary_points = self.map2World(np.array([col+map_xs, row+map_ys]).astype('float64'), worldRef, map_center)
            cost = self.computeDist(polynom=transformed_polynom, boundaryPoints=boundary_points.T, heading=worldRef[2],debug=debug)
            #combine (independent) measurements
            cost *= weight
        else:
            cost = 1e3
           
        return cost

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
                
                var_dist, _ = self.getVariance()
                var_cond = var_dist < 5
                self.uncertainty_flag = self.uncertainty_flag & var_cond
                sigma_factor_along_track = 1 if self.uncertainty_flag else 0
                sigma_factor_cross_track = 0.2
                sigma_along_track = (dt / 10) if np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2]) < 0.003 else (dt / 100)
                error_along_track = np.random.normal(0, sigma_along_track + sigma_factor_along_track)
                sigma_cross_track = 0
                error_cross_track = np.random.normal(0, sigma_factor_cross_track) if self.uncertainty_flag else 0
                xy_rotated_plus_errs = xy_rotated + np.array([error_along_track, error_cross_track])
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
                    
        self.uncertainty_flag = False

    def eval_sensor_model_old(self, worldRef, extTracks, roadMap):
        #rate each particle
        for ext_track in extTracks:
            weight = self.calcTrkWeight(ext_track)
            for particle in self.particles:
                cost_total = 1e-6 # for accumulating probabilities
                #transform map according to worldRef and particle info
                transformed_map, transformed_center = self.transformMap(roadMap, worldRef, particle)
                world_xs,world_xe,world_ys,world_ye = self.getXYLimits(ext_track)
                map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, transformed_center)
                map_xs,map_xe,map_ys,map_ye = int(min(map_limits[0,0], map_limits[1,0])), int(max(map_limits[0,0], map_limits[1,0])), int(min(map_limits[0,1], map_limits[1,1])), int(max(map_limits[0,1], map_limits[1,1]))
                (row,col) = np.where(transformed_map[map_ys:map_ye,map_xs:map_xe])
                for imap in range(0,row.shape[0]):
                    seg_pos = self.map2World(np.array([row[imap]+map_ys, col[imap]+map_xs]), worldRef, transformed_center)
                    sy = seg_pos[0]
                    sx = seg_pos[1]
                    #print("worldRef", worldRef, "particle", particle, "np.array([row[imap]+map_xs, col[imap]+map_ys])", np.array([row[imap]+map_xs, col[imap]+map_ys]), "map_xe", map_xe, "map_ye", map_ye)
                    #calculate likelihood of matching between polynom and map-segment
                    cost = self.computeDist(extTrack=ext_track, sx=sx, sy=sy, heading=worldRef[2])
                    #combine (independent) measurements
                    cost_total += weight * cost
                particle['weight'] = 1 / cost_total #cost instead of probability

        #normalize weights
        normalizer = sum([p['weight'] for p in self.particles])

        for particle in self.particles:
            particle['weight'] = particle['weight'] / normalizer
            
    def eval_sensor_model(self, worldRef, extTracks, roadMap):
        min_cost = 1e6
        for particle in self.particles:
            cost_total = 1e-6 # for accumulating probabilities
            for ext_track in extTracks:
                cost = self.eval_polynom_map_match(ext_track, particle, worldRef, roadMap)
                min_cost = min(min_cost, cost)
                cost_total += cost
            particle['weight'] = 1 / cost_total #cost instead of probability

        #normalize weights
        normalizer = sum([p['weight'] for p in self.particles])

        for particle in self.particles:
            particle['weight'] = particle['weight'] / normalizer
            
        self.uncertainty_flag = True if (min_cost > 12 and len(extTracks) > 1) else False

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
    
    def calcDistXY(self, p1, p2):
        distX = np.linalg.norm(np.array([p1['x']]) - np.array([p2['x']]))
        distY = np.linalg.norm(np.array([p1['y']]) - np.array([p2['y']]))
        
        return np.array([distX,distY])
    
    def getVariance(self):
        E = self.getMean(self.particles)
        var_dist = sum([self.calcDist(p, E) for p in self.particles]) / len(self.particles)
        
        return var_dist, E
    
    def getCovarianceMatrix(self):
        xy = np.array([[p['x'], p['y']] for p in self.particles]).T
        cov = np.cov(xy)

        return cov
    
    def getSigmaClipped(self, nSigma=1.05):
        var_dist, E = self.getVariance()
        particles_within_sigma = [p for p in self.particles if (self.calcDist(p, E) <= var_dist * nSigma)]
        E = self.getMean(particles_within_sigma)
        
        return E      

import cv2
from nuscenes.map_expansion.map_api import NuScenesMap

class MapMatching:
    def __init__(self, N=100):
        self.N = N
        self.pf = PF(N=N) 
        self.counter = 0
    
    def getMapReference(self, nuscMap, worldRef):
        edges1 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=200, layer_names=['walkway'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges2 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=200, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges = edges1 | edges2
        
        return edges #take only borders of drivable area
    
    def getDrivableArea(self, nuscMap, worldRef, layer_names = ['drivable_area'], patch_size=50):
        patch_angle = 0
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        
        return mask

    def run(self, extTracks, nuscMap, origWorldRef, worldRef, origRadarRef, radarRef, R, heading, odometry):
        orig_world_ref = np.array([origWorldRef[0],origWorldRef[1], np.deg2rad(heading-90)])
        world_ref = np.array([worldRef[0],worldRef[1], np.deg2rad(heading-90)])
        radar_ref = np.array([radarRef[0],radarRef[1], np.deg2rad(heading-90)])
        #print("radar_ref", radar_ref, "world_ref", world_ref)
        if self.pf.current_odom_xy_theta is None:
            self.pf.initialize_particles(self.N, world_ref)
        drivable_area = self.getDrivableArea(nuscMap, orig_world_ref)
        self.pf.sample_motion_model(world_ref, drivable_area)
        road_map = self.getMapReference(nuscMap, origRadarRef)
        self.pf.eval_sensor_model(radar_ref, extTracks, road_map)
        self.best_particle = self.pf.getBestParticle()
        self.pf.resample_particles()
        
        #DEBUG
        self.cost_true = []
        self.cost_mean = []
        debug = True if self.counter > 1000 else False
        for ext_track in extTracks:
            true_pos_particle = {"x": radar_ref[0], "y": radar_ref[1], "theta": radar_ref[2]}
            cost_true = self.pf.eval_polynom_map_match(ext_track, true_pos_particle, radar_ref, road_map, debug=debug)
            cost_best = self.pf.eval_polynom_map_match(ext_track, self.best_particle, radar_ref, road_map, debug=debug)
            mean_particle = self.pf.getSigmaClipped()
            cost_mean = self.pf.eval_polynom_map_match(ext_track, mean_particle, radar_ref, road_map, debug=debug)
            #print("cost_gt", cost_true, "cost_best", cost_best, "cost_mean", cost_mean)
            #print("gt", true_pos_particle, "best", self.best_particle, "mean", mean_particle)
            self.cost_true.append(cost_true)
            self.cost_mean.append(cost_mean)
        
    def getResults(self):
        covariance = self.pf.getCovarianceMatrix()
        mean_particle = self.pf.getSigmaClipped()
        best_particle = self.best_particle
        mean_pos = np.array([mean_particle['x'], mean_particle['y']])
        best_pos = np.array([best_particle['x'], best_particle['y']])
        results = {"all_particles": self.pf.particles, "pf_best_pos": best_pos, "pf_best_theta": best_particle['theta'], "pf_mean_pos": mean_pos, "pf_mean_theta": mean_particle['theta'], "cost_true": self.cost_true, "cost_mean": self.cost_mean, "covariance": covariance}
        
        return results
        