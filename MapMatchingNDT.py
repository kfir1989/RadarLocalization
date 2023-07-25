import numpy as np
import math
import copy
import time
import scipy.stats
from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splev

from utils import ExtObjectDataAssociator
from map_utils import getRoadBorders, getCombinedMap
from map_utils import build_probability_map, scatter_to_image, drawLanes

class PF:
    def __init__(self, N):
        self.N = N
        self.ext_object_associator = ExtObjectDataAssociator(dim=2,deltaL=5,deltaS=1,deltaE=1)
        #Radar measurements estimated errors
        self.sigma_r = 0.2
        self.sigma_theta = 0.01
        #IMU measurements estimated errors
        self.sigma_rot1 = 0.0005
        self.sigma_trans = 0.02
        self.sigma_rot2 = 0.01
        self.current_odom_xy_theta = None
        self.uncertainty_flag = False
        self.imu_bias_factor = 0
        self.cond3_counter = 0
        self.wheel_radius_bias = 1
        
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
    def sigma_clip(x, alpha=2):
        mean = np.mean(x)
        var = np.std(x)
        x[x-mean > alpha * var] = mean
        
        return x

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
    
    @staticmethod
    def transformDynTrack(tstate, worldRef, particle):
        dx = worldRef[0]-particle["x"]
        dy = worldRef[1]-particle["y"]
        dtheta = worldRef[2]-particle["theta"]
        center = (particle["x"], particle["y"])
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=dtheta, scale=1)
        transform_matrix = rotate_matrix
        transform_matrix[0,2] -= dx
        transform_matrix[1,2] -= dy
        
        b = np.ones((tstate.shape[0],1))
        trk_expanded = np.hstack((tstate,b))
        
        transformed_trk = np.dot(transform_matrix,trk_expanded.T)

        return transformed_trk.T
    
    def calcTrkWeight(self, extTrack):
        if 0:
            trk_age = extTrack.last_update_frame_idx - extTrack.create_frame_idx
            trk_n_updates = extTrack.counter_update
            alpha = 0.5
            weight = (alpha * trk_age + (1-alpha) * trk_n_updates) * 0.01
        else:
            weight = 1
        
        return weight
    
    def calcScore(self, probMap, scatter, mapRes, mapCenter):
        im = scatter_to_image(scatter, res_factor=mapRes, center = mapCenter, patch_size=300)
        score = im*probMap
        return score
        
    
    def computeScoreForDynamicTracks(self, trk):
        trk_on_map = trk
        score = self.calcScore(probMap=self.lanes_map, scatter=trk_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        score = np.sum(np.sum(score))
        return score
        
    def computeScoreForStaticTracks(self, polynom, boundaryPoints):
        polynom_on_map = polynom
        high_sigma_score = self.calcScore(probMap=self.static_sigma4_map, scatter=polynom_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        
        if boundaryPoints is not None:
            #Find median point on the polynom
            polynom_mid_point = polynom[int(polynom.shape[0]/2),:]
            #Find nearest neighbor point on the map
            eucld_dist = np.linalg.norm(boundaryPoints-polynom_mid_point,axis=1)
            it = eucld_dist.argmin()
            #Calc translation
            txy = boundaryPoints[it,:] - polynom_mid_point
            if(np.linalg.norm(txy)) < 4:
                #Translate polynom
                polynom_translated = polynom + txy
                #Compute low sigma score for translated polynom(shape)
                low_sigma_score = self.calcScore(probMap=self.static_sigma1_map, scatter=polynom_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
            else:
                #Compute low sigma score for non-translated polynom(shape)
                low_sigma_score = self.calcScore(probMap=self.static_sigma1_map, scatter=polynom_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        else:
            #Compute low sigma score for non-translated polynom(shape)
            low_sigma_score = self.calcScore(probMap=self.static_sigma1_map, scatter=polynom_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        
        return np.sum(np.sum(low_sigma_score * high_sigma_score))
        
    
    def getClosestPointToDrivableArea(self, pos, drivableArea):
        drivable_area_indices = np.argwhere(drivableArea > 0)
        dists = cdist(pos, drivable_area_indices)
        return drivable_area_indices[np.argmin(dists)]
    
    def eval_polynom_map_match(self, ext_track, particle, worldRef):
        height, width = self.road_map.shape[:2]
        map_center = (width/2, height/2)
        weight = self.calcTrkWeight(ext_track)
        polynom = ext_track.getElements()
        transformed_polynom = self.transformPolynom(polynom, worldRef, particle)
        #transform polynom to the particles' perspective
        world_xs,world_xe,world_ys,world_ye = transformed_polynom[0,0], transformed_polynom[-1,0], transformed_polynom[0,1], transformed_polynom[-1,1]
        #get relevant patch from map
        padding = 20
        map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, map_center)
        map_xs,map_xe,map_ys,map_ye = max(0, int(min(map_limits[0,0], map_limits[1,0])) - padding), min(width-1, int(max(map_limits[0,0], map_limits[1,0])) + padding), max(0, int(min(map_limits[0,1], map_limits[1,1])) - padding), min(height-1, int(max(map_limits[0,1], map_limits[1,1])) + padding)
        (row,col) = np.where(self.road_map[map_ys:map_ye,map_xs:map_xe])
        #if there are any boundary points in the map
        boundary_points = None
        if row.shape[0] > 0:
            boundary_points = self.map2World(np.array([col+map_xs, row+map_ys]).astype('float64'), worldRef, map_center)
            boundary_points = boundary_points.T
        
        score = self.computeScoreForStaticTracks(polynom=transformed_polynom, boundaryPoints=boundary_points)
        score *= weight
           
        return score
    
    def eval_dynamic_track_map_match(self, trk, particle, worldRef, firstWorldRef, debug=False):
        cost = 0
        if trk.confirmed and trk.hits > 10:
            tstate, tspeed = trk.getTranslatedState()
            abs_vel = np.mean(np.linalg.norm(tspeed,axis=1), axis=0)
            if abs_vel > 2:
                tstate = np.squeeze(tstate,axis=2) - firstWorldRef
                transformed_dyn_track = self.transformDynTrack(tstate, worldRef, particle)
                cost = self.computeScoreForDynamicTracks(transformed_dyn_track[-10:,:]) # consider only last 10 updates!
           
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
        self.ins_bounds = {"x1": 0, "x2": 0, "y1": 0, "y2": 0}
    
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
        
        dr1 = math.atan2(delta[1], delta[0]) - old_odom_xy_theta[2]
        dt = math.sqrt((delta[0]**2) + (delta[1]**2))
        #uncertainty test
        uncertainty_flag = self.improvishment_flag
        large_angle_turn_flag = np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2]) > 0.02
        for particle in self.particles:           
            dr1_noisy = dr1 + np.random.normal(0, self.sigma_rot1)
            dt_noisy = dt * self.wheel_radius_bias + np.random.normal(0, self.sigma_trans)
            #Add measurement noise
            particle["theta"] += dr1
            particle["x"] += dt_noisy * np.cos(particle["theta"] + dr1_noisy)
            particle["y"] += dt_noisy * np.sin(particle["theta"] + dr1_noisy)
            particle["theta"] += (delta[2] - dr1_noisy)
            #particle["theta"] += imu_angle_bias
            particle["theta"] = (particle["theta"] + np.pi) % (2 * np.pi) - np.pi
            #Add along-track and cross-track noise (for higher flexibility)
            R = np.array([[np.cos(-particle["theta"]), -np.sin(-particle["theta"])], [np.sin(-particle["theta"]), np.cos(-particle["theta"])]])
            xy = np.array([particle["x"], particle["y"]])
            xy_rotated = np.dot(R, xy)
            

            sigma_factor_along_track = 0.04 if uncertainty_flag else 1e-8
            sigma_factor_cross_track = 0.02 if uncertainty_flag else 1e-8
            sigma_along_track = (dt / 25) if not large_angle_turn_flag else (dt / 50)
            error_along_track = np.random.normal(0, sigma_along_track + sigma_factor_along_track)
            sigma_cross_track = dt / 80
            error_cross_track = np.random.normal(0, sigma_cross_track + sigma_factor_cross_track)
            xy_rotated_plus_errs = xy_rotated + np.array([error_along_track, error_cross_track])
            xy_rotated_back = np.dot(np.linalg.inv(R), xy_rotated_plus_errs)
            particle["x"] = xy_rotated_back[0]
            particle["y"] = xy_rotated_back[1]         
            #Add constraints: drivable area, max bias of INS
            #particle = self.getClosestToINSBoundaries(particle)
            x_on_map = round(particle["x"] - new_odom_xy_theta[0] + drivableArea.shape[0] / 2)
            y_on_map = round(particle["y"] - new_odom_xy_theta[1] + drivableArea.shape[1] / 2)
            closest_point = self.getClosestPointToDrivableArea([(y_on_map, x_on_map)], drivableArea)
            dx = closest_point[1] - x_on_map
            dy = closest_point[0] - y_on_map
            particle["x"] += dx
            particle["y"] += dy

    def eval_sensor_model(self, worldRef, extTracks, dynTracks, firstWorldRef, mapCenter):
        w = 5
        for particle in self.particles:
            total_score = 1e-6 # for accumulating probabilities
            for i_ext_track, ext_track in enumerate(extTracks):
                score = self.eval_polynom_map_match(ext_track, particle, worldRef)
                total_score += score
                
            for i_dyn, dyn_track in enumerate(dynTracks):
                score = self.eval_dynamic_track_map_match(dyn_track, particle, worldRef, firstWorldRef)
                total_score += w * score
            
            particle['weight'] = total_score

        #normalize weights
        normalizer = sum([p['weight'] for p in self.particles])

        for particle in self.particles:
            particle['weight'] = particle['weight'] / normalizer

    def resample_particles(self):
        # Returns a new set of particles obtained by performing
        # stochastic universal sampling, according to the particle 
        # weights.
        
        ESS = 1 / sum([p['weight']**2 for p in self.particles])
        
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
        
        self.improvishment_flag = False
        if ESS < len(self.particles)/2:
            #Add direct noise next time
            print("improvishment_flag is ON!")
            self.improvishment_flag = True
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
    
    def setNDTMaps(self, road_map, static_sigma1_map, static_sigma4_map, lanes_map, map_reference_point, res_factor=1.):
        self.road_map = road_map
        self.static_sigma1_map = static_sigma1_map
        self.static_sigma4_map = static_sigma4_map
        self.lanes_map = lanes_map
        self.map_res_factor = res_factor
        self.map_reference_point = map_reference_point

import cv2
from nuscenes.map_expansion.map_api import NuScenesMap

class MapMatching:
    def __init__(self, N=100):
        self.N = N
        self.pf = PF(N=N) 
        self.pf.counter = 0
        self.pf.imu_path = np.zeros([2000,2])
        self.pf.pf_path = np.zeros([2000,2])
    
    def getMapReference(self, nuscMap, worldRef):
        edges1 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=300, layer_names=['walkway'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges2 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=300, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges = edges1 | edges2
        
        return edges #take only borders of drivable area
    
    def getDrivableArea(self, nuscMap, worldRef, layer_names = ['drivable_area'], patch_size=50):
        patch_angle = 0
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        
        return mask

    def run(self, extTracks, nuscMap, dynTracks, lanes, firstWorldRef, IMURelativeRef):
        try:
            last_output = np.array([self.output_position['x'], self.output_position['y']])
        except:
            last_output = IMURelativeRef[0:2]
            
        self.pf.imu_path[self.pf.counter, :] = IMURelativeRef[0:2]
        self.pf.pf_path[self.pf.counter, :] = last_output

        res_factor = 5
        #lanes = np.concatenate(lanes, axis=0)
        if self.pf.current_odom_xy_theta is None:
            self.pf.initialize_particles(self.N, IMURelativeRef)
        drivable_area = self.getDrivableArea(nuscMap, firstWorldRef + last_output)
        self.pf.sample_motion_model(IMURelativeRef, drivable_area)
        #road_map = self.getMapReference(nuscMap, origRadarRef)
        self.road_map = getCombinedMap(nuscMap, firstWorldRef+last_output, patchSize=300, res_factor=res_factor)
        self.static_sigma1_map = build_probability_map(self.road_map, sigma=2.)
        self.static_sigma4_map = build_probability_map(self.road_map, sigma=8.)
        sparse_scatter = drawLanes(nuscMap, ego_trns=firstWorldRef+last_output)
        lanes = scatter_to_image(sparse_scatter, center=firstWorldRef+last_output, res_factor=res_factor, patch_size=300)
        lanes[lanes == 1] = 255
        self.lanes_map = build_probability_map(lanes, sigma=3.)
        self.pf.setNDTMaps(road_map=self.road_map, static_sigma1_map=self.static_sigma1_map, static_sigma4_map=self.static_sigma1_map, 
                          lanes_map=self.lanes_map, map_reference_point=last_output, res_factor=res_factor)
        self.pf.eval_sensor_model(IMURelativeRef, extTracks, dynTracks, firstWorldRef, mapCenter=last_output)
        self.best_particle = self.pf.getBestParticle()
        self.pf.resample_particles()
        self.pf.counter += 1
        
    def getResults(self, extTracks, dynTracks, firstWorldRef, gtRelativeRef, imuRelativeRef, lanes):
        covariance = self.pf.getCovarianceMatrix()
        mean_particle = self.pf.getSigmaClipped()
        self.output_position = mean_particle
        best_particle = self.best_particle
        mean_pos = np.array([mean_particle['x'], mean_particle['y']])
        best_pos = np.array([best_particle['x'], best_particle['y']])
        
        mean_particle = self.pf.getSigmaClipped()
        
        #DEBUG
        self.cost_true = []
        self.cost_mean = []
        debug = True if self.pf.counter > 2000 else False
        for ext_track in extTracks:
            true_pos_particle = {"x": gtRelativeRef[0], "y": gtRelativeRef[1], "theta": gtRelativeRef[2]}
            cost_true = self.pf.eval_polynom_map_match(ext_track, true_pos_particle, imuRelativeRef)
            cost_best = self.pf.eval_polynom_map_match(ext_track, self.best_particle, imuRelativeRef)
            cost_mean = self.pf.eval_polynom_map_match(ext_track, mean_particle, imuRelativeRef)
            #print("cost_gt", cost_true, "cost_best", cost_best, "cost_mean", cost_mean)
            #print("gt", true_pos_particle, "best", self.best_particle, "mean", mean_particle)
            self.cost_true.append(cost_true)
            self.cost_mean.append(cost_mean)
            
        self.cost_dyn_true = []
        self.cost_dyn_mean = []
        lanes = np.concatenate(lanes, axis=0)
        for dyn_track in dynTracks:
            if dyn_track.confirmed and dyn_track.hits > 10:
                true_pos_particle = {"x": gtRelativeRef[0], "y": gtRelativeRef[1], "theta": gtRelativeRef[2]}
                cost_true = self.pf.eval_dynamic_track_map_match(dyn_track, true_pos_particle, imuRelativeRef, firstWorldRef)
                cost_best = self.pf.eval_dynamic_track_map_match(dyn_track, self.best_particle, imuRelativeRef, firstWorldRef)
                cost_mean = self.pf.eval_dynamic_track_map_match(dyn_track, mean_particle, imuRelativeRef, firstWorldRef)
                #print("cost_gt", cost_true, "cost_best", cost_best, "cost_mean", cost_mean)
                #print("gt", true_pos_particle, "best", self.best_particle, "mean", mean_particle)
                self.cost_dyn_true.append(cost_true)
                self.cost_dyn_mean.append(cost_mean)
        
        results = {"all_particles": self.pf.particles, "pf_best_pos": best_pos, "pf_best_theta": best_particle['theta'], "pf_mean_pos": mean_pos, "pf_mean_theta": mean_particle['theta'], "cost_true": self.cost_true, "cost_mean": self.cost_mean, "covariance": covariance, "cost_dyn_true": self.cost_dyn_true, "cost_dyn_mean": self.cost_dyn_mean}
        
        return results
        