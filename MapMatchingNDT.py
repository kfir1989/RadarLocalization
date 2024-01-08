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
from shapes import test_line
import math

class PF:
    def __init__(self, N, method='_improved'):
        self.N = N
        self.method = method
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
    def getLineIndication(pos, N=50, thr=0.005):
        x = pos[-N:,0]
        y = pos[-N:,1]

        fit, cov = np.polyfit(x, y, 1, cov=True)
        if abs(fit[0]) > 2:
            fit, cov = np.polyfit(y, x, 1, cov=True)
            
        if np.sqrt(cov[0,0]) < thr:
            return True

        return False
    
    @staticmethod
    def computeIMUAngleBias(imuPos, pfPose, N=100):
        if imuPos.shape[0] >= N:
            l = np.linalg.norm(imuPos[-N,:]-imuPos[0,:])
            if l > 50 and PF.getLineIndication(imuPos, N=N):
                fit_imu = np.polyfit(imuPos[-N:,0], imuPos[-N:,1], 1, cov=False)
                fit_pf = np.polyfit(pfPose[-N:,0], pfPose[-N:,1], 1, cov=False)

                m1 = fit_imu[0]
                m2 = fit_pf[0]
                theta = np.arctan((m1 - m2)/(1 + m1*m2))
                if(abs(theta < 0.01)):
                    return theta

        return None
    
    @staticmethod
    def getXYLimits(extTrack):
        xs = extTrack.getStateVector()[3]
        xe = extTrack.getStateVector()[4]
        a0, a1, a2 = extTrack.getStateVector()[0], extTrack.getStateVector()[1], extTrack.getStateVector()[2]
        ys = a0 + a1 * xs + a2 * xs**2
        ye = a0 + a1 * xe + a2 * xe**2
        
        return xs,xe,ys,ye
    
    @staticmethod
    def world2Map(pos, worldRef, mapRef, mapRes=1.):
        pos[:, 0] -= worldRef[0]
        pos[:, 0] *= mapRes
        pos[:, 1] -= worldRef[1]
        pos[:, 1] *= mapRes
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
    def map2World(pos, worldRef, mapRef, mapRes=1.):
        pos[0] -= mapRef[0]
        pos[0] /= mapRes
        pos[1] -= mapRef[1]
        pos[1] /= mapRes
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
    
    def isTranslationInINSBoundaries(self, particle, tTh=300):
        t = self.ins_bounds["trns"]
        if t > tTh:
            if particle["trns"] < 0.985 * t or particle["trns"] > 1.015 * t:
                return False
        
        return True
                
    
    @staticmethod
    def calcScore(probMap, scatter, mapRes, mapCenter,patch_size=240):
        im = scatter_to_image(scatter, res_factor=mapRes, center = mapCenter, patch_size=patch_size)
        score = im*probMap
        valid_ind = (score != 0)
        valid_len = np.count_nonzero(valid_ind)
        score[valid_ind] = (1./score[valid_ind]) / mapRes
        return score      
    
    def computeScoreForDynamicTracks(self, trk):
        trk_on_map = trk
        score = self.calcScore(probMap=self.lanes_map, scatter=trk_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        score = np.sum(np.sum(score))
        return score
        
    def computeScoreForStaticTracks(self, polynom, boundaryPoints):
        polynom_on_map = polynom
        high_sigma_score = self.calcScore(probMap=self.static_sigma4_map, scatter=polynom_on_map, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        
        #is_line = all(test_line(polynom[0,:],polynom[-1,:],polynom))
        #second_step_flag = (boundaryPoints is not None and self.method == '_improved' and not is_line)
        #print(f"self.method={self.method} is_line={is_line} second_step_flag={second_step_flag}")
        
        if boundaryPoints is not None:
            #Find median point on the polynom
            polynom_mid_point = polynom[int(polynom.shape[0]/2),:]
            #Find nearest neighbor point on the map
            eucld_dist = np.linalg.norm(boundaryPoints-polynom_mid_point,axis=1)
            it = eucld_dist.argmin()
            #Calc translation
            txy = boundaryPoints[it,:] - polynom_mid_point
            if(np.linalg.norm(txy)) < 6:
                #Translate polynom
                polynom_translated = polynom + txy
                #Compute low sigma score for translated polynom(shape)
                low_sigma_score = self.calcScore(probMap=self.static_sigma1_map, scatter=polynom_translated, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
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
        polynom = ext_track.getElementsInFOV(worldRef)
        if not polynom.any():
            return 0,0
        
        polynom_len = np.sqrt(abs(polynom[-1,0]-polynom[0,0])**2 + abs(polynom[-1,1]-polynom[0,1])**2)
        transformed_polynom = self.transformPolynom(polynom, worldRef, particle)
        #transform polynom to the particles' perspective
        world_xs,world_xe,world_ys,world_ye = transformed_polynom[0,0], transformed_polynom[-1,0], transformed_polynom[0,1], transformed_polynom[-1,1]
        #get relevant patch from map
        padding = 20
        map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, map_center, mapRes=self.map_res_factor)
        map_xs,map_xe,map_ys,map_ye = max(0, int(min(map_limits[0,0], map_limits[1,0])) - padding), min(width-1, int(max(map_limits[0,0], map_limits[1,0])) + padding), max(0, int(min(map_limits[0,1], map_limits[1,1])) - padding), min(height-1, int(max(map_limits[0,1], map_limits[1,1])) + padding)
        (row,col) = np.where(self.road_map[map_ys:map_ye,map_xs:map_xe])
        #if there are any boundary points in the map
        boundary_points = None
        if row.shape[0] > 0:
            boundary_points = self.map2World(np.array([col+map_xs, row+map_ys]).astype('float64'), worldRef, map_center, mapRes=self.map_res_factor)
            boundary_points = boundary_points.T
        
        score = self.computeScoreForStaticTracks(polynom=transformed_polynom, boundaryPoints=boundary_points)
           
        return score, polynom_len
    
    def eval_polynom_map_match_new(self, ext_tracks, particle, worldRef):
        height, width = self.road_map.shape[:2]
        map_center = (width/2, height/2)
        
        #1. Build large scatters
        low_sigma_scatter = np.empty((0,2))
        high_sigma_scatter = np.empty((0,2))
        polynom_len = 1e-6
        
        for ext_track in ext_tracks:
            if not ext_track.isUpdated():
                continue # Use only updated tracks!
            polynom = ext_track.getElementsInFOV(worldRef)
            if not polynom.any():
                continue
            polynom_len += np.sqrt(abs(polynom[-1,0]-polynom[0,0])**2 + abs(polynom[-1,1]-polynom[0,1])**2)
            transformed_polynom = self.transformPolynom(polynom, worldRef, particle)
            #transform polynom to the particles' perspective
            world_xs,world_xe,world_ys,world_ye = transformed_polynom[0,0], transformed_polynom[-1,0], transformed_polynom[0,1], transformed_polynom[-1,1]
            #get relevant patch from map
            padding = 20
            map_limits = self.world2Map(np.array([[world_xs,world_ys],[world_xe,world_ye]]), worldRef, map_center, mapRes=self.map_res_factor)
            map_xs,map_xe,map_ys,map_ye = max(0, int(min(map_limits[0,0], map_limits[1,0])) - padding), min(width-1, int(max(map_limits[0,0], map_limits[1,0])) + padding), max(0, int(min(map_limits[0,1], map_limits[1,1])) - padding), min(height-1, int(max(map_limits[0,1], map_limits[1,1])) + padding)
            (row,col) = np.where(self.road_map[map_ys:map_ye,map_xs:map_xe])
            #if there are any boundary points in the map
            boundary_points = None
            if row.shape[0] > 0:
                boundary_points = self.map2World(np.array([col+map_xs, row+map_ys]).astype('float64'), worldRef, map_center, mapRes=self.map_res_factor)
                boundary_points = boundary_points.T
            
            
            high_sigma_scatter = np.concatenate((high_sigma_scatter, transformed_polynom))
            
            is_line = ext_track.getShape() == "Line"
            second_step_flag = (self.method == '_improved' and not is_line)
            #if second_step_flag:
                #print(f"Second step activated! ext_track.getShape()={ext_track.getShape()}")
            
            if second_step_flag and boundary_points is not None:
                #Find median point on the polynom
                polynom_mid_point = transformed_polynom[int(transformed_polynom.shape[0]/2),:]
                #Find nearest neighbor point on the map
                eucld_dist = np.linalg.norm(boundary_points-polynom_mid_point,axis=1)
                it = eucld_dist.argmin()
                #Calc translation
                txy = boundary_points[it,:] - polynom_mid_point
                if(np.linalg.norm(txy)) < 6:
                    #Translate polynom
                    polynom_translated = transformed_polynom + txy
                    low_sigma_scatter = np.concatenate((low_sigma_scatter, polynom_translated))
                else:
                    low_sigma_scatter = np.concatenate((low_sigma_scatter, transformed_polynom))
            elif second_step_flag:
                low_sigma_scatter = np.concatenate((low_sigma_scatter, transformed_polynom))
        
        total_score = self.eval_static_score(high_sigma_scatter, low_sigma_scatter)
   
        #print(f"total_score = {total_score} polynom_len = {polynom_len}")
        return total_score, polynom_len
    
    def eval_static_score(self, high_sigma_scatter, low_sigma_scatter):
        total_score = 1e-6
        if high_sigma_scatter.size > 0:
            high_sigma_score = self.calcScore(probMap=self.static_sigma4_map, scatter=high_sigma_scatter, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        if low_sigma_scatter.size > 0:
            low_sigma_score = self.calcScore(probMap=self.static_sigma1_map, scatter=low_sigma_scatter, mapRes=self.map_res_factor, mapCenter=self.map_reference_point)
        
        if high_sigma_scatter.size > 0 and low_sigma_scatter.size > 0:
            high_sigma_score = np.sum(np.sum(high_sigma_score))
            low_sigma_score = np.sum(np.sum(low_sigma_score))
            total_score = 0.5 * high_sigma_score + 0.5 * (min(low_sigma_score, high_sigma_score * 2))
            #total_score = np.sum(np.sum(0.5*low_sigma_score + 0.5*high_sigma_score))
        elif high_sigma_scatter.size > 0:
            total_score = np.sum(np.sum(high_sigma_score))
            
        return total_score
    
    def eval_dynamic_track_map_match(self, trk, particle, worldRef, firstWorldRef, debug=False):
        score = 1e-6
        if trk.confirmed and trk.hits > 10:
            tstate, tspeed = trk.getTranslatedState()
            abs_vel = np.mean(np.linalg.norm(tspeed,axis=1), axis=0)
            if abs_vel > 2:
                tstate = np.squeeze(tstate,axis=2) - firstWorldRef
                transformed_dyn_track = self.transformDynTrack(tstate, worldRef, particle)
                score = self.computeScoreForDynamicTracks(transformed_dyn_track[-10:,:]) # consider only last 10 updates!
           
        return score

    def initialize_particles(self, num_particles, worldRef):
        particles = []

        for i in range(num_particles):
            particle = dict()

            #random position inside a unity circle
            t = 2*np.pi*np.random.uniform(0, 1)
            u = np.random.uniform(0, 1)+np.random.uniform(0, 1)
            r = 2-u if u>1 else u

            diameter = 1 #Estimated error in the beginning
            xc = worldRef[0]
            yc = worldRef[1]
            #initialize pose: at the beginning, robot is certain it is at [0,0,0]
            particle['x'] = diameter*r*np.cos(t) + xc
            particle['y'] = diameter*r*np.sin(t) + yc
            particle['theta'] = worldRef[2]

            #initial weight
            particle['weight'] = 1.0 / num_particles
            particle["trns"] = 0

            #particle history aka all visited poses
            particle['history'] = []

            #add particle to set
            particles.append(particle)

        self.particles = particles
        self.ins_bounds = {"trns": 0}
        
    def predict_output(self, last_output, new_odom_xy_theta):
        if self.current_odom_xy_theta is not None:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])
        else:
            return last_output
        
        dr1 = math.atan2(delta[1], delta[0]) - old_odom_xy_theta[2]
        dt = math.sqrt((delta[0]**2) + (delta[1]**2))
        last_output["theta"] += dr1
        last_output["x"] += dt * np.cos(last_output["theta"] + dr1)
        last_output["y"] += dt * np.sin(last_output["theta"] + dr1)
        last_output["theta"] += (delta[2] - dr1)
        last_output["theta"] = (last_output["theta"] + np.pi) % (2 * np.pi) - np.pi
        
        return last_output
        
    
    def sample_motion_model(self, new_odom_xy_theta, drivableArea):
        imu_angle_bias = 0
        if 1 and self.imu_bias_factor == 0:
            b = self.computeIMUAngleBias(self.imu_path[:self.counter,:], self.pf_path[:self.counter,:]) 
            if b is not None:
                self.imu_bias_factor = -b
                imu_angle_bias = -b
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
        
        #compute IMU bounds constraint
        wheel_bias_factor = 0.03 # limit 3% deviation
        self.ins_bounds["trns"] += dt
        
        #uncertainty test
        uncertainty_flag = False#self.improvishment_flag
        large_angle_turn_flag = np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2]) > 0.02
        for particle in self.particles:           
            dr1_noisy = dr1 + np.random.normal(0, self.sigma_rot1)
            dt_noisy = dt + np.random.normal(0, self.sigma_trans)
            #Add measurement noise
            particle["theta"] += dr1
            particle["x"] += dt_noisy * np.cos(particle["theta"] + dr1_noisy)
            particle["y"] += dt_noisy * np.sin(particle["theta"] + dr1_noisy)
            particle["theta"] += (delta[2] - dr1_noisy)
            particle["theta"] += imu_angle_bias
            particle["theta"] = (particle["theta"] + np.pi) % (2 * np.pi) - np.pi
            #Add along-track and cross-track noise (for higher flexibility)
            R = np.array([[np.cos(-particle["theta"]), -np.sin(-particle["theta"])], [np.sin(-particle["theta"]), np.cos(-particle["theta"])]])
            xy = np.array([particle["x"], particle["y"]])
            xy_rotated = np.dot(R, xy)
            

            sigma_factor_along_track = 0.04 if uncertainty_flag else 0
            sigma_factor_cross_track = 0.1 if uncertainty_flag else 0
            sigma_along_track = (dt / 50) if not large_angle_turn_flag else (dt / 50)
            error_along_track = np.random.normal(0, sigma_along_track + sigma_factor_along_track)
            sigma_cross_track = dt / 80
            error_cross_track = np.random.normal(0, sigma_cross_track + sigma_factor_cross_track)
            xy_rotated_plus_errs = xy_rotated + np.array([error_along_track, error_cross_track])
            xy_rotated_back = np.dot(np.linalg.inv(R), xy_rotated_plus_errs)
            particle["x"] = xy_rotated_back[0]
            particle["y"] = xy_rotated_back[1]         
            particle["trns"] += dt_noisy + error_along_track
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
        for particle in self.particles:
            dynamic_static_res_factor = 5 #static points are 10 times in size than dynamic points
            static_score = 1e-6
            dynamic_score = 1e-6
            static_length = 1
            total_score = static_score + dynamic_score # for accumulating probabilities
            num_dyn_tracks = 0
            if 1 and not self.isTranslationInINSBoundaries(particle):
                particle['weight'] = total_score
                continue

            static_score, static_len = self.eval_polynom_map_match_new(extTracks, particle, worldRef)
                
            for i_dyn, dyn_track in enumerate(dynTracks):
                score = self.eval_dynamic_track_map_match(dyn_track, particle, worldRef, firstWorldRef)
                dynamic_score += score
                if score > 0:
                    num_dyn_tracks += 1
                    
            wd = (num_dyn_tracks * 100) / static_length
            total_weight = 1./(dynamic_static_res_factor * wd * dynamic_score + static_score)
                
            particle['weight'] = total_weight
            
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
        mean_particle['trns'] = sum([p['trns'] for p in particles]) / len(particles)
                
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
    
    def get_density(self, E, var_dist=2., nSigma=1.05):
        particles_within_sigma = [p for p in self.particles if (self.calcDist(p, E) <= var_dist * nSigma)]
        return len(particles_within_sigma) / len(self.particles)
    
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
    def __init__(self, N=100, method="_improved"):
        self.N = N
        self.pf = PF(N=N, method=method) 
        self.pf.counter = 0
        self.pf.imu_path = np.zeros([2000,2])
        self.pf.pf_path = np.zeros([2000,2])
    
    def getMapReference(self, nuscMap, worldRef):
        edges1 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=240, layer_names=['walkway'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges2 = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=240, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0.5, threshold2=0.7)
        edges = edges1 | edges2
        
        return edges #take only borders of drivable area
    
    def getDrivableArea(self, nuscMap, worldRef, layer_names = ['drivable_area'], patch_size=50):
        patch_angle = 0
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        
        return mask

    def run(self, extTracks, nuscMap, dynTracks, lanes, firstWorldRef, IMURelativeRef,lane_offset=[0,0]):
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
            self.last_output = self.predicted_output = self.pf.getSigmaClipped()
        else:
            self.predicted_output = self.pf.predict_output(self.output_position, IMURelativeRef)
            self.last_output = self.output_position
        drivable_area = self.getDrivableArea(nuscMap, firstWorldRef + IMURelativeRef[0:2])
        t1 = time.time()
        self.pf.sample_motion_model(IMURelativeRef, drivable_area)
        t2 = time.time()
        #road_map = self.getMapReference(nuscMap, origRadarRef)
        self.road_map = getCombinedMap(nuscMap, firstWorldRef+last_output, patchSize=240, res_factor=res_factor)
        self.static_sigma1_map = build_probability_map(self.road_map, sigma=5.)
        self.static_sigma4_map = build_probability_map(self.road_map, sigma=15.)
        sparse_scatter = drawLanes(nuscMap, ego_trns=firstWorldRef+last_output)
        sparse_scatter[:, 0] += lane_offset[0]
        sparse_scatter[:, 1] += lane_offset[1]
        lanes = scatter_to_image(sparse_scatter, center=firstWorldRef+last_output, res_factor=res_factor, patch_size=240)
        lanes[lanes == 1] = 255
        self.lanes_map = build_probability_map(lanes, sigma=5.)
        self.pf.setNDTMaps(road_map=self.road_map, static_sigma1_map=self.static_sigma1_map, static_sigma4_map=self.static_sigma1_map, 
                          lanes_map=self.lanes_map, map_reference_point=last_output, res_factor=res_factor)
        t3 = time.time()
        self.pf.eval_sensor_model(IMURelativeRef, extTracks, dynTracks, firstWorldRef, mapCenter=last_output)
        t4 = time.time()
        self.best_particle = self.pf.getBestParticle()
        self.pf.resample_particles()
        t5 = time.time()
        self.pf.counter += 1
        
        #print(f"Sample Motion: {t2-t1} maps: {t3-t2} eval_sensor_model: {t4-t3} resample: {t5-t4}")
        
    def smoothMeanParticle(self, mean_particle):
        #Smooth + Limit by IMU the reported position
        position_jump_thr = 0.5
        dist_cur_predicted = self.pf.calcDist(self.predicted_output, mean_particle)
        density_around_mean = self.pf.get_density(mean_particle)
        #print(f"dist_cur_predicted = {dist_cur_predicted} density_around_mean = {density_around_mean}")
        if dist_cur_predicted > position_jump_thr or \
        not self.pf.isTranslationInINSBoundaries(mean_particle, tTh=0):
            density_around_predicted = self.pf.get_density(self.predicted_output)
            #print(f"self.predicted_output = {self.predicted_output} mean_particle = {mean_particle}")
            #print(f"density_around_predicted = {density_around_predicted}")
            if density_around_predicted > 0.2:
                mean_particle = self.predicted_output
                
        return mean_particle
        
    def getResults(self, extTracks, dynTracks, firstWorldRef, gtRelativeRef, imuRelativeRef, lanes):
        covariance = self.pf.getCovarianceMatrix()
        mean_particle = self.pf.getSigmaClipped()
        mean_particle = self.smoothMeanParticle(mean_particle)
        self.output_position = mean_particle
        best_particle = self.best_particle
        mean_pos = np.array([mean_particle['x'], mean_particle['y']])
        best_pos = np.array([best_particle['x'], best_particle['y']])

        #DEBUG
        self.cost_true = []
        self.cost_mean = []
        debug = True if self.pf.counter > 2000 else False
        for ext_track in extTracks:
            true_pos_particle = {"x": gtRelativeRef[0], "y": gtRelativeRef[1], "theta": gtRelativeRef[2]}
            cost_true, _ = self.pf.eval_polynom_map_match_new([ext_track], true_pos_particle, imuRelativeRef)
            cost_best, _ = self.pf.eval_polynom_map_match_new([ext_track], self.best_particle, imuRelativeRef)
            cost_mean, _ = self.pf.eval_polynom_map_match_new([ext_track], mean_particle, imuRelativeRef)
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
        