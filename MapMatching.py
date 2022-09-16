import numpy as np
import math
import copy
import time
import scipy.stats
from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splev

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
    def computeWheelRadiusBias(imuPos, pfPose, N=100):
        if imuPos.shape[0] >= N:
            l = np.linalg.norm(imuPos[-N,:]-imuPos[0,:])
            if l > 100 and PF.getLineIndication(imuPos, N=N):
                fit_imu = np.polyfit(imuPos[-N:,0], imuPos[-N:,1], 1, cov=False)
                fit_pf = np.polyfit(pfPose[-N:,0], pfPose[-N:,1], 1, cov=False)
                imu_f = np.poly1d(fit_imu)
                pf_f = np.poly1d(fit_pf)
                imu_integral = np.array([1+imu_f[1]**2, 4*imu_f[2]*imu_f[1], 4*imu_f[2]**2])
                pf_integral = np.array([1+pf_f[1]**2, 4*pf_f[2]*pf_f[1], 4*pf_f[2]**2])
                imu_f = lambda x:imu_integral[0]+imu_integral[1]*x+imu_integral[2]*x**2
                pf_f = lambda x:pf_integral[0]+pf_integral[1]*x+pf_integral[2]*x**2
                imu_length, _ = scipy.integrate.quad(imu_f, np.min(imuPos[-N:,0]), np.max(imuPos[-N:,0]))
                pf_length, _ = scipy.integrate.quad(pf_f, np.min(pfPose[-N:,0]), np.max(pfPose[-N:,0]))
                b = abs(pf_length / imu_length)
                print(f"Detected wheel speed bias {b} imu_length={imu_length} pf_length={pf_length} min_pos_imu = {np.min(imuPos[-N:,0])} max_imu={np.max(imuPos[-N:,0])}  pf_min = {np.min(pfPose[-N:,0])} pf_max = {np.max(pfPose[-N:,0])}")
                #return 0.2 + 0.8 * b
                if b > 0.99 and b < 1:
                    return b

        return None

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
    
    def computeDistForDynamicTracks(self, trk, lanes):
        total_dist = 1e6
        dx = trk[:,0].reshape(-1,1)-lanes[:,0].reshape(1,-1)
        dy = trk[:,1].reshape(-1,1)-lanes[:,1].reshape(1,-1)
        
        trk_dx = trk[-1,0]-trk[0,0]
        trk_dy = trk[-1,1]-trk[0,1]
        trk_angle = np.arctan2(np.repeat(trk_dy, trk.shape[0]), np.repeat(trk_dx, trk.shape[0])) 
        lane_angle = np.arctan2(lanes[:,3], lanes[:,2])
        dv = trk_angle.reshape(-1,1)-lane_angle.reshape(1,-1)

        dxdy_norm = np.sqrt(dx**2 + dy**2 + 5*dv**2) #No one has any priority
        it = dxdy_norm.argmin(axis=1)

        dx_min = np.take_along_axis(dx, np.expand_dims(it, axis=-1), axis=1)
        dy_min = np.take_along_axis(dy, np.expand_dims(it, axis=-1), axis=1)
        dv_min = np.take_along_axis(dv, np.expand_dims(it, axis=-1), axis=1)
        dv_min = self.sigma_clip(dv_min)

        dxdy_min = np.array([dx_min, dy_min, dv_min])
        dxdy_min = np.squeeze(dxdy_min, -1) if dxdy_min.ndim == 3 else dxdy_min  
        
        #print("trk", trk, "dxdy_min", dxdy_min, "lanes[it,0]", lanes[it,0], "lanes[it,1]", lanes[it,1])

        sigma_x = 0.5
        sigma_y = 0.5
        sigma_grad = 0.2
        P = np.array([[sigma_x, 0, 0],[0, sigma_y, 0],[0, 0, sigma_grad]])
        dist = np.sum(np.dot((dxdy_min.T)**2,np.linalg.inv(P)), axis=1)
        #take 20% highest
        n_elements = int(max(1, trk.shape[0] / 5))
        #print(dist.shape, n_elements)
        top_ind = np.argpartition(dist, -n_elements)[-n_elements:]
        top20_dist = dist[top_ind]
        total_dist = np.mean(top20_dist)
        
        #print("total_dist", total_dist)
        
        return total_dist
        
    def computeDistForStaticTracks(self, polynom, boundaryPoints, heading, curve=0, debug=False):
        R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        
        polynom_rotated = np.dot(R, polynom.T).T
        boundary_rotated = np.dot(R, boundaryPoints.T).T
        dx = polynom_rotated[:,0].reshape(-1,1)-boundary_rotated[:,0].reshape(1,-1)
        #dx[polynom_rotated[:,0] > 0, :] -= 1
        #dx[polynom_rotated[:,0] < 0, :] += 1
        dy = np.abs(polynom_rotated[:,1].reshape(-1,1)-boundary_rotated[:,1].reshape(1,-1))#+1

        #Force association on certain axis according to heading
        max_incline = 2
        min_incline = 0.5
        derivative = (max(polynom_rotated[:, 0])-min(polynom_rotated[:, 0])) / (max(polynom_rotated[:, 1])-min(polynom_rotated[:, 1]))
        straight_line_indication = curve < 0.1
        if straight_line_indication and derivative > max_incline: #association should be on along track axis
            dxdy_norm = np.sqrt(10*dx**2 + 0.4*dy**2) #dx has higher priority
        elif straight_line_indication and derivative < min_incline:
            dxdy_norm = np.sqrt(0.4*dx**2 + 10*dy**2) #dy has higher priority
        else:
            dxdy_norm = np.sqrt(dx**2 + dy**2) #No one has any priority
            
        it = dxdy_norm.argmin(axis=1)
            
        dx_min = np.take_along_axis(dx, np.expand_dims(it, axis=-1), axis=1)
        dy_min = np.take_along_axis(dy, np.expand_dims(it, axis=-1), axis=1)
        dxdy_min = np.array([dx_min, dy_min])
        dxdy_min = np.squeeze(dxdy_min, -1) if dxdy_min.ndim == 3 else dxdy_min
        
        sigma_along_track = 0.5
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
    
    def getClosestToINSBoundaries(self, particle):
        xmin = min(self.ins_bounds["x1"], self.ins_bounds["x2"])
        xmax = max(self.ins_bounds["x1"], self.ins_bounds["x2"])
        ymin = min(self.ins_bounds["y1"], self.ins_bounds["y2"])
        ymax = max(self.ins_bounds["y1"], self.ins_bounds["y2"])
        if not (xmin <= particle["x"] <= xmax and ymin <= particle["y"] <= ymax):
            dx1, dx2 = xmin - particle["x"], particle["x"] - xmax
            dy1, dy2 = ymin - particle["y"], particle["y"] - ymax
            #print(f"Boundary limits reached! xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax} dx1={dx1}, dx2={dx2}, dx3={dx3}, dx4={dx4}")
            if dx1 > 0:
                particle["x"] = xmin
            if dx2 > 0:
                particle["x"] = xmax
            if dy1 > 0:
                particle["y"] = ymin
            if dy2 > 0:
                particle["y"] = ymax
                
        return particle
    
    def eval_polynom_map_match(self, ext_track, particle, worldRef, roadMap, debug=False):
        height, width = roadMap.shape[:2]
        map_center = (width/2, height/2)
        weight = self.calcTrkWeight(ext_track)
        polynom = ext_track.getElements()
        curve = ext_track.getStateVector()[2]
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
            cost = self.computeDistForStaticTracks(polynom=transformed_polynom, boundaryPoints=boundary_points.T, heading=worldRef[2], curve=curve,debug=debug)
            #combine (independent) measurements
            cost *= weight
        else:
            cost = 1e3
           
        return cost
    
    def eval_dynamic_track_map_match(self, trk, particle, worldRef, firstWorldRef, lanes, debug=False):
        cost = 0
        if trk.confirmed and trk.hits > 10:
            tstate, tspeed = trk.getTranslatedState()
            abs_vel = np.mean(np.linalg.norm(tspeed,axis=1), axis=0)
            if abs_vel > 2:
                tstate = np.squeeze(tstate,axis=2) - firstWorldRef
                transformed_dyn_track = self.transformDynTrack(tstate, worldRef, particle)
                cost = self.computeDistForDynamicTracks(transformed_dyn_track[-10:,:], lanes) # consider only last 10 updates!
           
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
        imu_angle_bias = 0
        if 1 and self.imu_bias_factor == 0:
            b = self.computeIMUAngleBias(self.imu_path[:self.counter,:], self.pf_path[:self.counter,:]) 
            if b is not None:
                self.imu_bias_factor = -b
                imu_angle_bias = -b
        if 1 and self.cond3_counter > 15 and self.wheel_radius_bias == 1:
            b = self.computeWheelRadiusBias(self.imu_path[:self.counter,:], self.pf_path[:self.counter,:]) 
            if b is not None:
                self.wheel_radius_bias = b
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
        self.ins_bounds["x1"] += (dt-dt*wheel_bias_factor)*np.cos(self.current_odom_xy_theta[2])
        self.ins_bounds["x2"] += (dt+dt*wheel_bias_factor)*np.cos(self.current_odom_xy_theta[2])
        self.ins_bounds["y1"] += (dt-dt*wheel_bias_factor)*np.sin(self.current_odom_xy_theta[2])
        self.ins_bounds["y2"] += (dt+dt*wheel_bias_factor)*np.sin(self.current_odom_xy_theta[2])
        #uncertainty test
        prev_num_polynoms = self.polynom_cost.shape[0]
        uncertainty_flag = False
        large_angle_turn_flag = np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2]) > 0.02
        if prev_num_polynoms > 0:
            min_cost = min(self.polynom_cost)
            median_cost = np.median(self.polynom_cost)
            var_dist, _ = self.getVariance()
            var_cond = var_dist < 5
            ind_max = np.argmax(self.polynom_cost)
            lateral_polynom_max_cost = 0 if self.polynom_longitudal[ind_max] else self.polynom_cost[ind_max]
            cond1 = min_cost > 10 and prev_num_polynoms > 1 and var_cond
            cond2 = large_angle_turn_flag and median_cost > 20 and min_cost > 5 and prev_num_polynoms > 1 and var_cond
            cond3 = lateral_polynom_max_cost > 15 and self.wheel_radius_bias == 1
            cond4 = False if len(self.polynom_dynamic_cost) == 0 else np.max(self.polynom_dynamic_cost) > 100
            if cond1 or cond2 or cond3 or cond4:
                print(f"uncertainty_flag is True! cond1={cond1} cond2={cond2} cond3={cond3} cond4={cond4} large_angle_turn_flag={large_angle_turn_flag} ang_diff = {np.abs(new_odom_xy_theta[2]-old_odom_xy_theta[2])}")
                uncertainty_flag = True
            if cond3:
                self.cond3_counter += 1

        for particle in self.particles:           
            dr1_noisy = dr1 + np.random.normal(0, self.sigma_rot1)
            dt_noisy = dt * self.wheel_radius_bias + np.random.normal(0, self.sigma_trans)
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
            

            sigma_factor_along_track = 0.1 if uncertainty_flag else 0
            sigma_factor_cross_track = 0.1
            sigma_along_track = (dt / 25) if not large_angle_turn_flag else (dt / 50)
            error_along_track = np.random.normal(0, sigma_along_track + sigma_factor_along_track)
            sigma_cross_track = dt / 80
            error_cross_track = np.random.normal(0, sigma_factor_cross_track) if uncertainty_flag else np.random.normal(0, sigma_cross_track)
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

    def eval_sensor_model(self, worldRef, extTracks, dynTracks, firstWorldRef, roadMap, lanes):
        w = 2
        self.polynom_cost = np.ones(len(extTracks)) * 1e6
        self.polynom_longitudal = np.ones(len(extTracks))
        self.polynom_dynamic_cost = np.ones(len(dynTracks)) * 1e6
        for particle in self.particles:
            cost_total = 1e-6 # for accumulating probabilities
            for i_ext_track, ext_track in enumerate(extTracks):
                cost = self.eval_polynom_map_match(ext_track, particle, worldRef, roadMap)
                self.polynom_cost[i_ext_track] = min(self.polynom_cost[i_ext_track], cost)
                cost_total += cost
                
            for i_dyn, dyn_track in enumerate(dynTracks):
                cost = self.eval_dynamic_track_map_match(dyn_track, particle, worldRef, firstWorldRef, lanes)
                self.polynom_dynamic_cost[i_dyn] = min(self.polynom_dynamic_cost[i_dyn], cost)
                cost_total += w * cost
            
            particle['weight'] = 1 / cost_total #cost instead of probability

        #normalize weights
        normalizer = sum([p['weight'] for p in self.particles])

        for particle in self.particles:
            particle['weight'] = particle['weight'] / normalizer
            
        
        for i_ext_track, ext_track in enumerate(extTracks):
            heading = worldRef[2]
            polynom = ext_track.getElements()
            R = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
            polynom_rotated = np.dot(R, polynom.T).T
            dx = np.max(polynom_rotated[:, 0])-np.min(polynom_rotated[:, 0])
            dy = np.max(polynom_rotated[:, 1])-np.min(polynom_rotated[:, 1])
            if abs(dy) > abs(dx) * 3:
                print("polynom is not longitudal!")
                self.polynom_longitudal[i_ext_track] = 0

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

        lanes = np.concatenate(lanes, axis=0)
        if self.pf.current_odom_xy_theta is None:
            self.pf.initialize_particles(self.N, IMURelativeRef)
        drivable_area = self.getDrivableArea(nuscMap, firstWorldRef + last_output)
        self.pf.sample_motion_model(IMURelativeRef, drivable_area)
        #road_map = self.getMapReference(nuscMap, origRadarRef)
        self.road_map = getCombinedMap(nuscMap, firstWorldRef + last_output, patchSize=300)
        self.pf.eval_sensor_model(IMURelativeRef, extTracks, dynTracks, firstWorldRef, self.road_map, lanes)
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
            cost_true = self.pf.eval_polynom_map_match(ext_track, true_pos_particle, imuRelativeRef, self.road_map, debug=debug)
            cost_best = self.pf.eval_polynom_map_match(ext_track, self.best_particle, imuRelativeRef, self.road_map, debug=debug)
            cost_mean = self.pf.eval_polynom_map_match(ext_track, mean_particle, imuRelativeRef, self.road_map, debug=debug)
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
                cost_true = self.pf.eval_dynamic_track_map_match(dyn_track, true_pos_particle, imuRelativeRef, firstWorldRef, lanes)
                cost_best = self.pf.eval_dynamic_track_map_match(dyn_track, self.best_particle, imuRelativeRef, firstWorldRef, lanes)
                cost_mean = self.pf.eval_dynamic_track_map_match(dyn_track, mean_particle, imuRelativeRef, firstWorldRef, lanes)
                #print("cost_gt", cost_true, "cost_best", cost_best, "cost_mean", cost_mean)
                #print("gt", true_pos_particle, "best", self.best_particle, "mean", mean_particle)
                self.cost_dyn_true.append(cost_true)
                self.cost_dyn_mean.append(cost_mean)
        
        results = {"all_particles": self.pf.particles, "pf_best_pos": best_pos, "pf_best_theta": best_particle['theta'], "pf_mean_pos": mean_pos, "pf_mean_theta": mean_particle['theta'], "cost_true": self.cost_true, "cost_mean": self.cost_mean, "covariance": covariance, "cost_dyn_true": self.cost_dyn_true, "cost_dyn_mean": self.cost_dyn_mean}
        
        return results
        