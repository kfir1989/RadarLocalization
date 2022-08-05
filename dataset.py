from utils import *
from tools import *
import os
import os.path
import numpy as np
import pandas as pd
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
import matplotlib.image as mpimg


class Dataset():
    def __init__(self, **kwargs):
        pass
    
    def getData(self, idx):
        pass

class DynamicSimulatedDataset(Dataset):
    def __init__(self, **kwargs):
        x0 = kwargs.pop('x0', (0,0))
        v0 = kwargs.pop('v0', 0)
        a0 = kwargs.pop('a0', 0)
        N = kwargs.pop('N', 150)
        self.prior2 = kwargs.pop('prior', [(1, 0.009, -0.004, 3, 40, True), (1, 0.009, -0.004, 60, 90, True)])
        self.prior = [(27.5,-5,0.3,10,16.1,True)]
        dR = kwargs.pop('dR', 0.4)
        dAz = kwargs.pop('dAz', 0.02)
        polynom_noise_ratio = kwargs.pop('polynom_noise_ratio', 0.2) #1:1
        seed = kwargs.pop('seed', None)
        self.__generateEgoMotion(x0=np.asarray(x0), v0=v0, a0=a0, path=self.prior[0], N=N, dT=0.1)
        self.dR, self.dAz, self.polynom_noise_ratio = dR, dAz, polynom_noise_ratio
        self.N = N
        
    def getData(self, t):
        pos = [self.t[0,t], self.t[1,t]]
        heading = np.rad2deg(np.arccos(self.R[0,0,t]))
        z, dz = np.array([], dtype=np.float).reshape(0,2), np.array([], dtype=np.float).reshape(0,2,2)
        xmin, xmax = 300, 0
        for prior in self.prior:
            zp,dzp = self.__generateData(prior=prior, dR=self.dR, dAz=self.dAz, pos=pos, R=self.R[:,:,t], N=100)
            z, dz = np.concatenate([z, zp]), np.concatenate([dz, dzp])
            xmin = min(xmin, np.min(z[:,0]))
            xmax = max(xmax, np.max(z[:,0]))
        
        data_size = int(z.shape[0])
        N_noise = int(self.polynom_noise_ratio * data_size)
        
        zp,dzp = self.__generateNoise(xRange=[xmin,xmax], dR=self.dR, dAz=self.dAz, pos=pos, R=self.R[:,:,t], N=N_noise)
        z, dz = np.concatenate([z, zp]), np.concatenate([dz, dzp])
        
        zw, covw = self.__convert2WorldCoordinates(z, dz, self.R[:,:,t], self.t[:,t].reshape(2,1))
        print("data_size",data_size,"zw.shape", zw.shape)
        video_data = {"polynom":zw[0:data_size,:],"dpolynom":covw[0:data_size,:,:], "other":zw[data_size:,:],"dother":covw[data_size:,:,:],
                     "pos":pos,"heading":heading}
        
        prior = self.__generatePrior(self.prior2)
        
        return zw, covw, prior, video_data
        
    def __generateData(self, prior, dR, dAz, pos, R, N=50):
        #if not prior[5]:
            #pos = np.array([-5,0])
        print("[prior[3],prior[4]]",[prior[3],prior[4]],pos)
        [_,_,x_poly,y_poly,polynom_cov] = generatePolynomNoisyPoints(N=N,a1=prior[0],a2=prior[1],a3=prior[2],dR=dR,dAz=dAz,xRange=[prior[3],prior[4]],pos=pos,R=np.linalg.inv(R))
        if prior[5]:
            z = np.array([x_poly, y_poly]).T
            dz = np.array(polynom_cov)
        else:
            z = np.array([y_poly, x_poly]).T
            dz = np.flip(np.array(polynom_cov))
        
        print("z",z)
        return z,dz
    
    def __generateNoise(self, xRange, dR, dAz, pos, R, N=50):
        xRange[0]=min(5, xRange[0])
        xRange[1]=max(50,xRange[1])
        
        [x_noise,y_noise,noise_cov] = generateRandomNoisyPoints(N=N,xRange=xRange,yRange=[-20,20],dR=dR,dAz=dAz)
        z = np.array([x_noise, y_noise]).T
        dz = np.array(noise_cov)
        
        return z,dz
    
    def __convert2WorldCoordinates(self, z, cov, R, t):
        #print(z.shape, cov.shape, R.shape, t.shape)
        zw = np.matmul(R,z.T) + t
        covw = np.matmul(np.matmul(R, cov), R.T)
        
        return zw.T, covw
    
    def __generatePrior(self, coeffList):
        prior_list = []
        for coeff in coeffList:
            prior_list.append({"c": (coeff[0], coeff[1], coeff[2]), "xmin": coeff[3], "xmax": coeff[4], "fx": coeff[5]})
            
        return prior_list
            
    
    def __generateEgoMotion(self, x0, v0, a0, path, N, dT):
        dist = np.arange(0,N)*dT*(v0+a0*dT)
        c,b,a = path[0],path[1],path[2]
        x = (np.power(3*a*dist + np.power(b+1,3./2), 2./3)-b-1)/(2*a)
        y = a*x**2+b*x+c - 5
        self.path = np.array([x, y])
        d0 = np.array([1,0]).reshape(1,2)
        di = np.array([np.repeat(1,x.shape), 2*a*x])
        
        angle = np.arctan2(d0[:,0]*di[1,:]-d0[:,1]*di[0,:],d0[:,0]*di[0,:]+d0[:,1]*di[1,:])
       
        if v0==0 and a0==0:
            angle=np.zeros(angle.shape)
        self.R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        self.t = self.path - x0.reshape(2,1)
        if v0==0 and a0==0:
            self.t = np.zeros(self.t.shape)
		
		
		

class NuscenesDataset(Dataset):
    def __init__(self, nusc, **kwargs):
        self.nusc = nusc
        scene_id = kwargs.pop('scene_id', 5)
        directory = kwargs.pop('directory')
        N = kwargs.pop('N', 800)
        scene_name = self.__getSceneName(scene_id)
        map_name = self.__getMapName(scene_name)
        print(r"scene_id={} scene_name={} map_name={}".format(scene_id, scene_name, map_name))
        self.nusc_map = NuScenesMap(dataroot=directory, map_name=map_name)
        nusc_can = NuScenesCanBus(dataroot=directory)
        self.imu = []
        self.veh_speed = []
        self.veh_pose = []
        for ii in range(scene_id, scene_id + 8):
            self.imu = self.imu + nusc_can.get_messages(self.__getSceneName(ii), 'ms_imu')
            self.veh_speed = self.veh_speed + nusc_can.get_messages(self.__getSceneName(ii), 'zoe_veh_info')
            self.veh_pose = self.veh_pose + nusc_can.get_messages(self.__getSceneName(ii), 'pose')
        #ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[5]['token']], verbose=False)
        self.rpath = os.path.join(directory, 'sweeps', 'RADAR_FRONT')
        self.cpath = os.path.join(directory, 'sweeps', 'CAM_FRONT')
        self.ego = self.__extractEgo(os.path.join(directory, 'v1.0-trainval', 'ego_pose.json'))
        self.radar_files = sorted(os.listdir(self.rpath))
        self.camera_files = sorted(os.listdir(self.cpath))
        self.radar_ts = self.__extractTimestamps("radar", self.radar_files)
        self.camera_ts = self.__extractTimestamps("camera", self.camera_files)
        
        my_scene = self.nusc.scene[scene_id]
        first_sample_token = my_scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        radar_front_data = self.nusc.get('sample_data', my_sample['data']['RADAR_FRONT'])
        self.cs_record = self.nusc.get('calibrated_sensor', radar_front_data['calibrated_sensor_token'])
        self.first_idx = self.__getFirstIdxOffset(scene_name)
        self.odometry = {'r1':0,'t':0,'r2':0}
        self.__getFirstPosition(self.first_idx)
        self.ego_path, self.ego_trns = self.__extractEgoPath(N)
        
    def getData(self, t, GT=False):
        t += self.first_idx
        dR = 0.4
        dAz = 0.04
        z = self.__getRadarSweep(t)[:,0:2]
        cov = getXYCovMatrix(z[:,1], z[:,0], dR, dAz)
        trns_gt,rot_gt = self.getEgoInfo(t, GT=True)
        veh_speed = self.odometry["speed"]
        R_gt = rot_gt.rotation_matrix[0:2,0:2] ##not great!
        trns_imu,rot_imu = self.getEgoInfo(t, GT=False)
        R_imu = rot_imu.rotation_matrix[0:2,0:2] ##not great!
        if GT:
            trns,rot,R = trns_gt, rot_gt, R_gt
        else:
            trns,rot,R = trns_imu, rot_imu, R_imu
                
        pc = self.__getTransformedRadarData(t,trns,rot)
        static_data = pc[:, pc[3,:]==1].T
        #dynamic_data = pc[:, np.where(pc[:,3]==0) or np.where(pc[:,3]==2) or np.where(pc[:,3]==6)].T
        pc = pc.T
        dw = np.squeeze(pc[np.where(pc[:,3]==0) or np.where(pc[:,3]==2) or np.where(pc[:,3]==6), :], axis=0)
        zw = static_data[:,0:2]
        #dw = np.squeeze(dynamic_data,axis=1)
        covw = np.matmul(np.matmul(R, cov), R.T)
        
        prior = self.__getPrior(t)
        img = self.__getSyncedImage(t)
        ts = self.radar_ts.iloc[t]["timestamp"] #take timestamp from radar
        heading_gt = 90 + np.sign(R_gt[1,0]) * (np.rad2deg(np.arccos(R_gt[0,0])))
        heading_imu = 90 + np.sign(R_imu[1,0]) * (np.rad2deg(np.arccos(R_imu[0,0])))
        
        video_data = {"pc": zw, "img": img, "prior": prior, "pos": trns_gt, "rot": R_gt, "heading": heading_gt, "heading_imu": heading_imu, "pos_imu" : trns_imu, "rot_imu" : rot_imu,  "odometry": self.odometry, "ego_path": self.ego_path, "ego_trns": self.ego_trns, "veh_speed": veh_speed, "timestamp": ts}

        return zw, covw, prior, dw, video_data, self.nusc_map
    
    def __getTransformedRadarData(self, t, trns, rot):
        f = os.path.join(self.rpath, self.radar_files[t])
        pc = RadarPointCloud.from_file(f)
        pc = self.__sensor2World(t, pc, trns, rot)
        
        return pc.points
    
    def getDynamicPoints(self, t):
        t += self.first_idx
        trns,rot = self.getEgoInfo(t, GT=True)
        pc = self.__getTransformedRadarData(t, trns, rot)
        pc = pc.T
        pc = np.squeeze(pc[np.where(pc[:,3]==0) or np.where(pc[:,3]==2) or np.where(pc[:,3]==6), :], axis=0)
        R = rot.rotation_matrix[0:2,0:2] ##not great!
        heading = np.sign(R[1,0]) * (np.rad2deg(np.arccos(R[0,0])))
        ego_speed = self.odometry['speed']
        ts = self.radar_ts.iloc[t]["timestamp"] #take timestamp from radar
        return pc, ts, heading, ego_speed
        
    def __extractPolynomFromLane(self, lane):
        lane_record = self.nusc_map.get_arcline_path(lane)
        poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=0.5)
        poses = np.asarray(poses)
        lane = np.polyfit(poses[:,0], poses[:,1], 2)
        poly = np.poly1d(lane)
        dx = np.diff(poses[:,0])
        dy = np.diff(poses[:,1])
        return {"x":poses[:,0], "y": poses[:,1], "dx": dx, "dy": dy, "poly":poly}
    
    def __sensor2EgoCoordinates(self, pc):
        pc.rotate(Quaternion(self.cs_record['rotation']).rotation_matrix)
        pc.rotate(Quaternion([0, 0, 1, 0]).rotation_matrix) # rotate by 180 degrees. not sure why!
        pc.translate(np.array(self.cs_record['translation']))
        
        return pc
        
    def __sensor2World(self, idx, pc, trns, rot):
        car_from_sensor = transform_matrix(self.cs_record['translation'], Quaternion(self.cs_record['rotation']), inverse=False)
        rotate_by_180 = transform_matrix([0,0,0], Quaternion([0, 0, 1, 0]), inverse=False)
        global_from_car = transform_matrix(trns, rot, inverse=False)
        
        #car_from_sensor = np.dot(rotate_by_180, car_from_sensor)
         # Combine tranformation matrices
        global_from_sensor = np.dot(global_from_car, car_from_sensor)
    
        # Transform pointcloud
        pc.transform(global_from_sensor)
        
        return pc
        
    def __extractEgo(self, file):
        with open(file) as ego_file:
            ego = pd.read_json(ego_file,convert_dates=False)

        ego.reset_index(level=0, inplace=True)
        return ego
    
    def __extractEgoPath(self, N):
        radar_first_ts = self.radar_ts.iloc[self.first_idx]["timestamp"]
        radar_last_ts = self.radar_ts.iloc[self.first_idx+N]["timestamp"]
        ego_first_idx = (self.ego['timestamp']-radar_first_ts).abs().argsort()[0]
        ego_last_idx = (self.ego['timestamp']-radar_last_ts).abs().argsort()[0]
        
        relevant_ego_path = self.ego.iloc[ego_first_idx:ego_last_idx+1][["translation", "timestamp"]]
        ego_path = []
        for idx in range(0,N):
            closest_ego_idx = (relevant_ego_path['timestamp']-self.radar_ts.iloc[self.first_idx + idx]["timestamp"]).abs().argsort().iloc[0]
            ego_path.append(relevant_ego_path.iloc[closest_ego_idx]['translation'])
                     
        ego_trns = [np.linalg.norm(np.array(x) - np.array(ego_path[i - 1])) for i, x in enumerate(ego_path)][1:]
        ego_trns = np.cumsum(ego_trns)
        ego_trns = np.insert(ego_trns, 0, 0)
        
        return np.array(ego_path), ego_trns
        
    def __extractTimestamps(self, sensor_name, filenames):
        db_file = f"/home/kfir/workspace/RadarLocalization/database/{sensor_name}_timestamps.csv"
        if os.path.isfile(db_file):
            df = pd.read_csv(db_file)
        else:
            ts = []
            for filename in filenames:
                ts.append(np.int64(filename.split('_')[-1].split('.')[0]))

            df = pd.DataFrame({'timestamp':ts})
            print("mkdir", os.path.dirname(db_file))
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            df.to_csv(db_file)
            
        return df
    
    def getEgoInfo(self, i, GT=True):
        ts = self.radar_ts.iloc[i]["timestamp"] #take timestamp from radar
        if GT:
            eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
            trns = self.ego.iloc[eidx]["translation"]
            rot = Quaternion(self.ego.iloc[eidx]["rotation"])
            timestamp = self.ego.iloc[eidx]["timestamp"] / 1e6
            #save vehicle speed
            veh_speed = np.array([(m['utime'], m['vel']) for m in self.veh_pose])
            pidx = np.argsort(np.abs(veh_speed[:,0]-ts))[0]
            self.odometry['speed'] = np.array(veh_speed[pidx,1])
        else:
            #based on IMU and odometry
            rot_imu = np.array([(m['utime'], m['q']) for m in self.imu])
            ridx = np.argsort(np.abs(rot_imu[:,0]-ts))[0]
            rot = Quaternion(self.imu[ridx]['q'])
            rot *= self.first_imu_rot.inverse #substract IMU intial offset
            rot *= self.first_gt_rot # add initial GT offset 

            #odometry (zoe vehicle info)
            wheel_speed = np.array([(m['utime'], m['FL_wheel_speed']) for m in self.veh_speed])
            radius = 0.31#0.305  # Known Zoe wheel radius in meters.
            circumference = 2 * np.pi * radius
            wheel_speed[:, 1] *= circumference / 60

            oidx = np.argsort(np.abs(wheel_speed[:,0]-ts))[0]
            trns = self.last_veh_trns
            timestamp = wheel_speed[oidx, 0] / 1e6
            for odomidx in range(self.veh_speed_oidx+1, oidx+1):
                dT = (wheel_speed[odomidx, 0] - wheel_speed[odomidx-1, 0]) / 1e6
                #print(f"dT {dT} odomidx {odomidx}")
                xnorm = wheel_speed[odomidx, 1] * dT
                x = np.array([xnorm, 0]).T.reshape((2,1))
                dt_trns = np.dot(rot.rotation_matrix[0:2,0:2], x)
                trns += np.array([dt_trns[0][0], dt_trns[1][0], 0]) # add initial GT offset
                #print(f"trns {trns}")
            self.veh_speed_oidx = oidx
            self.last_veh_trns = trns
            #save vehicle speed
            self.odometry['speed'] = wheel_speed[oidx, 1]
            
        self.odometry['r1'] = self.odometry['r2']
        R = rot.rotation_matrix[0:2,0:2] ##not great!
        self.odometry['r2'] = np.sign(R[1,0]) * np.arccos(R[0,0])
        self.odometry['t'] = np.linalg.norm(np.array(trns)-np.array(self.odometry['trns']))
        self.odometry['trns'] = trns
        self.odometry['timestamp'] = timestamp
        return np.array(trns),rot
    
    def getOdometry(self):
        return self.odometry
    
    def __getFirstPosition(self, firstIdx):
        ts = self.radar_ts.iloc[firstIdx]["timestamp"] #take timestamp from radar
        eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
        self.first_gt_trns = self.ego.iloc[eidx]["translation"]
        self.first_gt_rot = Quaternion(self.ego.iloc[eidx]["rotation"])
        #odometry
        rot_imu = np.array([(m['utime'], m['q']) for m in self.imu])
        ridx = np.argsort(np.abs(rot_imu[:,0]-ts))[0]
        self.first_imu_rot = Quaternion(rot_imu[ridx,1])
        odom_speed = np.array([(m['utime'], m['odom_speed']) for m in self.veh_speed])
        print("odom_speed",odom_speed)
        oidx = np.argsort(np.abs(odom_speed[:,0]-ts))[0]
        self.veh_speed_oidx = oidx
        self.last_veh_trns = self.first_gt_trns
        
        R = self.first_imu_rot.rotation_matrix[0:2,0:2]
        self.odometry['r2'] = np.sign(R[1,0]) * np.arccos(R[0,0])
        self.odometry['trns'] = self.first_gt_trns
        self.odometry['timestamp'] = odom_speed[oidx,0]
        
    def __getPrior(self, i):
        ts = self.radar_ts.iloc[i]["timestamp"] #take timestamp from radar
        eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
        trns = self.ego.iloc[eidx]["translation"]
        lanes = []
        if 0:
            closest_lane = self.nusc_map.get_closest_lane(trns[0], trns[1], radius=2)
            incoming_lane = self.nusc_map.get_incoming_lane_ids(closest_lane)
            outgoing_lane = self.nusc_map.get_outgoing_lane_ids(closest_lane)
            lanes.append(self.__extractPolynomFromLane(closest_lane))
            for lane in incoming_lane:
                lanes.append(self.__extractPolynomFromLane(lane))
                for ilane in self.nusc_map.get_incoming_lane_ids(lane):
                    lanes.append(self.__extractPolynomFromLane(ilane))
            for lane in outgoing_lane:
                lanes.append(self.__extractPolynomFromLane(lane))
                for olane in self.nusc_map.get_incoming_lane_ids(lane):
                    lanes.append(self.__extractPolynomFromLane(olane))
        else:
            lane_ids = self.nusc_map.get_records_in_radius(trns[0], trns[1], 80, ['lane', 'lane_connector'])
            nearby_lanes = lane_ids['lane'] + lane_ids['lane_connector']
            for lane in nearby_lanes:
                lanes.append(self.__extractPolynomFromLane(lane))
            
        return lanes
        
    def __getRadarSweep(self, i):
        f = os.path.join(self.rpath, self.radar_files[i])
        pc = RadarPointCloud.from_file(f)
        pc = pc.points.T
        
        return pc
        
    def __getSyncedImage(self, i):
        ts = self.radar_ts.iloc[i]["timestamp"] #take timestamp from radar
        cidx = (self.camera_ts['timestamp']-ts).abs().argsort()[0]
        img = mpimg.imread(os.path.join(self.cpath, self.camera_files[cidx]))
        
        return img
    
    def __getSceneName(self, scene_id):
        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]
        
        return recs[scene_id][1]["name"]
    
    def __getMapName(self, scene_name):
        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]

        for start_time, record in sorted(recs):
            if record["name"] == scene_name:
                location = self.nusc.get('log', record['log_token'])['location']
                return location
            
        raise ValueError(f"Map doesn't exist for scene_name={scene_name}")
        
    
    def __getFirstIdxOffset(self, scene_name):
        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]
        
        for start_time, record in sorted(recs):
            if record["name"] == scene_name:
                start_time = self.nusc.get('sample', record['first_sample_token'])['timestamp']
                ridx = (self.radar_ts['timestamp']-start_time).abs().argsort()[0]
                return ridx
            
        raise ValueError(f"First idx cannot be found for scene_name={scene_name}")
        
    def getEgoPath(self):
        return self.ego_path
    
    def getMap(self):
        return self.nusc_map