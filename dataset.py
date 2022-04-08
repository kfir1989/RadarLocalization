from utils import *
from tools import *
import os
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
        v0 = kwargs.pop('v0', 5)
        a0 = kwargs.pop('a0', 0)
        N = kwargs.pop('N', 150)
        prior = kwargs.pop('prior', (1, 0.009, -0.004))
        dR = kwargs.pop('dR', 0.4)
        dAz = kwargs.pop('dAz', 0.05)
        polynom_noise_ratio = kwargs.pop('polynom_noise_ratio', 0.5)
        seed = kwargs.pop('seed', None)
        self.__generateEgoMotion(x0=np.asarray(x0), v0=v0, a0=a0, path=prior, N=N, dT=0.1)
        self.prior, self.dR, self.dAz, self.polynom_noise_ratio = prior, dR, dAz, polynom_noise_ratio
        x, y = createPolynom(a1=prior[0],a2=prior[1],a3=prior[2],xstart=0,xend=200)
        self.N = N
        
    def getData(self, t):
        pos = [self.t[0,t], self.t[1,t]]
        heading = np.rad2deg(np.arccos(self.R[0,0,t]))
        z,dz = self.__generateData(prior=self.prior, dR=self.dR, dAz=self.dAz, pos=pos, R=self.R[:,:,t], polynom_noise_ratio=self.polynom_noise_ratio, N=100)
        zw, covw = self.__convert2WorldCoordinates(z, dz, self.R[:,:,t], self.t[:,t].reshape(2,1))
        video_data = {"polynom":zw[0:50,:],"dpolynom":covw[0:50,:,:], "other":zw[50:100,:],"dother":covw[50:100,:,:],
                     "pos":pos,"heading":heading}
        
        return zw, covw, self.prior, video_data
        
    def __generateData(self, prior, dR, dAz, pos, R, polynom_noise_ratio=0.5, N=100):
        [_,_,x_poly,y_poly,polynom_cov] = generatePolynomNoisyPoints(N=int(polynom_noise_ratio*N),a1=prior[0],a2=prior[1],a3=prior[2],dR=dR,dAz=dAz,pos=pos,R=np.linalg.inv(R))
        [x_noise,y_noise,noise_cov] = generateRandomNoisyPoints(N=int((1-polynom_noise_ratio)*N),xRange=[3,100],yRange=[-40,40],dR=dR,dAz=dAz)
        x_meas = np.concatenate([x_poly, x_noise])
        y_meas = np.concatenate([y_poly, y_noise])
        dz_meas = np.concatenate([polynom_cov, noise_cov])
        z = np.array([x_meas, y_meas]).T
        dz = np.array(dz_meas)
        
        return z,dz
    
    def __convert2WorldCoordinates(self, z, cov, R, t):
        #print(z.shape, cov.shape, R.shape, t.shape)
        zw = np.matmul(R,z.T) + t
        covw = np.matmul(np.matmul(R, cov), R.T)
        
        return zw.T, covw
    
    def __generateEgoMotion(self, x0, v0, a0, path, N, dT):
        dist = np.arange(0,N)*dT*(v0+a0*dT)
        c,b,a = path[0],path[1],path[2]
        x = (np.power(3*a*dist + np.power(b+1,3./2), 2./3)-b-1)/(2*a)
        y = a*x**2+b*x+c - 5
        self.path = np.array([x, y])
        d0 = np.array([1,0]).reshape(1,2)
        di = np.array([np.repeat(1,x.shape), 2*a*x])
        
        angle = np.arctan2(d0[:,0]*di[1,:]-d0[:,1]*di[0,:],d0[:,0]*di[0,:]+d0[:,1]*di[1,:])
       
        self.R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        self.t = self.path - x0.reshape(2,1)
		
		
		

class NuscenesDataset(Dataset):
    def __init__(self, **kwargs):
        directory = kwargs.pop('directory')
        scene_id = kwargs.pop('scene', 5)
        scene_name = self.__getSceneName(scene_id)
        map_name = self.__getMapName(scene_id)
        print(r"scene_id={} scene_name={} map_name={}",scene_id, scene_name, map_name)
        self.nusc = NuScenes(version="v1.0-mini", dataroot=directory, verbose=False)
        self.nusc_map = NuScenesMap(dataroot=directory, map_name=map_name)
        nusc_can = NuScenesCanBus(dataroot=directory)
        self.imu = nusc_can.get_messages(self.__getSceneName(scene_id), 'ms_imu')
        self.veh_speed = nusc_can.get_messages(scene_name, 'zoe_veh_info')
        #ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[5]['token']], verbose=False)
        self.rpath = os.path.join(directory, 'sweeps', 'RADAR_FRONT')
        self.cpath = os.path.join(directory, 'sweeps', 'CAM_FRONT')
        self.ego = self.__extractEgo(os.path.join(directory, 'v1.0-mini', 'ego_pose.json'))
        self.radar_files = os.listdir(self.rpath)
        self.camera_files = os.listdir(self.cpath)
        self.radar_ts = self.__extractTimestamps(self.radar_files)
        self.camera_ts = self.__extractTimestamps(self.camera_files)
        
        my_scene = self.nusc.scene[scene_id]
        first_sample_token = my_scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        radar_front_data = self.nusc.get('sample_data', my_sample['data']['RADAR_FRONT'])
        self.cs_record = self.nusc.get('calibrated_sensor', radar_front_data['calibrated_sensor_token'])
        self.first_idx = self.__getFirstIdxOffset(scene_id)
        self.odometry = {'r1':0,'t':0,'r2':0}
        self.__getFirstPosition(self.first_idx)
        
    def getData(self, t):
        t += self.first_idx
        dR = 0.4
        dAz = 0.05
        z = self.__getRadarSweep(t)[:,0:2]
        #z = z[:, [1, 0]]
        cov = getXYCovMatrix(z[:,1], z[:,0], dR, dAz)
        trns,rot = self.getEgoInfo(t)
        zw = self.__getTransformedRadarData(t)[:, 0:2]
        #zw = zw[:, [1, 0]]
        R = rot.rotation_matrix[0:2,0:2] ##not great!
        covw = np.matmul(np.matmul(R, cov), R.T)
        prior = self.__getPrior(t)
        img = self.__getSyncedImage(t)
        heading = 90 + np.sign(R[1,0]) * (np.rad2deg(np.arccos(R[0,0])))
        
        video_data = {"pc": zw, "img": img, "prior": prior, "pos": trns, "rot": R, "heading": heading, "odometry": self.odometry}

        return zw, covw, prior, video_data, self.nusc_map
    
    def __getTransformedRadarData(self, t):
        f = os.path.join(self.rpath, self.radar_files[t])
        pc = RadarPointCloud.from_file(f)
        pc = self.__sensor2World(t, pc)
        
        pc.points = pc.points[:, pc.points[3,:]==1]
        return pc.points.T
        
    def __extractPolynomFromLane(self, lane):
        lane_record = self.nusc_map.get_arcline_path(lane)
        poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=0.5)
        poses = np.asarray(poses)
        lane = np.polyfit(poses[:,0], poses[:,1], 2)
        poly = np.poly1d(lane) 
        return {"x":poses[:,0], "poly":poly}
    
    def __sensor2EgoCoordinates(self, pc):
        pc.rotate(Quaternion(self.cs_record['rotation']).rotation_matrix)
        pc.rotate(Quaternion([0, 0, 1, 0]).rotation_matrix) # rotate by 180 degrees. not sure why!
        pc.translate(np.array(self.cs_record['translation']))
        
        return pc
    
    def __sensor2WorldCoordinates(self, idx, pc):
        # Stage1: Sensor -> Ego
        pc = self.__sensor2EgoCoordinates(pc)
        # Stage2: Ego -> World
        trns,rot = self.getEgoInfo(idx)
        pc.rotate(rot.rotation_matrix)
        pc.translate(np.array(trns))
        
    def __sensor2World(self, idx, pc):
        trns,rot = self.getEgoInfo(idx)
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
        
    def __extractTimestamps(self, filenames):
        ts = []
        for filename in filenames:
            ts.append(np.int64(filename.split('_')[-1].split('.')[0]))
            
        df = pd.DataFrame({'timestamp':ts})
        return df
    
    def getEgoInfo(self, i, GT=True):
        ts = self.radar_ts.iloc[i].values[0] #take timestamp from radar
        if GT:
            eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
            trns = self.ego.iloc[eidx]["translation"]
            rot = Quaternion(self.ego.iloc[eidx]["rotation"])
        else:
            #based on IMU and odometry
            rot_imu = np.array([(m['utime'], m['q']) for m in self.imu])
            ridx = np.argsort(np.abs(rot_imu[:,0]-ts))[0]
            rot = Quaternion(self.imu[ridx]['q'])
            rot *= self.first_imu_rot.inverse #substract IMU intial offset
            rot *= self.first_gt_rot # add initial GT offset 

            odom_speed = np.array([(m['utime'], m['odom_speed']) for m in self.veh_speed])
            oidx = np.argsort(np.abs(odom_speed[:,0]-ts))[0]
            dT = (odom_speed[oidx, 0] - self.veh_speed_ts) / 1e6
            self.veh_speed_ts = odom_speed[oidx, 0]
            xnorm = (odom_speed[oidx, 1] / 3.6) * dT
            x = np.array([xnorm, 0]).T.reshape((2,1))
            trns = np.dot(rot.rotation_matrix[0:2,0:2], x)
            trns = self.last_veh_trns + np.array([trns[0][0], trns[1][0], 0]) # add initial GT offset
            self.last_veh_trns = trns
            
        
        self.odometry['r1'] = self.odometry['r2']
        R = rot.rotation_matrix[0:2,0:2] ##not great!
        self.odometry['r2'] = np.sign(R[1,0]) * np.arccos(R[0,0])
        self.odometry['t'] = np.linalg.norm(np.array(trns)-np.array(self.odometry['trns']))
        self.odometry['trns'] = trns
        return trns,rot
    
    def getOdometry(self):
        return self.odometry
    
    def __getFirstPosition(self, firstIdx):
        ts = self.radar_ts.iloc[firstIdx].values[0] #take timestamp from radar
        eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
        self.first_gt_trns = self.ego.iloc[eidx]["translation"]
        self.first_gt_rot = Quaternion(self.ego.iloc[eidx]["rotation"])
        #odometry
        rot_imu = np.array([(m['utime'], m['q']) for m in self.imu])
        ridx = np.argsort(np.abs(rot_imu[:,0]-ts))[0]
        self.first_imu_rot = Quaternion(rot_imu[ridx,1])
        odom_speed = np.array([(m['utime'], m['odom_speed']) for m in self.veh_speed])
        oidx = np.argsort(np.abs(odom_speed[:,0]-ts))[0]
        self.veh_speed_ts = odom_speed[oidx, 0]
        self.last_veh_trns = self.first_gt_trns
        
        R = self.first_imu_rot.rotation_matrix[0:2,0:2]
        self.odometry['r2'] = np.sign(R[1,0]) * np.arccos(R[0,0])
        self.odometry['trns'] = self.first_gt_trns
        
    def __getPrior(self, i):
        ts = self.radar_ts.iloc[i].values[0] #take timestamp from radar
        eidx = (self.ego['timestamp']-ts).abs().argsort()[0]
        trns = self.ego.iloc[eidx]["translation"]
        lanes = []
        if 1:
            closest_lane = self.nusc_map.get_closest_lane(trns[0], trns[1], radius=2)
            incoming_lane = self.nusc_map.get_incoming_lane_ids(closest_lane)
            outgoing_lane = self.nusc_map.get_outgoing_lane_ids(closest_lane)
            lanes.append(self.__extractPolynomFromLane(closest_lane))
            for lane in incoming_lane:
                lanes.append(self.__extractPolynomFromLane(lane))
            for lane in outgoing_lane:
                lanes.append(self.__extractPolynomFromLane(lane))
        else:
            lane_ids = self.nusc_map.get_records_in_radius(trns[0], trns[1], 2, ['lane', 'lane_connector'])
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
        ts = self.radar_ts.iloc[i].values[0] #take timestamp from radar
        cidx = (self.camera_ts['timestamp']-ts).abs().argsort()[0]
        img = mpimg.imread(os.path.join(self.cpath, self.camera_files[cidx]))
        
        return img
    
    def __getSceneName(self, scene_id):
        scene_names = ["scene-0103", "scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077" "scene-1094","scene-1100"]
        return scene_names[scene_id]
    
    def __getMapName(self, scene_id):
        map_names = ["boston-seaport", "boston-seaport", "boston-seaport", "boston-seaport", "boston-seaport", "singapore-queenstown", "singapore-queenstown", "boston-seaport" "boston-seaport","boston-seaport"]
        #map_name = kwargs.pop('map_name', 'singapore-onenorth')
        #map_name = kwargs.pop('map_name', 'singapore-hollandvillage')
        #map_name = kwargs.pop('map_name', 'boston-seaport')
        return map_names[scene_id]
    
    def __getFirstIdxOffset(self, scene_id):
        idx_offset = [0, 227, 455, 687, "unknown", 1125, 1344]
        return idx_offset[scene_id]