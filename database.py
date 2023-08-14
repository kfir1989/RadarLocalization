import pickle
import os

class DynamicTrackDatabase():
    def __init__(self, dyn_trk):
        self.confirmed = dyn_trk.confirmed
        self.hits = dyn_trk.hits
        self.saver_x, self.ego, self.ego_speed = dyn_trk.getHistory()
        
    def getHistory(self):
        return self.saver_x, self.ego, self.ego_speed
    
class ProcessedDatabase():
    def __init__(self):
        pass
    
    def run(self, N):
        pass
    
class SimulatedProcessedDatabase(ProcessedDatabase):
    def __init__(self, name, **kwargs):
        self.dir_name = os.path.join("images", f"{name}","database")    
        print("self.dir_name", self.dir_name)
        os.system("mkdir -p " + self.dir_name)
    
    def save(self, t, prior, video_data, points, polynoms, debug_info, pos):
        file_name = os.path.join(self.dir_name, f"frame_{t}.pkl")
        #relevant_keys = ["pc", "prior", "pos", "rot", "heading", "heading_imu", "pos_imu", "rot_imu", "ego_path", "ego_trns", "veh_speed", "timestamp"]
        #video_data_save = { relevant_key: video_data[relevant_key] for relevant_key in relevant_keys }
                         
        with open(file_name, 'wb') as f:
            pickle.dump([prior, video_data, points, polynoms, debug_info, pos], f)

    def load(self, t):
        file_name = os.path.join(self.dir_name, f"frame_{t}.pkl")
        with open(file_name, 'rb') as f:
            prior, video_data, points, polynoms, debug_info, pos = pickle.load(f)

        return prior, video_data, points, polynoms, debug_info, pos
    
    
    
    
            
class NuscenesProcessedDatabase(ProcessedDatabase):
    def __init__(self, scene_id, base_dir='images', **kwargs):
        self.dir_name = os.path.join(base_dir, f"{scene_id}","database")    
        print("self.dir_name", self.dir_name)
        os.system("mkdir -p " + self.dir_name)
    
    def save(self, t, video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info, generate_video=False):
        file_name = os.path.join(self.dir_name, f"frame_{t}.pkl")
        relevant_keys = ["pc", "prior", "pos", "rot", "heading", "heading_imu", "pos_imu", "rot_imu", "ego_path", "ego_trns", "veh_speed", "timestamp"]
        video_data_save = { relevant_key: video_data[relevant_key] for relevant_key in relevant_keys }
        
        dynamic_tracks_database = []
        for trk in dynamic_tracks:
            tmp = DynamicTrackDatabase(trk)
            dynamic_tracks_database.append(tmp)
                         
        with open(file_name, 'wb') as f:
            pickle.dump([video_data_save, polynoms, points, dynamic_tracks_database, dynamic_clusters, mm_results, translation, debug_info], f)

    def load(self, t):
        file_name = os.path.join(self.dir_name, f"frame_{t}.pkl")
        with open(file_name, 'rb') as f:
            video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info = pickle.load(f)

        return video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info