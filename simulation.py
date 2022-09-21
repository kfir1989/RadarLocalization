from dataset import *
from video import SimulationVideo
from video import NuscenesVideo, NuscenesVideoDebug, PFVideo, PFXYVideo, DynamicTrackerVideo
from database import NuscenesProcessedDatabase, SimulatedProcessedDatabase

class Simulation():
    def __init__(self, model, **kwargs):
        pass
    
    def run(self, N):
        pass
    
class DynamicSimulation():
    def __init__(self, model, **kwargs):
        self.model = model
        self.dataset = DynamicSimulatedDataset(**kwargs)
        self.video = SimulationVideo()
        self.video_flag = kwargs.pop('video_flag', False)
        self.save_processed = kwargs.pop('save_processed', False)
        self.sim_name = kwargs.pop('name', "sim")
        self.database = SimulatedProcessedDatabase(name=self.sim_name)
    
    def run(self, N):
        results = []
        for t in range(0,N):
            print(f"frame {t}")
            zw, covw, prior, video_data = self.dataset.getData(t)
            #print("prior", prior)
            points, polynoms = self.model.run(zw,covw,prior)
            prior = [{"c": (27.5,-5,0.3), "xmin": 5, "xmax": 16,"fx": True}]
            
            if self.video_flag:
                self.video.save(t, prior, video_data, points, polynoms, self.model.getDebugInfo(),pos=video_data["pos"])
            
            if self.save_processed:
                self.database.save(t, prior, video_data, points, polynoms, self.model.getDebugInfo(), video_data["pos"])
                
            results.append(polynoms)
            
        return results
            
class NuscenesSimulation():
    def __init__(self, nusc, model, scene_id, **kwargs):
        self.model = model
        directory = kwargs.pop('directory', r"/home/kfir/workspace/nuScenes/v1.0-trainval")
        self.video_list = kwargs.pop('video_list', {'video' : False, 'video_debug': False, 'video_pf': False, 'video_pf_xy': False})
        self.save_processed = kwargs.pop('save_processed', False)
        self.nmax = kwargs.pop('Nmax', 800)
        self.mm = model.mm
        self.dataset = NuscenesDataset(nusc=nusc, directory=directory, scene_id=scene_id, N=self.nmax)
        self.video = NuscenesVideo(history=True, scene=scene_id)
        self.video_debug = NuscenesVideoDebug(history=True, scene=scene_id)
        self.video_pf = PFVideo(history=True, scene=scene_id, N=self.nmax)
        self.video_pf_xy = PFXYVideo(history=True, scene=scene_id, N=self.nmax)
        self.video_dynamic_tracker = DynamicTrackerVideo(history=True, scene=scene_id, N=self.nmax)
        self.database = NuscenesProcessedDatabase(scene_id=scene_id)
        self.lane = None
        self.scene = scene_id
    
    def drawPlots(self, t, video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, nusc_map, video_with_priors, translation, debug_info, generate_video=False):
        if self.video_list["video"]:
            self.video.save(t,video_data, polynoms, nusc_map, video_with_priors=video_with_priors)
            if generate_video:
                self.video.generate(name=f"video\scene{self.scene}.avi", fps=5)
        if self.video_list["video_debug"]:
            self.video_debug.save(t,video_data, polynoms,points, nusc_map, debug_info, video_with_priors=video_with_priors)
            if generate_video:
                self.video_debug.generate(name=f"video\scene{self.scene}_debug.avi", fps=5)
        if self.mm and self.video_list['video_pf']:
            self.video_pf.save(t,video_data, mm_results, polynoms, dynamic_tracks, nusc_map)
            if generate_video:
                self.video_pf.generate(name=f"video\scene{self.scene}_pf.avi", fps=5)
        if self.mm and self.video_list['video_pf_xy']:
            self.video_pf_xy.save(t,video_data, mm_results, polynoms, nusc_map)
        if self.video_list["dynamic_tracker"]:
            self.video_dynamic_tracker.save(t, dynamic_tracks,dynamic_clusters, video_data, nusc_map)
    
    def run(self,start, N, generate_video=False, video_with_priors=False, debug=False, translate=True):
        start_idx = start
        first = True
        gt_pos = np.zeros([N,2])
        pf_pos = np.zeros([N,2])
        for t in range(start_idx,start_idx + N):
            print(f"frame {t}")
            #get data
            zw, covw, prior, dw, video_data, nusc_map = self.dataset.getData(t)
            translation = np.array(video_data["ego_path"][0])[0:2]
            #run model
            points, polynoms, dynamic_tracks, dynamic_clusters, debug_info, mm_results = self.model.run({"zw":zw, "covw":covw}, {"dw":dw}, video_data, prior, translation, nusc_map)
            #Draw plots
            self.drawPlots(t, video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, nusc_map, video_with_priors, translation, debug_info, generate_video=generate_video)
            if self.save_processed:
                self.database.save(t, video_data, polynoms, points, dynamic_tracks, dynamic_clusters, mm_results, translation, debug_info)
                
        
            
