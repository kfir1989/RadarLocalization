from dataset import *
from video import SimulationVideo
from video import NuscenesVideo, NuscenesVideoDebug, PFVideo, PFXYVideo

class Simulation():
    def __init__(self, model, **kwargs):
        pass
    
    def run(self, N):
        pass
    
class DynamicSimulation():
    def __init__(self, model, **kwargs):
        self.model = model
        self.dataset = DynamicSimulatedDataset()
        self.video = SimulationVideo()
    
    def run(self, N):
        for t in range(0,N):
            print(f"frame {t}")
            zw, covw, prior, video_data = self.dataset.getData(t)
            print("prior", prior)
            points, polynoms = self.model.run(zw,covw,prior)
            self.video.save(t, prior, video_data, points, polynoms, self.model.getDebugInfo())
            
class NuscenesSimulation():
    def __init__(self, nusc, model, scene_id, **kwargs):
        self.model = model
        directory = kwargs.pop('directory', r"/home/kfir/workspace/nuScenes/v1.0-trainval")
        self.video_list = kwargs.pop('video_list', {'video' : False, 'video_debug': False, 'video_pf': True, 'video_pf_xy': False})
        self.nmax = kwargs.pop('Nmax', 800)
        self.mm = model.mm
        self.dataset = NuscenesDataset(nusc=nusc, directory=directory, scene_id=scene_id, N=self.nmax)
        self.video = NuscenesVideo(history=True, scene=scene_id)
        self.video_debug = NuscenesVideoDebug(history=True, scene=scene_id)
        self.video_pf = PFVideo(history=True, scene=scene_id, N=self.nmax)
        self.video_pf_xy = PFXYVideo(history=True, scene=scene_id, N=self.nmax)
        self.lane = None
        self.scene = scene_id
    
    def drawPlots(self, t, video_data, polynoms, points, mm_results, nusc_map, video_with_priors, translation, debug_info, generate_video=False):
        if self.video_list["video"]:
            self.video.save(t,video_data, polynoms, nusc_map, video_with_priors=video_with_priors)
            if generate_video:
                self.video.generate(name=f"video\scene{self.scene}.avi", fps=5)
        if self.video_list["video_debug"]:
            self.video_debug.save(t,video_data, polynoms,points, nusc_map, debug_info, video_with_priors=video_with_priors)
            if generate_video:
                self.video_debug.generate(name=f"video\scene{self.scene}_debug.avi", fps=5)
        if self.mm and self.video_list['video_pf']:
            self.video_pf.save(t,video_data, mm_results, polynoms, nusc_map)
            if generate_video:
                self.video_pf.generate(name=f"video\scene{self.scene}_pf.avi", fps=5)
        if self.mm and self.video_list['video_pf_xy']:
            self.video_pf_xy.save(t,video_data, mm_results, polynoms, nusc_map)
    
    def run(self,start, N, generate_video=False, video_with_priors=False, debug=False, translate=True):
        start_idx = start
        first = True
        for t in range(start_idx,start_idx + N):
            print(f"frame {t}")
            #get data
            zw, covw, prior, video_data, nusc_map = self.dataset.getData(t)
            translation = np.array(video_data["ego_path"][0])[0:2]
            #run model
            points, polynoms, debug_info, mm_results = self.model.run({"zw":zw, "covw":covw}, video_data, prior, translation, nusc_map)
            #Draw plots
            self.drawPlots(t, video_data, polynoms, points, mm_results, nusc_map, video_with_priors, translation, debug_info, generate_video=generate_video)
            