from nuscenes.map_expansion.map_api import NuScenesMap
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import scipy.ndimage.morphology
import nuscenes.map_expansion.arcline_path_utils as path_utils
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve

def draw_line(mat, x0, y0, x1, y1, fill=255, inplace=False):
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = fill
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends
    mat[x0, y0] = fill
    mat[x1, y1] = fill
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[x, y] = fill
    if not inplace:
        return mat if not transpose else mat.T
    
def smoothEdges(edges):
    black = np.where(edges == 255)
    white = np.where(edges == 0)
    binary_map = np.copy(edges)
    binary_map[black] = 10
    binary_map[white] = 0

    kernel=np.array([[1, 1, 1],[1, 10, 1],[1, 1, 1]])
    output = cv2.filter2D(binary_map, -1, kernel)
    connectors = np.where((output == 110) | (output == 100))
    connectors = np.array(connectors).T
    dists = cdist(connectors, connectors, 'euclidean') + np.diag(np.full(connectors.shape[0],100))
    outputa = np.copy(edges)
    if len(dists) > 0:
        closest = np.argmin(dists,axis=1)
        for i in range(connectors.shape[0]):
            if dists[i,closest[i]] < 3:
                outputa = draw_line(outputa, connectors[i,0], connectors[i,1], connectors[closest[i],0], connectors[closest[i],1], inplace=False)
       
    return outputa

def getRoadBorders(nuscMap, worldRef, patchSize=200, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0, threshold2=0.1, smooth=True):
        patch_angle = 0
        patch_size = patchSize
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size, patch_size)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        mask_blur = cv2.GaussianBlur(mask, blur_area, 0)
        edges = cv2.Canny(image=mask_blur, threshold1=threshold1, threshold2=threshold2) # Canny Edge Detection
        if smooth:
            edges = smoothEdges(edges)
        
        return edges
    
def getLayer(nuscMap, worldRef, patchSize=200, layer_names=['drivable_area'],blur_area=(3,3),threshold1=0.5, threshold2=0.7, res_factor=1):
        patch_angle = 0
        patch_size = patchSize
        patch_box = (worldRef[0], worldRef[1], patch_size, patch_size)
        canvas_size = (patch_size*res_factor, patch_size*res_factor)
        map_mask = nuscMap.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        mask = map_mask[0]
        
        return mask
    
def getCombinedMap_old(nuscMap, worldRef, patchSize=200, smooth=True):
    road_borders = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=patchSize, layer_names=['drivable_area'],smooth=smooth)
    walkway = getRoadBorders(nuscMap=nuscMap, worldRef=worldRef, patchSize=patchSize, layer_names=['walkway'],smooth=smooth)
    edges = road_borders | walkway
    
    return edges



def getCombinedMap(nuscMap, worldRef, patchSize=200, smooth=True, res_factor=1):
    image1 = getLayer(nuscMap, worldRef, patchSize=patchSize, layer_names=['drivable_area'],res_factor=res_factor)
    image2 = getLayer(nuscMap, worldRef, patchSize=patchSize, layer_names=['walkway'],res_factor=res_factor)
    edges1 = image1 - scipy.ndimage.morphology.binary_dilation(image1)
    edges2 = image2 - scipy.ndimage.morphology.binary_dilation(image2)
    return edges1 | edges2

def drawLanes(nusc_map, ego_trns, sx=0, sy=0):
    lane_ids = nusc_map.get_records_in_radius(ego_trns[0], ego_trns[1], 80, ['lane', 'lane_connector'])
    nearby_lanes = lane_ids['lane'] + lane_ids['lane_connector']
    lanes_poses = np.array([])
    for lane_token in nearby_lanes:
        lane_record = nusc_map.get_arcline_path(lane_token)
        poses = path_utils.discretize_lane(lane_record, resolution_meters=0.5)
        poses = np.array(poses)
        
        poses[:, 0] += sx
        poses[:, 1] += sy
        lanes_poses = np.vstack([lanes_poses, poses]) if lanes_poses.size else poses
            
    return lanes_poses

def scatter_to_image(scatter, res_factor=5, center = [600,600], patch_size=200):
    minX = np.min(scatter[:, 0])  # Maximum x-coordinate
    maxX = np.max(scatter[:, 0])  # Maximum x-coordinate
    minY = np.min(scatter[:, 1])  # Maximum y-coordinate
    maxY = np.max(scatter[:, 1])  # Maximum y-coordinate

    matrix_width = matrix_height = int(patch_size * res_factor)
    image = np.zeros((matrix_height, matrix_width), dtype=int)  # Initialize output matrix
    for point in scatter:
        x, y = point[0], point[1]
        mat_y = int(matrix_height/2) + int((y-center[1]) * res_factor)
        mat_x = int(matrix_width/2) + int((x-center[0]) * res_factor)
        if(mat_x < 0 or mat_x >= matrix_width or mat_y < 0 or mat_y >= matrix_height):
            continue
            
        image[mat_y,mat_x] = 1  # Set corresponding element to 1

    return image

def build_probability_map(binary_map, sigma=1, N=1., kernel=None):
    # Create an empty probability map with the same size as the binary map
    probability_map = np.zeros_like(binary_map, dtype=float)
    
    # Set the highest probability (1) for pixels with a value of 1 in the binary map
    probability_map[np.logical_or(binary_map == 255, binary_map == 1)] = 1.0
    
    # Define the convolution kernel
    if kernel is None:
        kernel = np.array([[0.5, 0.5, 0.5],
                           [0.5, 1.0, 0.5],
                           [0.5, 0.5, 0.5]])
    
    # Apply convolution on the binary map to preserve the central 1s
    #convolved_map = convolve(binary_map, kernel, mode='constant', cval=0.0)
    
    # Apply Gaussian filter to spread the probabilities to neighboring pixels
    filtered_map = gaussian_filter(binary_map, sigma=sigma)
    
    # Normalize the probability map to range between 0 and 1
    norm_factor=scipy.ndimage.maximum_filter(filtered_map,size=[int(sigma*1.5),int(sigma*1.5)])
    probability_map = filtered_map / norm_factor
    probability_map = np.nan_to_num(probability_map, nan=0)
    probability_map[binary_map > 0] = 1.0
    
    c1,c2 = 0.99,0.01
    probability_map = c1 * probability_map + c2# add uniform distribution
    
    return probability_map