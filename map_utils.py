from nuscenes.map_expansion.map_api import NuScenesMap
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import scipy.ndimage.morphology

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