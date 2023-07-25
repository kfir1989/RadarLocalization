from pycpd import RigidRegistration, DeformableRegistration
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from map_utils import *
import open3d as o3d
import numpy as np

def runICP(A,B,trans_init, threshold=2):
    pA = o3d.geometry.PointCloud()
    pA.points = o3d.utility.Vector3dVector(A)

    pB = o3d.geometry.PointCloud()
    pB.points = o3d.utility.Vector3dVector(B)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
    pA, pB, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    #draw_registration_result(pA, pB, reg_p2p.transformation)
    
    return reg_p2p

def plotEGO(ax, trns, rot, center=[0,0], color='green',label='EGO'):
    angle_radians = np.deg2rad(rot)
    rect_size = [5.0, 2.5]
    #Rectangle
    rectangle_xy = (center[0]+trns[0],center[1]+trns[1])
    rectangle = patches.Rectangle(rectangle_xy, rect_size[0], rect_size[1], edgecolor=color, facecolor=color,angle=rot,label=label)
    ax.add_patch(rectangle)
    #Arrow
    arrow_len = 5
    arrow_start = (rect_size[0] + rectangle_xy[0], rect_size[1] / 2 + rectangle_xy[1])
    arrow_end = (rect_size[0] + arrow_len + rectangle_xy[0], rect_size[1] / 2 + rectangle_xy[1])

    delta_x = arrow_start[0] - rectangle_xy[0]
    delta_y = arrow_start[1] - rectangle_xy[1]
    rotated_delta_x = delta_x * np.cos(angle_radians) - delta_y * np.sin(angle_radians)
    rotated_delta_y = delta_x * np.sin(angle_radians) + delta_y * np.cos(angle_radians)
    rotated_arrow_start = (rectangle_xy[0]+rotated_delta_x, rectangle_xy[1]+rotated_delta_y)
    delta_x = arrow_end[0] - rectangle_xy[0]
    delta_y = arrow_end[1] - rectangle_xy[1]
    rotated_delta_x = delta_x * np.cos(angle_radians) - delta_y * np.sin(angle_radians)
    rotated_delta_y = delta_x * np.sin(angle_radians) + delta_y * np.cos(angle_radians)
    rotated_arrow_end = (rectangle_xy[0]+rotated_delta_x, rectangle_xy[1]+rotated_delta_y)
    
    arrow = patches.FancyArrowPatch(rotated_arrow_start, rotated_arrow_end, arrowstyle='->, head_length=5, head_width=5', color='blue')
    ax.add_patch(arrow)

def generatePolynom(x_start, x_end, a0, a1, a2, fx=True):
    xx = np.arange(x_start,x_end,0.1)
    yy = a0 + a1*xx + a2*xx**2
    zz = np.zeros(xx.shape)
    
    P = np.array([xx,yy,zz]).T
    
    if not fx:
        P = P[:, [1,0,2]]
        
    return P

def transformPolynom(B, tx, ty, theta, center=(0,0)):
    #center = (tx, ty)
    #print(center)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    transform_matrix = rotate_matrix
    transform_matrix[0,2] += tx
    transform_matrix[1,2] += ty

    b = np.ones((B.shape[0],1))
    Be = np.hstack((B,b))
    
    Bt = np.dot(transform_matrix,Be.T)\
    
    return Bt.T

def calcScore(LF, Bt, patch_size, res):
    Bt_int = np.round((Bt + patch_size/2)*res)
    Bt_int = Bt_int.astype(int)
    return np.sum(LF[Bt_int[:,1], Bt_int[:,0]])

def FindBestGrad(A, i0, i1, i2):
    t0 = min(A.shape[0]-1, i0+1)
    b0 = max(0, i0-1)
    t1 = min(A.shape[1]-1, i1+1)
    b1 = max(0, i1-1)
    t2 = min(A.shape[2]-1, i2+1)
    b2 = max(0, i2-1)
    
    nbhood = A[b0:t0+1,b1:t1+1,b2:t2+1]
    max_index = np.argmax(nbhood)
    (itx, ity, itheta) = np.unravel_index(max_index, nbhood.shape)
    itx += b0
    ity += b1
    itheta += b2
    
    status = False if itx == i0 and ity == i1 and itheta == i2 else True
    return (itx, ity, itheta), status

def runLF(A,B,trns_init,parts=None,max_trns=2,max_rot=5,res=10,patch_size=200,
          max_x_uncertainry=5,max_y_uncertainry=5,max_theta_uncertainry=8,):
    #1. Generate LF from A (sparse to binary + getProbabilityMap)
    bina = scatter_to_image(A, res_factor=res, center = [0,0], patch_size=patch_size)
    bina[bina == 1] = 255
    LFa1 = build_probability_map(bina, sigma=res*0.9)
    LFa2 = build_probability_map(bina, sigma=res/2)
    
    X0 = np.arange(-max_x_uncertainry,max_x_uncertainry,0.1)
    Y0 = np.arange(-max_y_uncertainry,max_y_uncertainry,0.1)
    TH0 = np.arange(-max_theta_uncertainry,max_theta_uncertainry,0.5)
    score = np.zeros([X0.shape[0], Y0.shape[0], TH0.shape[0]])
    for ix0, x0 in enumerate(np.nditer(X0)):
        for iy0, y0 in enumerate(np.nditer(Y0)):
            for ith0, th0 in enumerate(np.nditer(TH0)):
                #print(x0, y0, th0)
                Bt = transformPolynom(B[:, 0:2], np.asscalar(x0), np.asscalar(y0), np.asscalar(th0))
                high_sigma_score = calcScore(LFa1, Bt, patch_size, res)
                
                low_sigma_score = 0
                for ipart in range(1, len(parts)):
                    polynom = Bt[parts[ipart-1]:parts[ipart],:]
                    #Find median point on the polynom
                    polynom_mid_point = polynom[int(polynom.shape[0]/2),:]
                    #Find nearest neighbor point on the map
                    eucld_dist = np.linalg.norm(A[:, 0:2]-polynom_mid_point,axis=1)
                    it = eucld_dist.argmin()
                    #Calc translation
                    txy = A[it,:2] - polynom_mid_point
                    #print(f"txy = {txy}")
                    if(np.linalg.norm(txy)) < 4:
                        #Translate polynom
                        polynom_translated = polynom + txy
                        #Compute low sigma score for translated polynom(shape)
                        low_sigma_score += calcScore(LFa2, polynom_translated, patch_size, res)
                        if (ix0==18 and iy0 == 10 and ith0 == 27) or (ix0==13 and iy0 == 13 and ith0 == 6):
                            print(f"ith0 = {ith0} ipart = {ipart} low_sigma_score = {low_sigma_score} high_sigma_score = {high_sigma_score}")
                    else:
                        #Compute low sigma score for non-translated polynom(shape)
                        low_sigma_score += calcScore(LFa2, polynom, patch_size, res)
                    
                
                score[int(ix0), int(iy0), int(ith0)] = 0.99 * high_sigma_score + 0.01 * low_sigma_score
                
    
    global_max_flag = False
    if global_max_flag:
        max_index = np.argmax(score)
        (itx, ity, itheta) = np.unravel_index(max_index, score.shape)
    else:
        itx = int(score.shape[0]/2)
        ity = int(score.shape[1]/2)
        itheta = int(score.shape[2]/2)
        while True:
            cur_score = score[itx, ity, itheta]
            (itx, ity, itheta), status = FindBestGrad(score, itx, ity, itheta)
            if not status:
                break
    
    print("best is ",(itx, ity, itheta))
    return (X0[itx], Y0[ity], TH0[itheta]), LFa2, score
