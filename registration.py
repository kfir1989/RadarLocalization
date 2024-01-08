from pycpd import RigidRegistration, DeformableRegistration
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from map_utils import *
import open3d as o3d
import numpy as np
import cmath

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
    rect_size = [2.5, 1.5]
    #Rectangle
    rectangle_xy = (center[0]+trns[0],center[1]+trns[1])
    rectangle = patches.Rectangle(rectangle_xy, rect_size[0], rect_size[1], edgecolor=color, facecolor=color,angle=rot,label=label,zorder=10)
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
    angle = theta if theta < 0 else theta + 360
    #print(f"angle={type(angle)}")
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    transform_matrix = rotate_matrix
    transform_matrix[0,2] += tx
    transform_matrix[1,2] += ty

    b = np.ones((B.shape[0],1))
    Be = np.hstack((B,b))
    
    Bt = np.dot(transform_matrix,Be.T)\
    
    return Bt.T

class LF:
    def __init__(self):
        pass
    
    @staticmethod
    def FindBestGrad(A, i0, i1, i2,method="max"):
        t0 = min(A.shape[0]-1, i0+1)
        b0 = max(0, i0-1)
        t1 = min(A.shape[1]-1, i1+1)
        b1 = max(0, i1-1)
        t2 = min(A.shape[2]-1, i2+1)
        b2 = max(0, i2-1)

        nbhood = A[b0:t0+1,b1:t1+1,b2:t2+1]
        if method == "max":
            max_index = np.argmax(nbhood)
        else:
            max_index = np.argmin(nbhood)
        (itx, ity, itheta) = np.unravel_index(max_index, nbhood.shape)
        itx += b0
        ity += b1
        itheta += b2

        status = False if itx == i0 and ity == i1 and itheta == i2 else True
        return (itx-1, ity-1, itheta-1), status
    
    def buildMaps(self, A, res=10,patch_size=200):
        #1. Generate LF from A (sparse to binary + getProbabilityMap)
        bina = scatter_to_image(A, res_factor=res, center = [0,0], patch_size=patch_size)
        bina[bina == 1] = 255
        self.LFa1 = build_probability_map(bina, sigma=15,outlier_th=0.02)
        self.LFa2 = build_probability_map(bina, sigma=3,outlier_th=0.02)
        
    def calcScoreMatrix(self,A,B, \
                        X0, Y0, TH0,
                        parts=None,res=10,patch_size=200,
                        method="_basic"):
        score = np.zeros([X0.shape[0], Y0.shape[0], TH0.shape[0]])
        for ix0, x0 in enumerate(np.nditer(X0)):
            for iy0, y0 in enumerate(np.nditer(Y0)):
                for ith0, th0 in enumerate(np.nditer(TH0)):
                    #print(x0, y0, th0)
                    Bt = transformPolynom(B[:, 0:2], np.asscalar(x0), np.asscalar(y0), np.asscalar(th0))
                    high_sigma_score = calcScore(self.LFa1, Bt, patch_size, res)

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
                        if method == "_improved" and (np.linalg.norm(txy)) < 5:
                            #Translate polynom
                            polynom_translated = polynom + txy
                            #Compute low sigma score for translated polynom(shape)
                            low_sigma_score += calcScore(self.LFa2, polynom_translated, patch_size, res)
                        elif method == "_improved":
                            #Compute low sigma score for non-translated polynom(shape)
                            low_sigma_score += calcScore(self.LFa2, polynom, patch_size, res)

                    score[int(ix0), int(iy0), int(ith0)] = high_sigma_score + low_sigma_score
                    
        return score

    def run(self,A, B, parts=None,
            max_x_uncertainry=5,max_y_uncertainry=5,max_theta_uncertainry=8,
            method="_basic"):
        
        t_res = 0.1 #[m]
        angle_res = 0.2 #[deg]
        tx = 0
        ty = 0
        theta = 0   
        num_iterations = 0
        best_score = 1e6
        stop_th = 1e-3
        while True:
            num_iterations += 1
            X0 = np.arange(tx-t_res, tx+t_res+1e-6,t_res)
            Y0 = np.arange(ty-t_res, ty+t_res+1e-6,t_res)
            TH0 = np.arange(theta-angle_res,theta+angle_res+1e-6,angle_res)
            #print(f" it = {num_iterations} X0 = {X0} Y0 = {Y0} TH0 = {TH0} theta={theta}")
            cur_score = self.calcScoreMatrix(A, B, X0, Y0, TH0, parts=parts,method=method)
            (itx, ity, itheta), status = self.FindBestGrad(cur_score, 1, 1, 1, method="min")
            best_it_score = cur_score[itx+1,ity+1,itheta+1]
            tx += t_res * itx
            ty += t_res * ity
            theta += angle_res * itheta
            if not status:
                break
            if best_it_score < best_score - stop_th:
                best_score = best_it_score
            else:
                break

        return (tx, ty, theta), self.LFa2, cur_score

def calcScore(LF, Bt, patch_size, res):
    Bt_int = np.round((Bt + patch_size/2)*res)
    Bt_int = Bt_int.astype(int)
    Bt_int = np.unique(Bt_int, axis=0)
    return np.sum(1./LF[Bt_int[:,1], Bt_int[:,0]]) / Bt.shape[0]


epsLine = 1.
epsCircle = 1
epsCorner_inc = 0.1
epsCorner_offset = 3

def solve_quadratic_equation(a, b, c):
    # Calculate the discriminant
    discriminant = (b ** 2) - (4 * a * c)

    # Check the discriminant for nature of roots
    if discriminant > 0:
        # Two real and distinct roots
        root1 = (-b + cmath.sqrt(discriminant)) / (2 * a)
        root2 = (-b - cmath.sqrt(discriminant)) / (2 * a)
        return [root1, root2]
    elif discriminant == 0:
        # Two real and identical roots
        root = -b / (2 * a)
        return [root]
    else:
        return [np.nan]

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (np.array([cx, cy]), radius)

def getPoints(polynom):
    x = np.linspace(polynom['x_start'], polynom['x_end'], int(np.ceil(np.abs(polynom['x_end']-polynom['x_start']))*10))
    y = polynom['f'][0] + polynom['f'][1] * x + polynom['f'][2] * x**2
    points = np.array([x, y]).T if polynom['fxFlag'] else np.array([y, x]).T
    return points

def test_line(p1,p2,points):
    d = np.abs(np.cross(p2-p1, p1-points)) / np.linalg.norm(p2-p1)
    #print(max(d))
    return (d < epsLine)

def test_circle(p1,p2,points):
    #print(points.shape)
    p3 = points[int(points.shape[1]/2), :]
    xc, rc = define_circle(p1, p2, p3)
    d = np.abs(np.sqrt(((points-xc)**2).sum(1))-rc)
    #print(points.shape, xc.shape, rc)
    #print(max(d), rc, xc, p1, p2)
    return (d < epsLine)

def test_corner(p1,p2):
    offset = 2
    inc = 0.1
    xx1 = np.linspace(p1['x_start'], p1['x_end'], 10)
    xx2 = np.linspace(p2['x_start'], p2['x_end'], 10)
    pp1 = np.polyfit(xx1, p1['f'](xx1),1)
    pp2 = np.polyfit(xx2, p2['f'](xx2),1)
    b1 = pp1[0]
    c1 = pp1[1]
    b2 = pp2[0]
    c2 = pp2[1]
    if p1['fxFlag'] == p2['fxFlag']:
        if(abs((b2-b1)/(1+b1*b2)) > 0.5):
            x = (c2-c1)/(b1-b2)
            if (x >= p1['x_start']-offset and x <= p1['x_end']+offset and x >= p2['x_start']-offset and x <= p2['x_end']+offset):
                return True
    else:
        b1 = pp1[1] if p1['fxFlag'] else pp2[1]
        c1 = pp1[0] if p1['fxFlag'] else pp2[0]
        b2 = pp2[1] if p1['fxFlag'] else pp1[1]
        c2 = pp2[0] if p1['fxFlag'] else pp1[0]
        xs = p1['x_start'] if p1['fxFlag'] else p2['x_start']
        xe = p1['x_end'] if p1['fxFlag'] else p2['x_end']
        ys = p2['x_start'] if p1['fxFlag'] else p1['x_start']
        ye = p2['x_end'] if p1['fxFlag'] else p1['x_end']
        if (abs(b1*b2-1) > epsCorner):
            y = (b1*c2+c1)/(1-b1*b2)
            if(y >= ys-offset and y <= ye+offset and y >= min(xs,xe)-offset and y <= max(xs,xe)+offset):
                return True
            
    return False

def classifyShape(polynoms):
    lines = []
    circles = []
    clothoids = []
    corners = []
    #classify polynoms to lines, circles, clothoids
    for ipol,polynom in enumerate(polynoms):
        xs, xe = polynom['x_start'], polynom['x_end']
        p1 = np.array([xs, polynom['f'](xs)]).T
        p2 = np.array([xe, polynom['f'](xe)]).T
        points = getPoints(polynom)
        #print(points)
        if all(test_line(p1,p2,points)):
            lines.append(ipol)
        elif all(test_circle(p1,p2,points)):
            circles.append(ipol)
        else:
            clothoids.append(ipol)
    
    #check corners
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            if test_corner(polynoms[lines[i]],polynoms[lines[j]]):
                corners.append(lines[i])
                corners.append(lines[j])
    #remove lines if they form a corner            
    for i in range(len(lines)):
        if i in corners:
            lines.remove(i)
            
    return (lines, circles, clothoids, corners)
