import cmath
import numpy as np

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

def test_line(p1,p2,points,epsLine=1.):
    d = np.abs(np.cross(p2-p1, p1-points)) / np.linalg.norm(p2-p1)
    #print(max(d))
    return (d < epsLine)

def test_circle(p1,p2,points,epsLine=1.):
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

def extractPriorShape(polynoms):
    lines = []
    circles = []
    clothoids = []
    corners = []
    #classify polynoms to lines, circles, clothoids
    for ipol,polynom in enumerate(polynoms):
        if polynom["shape"] == "Line":
            lines.append(ipol)
        elif polynom["shape"] == "Arc":
            circles.append(ipol)
        elif polynom["shape"] == "Clothoid":
            clothoids.append(ipol)
        elif polynom["shape"] == "Corner":
            corners.append(ipol)
            
    return (lines, circles, clothoids, corners)