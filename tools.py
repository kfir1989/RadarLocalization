import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms

def drawEgo(x0, y0, angle, ax, n_std=1.0, facecolor='none',width=2,height=5, **kwargs):
    ego = Rectangle((x0, y0), width=2, height=5, angle=angle,
                      facecolor='black', **kwargs)

    ax.add_patch(ego)

    return ax

def confidence_ellipse(x0, y0, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if cov.shape != (2,2):
        raise ValueError("only 2x2 covariance matrices")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x0, y0)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
	
    return ax

def pol2car(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def errorPropagation(R,Az,dR,dAz):
    J = np.array([[np.sin(Az), R*np.cos(Az)],[np.cos(Az), -R*np.sin(Az)]]) #x and y are opposite here!
    orig_cov =  np.diag([dR*dR, dAz*dAz])

    return np.matmul(np.matmul(np.transpose(J, (2,0,1)), orig_cov), np.transpose(J, (2,1,0)))

def createPolynom(a1,a2,a3,xstart=0,xend=100):
    x = np.linspace(xstart,xend,50)
    y = a3*x**2 + a2*x + a1

    return x, y

def generatePolynom(a1,a2,a3,n,xstart=0,xend=100):
    x = np.sort(np.asarray(np.random.random_sample(n)))
    x = (xend - xstart) * x + xstart
    y = a3*x**2 + a2*x + a1

    return x, y

def getXYCovMatrix(x, y, dR, dAz):
    R = np.sqrt(x**2+y**2)
    Az = np.arctan(y/x)
    cov = errorPropagation(R,Az,dR,dAz) #back to xy covariance matrix
    return cov

def generatePolynomNoisyPoints(N, a1, a2, a3, dR, dAz, xRange=[0,100], pos=[0,0], R=np.eye(2)):
    x, y = generatePolynom(a1,a2,a3,N,xstart=xRange[0],xend=xRange[1])
    new_pos = np.matmul(R, np.array([x-pos[0],y-pos[1]]))
    x = new_pos[0,:]
    y = new_pos[1,:]
    R = np.sqrt(x**2+y**2)
    Az = np.arctan2(y,x)
    cov = errorPropagation(R,Az,dR,dAz)
    r_noisy = R + dR * np.random.randn(R.shape[0])
    az_noisy = Az + dAz * np.random.randn(Az.shape[0])
    x_noisy = r_noisy * np.cos(az_noisy) # x and y are tranverse
    y_noisy = r_noisy * np.sin(az_noisy) # x and y are tranverse

    return [x, y, x_noisy, y_noisy, cov]

def generateRandomNoisyPoints(N, xRange, yRange, dR, dAz):
    x = xRange[0]+(xRange[1]-xRange[0])*np.random.rand(N)
    y = yRange[0]+(yRange[1]-yRange[0])*np.random.rand(N)
    R = np.sqrt(x**2+y**2)
    Az = np.arctan(y/x)
    cov = errorPropagation(R,Az,dR,dAz)
    
    return [x,y,cov]