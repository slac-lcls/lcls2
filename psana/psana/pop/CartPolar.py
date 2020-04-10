import sys
import numpy as np
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors

def GenerateCartGrid(Rmax):

    Xs = np.arange(Rmax)+0.5
    Xs_mesh, Ys_mesh = np.meshgrid(Xs, Xs)
    Xs_mesh = Xs_mesh.flatten()
    Ys_mesh = Ys_mesh.flatten()   
    XYs_cart = np.hstack([Xs_mesh[:,np.newaxis],Ys_mesh[:,np.newaxis]])      
        
    return XYs_cart
    
def GeneratePolarGrid(Rmax):  

    Rarr = np.arange(Rmax)[::-1]+0.5
    num_elms_at_R = (np.round(0.5*np.pi*Rarr)).astype(int)
    angle_incs = 0.5*np.pi/num_elms_at_R 
    num_elms = num_elms_at_R.sum()
               
    Angles = np.zeros((num_elms,))
    Rarrs = np.zeros((num_elms,))        
        
    ind = 0
    for i, num in enumerate(num_elms_at_R):
        Angles[ind:(ind+num)] = np.arange(0,num)*angle_incs[i] + angle_incs[i]/2
        Rarrs[ind:(ind+num)] = np.ones((num,))*Rarr[i]
        ind += num      
                
    Xs = Rarrs*np.sin(Angles)
    Ys = Rarrs*np.cos(Angles)        
    XYs_polar = np.hstack([Xs[:,np.newaxis],Ys[:,np.newaxis]])         
    
    return Rarr, num_elms_at_R, num_elms, Rarrs, Angles, XYs_polar

def FindNbrs(xs,ys,n_neighbors=4,algorithm='ball_tree',metric='euclidean'):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,metric=metric).fit(xs)        
    ds, inds = nbrs.kneighbors(ys)
    ds += sys.float_info.epsilon
    cs = 1/ds
    cs = cs/(cs.sum(axis=1)[:,np.newaxis])
    
    return inds,cs
    
def Cart2Polar(Q_cart,inds_cart,cs_cart):
    
    Q_cart = Q_cart.flatten()        
    Q_polar = (Q_cart[inds_cart]*cs_cart).sum(1)
    
    return Q_polar
    
def Polar2Cart(Q_polar,inds_polar,cs_polar):
    
    Q_cart = (Q_polar[inds_polar]*cs_polar).sum(1)

    return Q_cart    
        
def Cart2Polar_interp(Q_cart,XYs_cart,XYs_polar,method='cubic'):

    Q_cart = Q_cart.flatten()
    Q_polar = interpolate.griddata(XY_cart,Q_cart,XY_polar,method=method) 

    return Q_polar         
        
def Polar2Cart_interp(Q_Polar,XYs_polar,XYs_cart,method='cubic'):
            
    Q_cart = interpolate.griddata(XY_polar,Q_polar,XY_cart,method=method) 
    
    return Q_cart       
    

