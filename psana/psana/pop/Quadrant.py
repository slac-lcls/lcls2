import numpy as np

def GetCenterR(img,X0=None,Y0=None,Rmax=None):

    if X0 is None:
        X0 = img.shape[1]//2
    if Y0 is None:
        Y0 = img.shape[0]//2   
    if Rmax is None:      
        Rmax = min(img.shape[1]-X0, img.shape[0]-Y0)
            
    return X0, Y0, Rmax


def GetQuadrant(img,X0,Y0,Rmax,s=[1,1,1,1]):

    i = 0
    Qs = np.zeros((s[0]+s[1]+s[2]+s[3], Rmax, Rmax))
    
    if img.shape[1] % 2 == 1:
        if s[0] == 1:
            Qs[i] = img[Y0-Rmax+1:Y0+1,X0:X0+Rmax]; i += 1
        if s[1] == 1:
            Qs[i] = np.fliplr(img[Y0-Rmax+1:Y0+1,X0-Rmax+1:X0+1]); i += 1
        if s[2] == 1:
            Qs[i] = np.flipud(img[Y0:Y0+Rmax,X0:X0+Rmax]); i += 1            
        if s[3] == 1:
            Qs[i] = np.flipud(np.fliplr(img[Y0:Y0+Rmax,X0-Rmax+1:X0+1]))
        
    elif img.shape[1] % 2 == 0:
        if s[0] == 1:
            Qs[i] = img[Y0-Rmax:Y0,X0:X0+Rmax]; i += 1 
        if s[1] == 1:
            Qs[i] = np.fliplr(img[Y0-Rmax:Y0,X0-Rmax:X0]); i += 1
        if s[2] == 1:
            Qs[i] = np.flipud(np.fliplr(img[Y0:Y0+Rmax,X0-Rmax:X0])); i += 1   
        if s[3] == 1:
            Qs[i] = np.flipud(img[Y0:Y0+Rmax,X0:X0+Rmax])        

    Q = np.flipud(Qs.sum(axis=0))
        
    return Q
    
def Quadrant2img(Q):
    Q = np.flipud(Q)
    Q = np.concatenate((np.fliplr(Q),Q),axis=1)
    img = np.concatenate((Q,np.flipud(Q)),axis=0)  
    return img
      
    
    
    
    
    
