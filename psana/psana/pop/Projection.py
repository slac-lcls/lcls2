import numpy as np
import pickle

def GenerateRBFs(rmax,num = int(1e6),fnm=None):

    rs = np.arange(2,rmax+1)
    randnum = np.random.random(num)
    Xs1,Ys1 = UnitSphereAbelProj(num) 
    
    RBFs = {}   
    for i,r in enumerate(rs):    
        r3 = r**3        
        rs = np.cbrt(r3 - (r3-(r-1)**3)*randnum)    
        Xs = Xs1*rs; Ys = Ys1*rs           
        Rs = np.sqrt(Xs**2+Ys**2)
        
        Rbins = np.arange(0,r+1)             
        RBFs_r,_ = np.histogram(Rs,bins = Rbins)  
        RBFs[r] = ((RBFs_r/RBFs_r[-1])[::-1])
       
        if i%50 ==0:
            print('r = '+str(r)+' finished.')
    
    if fnm is not None:
        with open(fnm,'wb') as f:
            pickle.dump(RBFs,f,protocol=pickle.HIGHEST_PROTOCOL)
     
    return RBFs        
        
def UnitSphereAbelProj(num):
    costheta = np.random.random(num)
    phi = np.random.random(num)*np.pi/2
    return np.sqrt(1-costheta**2)*np.sin(phi), costheta
