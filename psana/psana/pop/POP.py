import numpy as np
import pickle
from psana.pop.Projection import GenerateRBFs
from psana.pop.Legendre import Legendre
from psana.pop.Quadrant import GetCenterR, GetQuadrant, Quadrant2img
from psana.pop.CartPolar import GenerateCartGrid, GeneratePolarGrid, FindNbrs, Cart2Polar, Polar2Cart
from psana.pscalib.calib.MDBWebUtils import calib_constants

class POP:
    def __init__(self, lmax=4,reg=0,alpha=1,img=None,X0=None,Y0=None,Rmax=None,RBFs_db = False,RBFs_fnm=None,edge_w=10):
    
        print('Start initialization......')                          
        lnum = int(lmax/2 + 1)         
        ls = np.arange(0,lnum)*2 
        self.reg = reg 
        self.alpha = alpha         
                     
        self.X0, self.Y0, self.Rmax = GetCenterR(img,X0,Y0,Rmax) 
        
        if RBFs_db is True:
            print('Loading RBFs from DB......')
            self.RBFs, doc = calib_constants('ele_opal', exp='amox27716', ctype='pop_rbfs', run=85)
        elif RBFs_fnm is not None:
            print('Loading RBFs from '+RBFs_fnm+'......')
            with open(RBFs_fnm, 'rb') as f:
                self.RBFs = pickle.load(f)            
        else:
            print('Generating RBFs......')
            fnm = 'RBFs_5e6_'+str(self.Rmax)+'.pkl'
            self.RBFs = GenerateRBFs(self.Rmax,num = int(5e6),fnm=fnm)
            print('RBFs saved to fnm.')
        print('RBFs loaded.')    
                  
        print('Continue initialization......')                      
        XYs_cart = GenerateCartGrid(self.Rmax)
        self.Rarr, self.num_elms_at_R, num_elms, self.Rarrs, self.Angles, XYs_polar = GeneratePolarGrid(self.Rmax)      
        
        self.inds_cart,self.cs_cart = FindNbrs(XYs_cart,XYs_polar,n_neighbors=4,algorithm='ball_tree',metric='euclidean')  
        self.inds_polar,self.cs_polar = FindNbrs(XYs_polar,XYs_cart,n_neighbors=4,algorithm='ball_tree',metric='euclidean')  
        
        self.inds_ext = self.Rarrs>(self.Rmax-edge_w)                       
            
        self.LegMat_lst, self.LegMatUt_lst, self.LegMatS_lst, self.LegMatV_lst, self.LegMat_Rr_lst = \
        self.LegendreMat_SVD(lnum, ls)   
        
        self.rbins = np.arange(0,self.Rmax+1)
        self.Ebins = np.linspace(0,self.alpha*(self.Rmax+1)**2,int(len(self.rbins)/2))
        self.scf = np.sin(self.Angles)*np.sqrt(self.Rarrs)
        
        self.Q_polar_3D_slice_fit = np.zeros((num_elms,))          
        self.Q_polar = np.zeros((num_elms,))         
        print('Initialization completed, ready to peel!')
        
    def Peel(self, img, s=[1,1,1,1]):
    
        Q_cart = GetQuadrant(img,self.X0, self.Y0, self.Rmax,s=s)        
        self.Q_polar = Cart2Polar(Q_cart,self.inds_cart,self.cs_cart)        

        
        ind = 0          
        for i, num in enumerate(self.num_elms_at_R[:-1]):
            c_arr_i = np.dot(self.LegMatV_lst[i], 
                     np.dot(np.linalg.inv(self.LegMatS_lst[i]**2 +\
                                          self.reg*np.identity(self.LegMatS_lst[i].shape[0])), 
                     np.dot(self.LegMatS_lst[i], 
                     np.dot(self.LegMatUt_lst[i], self.Q_polar[ind:(ind+num)]))))             
       
            self.Q_polar_3D_slice_fit[ind:(ind+num)] = np.dot(self.LegMat_lst[i], c_arr_i)
       
            rbf = np.repeat((num/self.num_elms_at_R[(i+1):])*self.RBFs[self.Rmax-i][1:],\
                            self.num_elms_at_R[(i+1):])            
            ImgPolarFit_3D_i_proj = rbf*np.dot(self.LegMat_Rr_lst[i],c_arr_i)
            
            self.Q_polar[(ind+num):] -= ImgPolarFit_3D_i_proj
            self.Q_polar[(ind+num):][self.Q_polar[(ind+num):]<0]=0
            ind += num
            
        self.Q_polar[self.inds_ext] = 0
        self.Q_polar_3D_slice_fit[self.inds_ext] = 0            
                 
        
    def GetSlice(self,tp='fit'):
    
        if tp=='fit':
            Qp = self.Q_polar_3D_slice_fit    
        elif tp=='left_over':
            Qp = self.Q_polar
        else:
            raise ValueError('Please set <tp> to be either "fit" or "left_over".')
            
        Qc = Polar2Cart(Qp,self.inds_polar,self.cs_polar)         
        Q = np.reshape(Qc,(self.Rmax,self.Rmax))
        Q[np.isnan(Q)] = 0
        Q[Q<0] = 0
        Q = Q/Q.max() 
        slice_img = Quadrant2img(Q)    
        
        return slice_img  
    
    
    def GetRadialDist(self):
        DistR,_ = np.histogram(self.Rarrs,bins = self.rbins,weights=self.Q_polar_3D_slice_fit*self.scf)    
        return self.rbins,DistR
        
    def GetEnergyDist(self):
        DistE,_ = np.histogram(self.alpha*self.Rarrs**2,bins = self.Ebins,weights=self.Q_polar_3D_slice_fit*self.scf)    
        return self.Ebins,DistE        
    
        
    def LegendreMat_SVD(self, lnum, ls):
    
        LegMat_lst = []
        LegMatU_lst = []
        LegMatS_lst = []
        LegMatV_lst = []       
        LegMat_Rr_lst = []
        
        ind = 0            
        for i, num in enumerate(self.num_elms_at_R):
            LegMat = np.zeros((num,lnum))
            LegMat_Rr = np.zeros((self.num_elms_at_R[(i+1):].sum(),lnum))
            
            for j, l in enumerate(ls):  
                LegMat[:,j] = Legendre(l,np.cos(self.Angles[ind:(ind+num)]))
                factor = self.Rarrs[(ind+num):]/self.Rarr[i]
                LegMat_Rr[:,j] = Legendre(l,factor*np.cos(self.Angles[(ind+num):]))  
                          
            ind += num        
            U,S,Vt = np.linalg.svd(LegMat, False)       
            LegMat_lst.append(LegMat)
            LegMatU_lst.append(U.T)
            LegMatS_lst.append(np.diag(S))
            LegMatV_lst.append(Vt.T)               
            LegMat_Rr_lst.append(LegMat_Rr)
       
        return LegMat_lst, LegMatU_lst, LegMatS_lst, LegMatV_lst, LegMat_Rr_lst
