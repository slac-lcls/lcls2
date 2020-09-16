import numpy as np


class HitFinder:

    def __init__(self, params):

        self.uRunTime = params['runtime_u']
        
        self.vRunTime = params['runtime_v']        
        
        self.uTSumAvg = params['tsum_avg_u']
        self.uTSumLow = self.uTSumAvg - params['tsum_hw_u']
        self.uTSumHigh = self.uTSumAvg + params['tsum_hw_u']
        
        self.vTSumAvg = params['tsum_avg_v'] 
        self.vTSumLow = self.vTSumAvg - params['tsum_hw_v']
        self.vTSumHigh = self.vTSumAvg + params['tsum_hw_v']        
        
        self.f_u = params['f_u']
        self.f_v = params['f_v']        
        self.Rmax = params['Rmax']
         
        self.sqrt3 = np.sqrt(3.)
             
    def FindHits(self, McpSig, u1Sig, u2Sig, v1Sig, v2Sig):
       
        t1u = (-self.uRunTime+2*McpSig+self.uTSumAvg)/2
        t2u = (self.uRunTime+2*McpSig+self.uTSumAvg)/2
            
        t1v = (-self.vRunTime+2*McpSig+self.vTSumAvg)/2
        t2v = (self.vRunTime+2*McpSig+self.vTSumAvg)/2
        
        self.subf = {}
        self.sumf = {}
        
        for k in ['u','v']:
            self.subf[k] = np.array([])
            self.sumf[k] = np.array([])        
        
        self.Xf = np.array([])
        self.Yf = np.array([])
        self.Tf = np.array([])
               
        for i_McpT, McpT in enumerate(McpSig):
           
            u1 = u1Sig[(u1Sig>t1u[i_McpT]) & (u1Sig<t2u[i_McpT])]
            u2 = u2Sig[(u2Sig>t1u[i_McpT]) & (u2Sig<t2u[i_McpT])]
            v1 = v1Sig[(v1Sig>t1v[i_McpT]) & (v1Sig<t2v[i_McpT])]
            v2 = v2Sig[(v2Sig>t1v[i_McpT]) & (v2Sig<t2v[i_McpT])]   
           
            
            u1u2_sum = u1[:,np.newaxis] + u2[np.newaxis,:] - 2*McpT 
            v1v2_sum = v1[:,np.newaxis] + v2[np.newaxis,:] - 2*McpT            
            
            u1_ind, u2_ind = np.where(True | (u1u2_sum>self.uTSumLow) | (u1u2_sum<self.uTSumHigh))
            v1_ind, v2_ind = np.where(True | (v1v2_sum>self.vTSumLow) | (v1v2_sum<self.vTSumHigh))            
            
           
            sub_u = u1[u1_ind]-u2[u2_ind]
            sub_v = v1[v1_ind]-v2[v2_ind]
            
            sub_uf = sub_u*self.f_u/2
            sub_vf = sub_v*self.f_v/2
           
            sum_u = u1[u1_ind]+u2[u2_ind] - 2*McpT 
            sum_v = v1[v1_ind]+v2[v2_ind] - 2*McpT            

            self.subf['u'] = np.concatenate([self.subf['u'],sub_u],axis=0)
            self.subf['v'] = np.concatenate([self.subf['v'],sub_v],axis=0)
            
            self.sumf['u'] = np.concatenate([self.sumf['u'],sum_u],axis=0)
            self.sumf['v'] = np.concatenate([self.sumf['v'],sum_v],axis=0)
            
            Xuv = sub_uf[:,np.newaxis] + 0*sub_vf[np.newaxis,:]
            Yuv = 0*sub_uf[:,np.newaxis] + sub_vf[np.newaxis,:] 
            
            Xuv = np.ravel(Xuv)
            Yuv = np.ravel(Yuv)
            
            Ruv = np.sqrt(Xuv**2 + Yuv**2)
            ind_R = Ruv<self.Rmax               
            
            self.Xf = np.concatenate([self.Xf,Xuv[ind_R]],axis=0)
            self.Yf = np.concatenate([self.Yf,Yuv[ind_R]],axis=0)
            self.Tf = np.concatenate([self.Tf,McpT*np.ones(len(Xuv[ind_R]))],axis=0)
            
    def GetXYT(self):
        return self.Xf,self.Yf,self.Tf
        
    def GetSumSub(self):
        return self.sumf, self.subf




