import numpy as np
from psana.momentum.CalcPzArr import CalcPzArr
class IonMomentumFlat():
    def __init__(self,t0_ns=0,x0_mm=0,y0_mm=0,vjetx_mmPns=0,vjety_mmPns=0,
    l_mm=None,d_mm=None,ls_mm=None,U_V=None,Es_VPmm=None):        
        self.t0 = t0_ns
        self.x0 = x0_mm
        self.y0 = y0_mm
        self.vjetx = vjetx_mmPns
        self.vjety = vjety_mmPns
        self.U = U_V 
        self.l = l_mm
        self.d = d_mm
        if ls_mm is None:
            self.ls = np.array([self.l,self.d],dtype=np.float)
        else:
            self.ls = np.array(ls_mm,dtype=np.float)
        if Es_VPmm is None:
            self.Es = np.array([self.U/self.l,0],dtype=np.float)
        else:
            self.Es = np.array(Es_VPmm,dtype=np.float)
        self.mmPns2au = 0.4571028904957055
        self.VPmm2mmPns = 1.75882014896e-1        
        self.amu2au = 1836.15
         
    def CalcPx(self,m_amu,x_mm,t_ns):
        return self.amu2au*m_amu*self.mmPns2au*((x_mm-self.x0)/(t_ns-self.t0) - self.vjetx)
         
    def CalcPy(self,m_amu,y_mm,t_ns):
        return self.amu2au*m_amu*self.mmPns2au*((y_mm-self.y0)/(t_ns-self.t0) - self.vjety)    
         
    def CalcPzOneAcc(self,m_amu,q_au,t_ns):
        return self.amu2au*m_amu*self.mmPns2au*self.l/(t_ns-self.t0) - 8.04e-2*q_au*self.U*(t_ns-self.t0)/(2*self.l)
         
    def CalcPzOneAccApprox(self,ta_ns,ta0_ns,sfc=None,q_au=None):
        if sfc is not None:
            return sfc*(ta0_ns-ta_ns)
        elif q_au is not None:
            return 8.04e-2*q_au*self.U*(ta0_ns-ta_ns)/self.l
         
    #The algorithm for calculating Pz is adapted from that of Lutz Focar's CASS software.                 
    def CalcPzMultiAcc(self,m_amu,q_au,t_ns):
        if isinstance(t_ns,(list, tuple, np.ndarray)):
            if not isinstance(m_amu,(list, tuple, np.ndarray)):
                m_amu = np.ones((len(t_ns),),dtype=np.float)*m_amu
            if not isinstance(q_au,(list, tuple, np.ndarray)):
                q_au = np.ones((len(t_ns),),dtype=np.float)*q_au
            Pz_au = CalcPzArr(m_amu,q_au,t_ns,self.Es,self.ls,self.t0)
        else:
            t_ns = t_ns-self.t0        
            a = self.VPmm2mmPns*self.Es[0]*q_au/(self.amu2au*m_amu)
            v0 = self.ls[0]/t_ns-0.5*a*t_ns       
            v0 = self.Newton(v0,m_amu,q_au,t_ns)                  
            Pz_au = self.mmPns2au*v0*(self.amu2au*m_amu)
        return Pz_au     
                  
    def Newton(self,v0,m_amu,q_au,t_ns):
        td0 = self.CalcTofd(v0,m_amu,q_au,t_ns)
        while (abs(td0) > 0.01):
            v1 = 1.1*v0
            td1 = self.CalcTofd(v1,m_amu,q_au,t_ns)
            m = (td0-td1)/(v0-v1)
            v0 = v0-0.7*(td0)/m
            td0 = self.CalcTofd(v0,m_amu,q_au,t_ns)  
        return v0
    
    def CalcTofd(self,v0,m_amu,q_au,t_ns):
        v = v0
        t = 0
        for i, E in enumerate(self.Es):
            a = self.VPmm2mmPns*E*q_au/(self.amu2au*m_amu)
            l = self.ls[i]                             
            tt = 0
            if (a != 0):
                tt = (-v+np.sqrt(v*v+2*a*l))/a
            else:
                tt = l/v                        
            v += a*tt
            t += tt
        td = t-t_ns
        return td     
  
