import numpy as np

class EleMomentumRemi():
    def __init__(self,t0_ns=0,x0_mm=0,y0_mm=0,vjetx_mmPns=0,vjety_mmPns=0,
    l_mm=None,U_V=None):        
        self.t0 = t0_ns
        self.x0 = x0_mm
        self.y0 = y0_mm
        self.vjetx = vjetx_mmPns
        self.vjety = vjety_mmPns
        self.U = U_V 
        self.l = l_mm

        self.mmPns2au = 0.4571028904957055
        self.VPmm2mmPns = 1.75882014896e-1        
        self.amu2au = 1836.15
        self.au2tesla = 2.35e5
        self.au2mm = 5.28e-8
        
    def CalcPtr(self,B_tesla,omega_mns,x_mm,y_mm,t_ns):
        R = np.sqrt((x_mm-self.x0)**2 + (y_mm-self.y0)**2)
        angle = omega_mns*(t_ns-self.t0)
        angle %= 2*np.pi
        Ptr = (B_tesla/self.au2tesla)*R/self.au2mm/(2*np.abs(np.sin(angle/2)))
        
        theta = (np.arctan2(y_mm-self.y0,x_mm-self.x0)+2*np.pi)%(2*np.pi)
        phi = theta - angle/2
        Px = Ptr*np.cos(phi) - self.mmPns2au*self.vjetx
        Py = Ptr*np.sin(phi) - self.mmPns2au*self.vjety
        
        return Ptr, phi, Px, Py    
    
    def CalcR(self,B_tesla,omega_mns,Ptr,t_ns):
        angle = omega_mns*(t_ns-self.t0)
        angle %= 2*np.pi
        R = self.au2mm*(2*np.abs(np.sin(angle/2)))*Ptr/(B_tesla/self.au2tesla)
        return R        
         
    def CalcPzOneAcc(self,m_amu,t_ns):
        return self.amu2au*m_amu*self.mmPns2au*self.l/(t_ns-self.t0) - 8.04e-2*self.U*(t_ns-self.t0)/(2*self.l)
         
    def CalcPzOneAccApprox(self,ta_ns,ta0_ns,sfc=None):
        if sfc is not None:
            return sfc*(ta0_ns-ta_ns)
        else:
            return 8.04e-2*self.U*(ta0_ns-ta_ns)/self.l
