cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_flt
cdef double mmPns2au = 0.4571028904957055
cdef double VPmm2mmPns = 1.75882014896e-1        
cdef double amu2au = 1836.15  
    
@cython.boundscheck(False) 
@cython.wraparound(False)

def CalcPzArr(np.ndarray[DTYPE_flt, ndim=1] m_amu,np.ndarray[DTYPE_flt, ndim=1] q_au,np.ndarray[DTYPE_flt, ndim=1] t_ns,
np.ndarray[DTYPE_flt, ndim=1] Es,np.ndarray[DTYPE_flt, ndim=1] ls,double t0):

    cdef np.ndarray[DTYPE_flt, ndim=1] a, v0,Pz_au  
    
    t_ns = t_ns - t0        
    a = VPmm2mmPns*Es[0]*q_au/(amu2au*m_amu)
    v0 = ls[0]/t_ns-0.5*a*t_ns
        
    for i in range(len(t_ns)):
        v0[i] = Newton(m_amu[i],q_au[i],v0[i],t_ns[i],Es,ls)  
        
    Pz_au = mmPns2au*v0*(amu2au*m_amu)
    
    return Pz_au  
    
def Newton(double m_amu,double q_au,double v0,double t_ns,np.ndarray[DTYPE_flt, ndim=1] Es,np.ndarray[DTYPE_flt, ndim=1] ls):
    cdef double td0, v1,td1,slp
    td0 = CalcTofd(m_amu,q_au,t_ns,v0,Es,ls)
    while (abs(td0) > 0.01):
        v1 = 1.1*v0
        td1 = CalcTofd(m_amu,q_au,t_ns,v1,Es,ls)
        slp = (td0-td1)/(v0-v1)
        v0 = v0-0.7*(td0)/slp
        td0 = CalcTofd(m_amu,q_au,t_ns,v0,Es,ls)  
    return v0    
    
def CalcTofd(double m_amu,double q_au,double t_ns,double v0,np.ndarray[DTYPE_flt, ndim=1] Es,np.ndarray[DTYPE_flt, ndim=1] ls):
    cdef double t,tt,a
    t = 0
    for i in range(len(Es)):
        a = VPmm2mmPns*Es[i]*q_au/(amu2au*m_amu)
                                    
        tt = 0
        if (a != 0):
            tt = (-v0+np.sqrt(v0*v0+2*a*ls[i]))/a
        else:
            tt = ls[i]/v0                
        
        v0 += a*tt
        t += tt
    td = t-t_ns
    return td      
