import numpy as np

def Legendre(l,x): 
    if l==0:
        return np.ones(len(x))
    elif l==2:
        return (3*x**2-1)/2
    elif l==4:
        x2 = x**2
        return ((35*x2-30)*x2+3)/8
    elif l==6:
        x2 = x**2
        return (((231*x2-315)*x2+105)*x2-5)/16
    elif l==8:
        x2 = x**2
        return ((((6435*x2-12012)*x2+6930)*x2-1260)*x2+35)/128
    elif l==10:
        x2 = x**2
        return (((((46189*x2-109395)*x2+90090)*x2-30030)*x2+3465)*x2-63)/256
    elif l==12:
        x2 = x**2
        return ((((((676039*x2-1939938)*x2+2078505)*x2-1021020)*x2+225225)*x2-18018)*x2+231)/1024
    else:
        raise ValueError('l larger than 12 not implemented.')    
