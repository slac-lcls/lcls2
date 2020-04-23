import numpy as np
import matplotlib.pyplot as plt
from psana.momentum.IonMomentumFlat import IonMomentumFlat
from psana.momentum.Energy import CalcEnergy

names = ["Ion N","Events","TOF","Mass","Charge","X","Y","Z","Vx","Vy","Vz","KE"]
name2ind = {}
for i, name in enumerate(names):
    name2ind[name] = i
    
amu2au = 1836.15    
   
#load data      	
dat = np.loadtxt('/reg/d/psdm/AMO/amox27716/results/xiangli/psana_momentum/sim_N1_gauss_5000.dat',skiprows=12,delimiter=',')

X = dat[1::2,name2ind['X']]-80
Z = dat[1::2,name2ind['Z']]-80
T = dat[1::2,name2ind['TOF']]*1e3

Pbins = np.linspace(-300,300,50)
xP = (Pbins[1:]+Pbins[:-1])/2
Ebins = np.linspace(0,50,50)
xE = (Ebins[1:]+Ebins[:-1])/2

Pxa,_ = np.histogram(dat[0:-1:2,name2ind['Vx']]*amu2au*14/2187.7,bins=Pbins)
Pya,_ = np.histogram(dat[0:-1:2,name2ind['Vy']]*amu2au*14/2187.7,bins=Pbins)
Pza,_ = np.histogram(dat[0:-1:2,name2ind['Vz']]*amu2au*14/2187.7,bins=Pbins)
Ea,xx = np.histogram(dat[0:-1:2,name2ind['KE']],bins=50)

#spectrometer dimensions and voltage settings
ta0_ns = 2554.3
l_mm1 = 80.5
l_mm2 = 415.5
l_mm3 = 3.69
U_V = (-1090+23)/2+4313
Es_VPmm1 = U_V/l_mm1
Es_Vpmm2 = 0
Es_VPmm3 = (-4313+2300)/l_mm3

#N+
q_au = 1; m_amu = 14;

Momen = IonMomentumFlat(t0_ns=0,x0_mm=0,y0_mm=0,
                        l_mm=l_mm1,d_mm=l_mm2,ls_mm=[l_mm1,l_mm2,l_mm3],
                        U_V=U_V,Es_VPmm=[Es_VPmm1,0,Es_VPmm3])
  
#calculate momentum                        
Px = Momen.CalcPx(m_amu,X,T)
Pz = Momen.CalcPx(m_amu,Z,T)
Py = Momen.CalcPzMultiAcc(m_amu,q_au,T)
Py_apx = Momen.CalcPzOneAccApprox(T,ta0_ns,q_au=1)
Px_hist,_ = np.histogram(Px,bins=Pbins)
Pz_hist,_ = np.histogram(Pz,bins=Pbins)
Py_hist,_ = np.histogram(Py,bins=Pbins)
Py_apx_hist,_ = np.histogram(Py_apx,bins=Pbins)

#calculate energy                        
E = CalcEnergy(14,Px,Pz,Py)
E_hist,_ = np.histogram(E,bins=Ebins)
Ea,_ = np.histogram(dat[0:-1:2,name2ind['KE']],bins=Ebins)

plt.figure(figsize=(12,10))
plt.subplot(231)
plt.plot(xP,Pxa,'b',label='True Px')

plt.plot(xP,Px_hist,'r',label='Calculated Px')
plt.xlabel('$P_x$ (a.u.)')
plt.ylabel('Yield (arb. units)')
plt.legend(loc='best',prop={'size':8})

plt.subplot(232)
plt.plot(xP,Pza,'b',label='True Py')

plt.plot(xP,Pz_hist,'r',label='Calculated Py')
plt.xlabel('$P_y$ (a.u.)')
plt.legend(loc='best',prop={'size':8})

plt.subplot(233)
plt.plot(xP,Pya,'b',label='True Pz')
plt.plot(xP,Py_apx_hist,'g',label='Calculated Pz_apx')
plt.plot(xP,Py_hist,'r',label='Calculated Pz')
plt.xlabel('$P_z$ (a.u.)')
plt.legend(loc=1,prop={'size':6.5})

plt.subplot(212)
plt.plot(xE,Ea,'b',label='True Energy')
plt.plot(xE,E_hist,'r',label='Calculated Energy')
plt.xlabel('Energy (eV)')
plt.ylabel('Yield (arb. units)')
plt.legend(loc='best',prop={'size':12})        

fnm = 'ionMomentumFlatPlots.pdf'
plt.savefig(fnm,bbox_inches='tight')  
print('Results saved to '+fnm)           
