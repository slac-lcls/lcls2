import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from psana.pop.POP import POP
import pickle
from psana import DataSource

runinfo_data = {'expt': 'amox27716', 'runnum': 101}
fnm = '/reg/d/psdm/AMO/amox27716/results/xiangli/pop_files/pop_raw_imgs.xtc2'

ds = DataSource(files = fnm)
run = next(ds.runs())
camera = run.Detector('opal')
calib_dict, _ = camera.calibconst.get('pop_rbfs')

accum_num = 30
normalizeDist = True
pop = POP(lmax=4,reg=0,alpha=4e-4,img=None,X0=512,Y0=512,Rmax=512,RBFs_dict = calib_dict,RBFs_fnm=None,edge_w=10) 

img = np.zeros((1024,1024))
fig = plt.figure(figsize=(12,15))
plt.ion()
i = 0
for nevt,evt in enumerate(run.events()):
    if nevt>150:
        break
    if nevt < 2:
        continue
    img += evt._dgrams[0].opal[0].raw.img
    if (nevt%30==0):
        print('event num:',nevt,', starting peeling')
        pop.Peel(img)
        slice_img = pop.GetSlice()
        rbins,DistR = pop.GetRadialDist()
        Ebins,DistE = pop.GetEnergyDist()  
        print('peeling completed at event num',nevt)      
        fig.clf()

        plt.subplot(3,2,1)
        im = plt.imshow(img+1,extent=[0,1024,0,1024],norm=LogNorm(vmin=1, vmax=img.max()+1))
 
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('Raw VMI image')
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')

        plt.subplot(3,2,2)
        im = plt.imshow(slice_img,extent=[0,1024,0,1024])
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('3D slice (inverted image)')
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')

        plt.subplot(3,1,2)
        if normalizeDist:
            plt.plot((rbins[1:]+rbins[:-1])/2,DistR/DistR.max(),'r')
            plt.ylim([0,1])
        else:
            plt.plot((rbins[1:]+rbins[:-1])/2,DistR,'r')            
        plt.xlim([0,520])
        
        plt.title('Radial Distribution')
        plt.xlabel('Radius (pixel)')
        plt.ylabel('Yield (arb. units)')

        plt.subplot(3,1,3)
        if normalizeDist:        
            plt.plot((Ebins[1:]+Ebins[:-1])/2,DistE/DistR.max(),'r')
            plt.ylim([0,1])
        else:
            plt.plot((Ebins[1:]+Ebins[:-1])/2,DistE,'r')
        plt.title('Energy Distribution')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Yield (arb. units)')
        plt.show()
        plt.pause(0.5)
        
        img = np.zeros((1024,1024))        

fig.savefig("pop_example_xtc2_plot.pdf", bbox_inches='tight')
print('Last plots saved to "pop_example_plot.pdf".')
