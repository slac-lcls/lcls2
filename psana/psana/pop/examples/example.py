import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from psana.pop.POP import POP
import pickle


fnm = '/reg/d/psdm/AMO/amox27716/results/xiangli/pop_files/img.pkl'
with open(fnm, 'rb') as f:
    img = pickle.load(f)  
    
pop = POP(img=img,RBFs_fnm='/reg/d/psdm/AMO/amox27716/results/xiangli/pop_files/RBFs512.pkl')    
print('Peel...')
pop.Peel(img)
slice_img = pop.GetSlice()
DistR = pop.GetRadialDist()
print('Peel completed!')

fig = plt.figure(figsize=(12,9))
plt.subplot(2,2,1)
im = plt.imshow(img+1,extent=[0,1024,0,1024],norm=LogNorm(vmin=1, vmax=img.max()+1))
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_title('Raw VMI image')
ax.set_xlabel('X (pixel)')
ax.set_ylabel('Y (pixel)')

plt.subplot(2,2,2)
im = plt.imshow(slice_img)
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_title('3D slice (inverted image)')
ax.set_xlabel('X (pixel)')
ax.set_ylabel('Y (pixel)')

plt.subplot(2,1,2)
plt.plot(DistR,'r')
plt.xlim([0,520])
plt.ylim([0,2e5])
plt.title('Radial Distribution')
plt.xlabel('Radius (pixel)')
plt.ylabel('Yield (arb. units)')

fig.savefig("pop_example_plot.pdf", bbox_inches='tight')
print('Plots saved to "pop_example_plot.pdf".')
