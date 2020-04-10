import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from psana.pop.POP import POP
import abel
import pickle


fnm = '/reg/d/psdm/AMO/amox27716/results/xiangli/pop_files/img.pkl'
with open(fnm, 'rb') as f:
    img = pickle.load(f)  
    
pop = POP(img=img,RBFs_fnm='/reg/d/psdm/AMO/amox27716/results/xiangli/pop_files/RBFs512.pkl')    

methods = ['basex','hansenlaw','direct','two_point','three_point','onion_peeling','onion_bordas','psana-pop']
imgs = {}
ts = {}
for i, method in enumerate(methods):
    print(method,'#############################')
    if method =='psana-pop':
        t0 = time.time()
        pop.Peel(img)
        imgs[method] = pop.GetSlice()
        t1 = time.time()
        ts[method] = t1-t0
        continue
     
    t0 = time.time()
    imgs[method] = abel.Transform(img, direction='inverse', method=method).transform
    imgs[method][imgs[method]<0]=0
    imgs[method] = imgs[method]/imgs[method].max()    
    t1 = time.time()
    ts[method] = t1-t0
    
fig = plt.figure(figsize=(12,20))
for i, method in enumerate(methods):
    plt.subplot(4,2,i+1)
    im = plt.imshow(imgs[method])
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if method !='psana-pop':
        ax.set_title('PyAbel-'+method+':'+str(round(ts[method],2))+' s')
    else:
        ax.set_title(method+':'+str(round(ts[method],2))+' s')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')    
    
fig.savefig("comparison_with_PyAbel.pdf", bbox_inches='tight')
print('Plots saved to "comparison_with_PyAbel.pdf".')    
