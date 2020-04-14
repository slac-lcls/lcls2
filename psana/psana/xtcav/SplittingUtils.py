
import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.xtcav.Utils import *

def splitImage(image, n, islandsplitmethod, par1, par2):
    """
    Split an XTCAV image depending of different bunches. This function needs to be expanded
    Arguments:
      image: 3d numpy array with the image where the first index always has one dimension (it will become the bunch index), the second index correspond to y, and the third index corresponds to x
      n: number of bunches expected to find
    Output:
      outimage: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    """

    if islandsplitmethod == 'contourLabel' or islandsplitmethod == 'autothreshold':    #For one bunch, just the same image
        logger.warning('Method ' + islandsplitmethod + ' not currently implemented in this version of xtcav')
        outimages=np.zeros((n,image.shape[0],image.shape[1]))
        outimages[0,:,:]=image    
    
    else:       #In any other case just copies of the image, for debugging purposes
        Nx = image.shape[1]
        Ny = image.shape[0]
        transform = np.uint8(image)
        n_groups, groups = cv2.connectedComponents(transform)

        if n_groups == 1:
            logger.warning('No region of interest found')
            return None 
        
        #Structure for the areas and the images
        areas = np.zeros(n_groups-1, dtype=np.float64)

        #Obtain the areas
        for i in range(1,n_groups):    
            areas[i-1] = np.sum(groups==i)

        #Get the indices in descending area order
        orderareaind = np.argsort(areas)[::-1] 
        biggestarea = areas[orderareaind[0]]

        n_area_valid = 1
        for i in range(1, n_groups-1): 
            if areas[orderareaind[i]] < 1.0/20*biggestarea:
                break
            n_area_valid+=1

        n_valid = min(n, n_area_valid)
        outimages=np.zeros((n_valid, Ny, Nx))        

        #Assign the proper images to the output
        for i in range(n_valid):
            outimages[i,:,:] = groups==(orderareaind[i]+1)
            
    return outimages
