import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def epixquad_roi(xroi,yroi,vinner,vouter):
    _height = 192
    _width  = 176
    image    = np.zeros(((4*_width),(4*_height)),dtype=np.uint8)+vouter

    image[xroi[0]:xroi[1],yroi[0]:yroi[1]] = vinner

    #  break the quad into elements
    a = []
    for i in np.vsplit(image,2):
        a.extend(np.hsplit(i,2))
    #e = np.asarray([np.asarray(a[i],dtype=np.uint8) for i in (1,3,2,0)],dtype=np.uint8)
    e = np.asarray([np.asarray(a[i],dtype=np.uint8) for i in (3,2,1,0)],dtype=np.uint8)

    #  break the elements into asics
    #pca = []
    #for i in e:
    #    a = []
    #    for j in np.vsplit(i,2):
    #        a.extend(np.hsplit(j,2))
    #    pca.extend([a[3],
    #                np.flipud(np.fliplr(a[0])),
    #                np.flipud(np.fliplr(a[1])),
    #                a[2]])

    return (e,image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update gain map')
    parser.add_argument('--x', help='low gain x-bounds', type=int, nargs=2, required=True)
    parser.add_argument('--y', help='low gain x-bounds', type=int, nargs=2, required=True)
    parser.add_argument('--o', help='output file', default='gainmap.txt')
    parser.add_argument('--test', help='test transformation', action='store_true')
    args = parser.parse_args()

    #  Write gainmap
    e,image = epixquad_roi(args.x,args.y,0,1)

    if not args.test:
        np.savetxt(args.o,e.reshape((e.shape[0]*e.shape[1],e.shape[2])),fmt='%u')

    #  Show image
    plt.pcolormesh(image.transpose())
    plt.show()

