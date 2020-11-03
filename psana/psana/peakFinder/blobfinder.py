import numpy as np
import scipy.ndimage.measurements as smt
import scipy.stats as stats


def find_blobs_1d(waveform, threshold, min_sum):
    blobs, nblobs = smt.label(waveform > threshold)
    if nblobs > 0:
        index = np.arange(1, nblobs+1)
        adu_sum = smt.sum(waveform, blobs, index)
        x = smt.center_of_mass(waveform, blobs, index)
        x = np.array(x, dtype=np.float32).reshape(-1)
        ind, = np.where(adu_sum > min_sum)
        nblobs = ind.size
        adu_sum = adu_sum[ind]
        x = x[ind]
    else:
        nblobs = 0
        x = np.zeros(0, dtype=np.float32)
        adu_sum = np.zeros(0, dtype=np.float32)
    return nblobs, x, adu_sum


def find_blobs_2d(img,threshold,min_sum):
    blobs, nblobs = smt.label(img > threshold)
    if nblobs > 0:
        index = np.arange(1, nblobs+1)
        adu_sum = smt.sum(img, blobs, index)
        x, y = zip(*smt.center_of_mass(img, blobs, index))
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        ind, = np.where(adu_sum > min_sum)
        nblobs = ind.size
        adu_sum = adu_sum[ind]
        x = x[ind]
        y = y[ind]
    else:
        nblobs = 0
        x = np.zeros(0, dtype=np.float32)
        y = np.zeros(0, dtype=np.float32)
        adu_sum = np.zeros(0, dtype=np.float32)
    return nblobs, x, y, adu_sum

if __name__ == "__main__" :
    img = np.array(([1,1,1,1],[1,3,3,1],[1,3,3,1],[1,1,1,1]))
    print(find_blobs_2d(img,2,1))

    img = np.array([1,3,3,1,1,3,3,1])
    print(find_blobs_1d(img,2,1))
