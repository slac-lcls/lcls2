'''
New Filecreate file.
timestamp length will now match big data length
big data array will be filled with integers instead of zeros
'''
#logistical support
import h5py, random
import numpy as np

#Options imported here
#This is the filesize of the first h5 file. For small kB-size files set to 0
filesize = 8
number_of_files = 10

#initial variables
initialindex = 0
startonfilenumber = 2
tsFactor = 1
#evetn arrays are written in chunks of this number. DO NOT INSERT ODD NUMBERS HERE!
chunkSize = 4

#a filesize of 0 will create a small set of h5 files
if filesize == 0:
    datSize = 100
else:
    datSize = filesize*10000
    
#each event will be a megabyte in size
evtSize = 250000

#small data will be an array of random colors. "red" will be deemed as an interesting event.
smallDatColors = ['red', 'blue', 'green']

#introduces a sequential block to prevent large arrays of sequential timestamps
if datSize == 100:
    max_block_size = int(round(datSize*.05))
    bunch = int(round(datSize/5))
else:
    max_block_size = int(round(datSize*.005))
    bunch = int(round(datSize/100))

#imageing device prevents memory issues when writing large arrays
image = np.array([range(evtSize) for i in range(chunkSize)])

#creating the first h5 file
with h5py.File('file1.h5', 'w') as f:
    #creating small data
    smallDat = f.create_dataset('smalldata', (datSize,), dtype='S5')
    for event in range(datSize):
        smallDat[event] = (random.choice(smallDatColors))
    
    #creating first timestamp array
    firstStamp = f.create_dataset('timestamp1', (datSize,), dtype='i')
    for stamp in range(datSize):
        firstStamp[stamp] = initialindex+(stamp*tsFactor)
    
    #creating first event array
    maxshape = (None,) + image.shape[1:]
    print datSize, image.shape
    bigDat1 = f.create_dataset('bigdata1', shape=image.shape,
      maxshape=maxshape, chunks=image.shape, dtype=image.dtype)
    bigDat1[:] = image
    row_count = chunkSize
    for i in range(int(datSize/image.shape[0])-1):
        bigDat1.resize(row_count + image.shape[0], axis =0)
        bigDat1[row_count:] = image
        row_count += image.shape[0]
    
#creating the secondary h5 files
while startonfilenumber <= number_of_files:
    list = np.arange(datSize)[...]
    amendment = []
    while initialindex < len(list):
        sel_n = int(round(random.choice(range(initialindex, initialindex+bunch))))
        block_size = random.choice(range(max_block_size))
        if sel_n+block_size > len(list):
            block_size = len(list)-sel_n
        for i in range(sel_n, sel_n+block_size):
            amendment.append(list[i])
        initialindex = sel_n+block_size
        
    with h5py.File('file%s.h5' %startonfilenumber, 'w') as g:
        #creating timestamp arrays
        secondStamp = g.create_dataset('timestamp%s' %startonfilenumber, (len(amendment),), dtype='i')
        length = len(amendment)
        secondStamp[:length]=amendment[:]
        import math
        nChunks = int(math.floor(len(secondStamp)/chunkSize))
        nOffsets = len(secondStamp)%chunkSize
        
        #creating event data arrays
        maxshape = (None,) + image.shape[1:]
        bigDat = g.create_dataset('bigdata%s' %startonfilenumber, shape=image.shape,
          maxshape=maxshape, chunks=image.shape, dtype=image.dtype)
        bigDat[:] = image
        #create chunks
        row_count = chunkSize
        for i in range(nChunks-1):
          bigDat.resize(row_count + image.shape[0], axis =0)
          bigDat[row_count:] = image
          row_count += image.shape[0]
        #create offsets of the last chunk
        for i in range(nOffsets):
          bigDat.resize(row_count + 1, axis=0)
          bigDat[row_count] = np.arange(0, evtSize)
          row_count += 1
    startonfilenumber += 1
    initialindex = 0
