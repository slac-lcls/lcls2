import numpy as np
import matplotlib.pyplot as plt
import os

top_path                 = os.getcwd().split("lcls2-pcie-apps")[0]+"/lcls2-pcie-apps"
testing_package_path     = top_path+"/firmware/applications/TimeTool/tb/"
test_file_path           = [i for i in open(testing_package_path+"TestingPkg.vhd").read().split("\n") if "constant TEST_FILE_PATH" in i][0].split("\"")[1]



my_file = test_file_path+"/sim_input_data.dat"

n_frames                = 54
dropped_shot_rate       = 6
starting_drop           = 5
bits_per_pixel          = 8
pixels_per_transfer     = 16
pixels_per_frame        = 2048      #frame and packet are being used interchangeably

sigma                   = 800.0
jitter                  = 100.0
amplitude               = 128

my_file = open(my_file,'w')

to_file = []

#each line break will imply a tvalid. no special marker

def gaussian(x,u,s):
      return np.e**(-(x-u)**2/(2*s**2))

for i in range(n_frames):

      x = np.arange(pixels_per_frame)
      my_frame_array = amplitude*gaussian(x,pixels_per_frame/2.0,sigma)

      if(i%dropped_shot_rate!=starting_drop):
            edge_position = int(pixels_per_frame/2+(jitter*np.random.rand()-0.5))
            print("edge_position = {}".format(edge_position))
            my_frame_array[edge_position:] = my_frame_array[edge_position:] *0.2

      my_frame_array = np.convolve(np.ones(8)/8,my_frame_array,mode='same').astype(np.int)      
      
      my_frame_array[0]  = 67        #test values to makes sure no pixels are being lost
      my_frame_array[-1] = 73      #test values to makes sure no pixels are being lost

      if(i%400==1):
            my_frame_array = my_frame_array * 0

      my_frame_list = []
      for j in range(0,pixels_per_frame,pixels_per_transfer):
            my_transfer_string = ""
            for k in range(pixels_per_transfer): my_transfer_string += '{0:08b}'.format(int(my_frame_array[j+k]))
            #print(j)
            if(j<(pixels_per_frame-pixels_per_transfer-1)):
                  my_transfer_string += " 0\n"            
                  my_file.writelines(my_transfer_string)
            else:
                  print(j)
                  my_transfer_string += " 1\n"            
                  my_file.writelines(my_transfer_string)


            #my_frame_list.append(my_transfer_string)            
      
            

      
