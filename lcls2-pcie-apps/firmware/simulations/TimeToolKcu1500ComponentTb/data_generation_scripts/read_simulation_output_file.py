import numpy as np
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
import IPython
import os

plt.ion()

top_path                 = os.getcwd().split("lcls2-pcie-apps")[0]+"/lcls2-pcie-apps"
testing_package_path     = top_path+"/firmware/applications/TimeTool/tb/"
test_file_path           = [i for i in open(testing_package_path+"TestingPkg.vhd").read().split("\n") if "constant TEST_FILE_PATH" in i][0].split("\"")[1]

my_output_file = test_file_path+"/output_results.dat"
my_input_file  = test_file_path+"/sim_input_data.dat"


def read_sim_file(my_input_file,ncols,signed):

      unsigned          = not signed
      my_input_data     = []
      loaded_input_data = open(my_input_file)
      for i in loaded_input_data:
            #print(i)
            try:
                  if(unsigned):
                        my_input_data.append([int(i[j:j+8],2) for j in range(0,128,8)]) #unsigned reading
                  else:
                        my_input_data.append([int(i[j:j+8],2) for j in range(0,128,8)]) 
                        #my_input_data.append([((-1)**int(i[j]))*int(i[j+1:j+8],2) for j in range(0,128,8)]) #signed reading
                        #my_input_data.append([((-1)**int(i[j]))*int(i[j+1:j+8],2) for j in range(0,128,8)]) #signed reading
            except:
                  my_input_data.append([0 for i in range(0,128,8)])
                  

      loaded_input_data.close()

      my_input_data = np.array(my_input_data).flatten()

      return my_input_data

my_input_data = np.array(read_sim_file(my_input_file,2,signed=False))
my_output_data = np.array(read_sim_file(my_output_file,2,signed=False)).astype(dtype=np.int8)

plt.figure(0)
plt.plot(my_input_data)
plt.twinx()
plt.plot(my_output_data,'r-')
plt.xlim(9000,18000)
thismanager = plt.get_current_fig_manager()
thismanager.window.move(0,250)

plt.figure(1)
plt.plot(my_output_data,'r-')
plt.twinx()
plt.plot(np.convolve([-1,-1,-1,-1,1,1,1,1],my_input_data))
plt.xlim(9000,18000)
thismanager = plt.get_current_fig_manager()
thismanager.window.move(0,500)


plt.figure(2)
plt.plot(np.convolve([-1,-1,-1,-1,1,1,1,1],my_input_data))
plt.xlim(9000,18000)


plt.show()
