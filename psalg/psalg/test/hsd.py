import numpy as np
import matplotlib.pyplot as plt
#from psana.dgrammanager import DgramManager
import psana
ds = psana.DataSource('/reg/neh/home/yoon82/Software/lcls2/hsd_052218b.xtc') # chan0a
#ds = DgramManager('/reg/neh/home/cpo/hsd_040518.xtc') # chan0b
for i, evt in enumerate(ds.events()):
   print("Event: ", i)
   if i == 3:
      print(evt.dgrams[0].xpphsd.hsd.chan0[:150])
      x0 = evt.dgrams[0].xpphsd.hsd.chan0
      x1 = evt.dgrams[0].xpphsd.hsd.chan1
      x2 = evt.dgrams[0].xpphsd.hsd.chan2
      x3 = evt.dgrams[0].xpphsd.hsd.chan3

      b = x2.tobytes()
      c = np.frombuffer(b, dtype=np.uint16) # convert to uint16
      plt.plot(c,'ro-')
      plt.show()

      newFile = open("chan0_e"+str(i)+".bin", "wb")
      newFile.write(x0.tobytes())
      newFile.close()
      newFile = open("chan1_e"+str(i)+".bin", "wb")
      newFile.write(x1.tobytes())
      newFile.close()
      newFile = open("chan2_e"+str(i)+".bin", "wb")
      newFile.write(x2.tobytes())
      newFile.close()
      newFile = open("chan3_e"+str(i)+".bin", "wb")
      newFile.write(x3.tobytes())
      newFile.close()

