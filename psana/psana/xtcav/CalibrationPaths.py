from psana import *
#from PSCalib.CalibFileFinder import CalibFile, CalibFileFinder
from psana.pscalib.calib.CalibFileFinder import CalibFile, CalibFileFinder
import os
import psana.xtcav.Constants as cons

class CalibrationPaths:
    def __init__(self,env,calibdir=''):
        self.env = env
        self.calibgroup = cons.CALIB_GROUP
        self.src  = cons.DETNAME
        self.cdir = calibdir if calibdir else self.env.calibDir()
        

    def findCalFileName(self,type,rnum,method='default'):
        """
        Returns calibration file name, given run number and type
        """ 
        if method == 'latest':
            return self.findCalibFile(self.src, type, rnum)
        cff = CalibFileFinder(self.cdir, self.calibgroup, pbits=0)
        fname = cff.findCalibFile(self.src, type, rnum)
        return fname             
        


    def newCalFileName(self, type, runBegin, runEnd='end'):
        """
        Returns calibration file name, given run number and type
        (either 'pedestals' or 'nolasing' for XTCAV.)"
        """
        
        path=os.path.join(self.cdir)
        if not os.path.exists(path): 
            os.mkdir(path)
        path=os.path.join(self.cdir,self.calibgroup)
        if not os.path.exists(path): 
            os.mkdir(path)
        path=os.path.join(self.cdir,self.calibgroup,self.src)
        if not os.path.exists(path): 
            os.mkdir(path)
        path=os.path.join(self.cdir,self.calibgroup,self.src,type)
        if not os.path.exists(path): 
            os.mkdir(path)
        return path+'/'+str(runBegin)+'-'+str(runEnd)+'.data'


    def findCalibFile(self, src, type, rnum0) :
        """Find calibration file.
        """
        rnum = rnum0 if rnum0 <= 9999 else 9999

        # there have been problems with calib-dir mounts on the mon nodes.
        # raise an exception here to try to detect this problem
        assert os.path.isdir(self.cdir), 'psana calib-dir must exist: '+self.cdir

        if not self.calibgroup: 
            return ''

        dir_name = os.path.join(self.cdir, self.calibgroup, src, type)
        if not os.path.exists(dir_name) :
            return ''

        fnames = os.listdir(dir_name)
        files = [os.path.join(dir_name,fname) for fname in fnames]
        return self.selectCalibFile(files, rnum) 


    def selectCalibFile(self, files, rnum) :
        """Selects calibration file from a list of file names
        """
        list_cf = []
        for path in files : 
            fname = os.path.basename(path)

            if fname is 'HISTORY' : continue
            if os.path.splitext(fname)[1] != '.data' : continue

            cf = CalibFile(path)
            if cf.valid :
                modification_time = os.path.getmtime(path)
                list_cf.append((modification_time, cf))
           
        list_cf_ord = [x for _,x in sorted(list_cf)]
        
        # search for the calibration file
        for cf in list_cf_ord[::-1] :
            if cf.get_begin() <= rnum and rnum <= cf.get_end() :
                return cf.get_path()

        # if no matching found
        return ''

