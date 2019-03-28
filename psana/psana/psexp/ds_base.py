import weakref
import os, glob

class DataSourceBase(object):
    filter = 0
    batch_size = 1
    max_events = 0
    sel_det_ids = []
    exp = None
    run_dict = {}

    def __init__(self, **kwargs):
        """Initializes datasource base.
        
        Keyword arguments:
        filter      -- filtering callback that handles Event object.
        batch_size  -- length of batched offsets
        max_events  -- no. of maximum events
        sel_det_ids -- user-selected detector IDs.
        """
        if kwargs is not None:
            keywords = ('filter', 'batch_size', 'max_events', 'sel_det_ids', 'det_name')
            for k in keywords:
                if k in kwargs:
                    setattr(self, k, kwargs[k])
        assert self.batch_size > 0
    
    def events(self):
        for run in self.runs():
            for evt in run.events(): yield evt

    # FIXME: this supports cctbx code and will be removed in next update
    def Detector(self, det_name):
        return self.run.Detector(det_name)
    
    def parse_expstr(self, expstr):
        exp = None
        run_dict = {} # stores list of runs with corresponding xtc_files, smd_files, and epic file

        # Check if we are reading file(s) or an experiment
        read_exp = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                xtc_files = [expstr]
                smd_files = None
                epic_file = None
                run_dict[-1] = (xtc_files, smd_files, epic_file)
            else:
                read_exp = True
        elif isinstance(expstr, (list, np.ndarray)):
            xtc_files = expstr
            smd_files = None
            epic_file = None
            run_dict[-1] = (xtc_files, smd_files, epic_file)

        # Reads list of xtc files from experiment folder
        if read_exp:
            opts = expstr.split(':')
            exp_dict = {}
            for opt in opts:
                items = opt.split('=')
                assert len(items) == 2
                exp_dict[items[0]] = items[1]

            assert 'exp' in exp_dict
            exp = exp_dict['exp']

            if 'dir' in exp_dict:
                xtc_path = exp_dict['dir']
            else:
                xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                xtc_path = os.path.join(xtc_dir, exp_dict['exp'][:3], exp_dict['exp'], 'xtc')

            run_num = -1
            if 'run' in exp_dict:
                run_num = int(exp_dict['run'])

            # get a list of runs (or just one run if user specifies it) then
            # setup corresponding xtc_files and smd_files for each run in run_dict
            run_list = []
            if run_num > -1:
                run_list = [run_num]
            else:
                run_list = [int(os.path.splitext(os.path.basename(_dummy))[0].split('-r')[1].split('-')[0]) \
                        for _dummy in glob.glob(os.path.join(xtc_path, '*-r*.xtc2'))]
                run_list.sort()

            smd_dir = os.path.join(xtc_path, 'smalldata')
            for r in run_list:
                smd_files = glob.glob(os.path.join(smd_dir, '*r%s-s*.smd.xtc2'%(str(r).zfill(4))))
                xtc_files = [os.path.join(xtc_path, \
                             os.path.basename(smd_file).split('.smd')[0] + '.xtc2') \
                             for smd_file in smd_files \
                             if os.path.isfile(os.path.join(xtc_path, \
                             os.path.basename(smd_file).split('.smd')[0] + '.xtc2'))]
                all_files = glob.glob(os.path.join(xtc_path, '*r%s-*.xtc2'%(str(r).zfill(4))))
                other_files = [f for f in all_files if f not in xtc_files]
                run_dict[r] = (xtc_files, smd_files, other_files)

        return exp, run_dict

