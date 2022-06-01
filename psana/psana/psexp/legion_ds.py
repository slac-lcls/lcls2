from psana.psexp import RunLegion, DataSourceBase, SmdReaderManager, TransitionId, LSmd0, LEventBuilderNode, mode
from psana.dgrammanager import DgramManager
from psana.event import Event
import numpy as np

import logging
logger = logging.getLogger(__name__)
import os
if mode == 'legion':
    import pygion
    from psana.psexp import legion_node

class LegionDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(LegionDataSource, self).__init__(**kwargs)
        super()._setup_runnum_list()
        self.runnum_list_index = 0
        self.smd_fds = None
        # No. of event builder processors
        self.eb_size = int(os.environ.get('PS_EB_NODES', 1))
        # of big data processors = global_procs-(eb_procs+smd0) || 1
        self.bd_size =  pygion.Tunable.select(
            pygion.Tunable.GLOBAL_PYS).get() - (self.eb_size + 1)
        
        if self.bd_size <= 0:
            self.bd_size=1

        self._setup_run()
        super()._start_prometheus_client()

    def __del__(self):
        super()._close_opened_smd_files()
        super()._end_prometheus_client()

    def _setup_configs(self):
        super()._close_opened_smd_files()
        self.smd_fds  = np.array([os.open(smd_file, os.O_RDONLY) for smd_file in self.smd_files], dtype=np.int32)
        logger.debug(f'legion_ds: opened smd_fds: {self.smd_fds}')
        self.smdr_man = SmdReaderManager(self.smd_fds, self.dsparms)
        self._configs = self.smdr_man.get_next_dgrams()
        super()._setup_det_class_table()
        super()._set_configinfo()

        self.dm = DgramManager(self.xtc_files, configs=self._configs,
                found_xtc2_callback=super().found_xtc2_callback)

        # Legion Smd0
        self.smd0 = LSmd0(self.eb_size, self._configs, self.smdr_man, self.dsparms)
        # Legion Event Builder Node
        # Big Data Point Start Offset = # of eb processors + smd0
        offset = self.eb_size+1
        self.eb = LEventBuilderNode(self.bd_size, offset, self._configs, self.dsparms, self.dm)

        # Legion Reductions
        self.reduc = False

    def _setup_run(self):
        if self.runnum_list_index == len(self.runnum_list):
            return False
        runnum = self.runnum_list[self.runnum_list_index]
        self.runnum_list_index += 1
        super()._setup_run_files(runnum)
        super()._apply_detector_selection()
        self._setup_configs()

        return True
    
    def _setup_beginruns(self):
        """ Determines if there is a next run as
        1) New run found in the same smalldata files
        2) New run found in the new smalldata files
        """
        dgrams = self.smdr_man.get_next_dgrams() 
        while dgrams is not None:
            if dgrams[0].service() == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
            dgrams = self.smdr_man.get_next_dgrams() 
        return False

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():   # try to get next run from current files
            super()._setup_run_calibconst()
            found_next_run = True
        elif self._setup_run():       # try to get next run from next files 
            if self._setup_beginruns():
                super()._setup_run_calibconst()
                found_next_run = True
        return found_next_run

    def runs(self):
        while self._start_run():
            run = RunLegion(self, Event(dgrams=self.beginruns))
            yield run
    
    def is_mpi(self):
        return False

    def analyze(self, run_fn = None, event_fn=None):
        for run in self.runs():
            if run_fn is not None:
                run_fn(run)
            reduc_fn=None
            reduc_region=None
            reduc_pr=None
            reduc_final_fn=None
            reduc_shape = None
            reduc_fill_val = None
            reduc_rtype = None
            if self.reduc:
                run.reduc = True
                run.reduc_fn=self.reduc_fn
                run.reduc_privileges = self.reduc_pr
                run.reduc_final_fn = self.reduc_final_fn
                run.reduc_shape = self.reduc_shape
                run.reduc_fill_val = self.reduc_fill_val
                run.reduc_rtype = self.reduc_rtype
            legion_node.analyze(run, event_fn, run.det)
            import pygion
            pygion.execution_fence(block=True)

    # register a reduction operation
    def register_reduc(self, reduc_type, shape, rtype, fill_val, reduc_fn=None,
                       reduc_final=None):
        operators = ['+', '-', '*', '/', 'max', 'min']
        assert reduc_type in operators
        assert rtype != None
        assert fill_val != None
        assert shape != None
        assert self.reduc==False
        logger.debug(f'reduc_type={reduc_type}, shape={shape}, {rtype}, {fill_val}')
        self.reduc=True
        self.reduc_shape = shape
        self.reduc_fill_val = fill_val
        self.reduc_fn = reduc_fn
        self.reduc_final_fn = reduc_final
        self.reduc_pr = reduc_type
        self.reduc_rtype = rtype

