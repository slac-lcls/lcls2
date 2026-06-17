from psana.dgrammanager import DgramManager
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.run import RunSingleFile
from pathlib import Path
from psana import utils


class SingleFileDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.runnum_list = list(range(len(self.files)))

        # GPU mode: when gpu_det= is set and the provided files are SMD files,
        # populate dsparms.smd_files so that Run._gpu_events() can find them.
        # The corresponding bigdata XTC2 files are derived automatically.
        gpu_smd_files = None
        if (self.dsparms.gpu_det and self.files
                and all(isinstance(f, str) and f.endswith('.smd.xtc2')
                        for f in self.files)):
            gpu_smd_files = list(self.files)
            # Derive bigdata XTC2 paths from SMD paths.
            import os
            xtc_files = [
                os.path.join(
                    os.path.dirname(os.path.dirname(f)),
                    os.path.basename(f).split('.smd')[0] + '.xtc2'
                )
                for f in gpu_smd_files
            ]
            # Replace self.files with bigdata files so DgramManager reads them.
            self.files = [f for f in xtc_files if os.path.exists(f)]

        if gpu_smd_files:
            self.dsparms.update_smd_state(gpu_smd_files,
                                          [False] * len(gpu_smd_files))
        else:
            self.dsparms.update_smd_state([None],
                                          [False] * len(self.runnum_list))

        self.runnum_list_index = 0
        self._setup_run()
        super()._start_prometheus_client()

    def __del__(self):
        super()._end_prometheus_client()

    def _setup_run(self):
        """
        Prepare the data manager and internal path state for the next run.

        This method:
        - Checks if there are remaining files to process.
        - Verifies that the file exists.
        - Extracts the directory path of the current file and stores it in `self.xtc_path`.
        - Initializes a new DgramManager for the current file.
        - Advances the file index.

        Returns:
            bool: True if setup was successful, False if there are no more runs to process.
        """
        if self.runnum_list_index == len(self.runnum_list):
            self.logger.debug("No more files to process.")
            return False

        file = self.files[self.runnum_list_index]
        full_path = Path(file)

        if not full_path.exists():
            self.logger.error(f"File not found: {file}")
            raise FileNotFoundError(f"Cannot set up run; file does not exist: {file}")

        try:
            # Resolve full path and set xtc_path
            self.xtc_path = full_path.parent.resolve()
            self.logger.debug(f"Resolved xtc_path: {self.xtc_path}")

            # Initialize the data manager
            self.dm = DgramManager(file, config_consumers=[self.dsparms])
            self.logger.debug(f"Initialized DgramManager for: {file}")

        except Exception as e:
            self.logger.exception(f"Failed to set up run for file: {file}")
            raise e

        self.runnum_list_index += 1
        return True

    def _setup_beginruns(self):
        for dgrams in self.dm:
            if utils.first_service(dgrams) == TransitionId.BeginRun:
                self.beginruns = dgrams
                return True
        return False

    def _start_run(self):
        found_next_run = False
        if self._setup_beginruns():  # try to get next run from the current file
            found_next_run = True
        elif self._setup_run():  # try to get next run from next files
            if self._setup_beginruns():
                found_next_run = True
        return found_next_run

    def runs(self):
        if self.dsparms.gpu_det:
            # GPU mode: all files are one combined multi-stream run.
            # Open the first file to get BeginRun configs; the GPU pipeline
            # (Run._gpu_events) then re-opens all streams together via
            # gpu_events(dsparms.smd_files, ...).
            if self._start_run():
                expt, runnum, ts = self._get_runinfo()
                run = RunSingleFile(
                    expt, runnum, ts,
                    self.dsparms, self.dm, None,
                    self._configs, self.beginruns,
                )
                yield run
            return

        # Normal (CPU) path: one run per file.
        while self._start_run():
            # Pull (expt, runnum, ts) from the BeginRun dgrams
            expt, runnum, ts = self._get_runinfo()
            run = RunSingleFile(
                expt,                 # experiment string
                runnum,               # run number (int)
                ts,                   # begin-run timestamp
                self.dsparms,         # shared parameters / config tables
                self.dm,              # DgramManager
                None,                 # SmdReaderManager (may be None for non-SMD modes)
                self._configs,        # configs for this run
                self.beginruns,       # beginrun dgrams
            )
            yield run

    def is_mpi(self):
        return False
