"""Define `SingleFileDataSource`, the data source that reads the xtc2 files given with `files`."""
from psana.dgrammanager import DgramManager
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.run import RunSingleFile
from pathlib import Path
from psana import utils


class SingleFileDataSource(DataSourceBase):
    """Data source that reads the xtc2 files listed in `files` directly, one after another.

    Smalldata (smd) files are not used. The constructor passes the keyword arguments to
    `DataSourceBase.__init__`, opens the first file with a `DgramManager` (FileNotFoundError if it
    does not exist), and starts the Prometheus client.
    """
    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.runnum_list = list(range(len(self.files)))
        self.dsparms.update_smd_state([None], [False] * len(self.runnum_list))  # disable SMDs unsupported in single file mode
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
        """Yield a `RunSingleFile` for each BeginRun found, reading the files in order.

        When the current file has no more BeginRun dgrams the next file is opened and searched;
        iteration stops when no files remain or a newly opened file has no BeginRun. Each run gets the
        experiment, run number and timestamp from the BeginRun dgram, plus `dsparms`, the
        `DgramManager`, the configs and the BeginRun dgrams.
        """
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
        """Return False."""
        return False
