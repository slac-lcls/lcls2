from psana.dgrammanager import DgramManager
from psana.event import Event
from psana.psexp import TransitionId
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.run import RunSingleFile
from pathlib import Path


class SingleFileDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.runnum_list = list(range(len(self.files)))
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
        for evt in self.dm:
            if evt.service() == TransitionId.BeginRun:
                self.beginruns = evt._dgrams
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
        while self._start_run():
            run = RunSingleFile(self, Event(dgrams=self.beginruns))
            yield run

    def is_mpi(self):
        return False
