# Importing this module makes the epixuhr-3x2-readout-testing python tree importable process-wide.
# A process that uses more than one firmware package should use
# psdaq.utils.firmware_packages.isolated() instead.
from psdaq.utils.firmware_packages import enable

enable('epixuhr-3x2-readout-testing')
