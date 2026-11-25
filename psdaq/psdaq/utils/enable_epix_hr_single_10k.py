# Importing this module makes the epix-hr-single-10k python tree importable process-wide.
# A process that uses more than one firmware package should use
# psdaq.utils.firmware_packages.isolated() instead.
from psdaq.utils.firmware_packages import enable

enable('epix-hr-single-10k')
