# Importing this module makes the lcls2-epix-hr-pcie python tree importable process-wide.
# A process that uses more than one firmware package should use
# psdaq.utils.firmware_packages.isolated() instead.
from psdaq.utils.firmware_packages import enable

enable('lcls2-epix-hr-pcie')
