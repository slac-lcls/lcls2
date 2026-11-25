# Importing this module makes the high-rate-encoder-dev python tree importable process-wide.
# A process that uses more than one firmware package should use
# psdaq.utils.firmware_packages.isolated() instead.
from psdaq.utils.firmware_packages import enable

enable('high-rate-encoder-dev')
