from typing import Dict, List, Tuple, Union

import numpy  # type: ignore
from psana import DataSource, setOption
from psana.peakfinder8 import peakfinder_8

class Peakfinder8PeakDetection:
    """
    See documentation of the '__init__' function.
    """

    def __init__(
        self,
        max_num_peaks: int,
        asic_nx: int,
        asic_ny: int,
        nasics_x: int,
        nasics_y: int,
        adc_threshold: float,
        minimum_snr: float,
        min_pixel_count: int,
        max_pixel_count: int,
        local_bg_radius: int,
        min_res: int,
        max_res: int,
        bad_pixel_map: Union[numpy.ndarray, None],
        radius_pixel_map: numpy.ndarray,
    ) -> None:
        self._max_num_peaks: int = max_num_peaks
        self._asic_nx: int = asic_nx
        self._asic_ny: int = asic_ny
        self._nasics_x: int = nasics_x
        self._nasics_y: int = nasics_y
        self._adc_thresh: float = adc_threshold
        self._minimum_snr: float = minimum_snr
        self._min_pixel_count: int = min_pixel_count
        self._max_pixel_count: int = max_pixel_count
        self._local_bg_radius: int = local_bg_radius
        self._radius_pixel_map: numpy.ndarray = radius_pixel_map
        self._min_res: int = min_res
        self._max_res: int = max_res
        self._mask: numpy.ndarray = bad_pixel_map
        self._mask_initialized: bool = False

    def find_peaks(self, data: numpy.ndarray) -> Dict:
        if not self._mask_initialized:
            if self._mask is None:
                self._mask = numpy.ones_like(data, dtype=numpy.int8)
            else:
                self._mask = self._mask.astype(numpy.int8)

            res_mask: numpy.ndarray = numpy.ones(
                shape=self._mask.shape, dtype=numpy.int8
            )
            res_mask[numpy.where(self._radius_pixel_map < self._min_res)] = 0
            res_mask[numpy.where(self._radius_pixel_map > self._max_res)] = 0
            self._mask *= res_mask

        peak_list: Tuple[List[float], ...] = peakfinder_8(
            self._max_num_peaks,
            data.astype(numpy.float32),
            self._mask,
            self._radius_pixel_map,
            self._asic_nx,
            self._asic_ny,
            self._nasics_x,
            self._nasics_y,
            self._adc_thresh,
            self._minimum_snr,
            self._min_pixel_count,
            self._max_pixel_count,
            self._local_bg_radius,
        )

        return {
            "num_peaks": len(peak_list[0]),
            "fs": peak_list[0],
            "ss": peak_list[1],
            "intensity": peak_list[2],
            "num_pixels": peak_list[3],
            "max_pixel_intensity": peak_list[4],
            "snr": peak_list[5],
        }


# To be used in this way: 
#
# Instantiate the Peakfinder8PeakDetection class:

pixel_map_r = numpy.ones((5632, 384), dtype=float)

pf8 = Peakfinder8PeakDetection(
    max_num_peaks=2048,
    asic_nx=384,
    asic_ny=352,
    nasics_x=1,
    nasics_y=16,
    adc_threshold=500.0,
    minimum_snr=7.0,
    min_pixel_count=2,
    max_pixel_count=2,
    local_bg_radius=4,
    min_res=80,
    max_res=1300,
    bad_pixel_map=None,
    radius_pixel_map=pixel_map_r
)

# Call find_peaks member function on data and retrieve information:
#
# ... Iterating over data, do this for each frame ...
#

peaks = pf8.find_peaks(data)
print("Peaks found (x,y, intensity):")
for x, y, intensity in zip(peaks['fs'], peaks['ss'], peaks['intensity']):
   print(x,y, intensity) 








         


