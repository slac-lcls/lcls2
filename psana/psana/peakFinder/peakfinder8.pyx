# This file is part of OM.
#
# OM is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# OM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with OnDA.
# If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2020 SLAC National Accelerator Laboratory
#
# Based on OnDA - Copyright 2014-2019 Deutsches Elektronen-Synchrotron DESY,
# a research centre of the Helmholtz Association.
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.stdint cimport int8_t

import numpy

cdef extern from "include/peakfinder8.hh":

    ctypedef struct tPeakList:
        long	    nPeaks
        long	    nHot
        float		peakResolution
        float		peakResolutionA
        float		peakDensity
        float		peakNpix
        float		peakTotal
        int			memoryAllocated
        long		nPeaks_max

        float       *peak_maxintensity
        float       *peak_totalintensity
        float       *peak_sigma
        float       *peak_snr
        float       *peak_npix
        float       *peak_com_x
        float       *peak_com_y
        long        *peak_com_index
        float       *peak_com_x_assembled
        float       *peak_com_y_assembled
        float       *peak_com_r_assembled
        float       *peak_com_q
        float       *peak_com_res

    void allocatePeakList(tPeakList* peak_list, long max_num_peaks)
    void freePeakList(tPeakList peak_list)

cdef extern from "include/peakfinder8.hh":

   #
    int peakfinder8(tPeakList *peaklist, float *data, char *mask, float *pix_r,
                    long asic_nx, long asic_ny, long nasics_x, long nasics_y,
                    float ADCthresh, float hitfinderMinSNR,
                    long hitfinderMinPixCount, long hitfinderMaxPixCount,
                    long hitfinderLocalBGRadius, char *outliersMask)


def peakfinder_8(int max_num_peaks, float[:,::1] data, char[:,::1] mask,
                 float[:,::1] pix_r, long asic_nx, long asic_ny, long nasics_x,
                 long nasics_y, float adc_thresh, float hitfinder_min_snr,
                 long hitfinder_min_pix_count, long hitfinder_max_pix_count,
                 long hitfinder_local_bg_radius):
    # The peakfinder8 algorithm isdescribed in the following publication:
    #
    #     A. Barty, R. A. Kirian, F. R. N. C. Maia, M. Hantke, C. H. Yoon, T. A. White,
    #     and H. N. Chapman, "Cheetah: software for high-throughput reduction and
    #     analysis of serial femtosecond X-ray diffraction data", J Appl  Crystallogr,
    #     vol. 47, pp. 1118-1131 (2014).
    # 
    # Explanation of the peakfinder_8 arguments.
    #
    # max_num_peaks: the maximum number of peaks that will be detected. Additional peaks
    #            will be discarded.
    # 
    # data: an array storing the data on which to perform the peak search. Must be a 2D
    #            array storing panels as described by the asic_nx, asic_ny, nasics_x
    #            and nasics_y arguments.
    # 
    # mask: a mask used to mark areas of the 'data' array to  be excluded from the peak search.
    #            Each pixel in the mask must have a value of either 0, meaning that the
    #            corresponding pixel in the data frame must be ignored, or 1, meaning that the
    #            corresponding pixel must be included in the search. The map is only used to
    #            exclude areas from the peak search: the data is not modified in any way.
    #
    # pix_r: a radius map, an array of the same size as 'data', storing for each pixel the
    #            distance of the pixel from the center of the detector.
    #
    # asic_nx: The size in pixels of the x axis of each detector's ASIC.
    #
    # asic_ny: The size in pixels of the y axis of each detector's ASIC.
    #
    # nasics_x: The number of ASICs along the x axis of the 'data' array.
    # 
    # nasics_y: The number of ASICs along the y axis of the 'data' array.
    #
    # adc_thresh: the minimum ADC threshold for a pixel to be considered for peak detection.
    #
    # hitfinderi_min_snr: the minimum signal-to-noise ratio for the peak above the local
    #            background.
    # 
    # hitfinder_min_pix_count: the minimum number of pixels required for a peak.
    #
    # hitfinder_max_pix_count: the maximum number of pixels allowed for a peak.
    #
    # hitfinder_local_bg_radius: the radius (in pixels) for the estimation of the local
    #           background.
    cdef tPeakList peak_list
    allocatePeakList(&peak_list, max_num_peaks)

    peakfinder8(&peak_list, &data[0, 0], &mask[0,0], &pix_r[0, 0], asic_nx, asic_ny,
                nasics_x, nasics_y, adc_thresh, hitfinder_min_snr,
                hitfinder_min_pix_count, hitfinder_max_pix_count,
                hitfinder_local_bg_radius, NULL)

    cdef int i
    cdef float peak_x, peak_y, peak_value
    cdef vector[double] peak_list_x
    cdef vector[double] peak_list_y
    cdef vector[long] peak_list_index
    cdef vector[double] peak_list_value
    cdef vector[double] peak_list_npix
    cdef vector[double] peak_list_maxi
    cdef vector[double] peak_list_sigma
    cdef vector[double] peak_list_snr

    num_peaks = peak_list.nPeaks

    if num_peaks > max_num_peaks:
        num_peaks = max_num_peaks

    for i in range(0, num_peaks):

        peak_x = peak_list.peak_com_x[i]
        peak_y = peak_list.peak_com_y[i]
        peak_index = peak_list.peak_com_index[i]
        peak_value = peak_list.peak_totalintensity[i]
        peak_npix = peak_list.peak_npix[i]
        peak_maxi = peak_list.peak_maxintensity[i]
        peak_sigma = peak_list.peak_sigma[i]
        peak_snr = peak_list.peak_snr[i]

        peak_list_x.push_back(peak_x)
        peak_list_y.push_back(peak_y)
        peak_list_index.push_back(peak_index)
        peak_list_value.push_back(peak_value)
        peak_list_npix.push_back(peak_npix)
        peak_list_maxi.push_back(peak_maxi)
        peak_list_sigma.push_back(peak_sigma)
        peak_list_snr.push_back(peak_snr)

    freePeakList(peak_list)


    # Description of return value:
    #
    # The function returns a tuple storing:
    #
    # peak_list_x: a list with the x coordinates in the 'data' array of all the
    #            detected peaks
    #
    # peak_list_y: a list with the y coordinates in the 'data' array of all the
    #            detected peaks
    #
    # peak_list_value: a list storing the integrated intensity of all the
    #            detected peaks
    #
    # peak_list_index: a list storing an index indetifier for each detected peak
    #            (an integer number)
    # 
    # peak_list_npix: a list storing the number of pixels that make up every
    #            detected peak
    # 
    # peak_list_maxi: a list storing the maximum instensity of all detected
    #            peaks (the value of the highest pixel)
    #
    # peak_list_sigma: a list storing the standard deviation over the local
    #            background of each detected peak
    #
    # peak_list_snr: a list storing the signal-to-noise ratio of each peak
    #            against the local background
    return (peak_list_x, peak_list_y, peak_list_value, peak_list_index,
            peak_list_npix, peak_list_maxi, peak_list_sigma, peak_list_snr)
