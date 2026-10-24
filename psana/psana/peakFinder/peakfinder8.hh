/**
 * @file
 * @brief Declares the peakfinder8 peak finder (from OM/Cheetah), its tPeakList result structure and the allocate/free helpers.
 */
// This file is part of OM.
//
// OM is free software: you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// OM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with OnDA.
// If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2020 SLAC National Accelerator Laboratory
//
// Based on OnDA - Copyright 2014-2019 Deutsches Elektronen-Synchrotron DESY,
// a research centre of the Helmholtz Association.
#ifndef PEAKFINDER8_H
#define PEAKFINDER8_H

typedef struct {
public:
	long	    nPeaks;  ///< Number of peaks stored by peakfinder8() (at most nPeaks_max).
	long	    nHot;  ///< Set to 0 by allocatePeakList(); not set by peakfinder8().
	/** Set to 0 by allocatePeakList(); not set by peakfinder8(). The trailing comment calls it the radius of 80% of peaks. */
	float		peakResolution;			// Radius of 80% of peaks
	/** Set to 0 by allocatePeakList(); not set by peakfinder8(). The trailing comment calls it the radius of 80% of peaks. */
	float		peakResolutionA;		// Radius of 80% of peaks
	/** Set to 0 by allocatePeakList(); not set by peakfinder8(). The trailing comment calls it the density of peaks within the 80% radius. */
	float		peakDensity;			// Density of peaks within this 80% figure
	/** Set to 0 by allocatePeakList(); not set by peakfinder8(). The trailing comment calls it the number of pixels in peaks. */
	float		peakNpix;				// Number of pixels in peaks
	/** Set to 0 by allocatePeakList(); not set by peakfinder8(). The trailing comment calls it the total integrated intensity in peaks. */
	float		peakTotal;				// Total integrated intensity in peaks
	int			memoryAllocated;  ///< Set to 1 by allocatePeakList(); freePeakList() clears it only in its by-value copy.
	long		nPeaks_max;  ///< Capacity of the per-peak arrays, set by allocatePeakList(); peakfinder8() stores at most this many peaks.

	/** Per-peak maximum intensity, filled by peakfinder8(). */
	float		*peak_maxintensity;		// Maximum intensity in peak
	/** Per-peak integrated intensity, filled by peakfinder8(). */
	float		*peak_totalintensity;	// Integrated intensity in peak
	/** Per-peak sigma value from the peak search, filled by peakfinder8() (the trailing comment says signal-to-noise ratio). */
	float		*peak_sigma;			// Signal-to-noise ratio of peak
	/** Per-peak snr value from the peak search (trailing comment: signal-to-noise ratio), filled by peakfinder8(). */
	float		*peak_snr;				// Signal-to-noise ratio of peak
	/** Per-peak number of pixels, filled by peakfinder8(). */
	float		*peak_npix;				// Number of pixels in peak
	/** Per-peak center-of-mass coordinate com_fs from the peak search (trailing comment: x in raw layout), filled by peakfinder8(). */
	float		*peak_com_x;			// peak center of mass x (in raw layout)
	/** Per-peak center-of-mass coordinate com_ss from the peak search (trailing comment: y in raw layout), filled by peakfinder8(). */
	float		*peak_com_y;			// peak center of mass y (in raw layout)
	/** Per-peak com_index from the peak search (trailing comment: closest pixel to the peak), filled by peakfinder8(). */
	long		*peak_com_index;		// closest pixel corresponding to peak
	/** Allocated by allocatePeakList() but not filled by peakfinder8() (trailing comment: center of mass x in assembled layout). */
	float		*peak_com_x_assembled;	// peak center of mass x (in assembled layout)
	/** Allocated by allocatePeakList() but not filled by peakfinder8() (trailing comment: center of mass y in assembled layout). */
	float		*peak_com_y_assembled;	// peak center of mass y (in assembled layout)
	/** Allocated by allocatePeakList() but not filled by peakfinder8() (trailing comment: center of mass r in assembled layout). */
	float		*peak_com_r_assembled;	// peak center of mass r (in assembled layout)
	/** Allocated by allocatePeakList() but not filled by peakfinder8() (trailing comment: scattering vector of the peak). */
	float		*peak_com_q;			// Scattering vector of this peak
	/** Allocated by allocatePeakList() but not filled by peakfinder8() (trailing comment: resolution of the peak). */
	float		*peak_com_res;			// REsolution of this peak
} tPeakList;  ///< Peak list for peakfinder8(): per-peak arrays of capacity nPeaks_max allocated by allocatePeakList(), plus summary fields. peakfinder8() fills nPeaks and the first eight per-peak arrays only.

/** Zero the counters and summary fields of *peak, set nPeaks_max to NpeaksMax, calloc the 13 per-peak arrays with NpeaksMax elements each, and set memoryAllocated to 1. Allocation failures are not checked. */
void allocatePeakList(tPeakList *peak, long NpeaksMax);
/** Free the 13 per-peak arrays of peak. peak is passed by value, so setting memoryAllocated to 0 does not change the caller's struct. */
void freePeakList(tPeakList peak);

/**
 * Find peaks in data (nasics_y x nasics_x ASICs of asic_ny x asic_nx pixels) using mask, the per-pixel radius pix_r and the thresholds and limits passed in; the work is done by static helpers in peakfinder8.cc (radial statistics over 5 iterations, then a search per ASIC).
 * Copies at most peaklist->nPeaks_max peaks into peaklist and sets nPeaks; if outliersMask is not null, the internal map of pixels in peaks is copied into it.
 * @return 0 on success, 1 if an internal allocation fails.
 */
int peakfinder8(tPeakList *peaklist, float *data, char *mask, float *pix_r,
                long asic_nx, long asic_ny, long nasics_x, long nasics_y,
                float ADCthresh, float hitfinderMinSNR,
				long hitfinderMinPixCount, long hitfinderMaxPixCount,
				long hitfinderLocalBGRadius, char* outliersMask);

#endif // PEAKFINDER8_H
