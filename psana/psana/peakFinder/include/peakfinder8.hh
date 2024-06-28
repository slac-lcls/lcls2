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
	long	    nPeaks;
	long	    nHot;
	float		peakResolution;			// Radius of 80% of peaks
	float		peakResolutionA;		// Radius of 80% of peaks
	float		peakDensity;			// Density of peaks within this 80% figure
	float		peakNpix;				// Number of pixels in peaks
	float		peakTotal;				// Total integrated intensity in peaks
	int			memoryAllocated;
	long		nPeaks_max;

	float		*peak_maxintensity;		// Maximum intensity in peak
	float		*peak_totalintensity;	// Integrated intensity in peak
	float		*peak_sigma;			// Signal-to-noise ratio of peak
	float		*peak_snr;				// Signal-to-noise ratio of peak
	float		*peak_npix;				// Number of pixels in peak
	float		*peak_com_x;			// peak center of mass x (in raw layout)
	float		*peak_com_y;			// peak center of mass y (in raw layout)
	long		*peak_com_index;		// closest pixel corresponding to peak
	float		*peak_com_x_assembled;	// peak center of mass x (in assembled layout)
	float		*peak_com_y_assembled;	// peak center of mass y (in assembled layout)
	float		*peak_com_r_assembled;	// peak center of mass r (in assembled layout)
	float		*peak_com_q;			// Scattering vector of this peak
	float		*peak_com_res;			// REsolution of this peak
} tPeakList;

void allocatePeakList(tPeakList *peak, long NpeaksMax);
void freePeakList(tPeakList peak);

int peakfinder8(tPeakList *peaklist, float *data, char *mask, float *pix_r,
                long asic_nx, long asic_ny, long nasics_x, long nasics_y,
                float ADCthresh, float hitfinderMinSNR,
				long hitfinderMinPixCount, long hitfinderMaxPixCount,
				long hitfinderLocalBGRadius, char* outliersMask);

#endif // PEAKFINDER8_H
