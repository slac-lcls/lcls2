import numpy as np
import json
from scipy.linalg import toeplitz
from scipy.linalg import inv
from scipy.signal import chirp, find_peaks, peak_widths


class EdgeFinderResult(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class EdgeFinder(object):

    height = 0.25 # factor from max - used in finding peaks

    def __init__(self, calibconst, good_image=None, bgs=None):
        self.calibconst = calibconst
        self.kernel = self._get_fir_coefficients() 
        #self.kernel = self._calc_filter_weights(good_image, bgs)
        self.delayed_denominator, _ = self.calibconst['delayed_denom']

    def __call__(self, image, IIR):
        result = None

        #validating that we're getting the correct number of pixels
        if(image is not None):
            assert len(image) == 2048

            # check if IIR exists 
            if IIR is None:
                IIR = np.zeros(image.shape, dtype=image.dtype)

            # do the matchy filter and normalize
            convolved = np.convolve( \
                (image-IIR)/ self.delayed_denominator, \
                self.kernel)
            convolved /= np.max(convolved)

            # get first peak and its fwhm
            first_peak = np.argmax(convolved)

            first_fwhms = peak_widths(convolved, [first_peak], rel_height=0.5)
            
            # find more peaks by looking at anything > 25% of the first peak amplitude
            height = self.height * convolved[first_peak]
            peaks, _  = find_peaks(convolved, height=height, distance=first_fwhms[0][0]*2)
            peaks_amp = convolved[peaks]
            sort_indices = np.flip(np.argsort(peaks_amp))
            peaks_sort = peaks[sort_indices]
            peaks_amp_sort = peaks_amp[sort_indices]
            results_half = peak_widths(convolved, peaks_sort, rel_height=0.5)
            
            # check if we have second peak
            amplitude_next = None
            if len(peaks_sort) > 1:
                amplitude_next = peaks_amp_sort[1]

            # gather result
            result = EdgeFinderResult(edge=peaks_sort[0], \
                    amplitude=peaks_amp_sort[0], \
                    fwhm=first_fwhms[0][0], \
                    amplitude_next=amplitude_next, \
                    ref_amplitude=np.mean(IIR), \
                    convolved=convolved, \
                    results_half=results_half)

        return result

    def _twos_complement(self, hexstr,bits):
        value = int(hexstr,16)
        if value & (1 << (bits-1)):
            value -= 1 << bits
        return value

    def _get_fir_coefficients(self):
        my_kernel = []
        my_dict_str, _ = self.calibconst['fir_coefficients']
        my_dict = json.loads(my_dict_str.replace("'", '"'))
        for key, val in my_dict.items():
            mystring = my_dict[key]
            my_kernel.extend([self._twos_complement(mystring[i:i+2],8) for i in range(0,len(mystring),2)])
        return my_kernel
    
    def _calc_filter_weights(self, good_image, bgs):
        """ Calculated weighted image from a given good image and
        a list of backgrounds.
        detail here: https://confluence.slac.stanford.edu/display/PSDM/TimeTool
        """
        # Step 3: Normalize backgrounds
        bg_avg = np.average(bgs[:])
        bgs /= bg_avg
        results = np.zeros(bgs.shape)
        for i, bg in enumerate(bgs):
            result = np.correlate(bg, bg, mode='full')
            # Auto correlation is the second half in 'full mode' for np.correlate. 
            # See https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
            results[i] = result[int(result.size/2):]
        
        # Step 4: auto-correlation function (acf) is the average of all those products
        acf = np.average(results, axis=0)
        
        # Step 5: Collect your best averaged ratio (selected "good_image" 
        # divided by averaged background) with signal; this averaging smooths 
        # out any non-physics features
        acfm = toeplitz(acf)
        acfm_inv = inv(acfm)
        w = np.dot(acfm_inv, good_image)
        weights = np.flip(w) # filter weight is inverted (for computing convolution)
        norm = np.dot(weights, good_image) 
        weights = weights * 1.00 / norm
        
        return weights
        
