
# Conversion from LCLS1 -> LCLS2
# 1) hexanode_proxy/resort64c.h -> psalg/hexanode/resort64c.hh (6) ??? or roentdek/resort64c.h
# 2) hexanode/SortUtils.h -> psalg/hexanode/SortUtils.hh (1)
# 3) hexanode/LMF_IO.h -> psalg/hexanode/LMF_IO.hh (2)
# 4) hexanode/cfib.h -> psalg/hexanode/cfib.hh (1)
# 5) hexanode/ctest_nda.h -> psalg/hexanode/ctest_nda.hh (1)
# 6) hexanode/wrap_resort64c.h -> psalg/hexanode/wrap_resort64c.hh (1)
# 7) print ... -> print(...)
# 8) see psana/setup.py for hexanode

# library and header:
# /reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/include/roentdek/resort64c.h
# /reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/lib/libResort64c_x64.a
# /reg/g/psdm/sw/conda2/inst/pkgs/roentdek-0.0.3-1/include/roentdek/resort64c.h
# /reg/g/psdm/sw/conda2/inst/pkgs/roentdek-0.0.3-1/lib/libResort64c_x64.a

# passing numpy arrays:
# http://stackoverflow.com/questions/17855032/passing-and-returning-numpy-arrays-to-c-methods-via-cython

# issue with resort64c.h:
# - seen from hexanode/src and app, but 
# - not seen in compilation stage from hexanode/includes or hexanode/pyext/hexanode_ext.pyx
# Hack to fix:
# export CPATH=/reg/common/package/hexanodelib/0.0.1/x86_64-centos7-gcc485

#from cpython.bool cimport *
#from cpython.float cimport *
#from cpython.int cimport *
#from cpython.list cimport *
#from cpython.object cimport *
#from cpython.ref cimport *
#from cpython.string cimport *

from libcpp.string cimport string
#from libcpp.vector cimport vector
#from libcpp cimport bool

from libc.time cimport time_t, ctime

import numpy as np
cimport numpy as np

#np.import_array()

#from cython.operator cimport dereference as deref

#------------------------------

cdef extern from "<stdint.h>" nogil:
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long  uint64_t

ctypedef fused dtypes2d :
    np.ndarray[np.double_t, ndim=2]
    np.ndarray[np.int16_t,  ndim=2]
    np.ndarray[np.uint16_t, ndim=2]

#------------------------------
#------------------------------
#------------------------------
# TESTS
#------------------------------
#------------------------------
#------------------------------

def met1():
    """cython method for test call from python"""
    cdef int nelem = 1
    print("HELL of cython is here")


cdef extern from "psalg/hexanode/cfib.hh":
    double cfib(int n)


def fib(n):
    """Returns the nth Fibonacci number"""
    return cfib(n)

#------------------------------

cdef extern from "psalg/hexanode/ctest_nda.hh":
    #void ctest_nda[T](T *arr, int r, int c) except +
    void ctest_nda_f8(double   *arr, int r, int c) except +
    void ctest_nda_i2(int16_t  *arr, int r, int c) except +
    void ctest_nda_u2(uint16_t *arr, int r, int c) except +

#------------------------------
# Most compact working case 

#def test_nda_v1(np.ndarray nda):
#    print('nda.dtype =', str(nda.dtype))
#    if   nda.dtype == np.float64 : ctest_nda(<double*>   nda.data, nda.shape[0], nda.shape[1])
#    elif nda.dtype == np.int16   : ctest_nda(<int16_t*>  nda.data, nda.shape[0], nda.shape[1])
#    elif nda.dtype == np.uint16  : ctest_nda(<uint16_t*> nda.data, nda.shape[0], nda.shape[1])
#    else: raise ValueError("Array data type is unknown")

#------------------------------
# Specialized methods for each type
#------------------------------

#def test_nda_f8(np.ndarray[np.double_t, ndim=2] nda): ctest_nda_f8(<double*>  nda.data, nda.shape[0], nda.shape[1])
#def test_nda_i2(np.ndarray[np.int16_t, ndim=2] nda):  ctest_nda_i2(<int16_t*> nda.data, nda.shape[0], nda.shape[1])
#def test_nda_u2(np.ndarray[np.uint16_t, ndim=2] nda): ctest_nda_u2(<uint16_t*>nda.data, nda.shape[0], nda.shape[1])

#def test_nda_f8(np.ndarray[np.double_t, ndim=2, mode="c"] nda): ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])
#def test_nda_i2(np.ndarray[np.int16_t, ndim=2, mode="c"] nda):  ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])
#def test_nda_u2(np.ndarray[np.uint16_t, ndim=2, mode="c"] nda): ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])

def test_nda_f8(np.ndarray[np.double_t, ndim=2, mode="c"] nda): ctest_nda_f8(&nda[0,0], nda.shape[0], nda.shape[1])
def test_nda_i2(np.ndarray[np.int16_t, ndim=2, mode="c"] nda):  ctest_nda_i2(&nda[0,0], nda.shape[0], nda.shape[1])
def test_nda_u2(np.ndarray[np.uint16_t, ndim=2, mode="c"] nda): ctest_nda_u2(&nda[0,0], nda.shape[0], nda.shape[1])


#def test_nda(np.ndarray nda):
#def test_nda(dtypes2d nda):
#    print('nda.dtype =', str(nda.dtype))
#    if   nda.dtype == np.float64 : test_nda_f8(nda)
#    elif nda.dtype == np.int16   : test_nda_i2(nda)
#    elif nda.dtype == np.uint16  : test_nda_u2(nda)
#    else: raise ValueError("Array data type is unknown")

#------------------------------
#------------------------------
#------------------------------

#def test_nda(np.ndarray nda):
def test_nda(dtypes2d nda):
    #print('nda.dtype =', str(nda.dtype))
    if   nda.dtype == np.float64 : ctest_nda_f8(<double*>   nda.data, nda.shape[0], nda.shape[1])
    elif nda.dtype == np.int16   : ctest_nda_i2(<int16_t*>  nda.data, nda.shape[0], nda.shape[1])
    elif nda.dtype == np.uint16  : ctest_nda_u2(<uint16_t*> nda.data, nda.shape[0], nda.shape[1])
    else: raise ValueError("Array data type is unknown")

#------------------------------
#------------------------------
#------------------------------

## IT SOULD BUT DOES NOT WORK
#def test_nda_xxx(np.ndarray nda):
#    ctest_nda(nda.data, nda.shape[0], nda.shape[1])

#------------------------------

# Very compact, BUT GENERATES TOO MANY WARNINGS AT scons

#def test_nda_xxx(dtypes2d nda):
#    print('nda.dtype =', str(nda.dtype))
#    ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])

#------------------------------
#------------------------------
#------------------------------
#------------------------------

cdef extern from "psalg/hexanode/wrap_resort64c.hh":
    void test_resort()

def ctest_resort():
    print('In psalg/hexanode/hexanode_ext.pyx ctest_resort - calls cpp test_resort() from psalg/hexanode/wrap_resort64c.h')
    test_resort()

#------------------------------
#------------------------------
#--------- hit_class ----------
#------------------------------
#------------------------------

#cdef extern from "psalg/hexanode/resort64c.hh":
cdef extern from "roentdek/resort64c.h":
    cdef cppclass hit_class:
        double x, y, time
        int32_t method
        hit_class(sort_class *) except +

cdef class py_hit_class:
    """ Python wrapper for C++ class hit_class from resort64c.h. 
    """
    cdef hit_class* cptr  # holds a C++ instance

    def __cinit__(self, py_sort_class sorter, int i=0):
        #print("In py_hit_class.__cinit__ index: %d" % i)
        self.cptr = sorter.cptr.output_hit_array[i]

    def __dealloc__(self):
        print("In py_hit_class.__dealloc__")
        del self.cptr

    @property
    def x(self) : return self.cptr.x

    @property
    def y(self) : return self.cptr.y

    @property
    def time(self) : return self.cptr.time

    @property
    def method(self) : return self.cptr.method

##------------------------------
##------------------------------
## scalefactors_calibration_class
##------------------------------
##------------------------------
## /reg/g/psdm/sw/conda2/inst/pkgs/roentdek-0.0.3-1/lib/libResort64c_x64.a
## /reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/lib/libResort64c_x64.a
## /reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/include/roentdek/resort64c.h
## cdef extern from "/reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/include/roentdek/resort64c.h":
## cdef extern from "psalg/hexanode/resort64c.hh":

#cdef extern from "roentdek/resort64c.h":
cdef extern from "psalg/hexanode/resort64c.hh":
    cdef cppclass scalefactors_calibration_class:

        double best_fv, best_fw, best_w_offset
        int32_t binx, biny
        double detector_map_resol_FWHM_fill
        double detector_map_devi_fill
        scalefactors_calibration_class() except +
        scalefactors_calibration_class(bint BuildStack, double runtime_u, double runtime_v, double runtime_w, 
                                       double runtime_inner_fraction, 
                                       double fu,double fv,double fw) except +
        bint map_is_full_enough() except +
        void feed_calibration_data(double u_ns, double v_ns, double w_ns, double w_ns_minus_woffset) except +

        bint clone(scalefactors_calibration_class *) except +

#------------------------------

cdef class py_scalefactors_calibration_class:
    """ Python wrapper for C++ class scalefactors_calibration_class from resort64c.h. 
    """
    cdef scalefactors_calibration_class* cptr  # holds a C++ instance
    #cdef scalefactors_calibration_class* cptc  # holds a C++ instance

    def __cinit__(self, py_sort_class sorter):
        #print("In py_scalefactors_calibration_class.__cinit__")
        self.cptr = sorter.cptr.scalefactors_calibrator
        #st = self.cptr.clone(self.cptc)

    def __dealloc__(self):
        print("In py_scalefactors_calibration_class.__dealloc__")
        del self.cptr

    @property
    def best_fv(self) : return self.cptr.best_fv

    @property
    def best_fw(self) : return self.cptr.best_fw

    @property
    def best_w_offset(self) : return self.cptr.best_w_offset

    @property
    def binx(self) : return self.cptr.binx

    @property
    def biny(self) : return self.cptr.biny

    @property
    def detector_map_resol_FWHM_fill(self) : return self.cptr.detector_map_resol_FWHM_fill

    @property
    def detector_map_devi_fill(self) : return self.cptr.detector_map_devi_fill

    def map_is_full_enough(self) :
        print("In py_scalefactors_calibration_class.map_is_full_enough()")
        return self.cptr.map_is_full_enough()

#------------------------------
#------------------------------
#--------- sort_class ---------
#------------------------------
#------------------------------

#cdef extern from "roentdek/resort64c.h":
cdef extern from "psalg/hexanode/resort64c.hh":
    cdef cppclass sort_class:
        int Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp 
        bint use_sum_correction 
        bint use_pos_correction

        double TDC_resolution_ns
        int32_t tdc_array_row_length # the size of second dimension of TDC array 
        int32_t* count               # pointer to the array that counts the number of hits for each channel 
                                     # count[number_of_channels]
        double* tdc_pointer          # pointer to the tdc array tdc_ns[number_of_channels][max_number_of_hits]
        double runtime_u
        double runtime_v
        double runtime_w
        double fu
        double fv
        double fw

        bint common_start_mode
        bint use_HEX
        bint use_MCP
        double MCP_radius
        double uncorrected_time_sum_half_width_u
        double uncorrected_time_sum_half_width_v
        double uncorrected_time_sum_half_width_w
        double dead_time_anode
        double dead_time_mcp
        #* sum_corrector_U
        hit_class** output_hit_array  # pointer to the hit array [max_number_of_hits]
 
        scalefactors_calibration_class* scalefactors_calibrator

        bint use_reflection_filter_on_u1, use_reflection_filter_on_u2
        bint use_reflection_filter_on_v1, use_reflection_filter_on_v2
        bint use_reflection_filter_on_w1, use_reflection_filter_on_w2

        double  u1_reflection_time_position, u1_reflection_half_width_at_base
        double  u2_reflection_time_position, u2_reflection_half_width_at_base
        double  v1_reflection_time_position, v1_reflection_half_width_at_base
        double  v2_reflection_time_position, v2_reflection_half_width_at_base
        double  w1_reflection_time_position, w1_reflection_half_width_at_base
        double  w2_reflection_time_position, w2_reflection_half_width_at_base

        #----------------------

        sort_class() except +
        int32_t sort() except + # sorts data, returns the number of reconstructed particles 
        int32_t run_without_sorting() except +
        bint create_scalefactors_calibrator(bint , double& runtime_u,\
                                            double& runtime_v,\
                                            double& runtime_w, double& v,\
                                            double& fu, double& fv, double& fw) except +

        int32_t init_after_setting_parameters()
        void shift_sums(int32_t direction, double Sumu_offeset, double Sumv_offeset, double Sumw_offeset)
        void shift_layer_w(int32_t direction, double w_offeset)
        void shift_position_origin(int32_t direction, double x_pos_offeset, double y_pos_offeset)

        void get_error_text(int32_t error_code, int32_t buffer_length, char*& error_text)
        bint do_calibration()
        bint feed_calibration_data(bint build_map, double w_offset, int32_t number_of_correction_points)
        bint feed_calibration_data(bint build_map, double w_offset)

        bint are_all_single()
#        bint clone(sort_class *clone)

#------------------------------

cdef class py_sort_class:
    """ Python wrapper for C++ class sort_class from resort64c.h. 
    """
    cdef sort_class* cptr  # holds a C++ instance

    def __cinit__(self):
        #print("In py_sort_class.__cinit__")
        self.cptr = new sort_class();
        if self.cptr == NULL:
            raise MemoryError('In py_sort_class.__cinit__: Not enough memory.')

    def __dealloc__(self):
        #print("In py_sort_class.__dealloc__")
        del self.cptr

    @property
    def cu1(self) : return self.cptr.Cu1

    @property
    def cu2(self) : return self.cptr.Cu2

    @property
    def cv1(self) : return self.cptr.Cv1

    @property
    def cv2(self) : return self.cptr.Cv2

    @property
    def cw1(self) : return self.cptr.Cw1

    @property
    def cw2(self) : return self.cptr.Cw2

    @property
    def cmcp(self) : return self.cptr.Cmcp

    @property
    def use_sum_correction(self) : return self.cptr.use_sum_correction

    @property
    def use_pos_correction(self) : return self.cptr.use_pos_correction

    @property
    def runtime_u(self) : return self.cptr.runtime_u

    @property
    def runtime_v(self) : return self.cptr.runtime_v

    @property
    def runtime_w(self) : return self.cptr.runtime_w

    @property
    def fu(self) : return self.cptr.fu

    @property
    def fv(self) : return self.cptr.fv

    @property
    def fw(self) : return self.cptr.fw

    @property
    def common_start_mode(self) : return self.cptr.common_start_mode

    @property
    def use_hex(self) : return self.cptr.use_HEX

    @property
    def use_mcp(self) : return self.cptr.use_MCP

    @property
    def mcp_radius(self) : return self.cptr.MCP_radius

    @property
    def uncorrected_time_sum_half_width_u(self) : return self.cptr.uncorrected_time_sum_half_width_u

    @property
    def uncorrected_time_sum_half_width_v(self) : return self.cptr.uncorrected_time_sum_half_width_v

    @property
    def uncorrected_time_sum_half_width_w(self) : return self.cptr.uncorrected_time_sum_half_width_w

    @property
    def dead_time_anode(self) : return self.cptr.dead_time_anode

    @property
    def dead_time_mcp(self) : return self.cptr.dead_time_mcp

    @property
    def u1_reflection_time_position(self) : return self.cptr.u1_reflection_time_position

    @property
    def u2_reflection_time_position(self) : return self.cptr.u2_reflection_time_position

    @property
    def v1_reflection_time_position(self) : return self.cptr.v1_reflection_time_position

    @property
    def v2_reflection_time_position(self) : return self.cptr.v2_reflection_time_position

    @property
    def w1_reflection_time_position(self) : return self.cptr.w1_reflection_time_position

    @property
    def w2_reflection_time_position(self) : return self.cptr.w2_reflection_time_position

    @property
    def u1_reflection_half_width_at_base(self) : return self.cptr.u1_reflection_half_width_at_base

    @property
    def u2_reflection_half_width_at_base(self) : return self.cptr.u2_reflection_half_width_at_base

    @property
    def v1_reflection_half_width_at_base(self) : return self.cptr.v1_reflection_half_width_at_base

    @property
    def v2_reflection_half_width_at_base(self) : return self.cptr.v2_reflection_half_width_at_base

    @property
    def w1_reflection_half_width_at_base(self) : return self.cptr.w1_reflection_half_width_at_base

    @property
    def w2_reflection_half_width_at_base(self) : return self.cptr.w2_reflection_half_width_at_base

#    @property
#    def output_hit_array(self) : return self.cptr.output_hit_array

#    def output_hit_array(self) :
#        cdef hit_array* ptr2 = &self.cptr.output_hit_array[0]
#        return ptr2

#    def output_hit(self, int32_t i) :
#        return self.cptr.output_hit_array[i]

#    @property
#    def ???(self) : return self.cptr.???

#    @???.setter
#    def ???(self): 
#       def __set__(self, v): self.cptr.??? = v

#------------------------------
#------------------------------

    def set_use_reflection_filter_on_u1(self, bint v) : 
       self.cptr.use_reflection_filter_on_u1 = v

    def set_use_reflection_filter_on_v1(self, bint v) : 
       self.cptr.use_reflection_filter_on_v1 = v

    def set_use_reflection_filter_on_w1(self, bint v) : 
       self.cptr.use_reflection_filter_on_w1 = v

    def set_use_reflection_filter_on_u2(self, bint v) : 
       self.cptr.use_reflection_filter_on_u2 = v

    def set_use_reflection_filter_on_v2(self, bint v) : 
       self.cptr.use_reflection_filter_on_v2 = v

    def set_use_reflection_filter_on_w2(self, bint v) : 
       self.cptr.use_reflection_filter_on_w2 = v

#------------------------------

    def set_tdc_resolution_ns(self, double v) : 
       self.cptr.TDC_resolution_ns = v

    def set_tdc_array_row_length(self, int32_t v) : 
       self.cptr.tdc_array_row_length = v


    def set_count(self, np.ndarray[np.int32_t, ndim=1, mode="c"] nda) : 
       self.cptr.count = &nda[0]

    def set_tdc_pointer(self, np.ndarray[np.float64_t, ndim=2, mode="c"] nda) : 
       self.cptr.tdc_pointer = &nda[0,0]


    def get_error_text(self, int32_t error_code, int32_t buffer_size) : 
        cdef char txt[512]
        self.cptr.get_error_text(error_code, buffer_size, &txt[0])
        return txt

    def create_scalefactors_calibrator(self, bint bv, double& runtime_u,\
                                                double& runtime_v,\
                                                double& runtime_w,\
                                                double& v,\
                                                double& fu, double& fv, double& fw) :
        print("In py_sort_class.create_scalefactors_calibrator")
        return self.cptr.create_scalefactors_calibrator(bv, runtime_u, runtime_v, runtime_w, v, fu, fv, fw)


    def sort(self) :
        #ok print("In py_sort_class.sort")
        return self.cptr.sort()


    def run_without_sorting(self) :
        #ok print("In py_sort_class.run_without_sorting")
        return self.cptr.run_without_sorting()


    def init_after_setting_parameters(self) :
        """Returns (int) error_code"""
        print("In py_sort_class.init_after_setting_parameters")
        return self.cptr.init_after_setting_parameters()


    def shift_sums(self, int32_t direction, double Sumu_offeset, double Sumv_offeset, double Sumw_offeset=0) :
        #print("In py_sort_class.shift_sums")
        self.cptr.shift_sums(direction, Sumu_offeset, Sumv_offeset, Sumw_offeset)


    def shift_layer_w(self, int32_t direction, double& w_offeset) :
        #print("In py_sort_class.shift_layer_w")
        self.cptr.shift_layer_w(direction, w_offeset)


    def shift_position_origin(self, int32_t direction, double& x_pos_offeset, double& y_pos_offeset) :
        #print("In py_sort_class.shift_position_origin")
        self.cptr.shift_position_origin(direction, x_pos_offeset, y_pos_offeset)


    def do_calibration(self) :
        print("In py_sort_class.do_calibration")
        return self.cptr.do_calibration()


    def feed_calibration_data(self, bint build_map, double w_offset, int32_t number_of_correction_points=0) :
        #print("In py_sort_class.feed_calibration_data")
        #return self.cptr.feed_calibration_data(build_map, w_offset, number_of_correction_points)
        return self.cptr.feed_calibration_data(build_map, w_offset)


    def are_all_single(self) :
        return self.cptr.are_all_single()


#    def clone(self, sort_class clone) :    	
#        return self.cptr.clone(self)

#------------------------------
#------------------------------
#--------- SortUtils.hh --------
#------------------------------
#------------------------------

cdef extern from "psalg/hexanode/SortUtils.hh":
    bint read_config_file(const char* fname, sort_class *& sorter, int& command,
                          double& offset_sum_u, double& offset_sum_v, double& offset_sum_w, 
                          double& w_offset, double& pos_offset_x, double& pos_offset_y)

    bint read_calibration_tables(const char* fname, sort_class* sorter)

    bint create_calibration_tables(const char* fname, sort_class* sorter)

    bint sorter_scalefactors_calibration_map_is_full_enough(sort_class* sorter)

#------------------------------

def py_read_config_file(const char* fname, py_sort_class sorter) :
    print("In py_read_config_file from %s" % fname)
    cdef int command = 0
    cdef double offset_sum_u = 0
    cdef double offset_sum_v = 0
    cdef double offset_sum_w = 0
    cdef double w_offset = 0
    cdef double pos_offset_x = 0
    cdef double pos_offset_y = 0
    status = read_config_file(fname, sorter.cptr, command,\
                              offset_sum_u, offset_sum_v, offset_sum_w,\
                              w_offset, pos_offset_x, pos_offset_y)
    return status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y


def py_read_calibration_tables(const char* fname, py_sort_class sorter) :
    print("In py_read_calibration_tables from file %s" % fname)
    return read_calibration_tables(fname, sorter.cptr)


def py_create_calibration_tables(const char* fname, py_sort_class sorter) :
    print("In py_create_calibration_tables in file %s" % fname)
    return create_calibration_tables(fname, sorter.cptr)


def py_sorter_scalefactors_calibration_map_is_full_enough(py_sort_class sorter) :
    return sorter_scalefactors_calibration_map_is_full_enough(sorter.cptr)

#------------------------------
#------------------------------
#--- signal_corrector_class ---
#------------------------------
#------------------------------

#cdef extern from "roentdek/resort64c.h":
cdef extern from "psalg/hexanode/resort64c.hh":
    cdef cppclass signal_corrector_class:
        signal_corrector_class()


cdef class py_signal_corrector_class:
    """ Python wrapper for C++ class sort_class from resort64c.h. 
    """

    cdef signal_corrector_class* cptr  # holds a C++ instance which we're wrapping

    def __cinit__(self):
        print("In py_signal_corrector_class.__cinit__"\
              " - direct test of methods from roentdek/resort64c.h in hexanode_ext.class py_signal_corrector_class")
        self.cptr = new signal_corrector_class();

    def __dealloc__(self):
        print("In py_signal_corrector_class.__dealloc__")
        del self.cptr

#------------------------------
#------------------------------
#--------- LMF_IO -------------
#------------------------------
#------------------------------

cdef extern from "psalg/hexanode/LMF_IO.hh":
    cdef struct TDC8HP_struct:
        int32_t UserHeaderVersion
        int32_t TriggerChannel_p64
        double  TriggerDeadTime_p68
        double  GroupRangeStart_p69
        double  GroupRangeEnd_p70

cdef class tdc8hp_struct:
    cdef TDC8HP_struct cobj

    @property
    def user_header_version(self) :   return self.cobj.UserHeaderVersion

    @property
    def trigger_channel_p64(self) :   return self.cobj.TriggerChannel_p64

    @property
    def trigger_dead_time_p68(self) : return self.cobj.TriggerDeadTime_p68

    @property
    def group_range_start_p69(self) : return self.cobj.GroupRangeStart_p69

    @property
    def group_range_end_p70(self) :   return self.cobj.GroupRangeEnd_p70

#------------------------------

cdef extern from "psalg/hexanode/LMF_IO.hh":
    cdef cppclass LMF_IO:
        time_t Starttime
        time_t Stoptime
        int32_t errorflag
        int32_t data_format_in_userheader
        string FilePathName
        string OutputFilePathName

        string Versionstring
        string Comment
        uint32_t Headersize
        int32_t Numberofcoordinates
        uint32_t timestamp_format
        int32_t common_mode
        uint64_t uint64_Numberofevents
        uint32_t DAQ_ID
        double tdcresolution
        TDC8HP_struct TDC8HP

        LMF_IO(int32_t, int32_t) except +
        bint OpenInputLMF(char*)
        bint ReadNextEvent()
        unsigned int GetEventNumber()
        void GetNumberOfHitsArray(int32_t*)    # (number_of_hits)
        #void GetNumberOfHitsArray(uint32_t*)   # (number_of_hits)
        void GetTDCDataArray(double*)          # &dTDC[0][0])
        #void GetTDCDataArray(int32_t*)         # &iTDC[0][0])
        #void GetTDCDataArray(uint16_t*)        # &sTDC[0][0])
        #void GetErrorText(int32_t, char*)      # LMF->errorflag, error_text
        const char* GetErrorText(int32_t)
        uint32_t GetNumberOfChannels()
        uint32_t GetMaxNumberOfHits()
        uint64_t GetLastLevelInfo()
        double GetDoubleTimeStamp()

cdef class lmf_io:
    cdef LMF_IO* cptr # holds a C++ instance

    def __cinit__(self, int number_of_channels, int number_of_hits):
        print("In LMF_IO.__cinit__ pars %d, %d" % (number_of_channels, number_of_hits))
        self.cptr = new LMF_IO(number_of_channels, number_of_hits)
        if self.cptr == NULL:
            raise MemoryError('Not enough memory.')

    def __dealloc__(self):
        print("In LMF_IO.__dealloc__")
        del self.cptr

    def open_input_lmf(self, const char* fname):
        print("In LMF_IO.cptr.open_input_lmf, open file %s" % fname)
        s = self.cptr.OpenInputLMF(fname)
        #print("file opening status %d" % s)
        return s

    def start_time(self):
        return ctime(&self.cptr.Starttime).decode().strip('\n')

    def stop_time(self):
        return ctime(&self.cptr.Stoptime).decode().strip('\n')

    def read_next_event(self):
        return self.cptr.ReadNextEvent()
        #cdef int s = self.cptr.ReadNextEvent()
        #return s

    def get_event_number(self):
        return self.cptr.GetEventNumber()

    def get_number_of_hits_array(self, np.ndarray[int32_t, ndim=1] arr1d not None):
        self.cptr.GetNumberOfHitsArray(&arr1d[0])

    def get_tdc_data_array(self, np.ndarray[double, ndim=2, mode="c"] arr2d not None):
        self.cptr.GetTDCDataArray(&arr2d[0,0])
        #self.cptr.GetTDCDataArray(<double*>arr2d.data) #depricated

    def get_error_text(self, errcode):
        return self.cptr.GetErrorText(errcode)

    def get_number_of_channels(self):
        return self.cptr.GetNumberOfChannels()

    def get_max_number_of_hits(self) :
        return self.cptr.GetMaxNumberOfHits()

    def get_last_level_info(self):
        return self.cptr.GetLastLevelInfo()

    def get_double_time_stamp(self):
        return self.cptr.GetDoubleTimeStamp()

    @property
    def error_flag(self) : return self.cptr.errorflag

    @property
    def data_format_in_user_header(self) : return self.cptr.data_format_in_userheader

    @property
    def file_path_name(self) : return self.cptr.FilePathName

    @property
    def output_file_path_name(self) : return self.cptr.OutputFilePathName

    @property
    def version_string(self) : return self.cptr.Versionstring

    @property
    def comment(self) : return self.cptr.Comment

    @property
    def header_size(self) : return self.cptr.Headersize

    @property
    def number_of_coordinates(self) : return self.cptr.Numberofcoordinates

    @property
    def timestamp_format(self) : return self.cptr.timestamp_format

    @property
    def common_mode(self) : return self.cptr.common_mode

    @property
    def uint64_number_of_events(self) : return self.cptr.uint64_Numberofevents

    @property
    def daq_id(self) : return self.cptr.DAQ_ID

    @property
    def tdc_resolution(self) : return self.cptr.tdcresolution

    @property
    def tdc8hp(self) : return tdc8hp_struct(self.cptr.TDC8HP)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

