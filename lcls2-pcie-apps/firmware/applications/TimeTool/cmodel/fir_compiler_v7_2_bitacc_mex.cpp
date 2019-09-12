//-----------------------------------------------------------------------------
//  (c) Copyright 2011 Xilinx, Inc. All rights reserved.
//
//  This file contains confidential and proprietary information
//  of Xilinx, Inc. and is protected under U.S. and
//  international copyright and other intellectual property
//  laws.
//
//  DISCLAIMER
//  This disclaimer is not a license and does not grant any
//  rights to the materials distributed herewith. Except as
//  otherwise provided in a valid license issued to you by
//  Xilinx, and to the maximum extent permitted by applicable
//  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
//  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
//  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
//  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
//  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
//  (2) Xilinx shall not be liable (whether in contract or tort,
//  including negligence, or under any other theory of
//  liability) for any loss or damage of any kind or nature
//  related to, arising under or in connection with these
//  materials, including for any direct, or any indirect,
//  special, incidental, or consequential loss or damage
//  (including loss of data, profits, goodwill, or any type of
//  loss or damage suffered as a result of any action brought
//  by a third party) even if such damage or loss was
//  reasonably foreseeable or Xilinx had been advised of the
//  possibility of the same.
//
//  CRITICAL APPLICATIONS
//  Xilinx products are not designed or intended to be fail-
//  safe, or for use in any application requiring fail-safe
//  performance, such as life-support or safety devices or
//  systems, Class III medical devices, nuclear facilities,
//  applications related to the deployment of airbags, or any
//  other applications that could lead to death, personal
//  injury, or severe property or environmental damage
//  (individually and collectively, "Critical
//  Applications"). Customer assumes the sole risk and
//  liability of any use of Xilinx products in Critical
//  Applications, subject only to applicable laws and
//  regulations governing limitations on product liability.
//
//  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
//  PART OF THIS FILE AT ALL TIMES.
//-----------------------------------------------------------------------------
//
//  Note the following:
//   * In MEX functions mxMalloc/mxCalloc/mxRealloc will not return if the allocation fails,
//       hence checks for null pointers are unnecessary
//   * Matlab performs automatic garbage collection of non-persistent memory and arrays within MEX functions,
//       hence calls to mxFree are generally not required

#include "mex.h"
#include "fir_compiler_v7_2_bitacc_cmodel.h"
#include <stdlib.h>
#include <string.h>

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//mxarray helper functions

//True if mx can be interpreted as a real numeric scalar
bool is_real_scalar(const mxArray* mx)
{
  return mx && mxGetNumberOfElements(mx)==1 && mxIsNumeric(mx) && !mxIsComplex(mx) && !mxIsSparse(mx);
}

//True if mx can be interpreted as a real numeric vector
bool is_numeric_vector(const mxArray* mx)
{
  return mx && mxIsNumeric(mx) && !mxIsSparse(mx);
}

//True if mx can be interpreted as a single string
bool is_string(const mxArray* mx)
{
  return mx && mxIsChar(mx) && mxGetM(mx)==1 && mxGetN(mx)==mxGetNumberOfElements(mx);
}

//Concatenate two C style strings in an mxMalloc allocated area of memory
char* mxmalloc_strcat(const char* s1,const char* s2)
{
  size_t s1len;
  size_t s2len;
  char* res;

  s1len=strlen(s1);
  s2len=strlen(s2);
  res=(char*)mxMalloc(s1len+s2len+1);

  strcpy(res      ,s1);
  strcpy(res+s1len,s2);

  return res;
}

//Convert an mxArray to a C-style string
char* mxarray_to_string(const mxArray* mx)
{
  char* res;
  unsigned int res_len;

  if (!is_string(mx)) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_string","ERROR:fir_compiler_v7_2_bitacc_mex:Expecting string");

  res_len=(unsigned int)mxGetNumberOfElements(mx)+1;
  res=(char*)mxMalloc(res_len);

  if (mxGetString(mx,res,res_len)) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_call","ERROR:fir_compiler_v7_2_bitacc_mex:Unexpected failure of mxGetString");
  return res;
}

//Convert a numeric real scalar mxArray to an int
int mxarray_to_int(const mxArray* mx)
{
  if (is_real_scalar(mx))
  {
    double x=mxGetScalar(mx);
    return (int)(x);
  }
  else if (mxIsLogicalScalar(mx))
  {
    //Convert logical to 0 or 1
    return (mxIsLogicalScalarTrue(mx) ? 1 : 0);
  }

  mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_scalar","ERROR:fir_compiler_v7_2_bitacc_mex:Expecting real numeric or logical scalar");
  return 0;  //Will not return
}

//Convert a numeric real scalar mxArray to an unsigned int
unsigned int mxarray_to_uint(const mxArray* mx)
{
  if (is_real_scalar(mx))
  {
    double x=mxGetScalar(mx);
    if (x>=0.0) return (unsigned int)(x);
  }
  else if (mxIsLogicalScalar(mx))
  {
    //Convert logical to 0 or 1
    return (mxIsLogicalScalarTrue(mx) ? 1 : 0);
  }

  mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_scalar","ERROR:fir_compiler_v7_2_bitacc_mex:Expecting unsigned real numeric or logical scalar");
  return 0;  //Will not return
}

//Create a real numeric scalar with the given value
mxArray* mxarray_create_scalar(double value)
{
  mxArray* res=mxCreateDoubleScalar(value);
  if (!res) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_scalar","ERROR:fir_compiler_v7_2_bitacc_mex:Could not create numeric scalar");
  return res;
}

//Create an empty scalar structure
mxArray* mxstruct_create()
{
  const char* FIELDNAMES[]={""};
  mxArray* res=mxCreateStructMatrix(1,1,0,FIELDNAMES);
  if (!res) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_struct","ERROR:fir_compiler_v7_2_bitacc_mex:Could not create empty structure");
  return res;
}

//Add a string field to an mxarray struct
void mxstruct_add_field_string(mxArray* mx, const char* fieldname, const char* value)
{
  int ix=mxAddField(mx,fieldname);
  if (ix==-1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_field","ERROR:fir_compiler_v7_2_bitacc_mex:Could not add field %s",fieldname);
  mxSetFieldByNumber(mx,0,ix,mxCreateString(value));
}

//Add an integer field to an mxarray struct
void mxstruct_add_field_int(mxArray* mx, const char* fieldname, int value)
{
  int ix=mxAddField(mx,fieldname);
  if (ix==-1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_field","ERROR:fir_compiler_v7_2_bitacc_mex:Could not add field %s",fieldname);
  mxSetFieldByNumber(mx,0,ix,mxCreateDoubleScalar((double)value));
}

//Add an unsigned integer field to an mxarray struct
void mxstruct_add_field_uint(mxArray* mx, const char* fieldname, unsigned int value)
{
  int ix=mxAddField(mx,fieldname);
  if (ix==-1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_field","ERROR:fir_compiler_v7_2_bitacc_mex:Could not add field %s",fieldname);
  mxSetFieldByNumber(mx,0,ix,mxCreateDoubleScalar((double)value));
}

//Add a mxarray field to an mxarray struct
void mxstruct_add_field_mxarray(mxArray* mx, const char* fieldname, const mxArray* mx_value)
{
  int ix=mxAddField(mx,fieldname);
  if (ix==-1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_field","ERROR:fir_compiler_v7_2_bitacc_mex:Could not add field %s",fieldname);
  mxSetFieldByNumber(mx,0,ix,(mxArray*)mx_value);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Functions for manipulating model handles

//Model handle type
typedef unsigned int model_handle;

//Simple dynamic vector for storing model structures attached to model handles
typedef struct
{
  xip_fir_v7_2** m_data;
  size_t m_size;
} model_handle_vector;

//Destroy a model_handle_vector
void model_handle_vector_destroy(model_handle_vector* mhv)
{
  size_t i;

  if (mhv)
  {
    //Destroy any model objects in the model_handle_vector
    if (mhv->m_data)
    {
      for (i=0; i<mhv->m_size; i++)
      {
        if (mhv->m_data[i]) xip_fir_v7_2_destroy(mhv->m_data[i]);
      }

      free(mhv->m_data);
    }

    free(mhv);
  }
}

//Create a model_handle_vector
model_handle_vector* model_handle_vector_create()
{
  const unsigned int DEFAULT_SIZE=16;
  model_handle_vector* mhv;

  mhv=(model_handle_vector*)calloc(1,sizeof(model_handle_vector));
  if (mhv)
  {
    mhv->m_data=(xip_fir_v7_2**)calloc(DEFAULT_SIZE,sizeof(xip_fir_v7_2*));
    mhv->m_size=DEFAULT_SIZE;
  }

  if (!mhv || !mhv->m_data)
  {
    model_handle_vector_destroy(mhv);
    mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_alloc","ERROR:fir_compiler_v7_2_bitacc_mex:Could not allocate memory for model handle vector");
    return 0;
  }

  return mhv;
}

//Get next unused model_handle in model_handle_vector
model_handle model_handle_vector_get_next(model_handle_vector* mhv)
{
  model_handle res=0;
  xip_fir_v7_2** p=0;

  //Scan array looking for free slot (handle 0 is always unused)
  for (res=1; res<mhv->m_size; res++) if (!mhv->m_data[res]) return res;

  //No spare slot found in array, so extend it
  p=(xip_fir_v7_2**)realloc((void*)mhv->m_data,sizeof(xip_fir_v7_2*)*mhv->m_size*2);
  if (!p) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:Could not allocate a free model handle");

  //Resize was successful
  memset((void*)(p+mhv->m_size),0,sizeof(xip_fir_v7_2*)*mhv->m_size);
  res=(model_handle)mhv->m_size;
  mhv->m_data=p;
  mhv->m_size=mhv->m_size*2;
  return res;
}

//Set xip_fir_v7_2 associated with model_handle
void model_handle_vector_set_structure(model_handle_vector* mhv, model_handle mh, xip_fir_v7_2* s)
{
  if (mh>0 && mh<mhv->m_size) mhv->m_data[mh]=s;
}

//Get xip_fir_v7_2 associated with model_handle
xip_fir_v7_2* model_handle_vector_get_structure(model_handle_vector* mhv, model_handle mh)
{
  if (mh>0 && mh<mhv->m_size) return mhv->m_data[mh];
  return 0;
}

//Get a model_handle from the numeric scalar array mx (or zero if error or invalid)
model_handle mxarray_get_model_handle(const mxArray* mx)
{
  if (!is_real_scalar(mx))
  {
    mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:Model handle must be a real numeric scalar");
    return 0;
  }

  //Get handle as an integer
  return (model_handle)mxGetScalar(mx);
}

//Global model_handle_vector for this MEX function
model_handle_vector* the_model_handle_vector=0;

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Functions for converting between Matlab and Xilinx arrays

//Convert dimensions vector from mwSize* to size_t*
const size_t* mxarray_dimensions_to_xip_array_dimensions(const mxArray* mx)
{
  size_t* res;
  mwSize len;
  const mwSize* p;
  const mwSize* q;
  size_t* r;

  if (sizeof(size_t)==sizeof(mwSize))
  {
    //If sizeof(mwSize)==sizeof(size_t) the vector can be used directly (read only)
    return (const size_t*)mxGetDimensions(mx);
  }
  else
  {
    //Otherwise we need a copy
    len=mxGetNumberOfDimensions(mx);
    res=(size_t*)mxMalloc(len*sizeof(size_t));
    for (p=mxGetDimensions(mx), q=p+len, r=res; p!=q; ++p, ++r) *r=*p;
    return res;
  }
}

//Convert dimensions vector from size_t* to mwSize*
const mwSize* xip_array_dimensions_to_mxarray_dimensions(const size_t* dim, const size_t dim_size)
{
  mwSize* res;
  const size_t* p;
  const size_t* q;
  mwSize* r;

  if (sizeof(size_t)==sizeof(mwSize))
  {
    //If sizeof(mwSize)==sizeof(size_t) the vector can be used directly (read only)
    return (const mwSize*)dim;
  }
  else
  {
    //Otherwise we need a copy
    res=(mwSize*)mxMalloc(dim_size*sizeof(mwSize));
    for (p=dim, q=p+dim_size, r=res; p!=q; ++p, ++r) *r=(mwSize)(*p);
    return res;
  }
}

//Convert a mxArray to a static xip_array_uint
xip_array_uint* mxarray_to_xip_array_uint(const mxArray* mx, const char* name)
{
  xip_array_uint* res=0;

  //Create an empty xip_array object on the Matlab heap
  res=(xip_array_uint*)mxCalloc(1,sizeof(xip_array_uint));
  res->owner=1;  //Memory is owned by Matlab
  res->ops=xip_array_uint_get_default_ops();

  if (mx)
  {
    if (!is_numeric_vector(mx) || mxIsComplex(mx)) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting unsigned int vector",name);
    if (mxGetClassID(mx)!=mxUINT32_CLASS) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting class unsigned int",name);

    //Create xip_array
    res->data         =(unsigned int*)mxGetData(mx); // Cast to unsigned int
    res->data_size    =mxGetNumberOfElements(mx);
    res->data_capacity=mxGetNumberOfElements(mx);
    res->dim          =(size_t*)mxarray_dimensions_to_xip_array_dimensions(mx);  //Discard const qualifier, however model will not modify as owner is not zero
    res->dim_size     =mxGetNumberOfDimensions(mx);
    res->dim_capacity =mxGetNumberOfDimensions(mx);
  }

  return res;
}

//Convert a mxArray to a static xip_array_real
xip_array_real* mxarray_to_xip_array_real(const mxArray* mx, const char* name)
{
  xip_array_real* res=0;

  //Create an empty xip_array object on the Matlab heap
  res=(xip_array_real*)mxCalloc(1,sizeof(xip_array_real));
  res->owner=1;  //Memory is owned by Matlab
  res->ops=xip_array_real_get_default_ops();

  if (mx)
  {
    if (!is_numeric_vector(mx) || mxIsComplex(mx)) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting real vector",name);
    if (mxGetClassID(mx)!=mxDOUBLE_CLASS) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting class double",name);

    //Create xip_array
    //  xip_array_real uses same data representation as mxarray, so we can link directly to Matlab data
    res->data         =mxGetPr(mx);
    res->data_size    =mxGetNumberOfElements(mx);
    res->data_capacity=mxGetNumberOfElements(mx);
    res->dim          =(size_t*)mxarray_dimensions_to_xip_array_dimensions(mx);  //Discard const qualifier, however model will not modify as owner is not zero
    res->dim_size     =mxGetNumberOfDimensions(mx);
    res->dim_capacity =mxGetNumberOfDimensions(mx);
  }

  return res;
}

//Convert a xip_array_real to a mxArray
mxArray* xip_array_real_to_mxarray(const xip_array_real* x)
{
  mxArray* res;
  const xip_real* p;
  const xip_real* q;
  double* r;

  res=mxCreateNumericArray((int)x->dim_size,xip_array_dimensions_to_mxarray_dimensions(x->dim,x->dim_size),mxDOUBLE_CLASS,mxREAL);
  for (p=x->data, q=x->data+mxGetNumberOfElements(res), r=(double*)mxGetPr(res); p!=q; ++p, ++r) *r=(double)*p;
  return res;
}

//Convert a xip_array_real to a mxArray
mxArray* xip_array_complex_to_mxarray(const xip_array_complex* x)
{
  mxArray* res;
  const xip_complex* p;
  const xip_complex* q;
  double* re;
  double* im;

  res=mxCreateNumericArray((int)x->dim_size,xip_array_dimensions_to_mxarray_dimensions(x->dim,x->dim_size),mxDOUBLE_CLASS,mxCOMPLEX);
  for (p=x->data, q=x->data+mxGetNumberOfElements(res), re=(double*)mxGetPr(res), im=(double*)mxGetPi(res); p!=q; ++p, ++re,++im) {
    *re=(double)(p->re);
    *im=(double)(p->im);
  }
  return res;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Functions to handle xip_fir_v7_2_* structures

//Convert a mxArray to a xip_fir_v7_2_config structure
xip_fir_v7_2_config* mxarray_to_xip_fir_v7_2_config(const mxArray* mx, const char* name)
{
  xip_fir_v7_2_config* res=0;
  int i;

  res=(xip_fir_v7_2_config*)mxCalloc(1,sizeof(xip_fir_v7_2_config));
  xip_fir_v7_2_default_config(res);

  if (mx)
  {
    if (!mxIsStruct(mx) || mxGetNumberOfElements(mx)!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_structure","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting scalar structure",name);

    for (i=0; i<mxGetNumberOfFields(mx); i++)
    {
      const char* fieldname=mxGetFieldNameByNumber(mx,i);
           if (!strcmp(fieldname,"name"                )) res->name            =mxarray_to_string(mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"filter_type"         )) res->filter_type     =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"rate_change"         )) res->rate_change     =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"interp_rate"         )) res->interp_rate     =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"decim_rate"          )) res->decim_rate      =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"zero_pack_factor"    )) res->zero_pack_factor=mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"coeff"               )) {
        if (!is_numeric_vector(mxGetFieldByNumber(mx,0,i))) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting numeric vector",name);
        if (mxGetClassID(mxGetFieldByNumber(mx,0,i))!=mxDOUBLE_CLASS) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting class double",name);
        if (mxGetNumberOfDimensions(mxGetFieldByNumber(mx,0,i))>2) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid coefficient array, expecting 1 by n or n by 1 array");
        if ( ((*mxGetDimensions(mxGetFieldByNumber(mx,0,i)) == 1) ^ (*(mxGetDimensions(mxGetFieldByNumber(mx,0,i))+1) == 1)) == 0 ) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid coefficient array, expecting 1 by n or n by 1 array");
        res->coeff = mxGetPr(mxGetFieldByNumber(mx,0,i));
      }
      else if (!strcmp(fieldname,"coeff_padding"       )) res->coeff_padding       =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"num_coeffs"          )) res->num_coeffs          =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"coeff_sets"          )) res->coeff_sets          =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"reloadable"          )) res->reloadable          =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"is_halfband"         )) res->is_halfband         =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"quantization"        )) res->quantization        =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"coeff_width"         )) res->coeff_width         =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"coeff_fract_width"   )) res->coeff_fract_width   =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"chan_seq"            )) res->chan_seq            =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"num_channels"        )) res->num_channels        =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"init_pattern"        )) res->init_pattern        =(xip_fir_v7_2_pattern)mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"num_paths"           )) res->num_paths           =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"data_width"          )) res->data_width          =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"data_fract_width"    )) res->data_fract_width    =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"output_rounding_mode")) res->output_rounding_mode=mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"output_width"        )) res->output_width        =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"output_fract_width"  )) res->output_fract_width  =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"config_method"       )) res->config_method       =mxarray_to_int   (mxGetFieldByNumber(mx,0,i));

      else mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_fieldname","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; unexpected fieldname %s",name,fieldname);
    }
  }

  return res;
}

//Convert a xip_fir_v7_2_config structure to a mxArray
mxArray* xip_fir_v7_2_config_to_mxarray(const xip_fir_v7_2_config* s)
{
  mxArray* mx;
  mxArray* coeff;
  mwSize*  coeff_dims;
  double*  r;

  //Create structure and populate
  mx=mxstruct_create();
  mxstruct_add_field_string(mx,"name",s->name);
  mxstruct_add_field_int(mx,"filter_type"         ,s->filter_type);
  mxstruct_add_field_int(mx,"rate_change"         ,s->rate_change);
  mxstruct_add_field_int(mx,"interp_rate"         ,s->interp_rate);
  mxstruct_add_field_int(mx,"decim_rate"          ,s->decim_rate);
  mxstruct_add_field_int(mx,"zero_pack_factor"    ,s->zero_pack_factor);
  if ( s->coeff != NULL ) {
    coeff_dims=(mwSize*)mxMalloc(1*sizeof(mwSize));
    coeff_dims[0]=(mwSize)s->num_coeffs;
    coeff=mxCreateNumericArray(1,coeff_dims,mxDOUBLE_CLASS,mxREAL);
    int coeff_i;
    for (coeff_i = 0, r=(double*)mxGetPr(coeff); coeff_i < s->num_coeffs;coeff_i++,r++) *r=s->coeff[coeff_i];
    mxstruct_add_field_mxarray(mx,"coeff",coeff);
  }
  mxstruct_add_field_int(mx,"coeff_padding"       ,s->coeff_padding);
  mxstruct_add_field_int(mx,"num_coeffs"          ,s->num_coeffs);
  mxstruct_add_field_int(mx,"coeff_sets"          ,s->coeff_sets);
  mxstruct_add_field_int(mx,"reloadable"          ,s->reloadable);
  mxstruct_add_field_int(mx,"is_halfband"         ,s->is_halfband);
  mxstruct_add_field_int(mx,"quantization"        ,s->quantization);
  mxstruct_add_field_int(mx,"coeff_width"         ,s->coeff_width);
  mxstruct_add_field_int(mx,"coeff_fract_width"   ,s->coeff_fract_width);
  mxstruct_add_field_int(mx,"chan_seq"            ,s->chan_seq);
  mxstruct_add_field_int(mx,"num_channels"        ,s->num_channels);
  mxstruct_add_field_int(mx,"init_pattern"        ,(int)s->init_pattern);
  mxstruct_add_field_int(mx,"num_paths"           ,s->num_paths);
  mxstruct_add_field_int(mx,"data_width"          ,s->data_width);
  mxstruct_add_field_int(mx,"data_fract_width"    ,s->data_fract_width);
  mxstruct_add_field_int(mx,"output_rounding_mode",s->output_rounding_mode);
  mxstruct_add_field_int(mx,"output_width"        ,s->output_width);
  mxstruct_add_field_int(mx,"output_fract_width"  ,s->output_fract_width);
  mxstruct_add_field_int(mx,"config_method"       ,s->config_method);

  return mx;
}

//Convert a mxArray to a xip_fir_v7_2_cnfg_packet structure
xip_fir_v7_2_cnfg_packet* mxarray_to_xip_fir_v7_2_cnfg_packet(const mxArray* mx, const char* name)
{
  xip_fir_v7_2_cnfg_packet* res=0;
  int i;

  res=(xip_fir_v7_2_cnfg_packet*)mxCalloc(1,sizeof(xip_fir_v7_2_cnfg_packet));

  if (mx)
  {
    if (!mxIsStruct(mx) || mxGetNumberOfElements(mx)!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_structure","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting scalar structure",name);

    for (i=0; i<mxGetNumberOfFields(mx); i++)
    {
      const char* fieldname=mxGetFieldNameByNumber(mx,i);
           if (!strcmp(fieldname,"chanpat")) res->chanpat=(xip_fir_v7_2_pattern)mxarray_to_int(mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"fsel" )) {
        res->fsel   =mxarray_to_xip_array_uint(mxGetFieldByNumber(mx,0,i),"fsel");
        // Need to check the dimension and modify to match the models expectation
        if (res->fsel->dim_size>2 || res->fsel->dim[0] != 1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s.%s; expecting 1 by n array",name,"fsel");
        res->fsel->dim_size=1; // Not modifying capacity
        res->fsel->dim[0] = res->fsel->dim[1];
      }
      else mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_fieldname","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; unexpected fieldname %s",name,fieldname);
    }
  }

  return res;
}

//Convert a mxArray to a xip_fir_v7_2_rld_packet structure
xip_fir_v7_2_rld_packet* mxarray_to_xip_fir_v7_2_rld_packet(const mxArray* mx, const char* name)
{
  xip_fir_v7_2_rld_packet* res=0;
  int i;

  res=(xip_fir_v7_2_rld_packet*)mxCalloc(1,sizeof(xip_fir_v7_2_rld_packet));

  if (mx)
  {
    if (!mxIsStruct(mx) || mxGetNumberOfElements(mx)!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_structure","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; expecting scalar structure",name);

    for (i=0; i<mxGetNumberOfFields(mx); i++)
    {
      const char* fieldname=mxGetFieldNameByNumber(mx,i);
           if (!strcmp(fieldname,"fsel"))   res->fsel  = mxarray_to_int(mxGetFieldByNumber(mx,0,i));
      else if (!strcmp(fieldname,"coeff" )) {
        res->coeff = mxarray_to_xip_array_real(mxGetFieldByNumber(mx,0,i),"coeff");
        // Need to check the dimension and modify to match the models expectation
        if (res->coeff->dim_size>2 || res->coeff->dim[0] != 1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_array","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s.%s; expecting 1 by n array",name,"coeff");
        res->coeff->dim_size=1; // Not modifying capacity
        res->coeff->dim[0] = res->coeff->dim[1];
      }
      else mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_fieldname","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid %s; unexpected fieldname %s",name,fieldname);
    }
  }

  return res;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Main body of MEX function

//Opcodes
typedef enum
{
  OP_GET_VERSION=0,
  OP_GET_DEFAULT_CONFIG=1,
  OP_CREATE=2,
  OP_DESTROY=3,
  OP_RESET=4,
  OP_CONFIG_SEND=5,
  OP_RELOAD_SEND=6,
  OP_FILTER=7,
  OP_GET_CONFIG=8,
} mex_opcode;

//Called at MEX exit
void mex_at_exit()
{
  //Destroy all outstanding model objects
  model_handle_vector_destroy(the_model_handle_vector);
}

//Pass messages to Matlab
void mex_message_handler(void* handle, int error, const char* msg)
{
  if (error)
  {
    mexErrMsgTxt(msg);
  }
  else
  {
    mexPrintf("%s\n",msg);
  }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_get_version(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  if (nrhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:get_version:Expecting zero input arguments");

  plhs[0]=mxCreateString(xip_fir_v7_2_get_version());
  if (!plhs[0])
  {
    mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_string","ERROR:fir_compiler_v7_2_bitacc_mex:get_version:Could not create string array");
    return;
  }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_get_default_config(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  xip_fir_v7_2_config* config;

  if (nrhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:get_default_config:Expecting zero input arguments");

  //Convert default config to mxArray structure
  config=mxarray_to_xip_fir_v7_2_config(0,"default_config");
  plhs[0]=xip_fir_v7_2_config_to_mxarray(config);
  mxFree(config);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_create(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  xip_fir_v7_2_config* config;
  model_handle mh;
  xip_fir_v7_2* model;

  if (nrhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:create:Expecting one input argument");

  //Get config
  config=mxarray_to_xip_fir_v7_2_config(prhs[0],"config");

  //Get next model_handle to use
  mh=model_handle_vector_get_next(the_model_handle_vector);

  //Now create the model
  model=xip_fir_v7_2_create(config,mex_message_handler,0);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_create","ERROR:fir_compiler_v7_2_bitacc_mex:create:Could not create model");

  //Register pointer and return handle
  model_handle_vector_set_structure(the_model_handle_vector,mh,model);
  plhs[0]=mxarray_create_scalar(mh);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_destroy(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;

  if (nrhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:destroy:Expecting one input argument");
  if (nlhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:destroy:Expecting zero output arguments");

  //Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);

  //Destroy object if it still exists
  xip_fir_v7_2_destroy(model);
  model_handle_vector_set_structure(the_model_handle_vector,mh,0);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_reset(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;

  if (nrhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:reset:Expecting one input argument");
  if (nlhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:reset:Expecting zero output arguments");

  //Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:reset:Invalid model pointer");

  xip_fir_v7_2_reset(model);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_config_send(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;
  xip_fir_v7_2_cnfg_packet* cnfg_packet=0;

  if (nrhs!=2) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:config_send:Expecting two input arguments");
  if (nlhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:config_send:Expecting zero output arguments");

  //Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:config_send:Invalid model pointer");

  cnfg_packet=mxarray_to_xip_fir_v7_2_cnfg_packet(prhs[1],"cnfg_packet");

  xip_fir_v7_2_config_send(model,cnfg_packet);
  mxFree(cnfg_packet);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_reload_send(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;
  xip_fir_v7_2_rld_packet* rld_packet=0;

  if (nrhs!=2) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:reload_send:Expecting two input arguments");
  if (nlhs!=0) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:reload_send:Expecting zero output arguments");

  //Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:reload_send:Invalid model pointer");

  rld_packet=mxarray_to_xip_fir_v7_2_rld_packet(prhs[1],"rld_packet");

  xip_fir_v7_2_reload_send(model,rld_packet);
  mxFree(rld_packet);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_filter(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;
  xip_array_real* data_in;
  xip_array_real* data_out;
  xip_array_complex* cmplx_data_out;
  xip_fir_v7_2_config config;

  if (nrhs!=2) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:data_send:Expecting two input arguments");
  if (nlhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:data_send:Expecting one output argument");

  // Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:data_send:Invalid model pointer");

  // Inputs
  data_in = mxarray_to_xip_array_real(prhs[1],"data_in");

  // Need to get config to determine if Hilbert filter
  xip_fir_v7_2_get_config(model,&config);

  // Register output array
  if ( config.filter_type == XIP_FIR_HILBERT ) {
    cmplx_data_out = xip_array_complex_create();
    xip_fir_v7_2_set_data_sink(model,0,cmplx_data_out);
  } else {
    data_out = xip_array_real_create();
    xip_fir_v7_2_set_data_sink(model,data_out,0);
  }

  // Run model
  xip_fir_v7_2_data_send(model,data_in);

  //Convert results to mxarray
  if ( config.filter_type == XIP_FIR_HILBERT ) {
    plhs[0]=xip_array_complex_to_mxarray(cmplx_data_out);
    xip_array_complex_destroy(cmplx_data_out);
  } else {
    plhs[0]=xip_array_real_to_mxarray(data_out);
    xip_array_real_destroy(data_out);
  }
  mxFree(data_in);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void do_get_config(mex_opcode opcode, int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  model_handle mh;
  xip_fir_v7_2* model;
  xip_fir_v7_2_config* config;

  config=(xip_fir_v7_2_config*)mxCalloc(1,sizeof(xip_fir_v7_2_config));

  if (nrhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_input","ERROR:fir_compiler_v7_2_bitacc_mex:get_config:Expecting one input argument");
  if (nlhs!=1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_output","ERROR:fir_compiler_v7_2_bitacc_mex:get_config:Expecting one output argument");

  // Get model_handle
  mh=mxarray_get_model_handle(prhs[0]);
  model=model_handle_vector_get_structure(the_model_handle_vector,mh);
  if (!model) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_model_handle","ERROR:fir_compiler_v7_2_bitacc_mex:do_get_config:Invalid model pointer");

  xip_fir_v7_2_get_config(model,config);

  //Convert results to mxarray
  plhs[0]=xip_fir_v7_2_config_to_mxarray(config);
  mxFree(config);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  double x;
  mex_opcode opcode;

  //First time initialisation
  if (!the_model_handle_vector)
  {
    //Register our exit function
    mexAtExit(mex_at_exit);

    //Ensure datatypes are equivalent
    if (sizeof(xip_uint)!=sizeof(unsigned int )) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_mex","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid MEX function; size of xip_uint and unsigned int types are incompatible");
    if (sizeof(xip_real)!=sizeof(double       )) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_mex","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid MEX function; size of xip_real and double types are incompatible");

    //Create model repository
    the_model_handle_vector=model_handle_vector_create();
  }

  //Consume and check opcode
  if (nrhs<1) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_opcode","ERROR:fir_compiler_v7_2_bitacc_mex:Missing opcode");
  if (!is_real_scalar(prhs[0])) mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_opcode","ERROR:fir_compiler_v7_2_bitacc_mex:opcode must be a real numeric scalar");
  x=mxGetScalar(prhs[0]);
  opcode=(mex_opcode)((int)x);
  nrhs--;
  prhs++;

  //Dispatch opcode
  switch (opcode)
  {
    //version=get_version()
    case OP_GET_VERSION:
    {
      do_get_version(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //config=get_default_config()
    case OP_GET_DEFAULT_CONFIG:
    {
      do_get_default_config(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //model=create(config)
    case OP_CREATE:
    {
      do_create(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //destroy(model)
    case OP_DESTROY:
    {
      do_destroy(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //reset(model)
    case OP_RESET:
    {
      do_reset(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //config_send(model,cnfg_packet)
    case OP_CONFIG_SEND:
    {
      do_config_send(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //reload_send(model,rld_packet)
    case OP_RELOAD_SEND:
    {
      do_reload_send(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //[data_out]=filter(model,data_in)
    case OP_FILTER:
    {
      do_filter(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    //[config]=get_config(model)
    case OP_GET_CONFIG:
    {
      do_get_config(opcode,nlhs,plhs,nrhs,prhs);
    } break;

    default:
    {
      mexErrMsgIdAndTxt("fir_compiler_v7_2_bitacc_mex:bad_opcode","ERROR:fir_compiler_v7_2_bitacc_mex:Invalid opcode %d",opcode);
    } break;
  }
}
