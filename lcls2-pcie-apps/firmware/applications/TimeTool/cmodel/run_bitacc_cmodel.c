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
// Purpose : Smoke test program for bit accurate C model

#include <stdio.h>
#include "fir_compiler_v7_2_bitacc_cmodel.h"

//---------------------------------------------------------------------------------------------------------------------
// Example message handler
static void msg_print(void* handle, int error, const char* msg)
{
  printf("%s\n",msg);
}

//---------------------------------------------------------------------------------------------------------------------
//Print a xip_array_real
void print_array_real(const xip_array_real* x)
{
  putchar('[');
  if (x && x->data && x->data_size)
  {
    const xip_real* p=x->data;
    const xip_real* q=x->data;
    const xip_real* r=x->data+x->data_size;
    while (q!=r)
    {
      if (q!=p) putchar(' ');
      printf("%g",*q);
      q++;
    }
  }
  putchar(']');
  putchar('\n');
}
//---------------------------------------------------------------------------------------------------------------------
// Print a xip_array_complex
void print_array_complex(const xip_array_complex* x)
{
  putchar('[');
  if (x && x->data && x->data_size)
  {
    const xip_complex* p=x->data;
    const xip_complex* q=x->data;
    const xip_complex* r=x->data+x->data_size;
    while (q!=r)
    {
      if (q!=p) putchar(' ');
           if (!q->re    ) printf("%gi"   ,q->im      );  // Bi
      else if ( q->im<0.0) printf("%g%gi" ,q->re,q->im);  // A-Bi
      else if ( q->im>0.0) printf("%g+%gi",q->re,q->im);  // A+Bi
      else                 printf("%g"    ,q->re      );  // A
      q++;
    }
  }
  putchar(']');
  putchar('\n');
}
//---------------------------------------------------------------------------------------------------------------------
// Print a xip_array_mpz
void print_array_mpz(const xip_array_mpz* x)
{
  putchar('[');
  if (x && x->data && x->data_size)
  {
    int i;
    for(i=0;i<x->data_size;i++) {
      if (i!=x->data_size+1) putchar(' ');
      mpz_out_str(0,10,x->data[i]);
    }
  }
  putchar(']');
  putchar('\n');
}
//---------------------------------------------------------------------------------------------------------------------
// Print a xip_array_mpz_complex
void print_array_mpz_complex(const xip_array_mpz_complex* x)
{
  putchar('[');
  if (x && x->data && x->data_size)
  {
    const xip_mpz_complex* p=x->data;
    const xip_mpz_complex* q=x->data;
    const xip_mpz_complex* r=x->data+x->data_size;
    while (q!=r)
    {
      if (q!=p) putchar(' ');
      if (!q->re    ) {
        mpz_out_str(0,10,q->im);
        printf("i");
      }
      else if ( mpz_sgn(q->im)<0.0) {
        mpz_out_str(0,10,q->re);
        mpz_out_str(0,10,q->im);
        printf("i");
      }
      else if ( mpz_sgn(q->im)>0.0) {
        mpz_out_str(0,10,q->re);
        printf("+");
        mpz_out_str(0,10,q->im);
        printf("i");
      }
      else
        mpz_out_str(0,10,q->re);
      q++;
    }
  }
  putchar(']');
  putchar('\n');
}

//----------------------------------------------------------------------------------------------------------------------
// Fill data array with a scaled impulse. Assumes 3-D array.
int create_impulse(xip_array_real* x) {
  int path;
  int chan;
  int i;
  for (path = 0; path < x->dim[0];path++) {
    for (chan = 0; chan < x->dim[1];chan++) {
       //xip_fir_v7_2_xip_array_real_set_chan(x,(double)((path+1)*(chan+1)),path,chan,0,P_BASIC);
       xip_fir_v7_2_xip_array_real_set_chan(x,(double)(2),path,chan,0,P_BASIC);
       for (i = 1; i < x->dim[2];i++) {
         xip_fir_v7_2_xip_array_real_set_chan(x,0,path,chan,i,P_BASIC);
       }
    }
  }
  return 0;
}

//----------------------------------------------------------------------------------------------------------------------
// String arrays used by the print_config funtion
const char *filt_desc[5] = { "SINGLE RATE", "INTERPOLATION", "DECIMATION", "HILBERT", "INTERPOLATED" };
const char *seq_desc[2] = { "Basic" , "Advanced" };

//----------------------------------------------------------------------------------------------------------------------
// Print a summary of a filter configuration
int print_config(const xip_fir_v7_2_config* cfg) {
  printf("Configuration of %s:\n",cfg->name);
  printf("\tFilter       : ");
  if ( cfg->filter_type == XIP_FIR_SINGLE_RATE || cfg->filter_type == XIP_FIR_HILBERT ) {
    printf("%s\n",filt_desc[cfg->filter_type]);
  } else if ( cfg->filter_type == XIP_FIR_INTERPOLATED ) {
    printf("%s by %d\n",filt_desc[cfg->filter_type],cfg->zero_pack_factor);
  } else {
    printf("%s up by %d down by %d\n",filt_desc[cfg->filter_type],cfg->interp_rate,cfg->decim_rate);
  }
  printf("\tCoefficients : %d ",cfg->coeff_sets);
  if ( cfg->is_halfband ) {
    printf("Halfband ");
  }
  if (cfg->reloadable) {
    printf("Reloadable ");
  }
  printf("coefficient set(s) of %d taps\n",cfg->num_coeffs);
  printf("\tData         : %d path(s) of %d %s channel(s)\n",cfg->num_paths,cfg->num_channels,seq_desc[cfg->chan_seq]);

  return 0;
}

//----------------------------------------------------------------------------------------------------------------------
int main () {

  const char* ver_str = xip_fir_v7_2_get_version();
  printf("-------------------------------------------------------------------------------\n");
  printf("FIR Compiler C Model version: %s\n",ver_str);

  printf("-------------------------------------------------------------------------------\n");
  printf("Default core....\n");
  printf("-------------------------------------------------------------------------------\n");

  // Define filter configuration
  xip_fir_v7_2_config fir_default_cnfg;
  xip_fir_v7_2_default_config(&fir_default_cnfg);

  fir_default_cnfg.name          = "fir_default";
  print_config(&fir_default_cnfg);

  //Create filter instances
  xip_fir_v7_2* fir_default = xip_fir_v7_2_create(&fir_default_cnfg,&msg_print,0);
  if (!fir_default) {
    printf("Error creating instance\n",fir_default_cnfg.name);
    return -1;
  } else {
    printf("Created instance\n",fir_default_cnfg.name);
  }

  // Create input data packet
  xip_array_real* din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_default_cnfg.num_paths;
  din->dim[1] = fir_default_cnfg.num_channels;
  din->dim[2] = fir_default_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output data packet
  //  - Automatically sized using xip_fir_v7_2_calc_size
  xip_array_real* fir_default_out = xip_array_real_create();
  xip_array_real_reserve_dim(fir_default_out,3);
  fir_default_out->dim_size = 3; // 3D array
  if(xip_fir_v7_2_calc_size(fir_default,din,fir_default_out,0)== XIP_STATUS_OK) {
    printf("Calculated output data size\n");
    if (xip_array_real_reserve_data(fir_default_out,fir_default_out->data_size) == XIP_STATUS_OK) {
      printf("Reserved data\n");
    } else {
      printf("Unable to reserve data!\n");
      return -1;
    }
  } else {
    printf("Unable to calculate output date size\n");
    return -1;
  }

  // Populate data in with an impulse
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_default,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Retrieve filtered data
  if ( xip_fir_v7_2_data_get(fir_default,fir_default_out,0)== XIP_STATUS_OK) {
    printf("Fetched result: ");
    print_array_real(fir_default_out);
  } else {
    printf("Error getting data\n");
    return -1;
  }

  printf("\nTest reset....\n");

  // Send the same data but only fetch first half of the output, reset and then repeat but fetch all the data

  if ( xip_fir_v7_2_data_send(fir_default,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Create a new output data packet
  xip_array_real* fir_default_out2 = xip_array_real_create();
  xip_array_real_reserve_dim(fir_default_out2,3);
  fir_default_out2->dim_size = 3; // 3D array
  if(xip_fir_v7_2_calc_size(fir_default,din,fir_default_out2,0)== XIP_STATUS_OK) {
    printf("Calculated output data size\n");
    // Reduce output packet to only read half the generated data
    fir_default_out2->dim[2]    = fir_default_out2->dim[2] / 2;
    fir_default_out2->data_size = fir_default_out2->dim[1] * fir_default_out2->dim[1] * fir_default_out2->dim[2];
    if (xip_array_real_reserve_data(fir_default_out2,fir_default_out2->data_size) == XIP_STATUS_OK) {
      printf("Reserved data\n");
    } else {
      printf("Unable to reserve data!\n");
      return -1;
    }
  } else {
    printf("Unable to calculate output date size\n");
    return -1;
  }

  // Get reduced output data packet
  if ( xip_fir_v7_2_data_get(fir_default,fir_default_out2,0)== XIP_STATUS_OK) {
    printf("Fetched result: ");
    print_array_real(fir_default_out2);
  } else {
    printf("Error getting data\n");
    return -1;
  }

  // Apply reset to model
  printf("Apply reset\n");
  if ( xip_fir_v7_2_reset(fir_default) != XIP_STATUS_OK) {
    printf("Error applying reset\n");
    return -1;
  }

  // Try fetching some data
  if ( xip_fir_v7_2_data_get(fir_default,fir_default_out2,0)== XIP_STATUS_OK) {
    if ( fir_default_out2->dim[2] == 0 ) {
      printf("Zero data fetched\n");
    } else {
      printf("Error, fetched result: ");
      print_array_real(fir_default_out2);
      return -1;
    }
  } else {
    printf("Error getting data\n");
    return -1;
  }

  // Send input data again and read output
  if ( xip_fir_v7_2_data_send(fir_default,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
  } else {
    printf("Error sending data\n");
    return -1;
  }
  print_array_real(din);

  // Read full output packet size
  fir_default_out2->dim[2]    = fir_default_cnfg.num_coeffs;
  fir_default_out2->data_size = fir_default_out2->dim[1] * fir_default_out2->dim[1] * fir_default_out2->dim[2];
  if (xip_array_real_reserve_data(fir_default_out2,fir_default_out2->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  if ( xip_fir_v7_2_data_get(fir_default,fir_default_out2,0)== XIP_STATUS_OK) {
    printf("Fetched result: ");
  } else {
    printf("Error getting data\n");
    return -1;
  }
  print_array_real(fir_default_out2);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_real_destroy(fir_default_out);
  xip_array_real_destroy(fir_default_out2);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_default);

  printf("-------------------------------------------------------------------------------\n");
  printf("Config and reload example....\n");
  printf("-------------------------------------------------------------------------------\n");

  // Define filter configurations
  xip_fir_v7_2_config fir1_cnfg,fir2_cnfg;
  xip_fir_v7_2_default_config(&fir1_cnfg);

  fir1_cnfg.name              = "fir1";
  fir1_cnfg.num_channels      = 3;
  fir1_cnfg.coeff_sets        = 2;
  fir1_cnfg.num_coeffs        = 10;
  fir1_cnfg.reloadable        = 1;
  fir1_cnfg.config_method     = XIP_FIR_CONFIG_BY_CHANNEL;

  xip_fir_v7_2_default_config(&fir2_cnfg);
  fir2_cnfg.name              = "fir2";
  fir2_cnfg.num_channels      = 3;
  const double fir2_coeffs[2] = { 0,2 };
  fir2_cnfg.coeff             = &fir2_coeffs[0];
  fir2_cnfg.num_coeffs        = 2;

  //Create filter instances
  print_config(&fir1_cnfg);
  xip_fir_v7_2* fir1 = xip_fir_v7_2_create(&fir1_cnfg,&msg_print,0);
  if (!fir1) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  print_config(&fir2_cnfg);
  xip_fir_v7_2* fir2 = xip_fir_v7_2_create(&fir2_cnfg,&msg_print,0);
  if (!fir2) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir1_cnfg.num_paths; // Number of paths
  din->dim[1] = fir1_cnfg.num_channels; // Number of channels
  din->dim[2] = fir1_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output data packet
  //  - Automatically sized using xip_fir_v7_2_calc_size
  xip_array_real* fir1_out = xip_array_real_create();
  xip_array_real_reserve_dim(fir1_out,3);
  fir1_out->dim_size = 3; // 3D array
  if(xip_fir_v7_2_calc_size(fir1,din,fir1_out,0)== XIP_STATUS_OK) {
    printf("Calculated output data size\n");
    if (xip_array_real_reserve_data(fir1_out,fir1_out->data_size) == XIP_STATUS_OK) {
      printf("Reserved data\n");
    } else {
      printf("Unable to reserve data!\n");
      return -1;
    }
  } else {
    printf("Unable to calculate output date size\n");
    return -1;
  }

  // Create config packet
  xip_array_uint* fsel = xip_array_uint_create();
  xip_array_uint_reserve_dim(fsel,1);
  fsel->dim_size = 1;
  fsel->dim[0] = fir1_cnfg.num_channels;
  fsel->data_size = fsel->dim[0];
  if (xip_array_uint_reserve_data(fsel,fsel->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }
  xip_fir_v7_2_cnfg_packet cnfg;
  cnfg.fsel = fsel;
  cnfg.fsel->data[0] = 0;
  cnfg.fsel->data[1] = 1; // Set 2nd channel to use 2nd coefficient set
  cnfg.fsel->data[2] = 4; // Set invalid filter set to demonstrate error generated by model

  // Create reload packet
  xip_real new_coeffs[10] = {1,2,3,4,5,6,7,8,9,10};

  xip_fir_v7_2_rld_packet rld;
  rld.fsel = 0;
  rld.coeff = xip_array_real_create();
  xip_array_real_reserve_dim(rld.coeff,1);
  rld.coeff->dim_size=1;
  rld.coeff->dim[0]=10;
  rld.coeff->data_size = rld.coeff->dim[0];
  if (xip_array_real_reserve_data(rld.coeff,rld.coeff->data_size) == XIP_STATUS_OK) {
    printf("Reserved coeff\n");
  } else {
    printf("Unable to reserve coeff!\n");
    return -1;
  }
  // Copy coefficients into reload packet
  int coeff_i;
  for (coeff_i=0;coeff_i<10;coeff_i++) {
    rld.coeff->data[coeff_i] = new_coeffs[coeff_i];
  }

  printf("Test default configuration of %s......\n",fir1_cnfg.name);

  // Define input data
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir1,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Retrieve filtered data
  if ( xip_fir_v7_2_data_get(fir1,fir1_out,0)== XIP_STATUS_OK) {
    printf("Fetched result: ");
    print_array_real(fir1_out);
  } else {
    printf("Error getting data\n");
    return -1;
  }

  // Connect the 1st filter's output to the 2nd filters input
  printf("\nConnect %s and %s and test in a chain......\n",fir1_cnfg.name,fir2_cnfg.name);

  // Register fir1_out array as data sink of fir1
  xip_fir_v7_2_set_data_sink(fir1,fir1_out,0);

  // Set xip_fir_v7_2_data_send_handler of fir2 as the data handler of fir1's output array
  xip_fir_v7_2_set_data_handler(fir1,&xip_fir_v7_2_data_send_handler,fir2,0);

  // Create output data structure for 2nd filter and use as a data sink, i.e. filter data does not need to be "pulled" using data_get
  xip_array_real* fir2_out = xip_array_real_create();
  xip_fir_v7_2_set_data_sink(fir2,fir2_out,0);

  printf("Update configuration of %s\n",fir1_cnfg.name);
  // Send config data
  if ( xip_fir_v7_2_config_send(fir1,&cnfg)== XIP_STATUS_OK) {
    printf("Sent config packet\n");
  } else {
    printf("Error sending config packet\n");
    return -1;
  }

  printf("Reload new coefficients for set %d of %s\n",rld.fsel,fir1_cnfg.name);
  // Send reload data
  if ( xip_fir_v7_2_reload_send(fir1,&rld)== XIP_STATUS_OK) {
    printf("Sent reload packet\n");
  } else {
    printf("Error sending reload packet\n");
    return -1;
  }

  // Send input data and filter
  //  - Once fir1 has processed the input data and generated an output fir2 will be called via the registered data
  //    handler and will in turn generate an output
  if ( xip_fir_v7_2_data_send(fir1,din)== XIP_STATUS_OK) {
    printf("Sent data: ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Filter output written directly to output data structures
  printf("%s output: ",fir1_cnfg.name);
  print_array_real(fir1_out);
  printf("%s output: ",fir2_cnfg.name);
  print_array_real(fir2_out);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_real_destroy(fir1_out);
  xip_array_real_destroy(fir2_out);
  xip_array_uint_destroy(fsel);

  //De-allocate reload coeff array
  xip_array_real_destroy(rld.coeff);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir1);
  xip_fir_v7_2_destroy(fir2);

  printf("-------------------------------------------------------------------------------\n");
  printf("Rate change exmaple....\n");
  printf("-------------------------------------------------------------------------------\n");

  xip_fir_v7_2_config fir_up_cnfg, fir_sr_cnfg, fir_down_cnfg;

  xip_fir_v7_2_default_config(&fir_up_cnfg);
  fir_up_cnfg.name              = "fir_up";
  fir_up_cnfg.filter_type       = XIP_FIR_INTERPOLATION;
  fir_up_cnfg.rate_change       = XIP_FIR_INTEGER_RATE;
  fir_up_cnfg.interp_rate       = 4;

  xip_fir_v7_2_default_config(&fir_sr_cnfg);
  fir_sr_cnfg.name              = "fir_sr";
  const double fir_sr_coeffs[2] = { 0,2 };
  fir_sr_cnfg.coeff             = &fir_sr_coeffs[0];
  fir_sr_cnfg.num_coeffs        = 2;

  xip_fir_v7_2_default_config(&fir_down_cnfg);
  fir_down_cnfg.name            = "fir_down";
  fir_down_cnfg.filter_type       = XIP_FIR_DECIMATION;
  fir_down_cnfg.rate_change       = XIP_FIR_INTEGER_RATE;
  fir_down_cnfg.decim_rate       = 3;

  //Create filter instances
  print_config(&fir_up_cnfg);
  xip_fir_v7_2* fir_up = xip_fir_v7_2_create(&fir_up_cnfg,&msg_print,0);
  if (!fir_up) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  print_config(&fir_sr_cnfg);
  xip_fir_v7_2* fir_sr = xip_fir_v7_2_create(&fir_sr_cnfg,&msg_print,0);
  if (!fir_sr) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  print_config(&fir_down_cnfg);
  xip_fir_v7_2* fir_down = xip_fir_v7_2_create(&fir_down_cnfg,&msg_print,0);
  if (!fir_down) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_up_cnfg.num_paths; // Number of paths
  din->dim[1] = fir_up_cnfg.num_channels; // Number of channels
  din->dim[2] = fir_up_cnfg.num_coeffs/fir_up_cnfg.interp_rate+1; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output arrays and register as data sinks
  xip_array_real* fir_up_out = xip_array_real_create();
  xip_fir_v7_2_set_data_sink(fir_up,fir_up_out,0);
  xip_array_real* fir_sr_out = xip_array_real_create();
  xip_fir_v7_2_set_data_sink(fir_sr,fir_sr_out,0);
  xip_array_real* fir_down_out = xip_array_real_create();
  xip_fir_v7_2_set_data_sink(fir_down,fir_down_out,0);

  // Connect filters in a single chain
  xip_fir_v7_2_set_data_handler(fir_up,&xip_fir_v7_2_data_send_handler,fir_sr,0);
  xip_fir_v7_2_set_data_handler(fir_sr,&xip_fir_v7_2_data_send_handler,fir_down,0);

  // Define input data
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_up,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Filter outputs written directly to output data structures
  printf("%s output: ",fir_up_cnfg.name);
  print_array_real(fir_up_out);
  printf("%s output: ",fir_sr_cnfg.name);
  print_array_real(fir_sr_out);
  printf("%s output: ",fir_down_cnfg.name);
  print_array_real(fir_down_out);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_real_destroy(fir_up_out);
  xip_array_real_destroy(fir_sr_out);
  xip_array_real_destroy(fir_down_out);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_up);
  xip_fir_v7_2_destroy(fir_sr);
  xip_fir_v7_2_destroy(fir_down);

  printf("-------------------------------------------------------------------------------\n");
  printf("Hilbert filter....\n");
  printf("-------------------------------------------------------------------------------\n");

  xip_fir_v7_2_config fir_hilb_cnfg;
  xip_fir_v7_2_default_config(&fir_hilb_cnfg);

  fir_hilb_cnfg.name        = "fir_hilb";
  fir_hilb_cnfg.filter_type = XIP_FIR_HILBERT;

  //Create filter instances
  print_config(&fir_hilb_cnfg);
  xip_fir_v7_2* fir_hilb = xip_fir_v7_2_create(&fir_hilb_cnfg,&msg_print,0);
  if (!fir_hilb) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_hilb_cnfg.num_paths; // Number of paths
  din->dim[1] = fir_hilb_cnfg.num_channels; // Number of channels
  din->dim[2] = fir_hilb_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output data packet structure
  //  - Will be used as a data sink. The model will allocate appropriate memory
  xip_array_complex* fir_hilb_out = xip_array_complex_create();
  xip_fir_v7_2_set_data_sink(fir_hilb,0,fir_hilb_out);

  // Define input data, scaled impulse on each channel
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_hilb,din)== XIP_STATUS_OK) {
    printf("Sent data: ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Output data is written directly to registered data sink array
  printf("%s output: ",fir_hilb_cnfg.name);
  print_array_complex(fir_hilb_out);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_complex_destroy(fir_hilb_out);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_hilb);

  printf("-------------------------------------------------------------------------------\n");
  printf("MPZ based filter....\n");
  printf("-------------------------------------------------------------------------------\n");

  xip_fir_v7_2_config fir_mpz_cnfg;
  xip_fir_v7_2_default_config(&fir_mpz_cnfg);

  fir_mpz_cnfg.name          = "fir_mpz";
  fir_mpz_cnfg.num_channels  = 3;
  fir_mpz_cnfg.data_width    = 45;
  fir_mpz_cnfg.coeff_width   = 32;

  //Create filter instances
  print_config(&fir_mpz_cnfg);
  xip_fir_v7_2* fir_mpz = xip_fir_v7_2_create(&fir_mpz_cnfg,&msg_print,0);
  if (!fir_mpz) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_mpz_cnfg.num_paths; // Number of paths
  din->dim[1] = fir_mpz_cnfg.num_channels; // Number of channels
  din->dim[2] = fir_mpz_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output data packet (MPZ data type)
  //  - Automatically sized using xip_fir_v7_2_calc_size
  xip_array_mpz* fir_mpz_out = xip_array_mpz_create();
  xip_array_mpz_reserve_dim(fir_mpz_out,3);
  fir_mpz_out->dim_size = 3; // 3D array
  if(xip_fir_v7_2_calc_size_mpz(fir_mpz,din,fir_mpz_out,0)== XIP_STATUS_OK) {
    printf("Calculated output data size\n");
    if (xip_array_mpz_reserve_data(fir_mpz_out,fir_mpz_out->data_size) == XIP_STATUS_OK) {
      printf("Reserved data\n");
    } else {
      printf("Unable to reserve data!\n");
      return -1;
    }
  } else {
    printf("Unable to calculate output date size\n");
    return -1;
  }

  // Define input data, scaled impulse on each channel
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_mpz,din)== XIP_STATUS_OK) {
    printf("Sent data     :");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Retrieve filtered data
  if ( xip_fir_v7_2_data_get_mpz(fir_mpz,fir_mpz_out,0)== XIP_STATUS_OK) {
    printf("Fetched result:\n");
    print_array_mpz(fir_mpz_out);
  } else {
    printf("Error getting data\n");
    return -1;
  }

  // Use helper function to extract the output value of particular channel
  xip_mpz chan_val;
  mpz_init(chan_val);
  xip_fir_v7_2_xip_array_mpz_get_chan(fir_mpz_out,&chan_val,0,1,2,P_BASIC);
  printf("Output value of index %d path %d channel %d: ",2,0,1);
  mpz_out_str(0,10,chan_val);
  printf("\n");
  mpz_clear(chan_val);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_mpz_destroy(fir_mpz_out);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_mpz);

  printf("-------------------------------------------------------------------------------\n");
  printf("MPZ based Hilbert filter....\n");
  printf("-------------------------------------------------------------------------------\n");

  xip_fir_v7_2_config fir_mpz_hilb_cnfg;
  xip_fir_v7_2_default_config(&fir_mpz_hilb_cnfg);

  fir_mpz_hilb_cnfg.name          = "fir_mpz_hilb";
  fir_mpz_hilb_cnfg.filter_type   = XIP_FIR_HILBERT;
  fir_mpz_hilb_cnfg.data_width    = 45;
  fir_mpz_hilb_cnfg.coeff_width   = 32;

  //Create filter instances
  print_config(&fir_mpz_hilb_cnfg);
  xip_fir_v7_2* fir_mpz_hilb = xip_fir_v7_2_create(&fir_mpz_hilb_cnfg,&msg_print,0);
  if (!fir_mpz_hilb) {
    printf("Error creating instance\n");
    return -1;
  } else {
    printf("Created instance\n");
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_mpz_hilb_cnfg.num_paths; // Number of paths
  din->dim[1] = fir_mpz_hilb_cnfg.num_channels; // Number of channels
  din->dim[2] = fir_mpz_hilb_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output array and register as data sink
  xip_array_mpz_complex* fir_mpz_hilb_out = xip_array_mpz_complex_create();
  xip_fir_v7_2_set_data_sink_mpz(fir_mpz_hilb,0,fir_mpz_hilb_out);

  // Define input data, scaled impulse on each channel
  printf("Create impulse\n");
  create_impulse(din);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_mpz_hilb,din)== XIP_STATUS_OK) {
    printf("Sent data: ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Output data is written directly to registered data sink array
  printf("%s output: ",fir_mpz_hilb_cnfg.name);
  print_array_mpz_complex(fir_mpz_hilb_out);

  // Use helper function to extract the output value of particular channel
  xip_mpz_complex chan_val_cmplex;
  mpz_init(chan_val_cmplex.re);
  mpz_init(chan_val_cmplex.im);
  xip_fir_v7_2_xip_array_mpz_complex_get_chan(fir_mpz_hilb_out,&chan_val_cmplex,0,0,2,P_BASIC);
  printf("Output imaginary value of index %d path %d channel %d: ",2,0,0);
  mpz_out_str(0,10,chan_val_cmplex.im);
  printf("\n");
  mpz_clear(chan_val_cmplex.re);
  mpz_clear(chan_val_cmplex.im);

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_mpz_complex_destroy(fir_mpz_hilb_out);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_mpz_hilb);

  printf("-------------------------------------------------------------------------------\n");
  printf("Advanced channel configuration....\n");
  printf("-------------------------------------------------------------------------------\n");

  // Define filter configurations
  xip_fir_v7_2_config fir_adv_cnfg;
  xip_fir_v7_2_default_config(&fir_adv_cnfg);

  fir_adv_cnfg.name          = "fir_adv";
  fir_adv_cnfg.chan_seq      = XIP_FIR_ADVANCED_CHAN_SEQ;
  fir_adv_cnfg.num_channels  = 4;
  fir_adv_cnfg.init_pattern  = P4_3;

  //Create filter instances
  print_config(&fir_adv_cnfg);
  xip_fir_v7_2* fir_adv = xip_fir_v7_2_create(&fir_adv_cnfg,&msg_print,0);
  if (!fir_adv) {
    printf("Error creating instance\n",fir_adv_cnfg.name);
    return -1;
  } else {
    printf("Created instance\n",fir_adv_cnfg.name);
  }

  // Create input data packet
  din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3);
  din->dim_size = 3; // 3D array
  din->dim[0] = fir_adv_cnfg.num_paths; // Number of paths
  din->dim[1] = fir_adv_cnfg.num_channels; // Number of channels
  din->dim[2] = fir_adv_cnfg.num_coeffs; // vectors in a single packet
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    printf("Reserved data\n");
  } else {
    printf("Unable to reserve data!\n");
    return -1;
  }

  // Create output data packet
  //  - Automatically sized using xip_fir_v7_2_calc_size
  xip_array_real* fir_adv_out = xip_array_real_create();
  xip_array_real_reserve_dim(fir_adv_out,3);
  fir_adv_out->dim_size = 3; // 3D array
  if(xip_fir_v7_2_calc_size(fir_adv,din,fir_adv_out,0)== XIP_STATUS_OK) {
    printf("Calculated output data size\n");
    if (xip_array_real_reserve_data(fir_adv_out,fir_adv_out->data_size) == XIP_STATUS_OK) {
      printf("Reserved data\n");
    } else {
      printf("Unable to reserve data!\n");
      return -1;
    }
  } else {
    printf("Unable to calculate output date size\n");
    return -1;
  }

  // Populate input data with an impulse
  printf("Create impulse\n");
  // First set all the locations to zero
  int path;
  int chan;
  int i;
  for (path = 0; path < din->dim[0];path++) {
    for (chan = 0; chan < din->dim[1];chan++) {
       for (i = 0; i < din->dim[2];i++) {
         xip_fir_v7_2_xip_array_real_set_chan(din,0,path,chan,i,P_BASIC);
       }
    }
  }
  // Selected pattern P4_3 for fir_adv initialization pattern which contains 3 channels; 1 x fs/2 and 2 x fs/4
  // Set a scaled impulse for each channel
  xip_fir_v7_2_xip_array_real_set_chan(din,1.0,0,0,0,P4_3);
  xip_fir_v7_2_xip_array_real_set_chan(din,2.0,0,1,0,P4_3);
  xip_fir_v7_2_xip_array_real_set_chan(din,3.0,0,2,0,P4_3);

  // Send input data and filter
  if ( xip_fir_v7_2_data_send(fir_adv,din)== XIP_STATUS_OK) {
    printf("Sent data     : ");
    print_array_real(din);
  } else {
    printf("Error sending data\n");
    return -1;
  }

  // Retrieve filtered data
  if ( xip_fir_v7_2_data_get(fir_adv,fir_adv_out,0)== XIP_STATUS_OK) {
    printf("Fetched result: ");
    print_array_real(fir_adv_out);
  } else {
    printf("Error getting data\n");
    return -1;
  }

  //De-allocate data
  xip_array_real_destroy(din);
  xip_array_real_destroy(fir_adv_out);

  //De-allocate fir instances
  xip_fir_v7_2_destroy(fir_adv);

  printf("...End\n");
}
