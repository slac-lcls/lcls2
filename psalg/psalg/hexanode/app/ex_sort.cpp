
#define NUM_CHANNELS 32
#define NUM_IONS 16

#include "math.h"
#include "hexanode_proxy/resort64c.h"
#include <stdio.h>
#include "stdlib.h"
#include "memory.h"
#include <iostream> // for cout, puts etc.
#include <time.h>

#include "hexanode/LMF_IO.h"
#include "hexanode/SortUtils.h"

double	offset_sum_u, offset_sum_v, offset_sum_w;
double	w_offset, pos_offset_x, pos_offset_y;
int	command;
sort_class * sorter;

//-----------------------------


//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[], char* envp[])
//////////////////////////////////////////////////////////////////////////////////////////
{
	printf("syntax: sort_LMF filename\n");
	printf("        This file will be sorted and\n");
	printf("        a new file will be written.\n\n");

	if (argc < 2)	{
		printf("Please provide a filename.\n");
		return 0;
	}
	if (argc > 2)	{
		printf("too many arguments\n");
		return 0;
	}

	double	tdc_ns[NUM_CHANNELS][NUM_IONS];
	__int32	number_of_hits[NUM_CHANNELS];
	command = -1;

	// The "command"-value is set in the first line of "sorter.txt"
	// 0 = only convert to new file format
	// 1 = sort and write new file 
	// 2 = calibrate fv, fw, w_offset
	// 3 = create calibration table files


	// create the sorter:
	sorter = new sort_class();

	if (!read_config_file("sorter.txt", sorter, command, 
                              offset_sum_u, offset_sum_v, offset_sum_w, 
                              w_offset, pos_offset_x, pos_offset_y)) {
		if (sorter) {delete sorter; sorter = 0;}
		return 0;
	}
	if (sorter) {
	if (sorter->use_sum_correction || sorter->use_pos_correction) {
		read_calibration_tables("calibration_table.txt", sorter);
	}
	}
	
	if (command == -1) {
		printf("no config file was read. Nothing to do.\n");
		if (sorter) {delete sorter; sorter = 0;}
		return 0;
	}
	
	int Cu1 = sorter->Cu1;
	int Cu2 = sorter->Cu2;
	int Cv1 = sorter->Cv1;
	int Cv2 = sorter->Cv2;
	int Cw1 = sorter->Cw1;
	int Cw2 = sorter->Cw2;
	int Cmcp= sorter->Cmcp;

        printf("Numeration of channels - u1:%i  u2:%i  v1:%i  v2:%i  w1:%i  w2:%i  mcp:%i\n",
	       Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp);

	char error_text[512];
	char LMF_Filename[500];
	sprintf(LMF_Filename, (char*)argv[1]);

	LMF_IO* LMF = new LMF_IO(NUM_CHANNELS, NUM_IONS);

	if (!LMF->OpenInputLMF(LMF_Filename)) {	  
	   //std::cout << "Can't open file: " << LMF_Filename << '\n';
	   printf("Can't open file: %s\n",LMF_Filename);
	   return false;
	}

	printf("LMF starttime: %s",ctime(&LMF->Starttime));
	printf("LMF stoptime : %s",ctime(&LMF->Stoptime));

	//return 0;

	// initialization of the sorter:
	printf("init sorter... ");
	sorter->TDC_resolution_ns = 0.025;
	sorter->tdc_array_row_length = NUM_IONS;
	sorter->count = (__int32*)number_of_hits;
	sorter->tdc_pointer = &tdc_ns[0][0];
	if (command >= 2) {
		sorter->create_scalefactors_calibrator(true, sorter->runtime_u, 
                                                             sorter->runtime_v, 
                                                             sorter->runtime_w, 0.78, 
                                                             sorter->fu, sorter->fv, sorter->fw); 
	}
 	int error_code = sorter->init_after_setting_parameters();
	if (error_code) {
		printf("sorter could not be initialized\n");
		sorter->get_error_text(error_code, 512, error_text);
		printf("Error %i: %s\n",error_code, error_text);
		return 0;
	}
	printf("ok\n");
	

	printf("LMF->tdcresolution %f\n", LMF->tdcresolution);
		

	while (my_kbhit()); // empty keyboard buffer
	unsigned __int64 event_counter = 0;

        //time_t t_sec;
        //t_sec = time (NULL);
        struct timespec start, stop;
        clock_gettime(CLOCK_REALTIME, &start);

	printf("reading event data... \n");
	while (LMF->ReadNextEvent()) {

	        unsigned int event_number = LMF->GetEventNumber();
		event_counter++;

		if (event_number%10000 == 0) printf("Event number: %6i\n", event_number);
		//if (event_counter%10000 == 0) {if (my_kbhit()) break;}

                //==================================
		//#error	TODO by end user:
		// Here you must read in a data block from your data file
		// and fill the array tdc_ns[][] and number_of_hits[]

		LMF->GetNumberOfHitsArray(&number_of_hits[0]);
		if (LMF->errorflag) {
			LMF->GetErrorText(LMF->errorflag, error_text);
			printf("%s",error_text);
			return false;
		}

                LMF->GetTDCDataArray(&tdc_ns[0][0]);
		if (LMF->errorflag) {
			LMF->GetErrorText(LMF->errorflag,error_text);
			printf("%s",error_text);
			return false;
		}

		// apply conversion to ns
		if (true) {
       		  for(unsigned int i=0; i<LMF->GetNumberOfChannels(); i++)
		    for(int j=0; j<number_of_hits[i]; j++)
                      tdc_ns[i][j] *= LMF->tdcresolution;	
		}		

		//printf("error	Seaqrch for TODO by end user...");
                //==================================
		
		if (sorter->use_HEX) {
			// shift the time sums to zero:
			sorter->shift_sums(+1, offset_sum_u, offset_sum_v, offset_sum_w);
			// shift layer w so that the middle lines of all layers intersect in one point:
			sorter->shift_layer_w(+1, w_offset);
		} else {
			// shift the time sums to zero:
			sorter->shift_sums(+1, offset_sum_u, offset_sum_v);
		}
		// shift all signals from the anode so that the center of the detector is at x=y=0:
		sorter->shift_position_origin(+1, pos_offset_x, pos_offset_y);

		sorter->feed_calibration_data(true, w_offset); // for calibration of fv, fw, w_offset and correction tables
		if (sorter->scalefactors_calibrator) {
		        bool status = sorter->scalefactors_calibrator->map_is_full_enough();
			if (status) {
                            printf("map_is_full_enough(): %d  event number: %d\n", status, event_number);  
                            break;
                        }
		}

		int number_of_particles = 0;
		if (command == 1) {  // sort the TDC-Data and reconstruct missing signals
			// sorts/reconstructs the detector signals and apply the sum- and NL-correction.
			number_of_particles = sorter->sort();
			// "number_of_particles" is the number of reconstructed particles
		} else {
			number_of_particles = sorter->run_without_sorting();
		}

		if (false) {
		  printf("  Event %5i  number_of_particles: %i", event_number, number_of_particles);
		  for(int i=0; i<number_of_particles; i++) {
		    printf("\n    p:%1i x:%.3f y:%.3f t:%.3f met:%d", i,
			   sorter->output_hit_array[i]->x, 
			   sorter->output_hit_array[i]->y,
			   sorter->output_hit_array[i]->time,
			   sorter->output_hit_array[i]->method);
		  }

		  double u = tdc_ns[Cu1][0] + tdc_ns[Cu2][0] - 2*tdc_ns[Cmcp][0];
		  double v = tdc_ns[Cv1][0] + tdc_ns[Cv2][0] - 2*tdc_ns[Cmcp][0];
		  double w = tdc_ns[Cw1][0] + tdc_ns[Cw2][0] - 2*tdc_ns[Cmcp][0];

		  printf("\n    part1  u:%.3f v:%.3f w:%.3f\n", u, v, w);
		} 


		//printf("error	Seaqrch for TODO by end user...");
		//#error	TODO by end user:
		// write the results into a new data file.
		// the variable "number_of_particles" contains the number of
		// reconstructed particles.
		// the x and y  (in mm) and TOF (in ns) is stored in the array sorter->output_hit_array:
		// 
		// for the first particle:
		// sorter->output_hit_array[0]->x;
		// sorter->output_hit_array[0]->y;
		// sorter->output_hit_array[0]->time;
		// 
		// for the 2nd particle:
		// sorter->output_hit_array[1]->x;
		// sorter->output_hit_array[1]->y;
		// sorter->output_hit_array[1]->time;
		//
		// for each particle you can also retrieve the information about how the particle
		// was reconstructed (tog et some measure of the confidence):
		// sorter->output_hit_array[0]->method;


	} // end of the while loop


	if (command == 2) {
		printf("calibrating detector... ");
		sorter->do_calibration();
		printf("ok\n");
		if (sorter->scalefactors_calibrator) {
		       printf("Good calibration factors are:\nf_U =%lg\nf_V =%lg\nf_W =%lg\nOffset on layer W=%lg\n",
                              2.*sorter->fu, 
                              2.*sorter->scalefactors_calibrator->best_fv,
                              2.*sorter->scalefactors_calibrator->best_fw, 
                              sorter->scalefactors_calibrator->best_w_offset);
		}
	}


	if (command == 3) {   // generate and print correction tables for sum- and position-correction
	  std::string fname_calib_tables("calibration_table.txt");
	  printf("creating calibration tables...\n");
	  create_calibration_tables(fname_calib_tables.c_str(), sorter);
	  printf("finished creating calibration tables: %s\n", fname_calib_tables.c_str());
	}

        clock_gettime(CLOCK_REALTIME, &stop);

	double dt_sec = stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec- start.tv_nsec);
	printf("consumed time (sec) = %.6f\n", dt_sec);

	if (sorter) {delete sorter; sorter=0;}
	
	return 0;
}
