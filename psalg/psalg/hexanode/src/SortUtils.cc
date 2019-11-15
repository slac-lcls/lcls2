
//-----------------------------

#include "psalg/hexanode/SortUtils.hh"

#include <termios.h> // termios
#include <string.h>  // memcpy
#include <stdlib.h>  // atof, atof

//-----------------------------

__int32 my_kbhit(void)
{
	struct termios term, oterm;
	__int32 fd = 0;
	__int32 c = 0;
	tcgetattr(fd, &oterm);
	memcpy(&term, &oterm, sizeof(term));
	term.c_lflag = term.c_lflag & (!ICANON);
	term.c_cc[VMIN] = 0;
	term.c_cc[VTIME] = 1;
	tcsetattr(fd, TCSANOW, &term);
	c = getchar();
	tcsetattr(fd, TCSANOW, &oterm);
	if (c == -1) return 0;
	return c;
}

//-----------------------------

void readline_from_config_file(FILE * ffile, char * text, __int32 max_len) {
	int i = -1;
	text[0] = 0;

	while (true) {
		i = -1;
		char c = 1;
		bool real_numbers_found = false;
		bool start_of_line = true;

		while (true) {
			fread(&c, 1, 1, ffile);
			if (c == 13) continue;
			if (start_of_line) {
				if (c==' ' || c==9 || c==10) continue;
				start_of_line = false;
			}
			if (c=='/') { // if there is a comment then read until end of the line
				while (c!=10) {
					fread(&c,1,1,ffile);
					if (c == 13) continue;
				}
				if (real_numbers_found) break;
				start_of_line = true;
				continue;
			}
			real_numbers_found = true;
			if (c==' ' || c==9) break;
			if (c == 10) break;
			if (c == ',') break;
			++i;
			if (i < max_len -1) {text[i] = c; text[i+1] = 0;}
		}
		if (real_numbers_found) break;
	}

	return;
}

//-----------------------------

int read_int(FILE * ffile) {
	char a[1024];
	readline_from_config_file(ffile, a, 1024);
	return atoi(a);
}

//-----------------------------

double read_double(FILE * ffile) {
	char a[1024];
	readline_from_config_file(ffile, a, 1024);
	return double(atof(a));
}

//-----------------------------

bool read_config_file(const char * name, sort_class *& sorter, int& command,
                      double& offset_sum_u, double& offset_sum_v, double& offset_sum_w,
                      double& w_offset, double& pos_offset_x, double& pos_offset_y)
{
	// read the config file:
	printf("opening %s... ",name);
	FILE * parameterfile_handle = fopen(name, "rt");

	if (!parameterfile_handle) {
		printf("file %s was not found.%c\n",name, 7);
		return false;
	}
	printf("ok\n");
	printf("reading %s... ",name);
	int int_dummy;
	command = read_int(parameterfile_handle);

	printf("\ncommand %i\n", command);

	if (command == -1) {
		printf("ok\n");
		return false;
	}

	int_dummy = read_int(parameterfile_handle);
	sorter->use_HEX = int_dummy ? true: false;

	int_dummy = read_int(parameterfile_handle);
	sorter->common_start_mode = (!int_dummy) ? true:false;

	sorter->Cu1 = read_int(parameterfile_handle) -1;
	sorter->Cu2 = read_int(parameterfile_handle) -1;
	sorter->Cv1 = read_int(parameterfile_handle) -1;
	sorter->Cv2 = read_int(parameterfile_handle) -1;
	sorter->Cw1 = read_int(parameterfile_handle) -1;
	sorter->Cw2 = read_int(parameterfile_handle) -1;

	sorter->Cmcp   = read_int(parameterfile_handle) -1;
	sorter->use_MCP = (sorter->Cmcp > -1) ? true : false;

	offset_sum_u = read_double(parameterfile_handle);
	offset_sum_v = read_double(parameterfile_handle);
	offset_sum_w = read_double(parameterfile_handle);

	pos_offset_x = read_double(parameterfile_handle);
	pos_offset_y = read_double(parameterfile_handle);

	sorter->uncorrected_time_sum_half_width_u = read_double(parameterfile_handle);
	sorter->uncorrected_time_sum_half_width_v = read_double(parameterfile_handle);
	sorter->uncorrected_time_sum_half_width_w = read_double(parameterfile_handle);

	sorter->fu = 0.5*read_double(parameterfile_handle);
	sorter->fv = 0.5*read_double(parameterfile_handle);
	sorter->fw = 0.5*read_double(parameterfile_handle);
	w_offset   = read_double(parameterfile_handle);
	sorter->runtime_u = read_double(parameterfile_handle);
	sorter->runtime_v = read_double(parameterfile_handle);
	sorter->runtime_w = read_double(parameterfile_handle);
	
	sorter->MCP_radius  = read_double(parameterfile_handle);

	sorter->dead_time_anode = read_double(parameterfile_handle);
	sorter->dead_time_mcp   = read_double(parameterfile_handle);

	int_dummy = read_int(parameterfile_handle);
	sorter->use_sum_correction = (int_dummy != 0) ? true: false;
	int_dummy = read_int(parameterfile_handle);
	sorter->use_pos_correction = (int_dummy != 0) ? true: false;

	__int32 check_dummy = read_int(parameterfile_handle);
	if (check_dummy != 88888) {
		printf("File %s was not correctly read.\n", name);
		fclose(parameterfile_handle);
		if (sorter) {delete sorter; sorter = 0;}
		return false;
	}

	fclose(parameterfile_handle);
	// end of reading the config file:
	printf("ok\n");

	return true;
}

//-----------------------------

bool read_calibration_tables(const char * filename, sort_class * sorter)
{
	if (!filename) return false;
	if (!sorter) return false;

	FILE * infile_handle = fopen(filename,"rt");
	if (!infile_handle) return false;
	int points = 0;

	points = read_int(infile_handle);
	for (int j=0;j<points;++j) {
		double x = read_double(infile_handle);
		double y = read_double(infile_handle);
		if (sorter->use_sum_correction) sorter->signal_corrector->sum_corrector_U->set_point(x,y);
	}
	points = read_int(infile_handle);
	for (int j=0;j<points;++j) {
		double x = read_double(infile_handle);
		double y = read_double(infile_handle);
		if (sorter->use_sum_correction) sorter->signal_corrector->sum_corrector_V->set_point(x,y);
	}
	if (sorter->use_HEX) {
		points = read_int(infile_handle);
		for (int j=0;j<points;++j) {
			double x = read_double(infile_handle);
			double y = read_double(infile_handle);
			if (sorter->use_sum_correction) sorter->signal_corrector->sum_corrector_W->set_point(x,y);
		}
	}

	points = read_int(infile_handle);
	for (int j=0;j<points;++j) {
		double x = read_double(infile_handle);
		double y = read_double(infile_handle);
		if (sorter->use_pos_correction) sorter->signal_corrector->pos_corrector_U->set_point(x,y);
	}
	points = read_int(infile_handle);
	for (int j=0;j<points;++j) {
		double x = read_double(infile_handle);
		double y = read_double(infile_handle);
		if (sorter->use_pos_correction) sorter->signal_corrector->pos_corrector_V->set_point(x,y);
	}
	if (sorter->use_HEX) {
		points = read_int(infile_handle);
		for (int j=0;j<points;++j) {
			double x = read_double(infile_handle);
			double y = read_double(infile_handle);
			if (sorter->use_pos_correction) sorter->signal_corrector->pos_corrector_W->set_point(x,y);
		}
	}

	if (infile_handle) {fclose(infile_handle); infile_handle = 0;}
	return true;
}

//-----------------------------

bool create_calibration_tables(const char* filename, sort_class* sorter) 
{
        //printf("In create_calibration_tables file: %s\n", filename);

	if (!sorter) return false;
	if (!filename) return false;

        char fmt[] = "%lg  %lg\n";
        //char fmt[] = "%10.4f  %10.5f\n";

	FILE * fo = fopen(filename,"wt");
	sorter->do_calibration();
	int number_of_columns = sorter->sum_walk_calibrator->sumu_profile->number_of_columns;
	fprintf(fo,"\n\n%i  	// number of sum calibration points for layer U\n",number_of_columns);
	for (int binx=0; binx < number_of_columns; ++binx) {
		double x,y;
		sorter->sum_walk_calibrator->get_correction_point(x,y,binx,0); // 0 = layer u
		fprintf(fo,fmt,x,y);
	}

	number_of_columns = sorter->sum_walk_calibrator->sumv_profile->number_of_columns;
	fprintf(fo,"\n\n%i  	// number of sum calibration points for layer V\n",number_of_columns);
	for (int binx=0; binx < number_of_columns; ++binx) {
		double x,y;
		sorter->sum_walk_calibrator->get_correction_point(x,y,binx,1); // 1 = layer v
		fprintf(fo,fmt,x,y);
	}


	number_of_columns = (sorter->use_HEX) ? sorter->sum_walk_calibrator->sumw_profile->number_of_columns : 0;

	fprintf(fo,"\n\n%i  	// number of sum calibration points for layer W (only needed for HEX-detectors)\n",number_of_columns);
	if (sorter->use_HEX) {
		for (int binx=0; binx < number_of_columns; ++binx) {
			double x,y;
			sorter->sum_walk_calibrator->get_correction_point(x,y,binx,2); // 2 = layer w
			fprintf(fo,fmt,x,y);
		}
	}


	//number_of_columns = sorter->pos_walk_calibrator->number_of_columns;
	number_of_columns = (sorter->use_HEX) ? sorter->pos_walk_calibrator->number_of_columns : 0;

	fprintf(fo,"\n\n%i  	// number of pos-calibration points for layer U\n",number_of_columns);
	for (int binx=0; binx < number_of_columns; ++binx) {
		double x,y;
		sorter->pos_walk_calibrator->get_correction_point(x,y,binx,0); // 0 = layer u
		fprintf(fo,fmt,x,y);
	}


	fprintf(fo,"\n\n%i  	// number of pos-calibration points for layer V\n",number_of_columns);
	for (int binx=0; binx < number_of_columns; ++binx) {
		double x,y;
		sorter->pos_walk_calibrator->get_correction_point(x,y,binx,1); // 1 = layer v
		fprintf(fo,fmt,x,y);
	}

	fprintf(fo,"\n\n%i  	// number of pos-calibration points for layer W (only needed for HEX-detectors)\n",number_of_columns);
	if (sorter->use_HEX) {
		for (int binx=0; binx < number_of_columns; ++binx) {
			double x,y;
			sorter->pos_walk_calibrator->get_correction_point(x,y,binx,2); // 2 = layer w
			fprintf(fo,fmt,x,y);
		}
	}
	fclose(fo); fo = 0;
	return true;
}

//-----------------------------

bool sorter_scalefactors_calibration_map_is_full_enough(sort_class* sorter)
{
	return (sorter) ? sorter->scalefactors_calibrator->map_is_full_enough() : false;
}

//-----------------------------
