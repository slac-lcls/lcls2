
//#include "conio.h"
#include "psalg/hexanode/LMF_IO.hh"

#define NUM_CHANNELS 32
#define NUM_IONS 16
#define FPRINT fprintf(ascii_outputfile_handle,
//#define FPRINT if (flag) printf(

char DEFAULT_FILE_NAME[] = "/reg/g/psdm/detector/data_test/lmf/hexanode-example-CO_4.lmf";

int main(int argc, char* argv[])
{
	#ifdef _DEBUG
		printf("\n***********************\n    SLOW DEBUG VERSION USED\n***********************\n");
	#endif

	printf("syntax: ex03_lmf2txt [filename] [-f] [-ns]\n");
	printf("        if -f is omitted then only the header and the first 100 events will be written.\n");
	printf("        -ns converts the values to nanoseconds.\n");

	//if (argc < 2 || argc > 4) {
	if (argc > 4) {
	  printf("wrong number of arguments: %d\n", argc);
	  return 0;
	}

	bool only_100_events_flag = true;
	bool time_conversion_flag = false;

	if (argc >= 3) {
		if (strcmp((char*)argv[2],"-f") == 0) only_100_events_flag = false;
		if (strcmp((char*)argv[2],"-ns") == 0) time_conversion_flag = true;		
	}
	if (argc == 4) {
		if (strcmp((char*)argv[3],"-f") == 0) only_100_events_flag = false;
		if (strcmp((char*)argv[3],"-ns") == 0) time_conversion_flag = true;
	}

	char LMF_Filename[500];

	//sprintf(LMF_Filename,(char*)argv[1]);
	sprintf(LMF_Filename, (argc<2) ? DEFAULT_FILE_NAME : (char*)argv[1]);

        printf("Input file: %s\n", LMF_Filename);

	unsigned int i,j;
	unsigned int number_of_hits[NUM_CHANNELS];
	memset(number_of_hits,0,NUM_CHANNELS*4);
	int		iTDC[NUM_CHANNELS][NUM_IONS];
	double		dTDC[NUM_CHANNELS][NUM_IONS];

	LMF_IO * LMF = new LMF_IO(NUM_CHANNELS,NUM_IONS);

	if (!LMF->OpenInputLMF(LMF_Filename)) {
		return false;
	}

	char error_text[512];
	std::string ascii_output_filename = LMF_Filename;
	ascii_output_filename += std::string(".txt");
	FILE * ascii_outputfile_handle = fopen(ascii_output_filename.c_str(),"wt");


	//bool flag = false;

// Print parts of the header information:
// ---------------------------------------
	FPRINT"File name = %s\n",LMF->FilePathName.c_str());
	FPRINT"Versionstring = %s\n",LMF->Versionstring.c_str());
	FPRINT"Comment = %s\n",LMF->Comment.c_str());

	FPRINT"Headersize = %i\n",LMF->Headersize);

	FPRINT"Number of channels = %i\n",LMF->GetNumberOfChannels());
	FPRINT"Number of hits = %i\n",LMF->GetMaxNumberOfHits());
	FPRINT"Number of Coordinates = %i\n",LMF->Numberofcoordinates);
	
	
	FPRINT"Timestamp info = %i\n",LMF->timestamp_format);
	if (LMF->common_mode == 0) FPRINT"Common start\n"); else FPRINT"Common stop\n");
        //FPRINT"Number of events = %I64i\n",LMF->uint64_Numberofevents);
        FPRINT"Number of events = %i\n", (int)LMF->uint64_Numberofevents);
	FPRINT"Data format = %i\n",LMF->data_format_in_userheader);
	FPRINT"DAQ_ID = %i\n",LMF->DAQ_ID);

	FPRINT"TDC resolution = %lf ns\n",LMF->tdcresolution);
	if (LMF->DAQ_ID == 0x000008 || LMF->DAQ_ID == 0x000010) {
		//FPRINT"TDC8HP Header Version %i\n",LMF->TDC8HP.UserHeaderVersion);
		FPRINT"Trigger channel = %i (counting from 1)\n",LMF->TDC8HP.TriggerChannel_p64+1);
		FPRINT"Trigger dead time = %lf ns\n",LMF->TDC8HP.TriggerDeadTime_p68);
		FPRINT"Group range start = %lf ns\n",LMF->TDC8HP.GroupRangeStart_p69);
		FPRINT"Group range end = %lf ns\n",LMF->TDC8HP.GroupRangeEnd_p70);
	}
	
	FPRINT"Starttime: %s",ctime(&LMF->Starttime));
	FPRINT"Stoptime:  %s",ctime(&LMF->Stoptime));
	FPRINT"\n");


//	LMF->OpenOutputLMF("test.lmf");

	double first_timestamp = 0.;
	double last_timestamp = 0.;

// Start reading event data:
// ---------------------------------------
	while(true) {
		if (LMF->ReadNextEvent())  {
		  FPRINT"------- #%i -------\n", (int)LMF->GetEventNumber());
			double new_timestamp = LMF->GetDoubleTimeStamp();
			if (first_timestamp == 0.) first_timestamp = new_timestamp;
			new_timestamp -= first_timestamp;
			if (LMF->timestamp_format != 0) FPRINT"T  = %.3lf ns = \t%lf s\n",new_timestamp*1.e9,new_timestamp);
			if (LMF->timestamp_format != 0) FPRINT"dT = %.3lf ns = \t%lf s\n",(new_timestamp-last_timestamp)*1.e9,new_timestamp-last_timestamp);

			last_timestamp = new_timestamp;

			LMF->GetNumberOfHitsArray(number_of_hits);
			if (LMF->errorflag) {
				LMF->GetErrorText(LMF->errorflag,error_text);
				FPRINT"%s",error_text);
				return false;
			}
			if (LMF->data_format_in_userheader==5) {
				LMF->GetTDCDataArray(&dTDC[0][0]);
			} else LMF->GetTDCDataArray(&iTDC[0][0]);

			if (LMF->errorflag) {
				LMF->GetErrorText(LMF->errorflag,error_text);
				FPRINT"%s",error_text);
				return false;
			}

			for (i=0;i<LMF->GetNumberOfChannels();i++) {
				FPRINT"chan %5i",i+1);
				FPRINT" %5i",number_of_hits[i]);
				if (LMF->data_format_in_userheader==5) {
					for (j=0;j<number_of_hits[i];j++) FPRINT" %lf",dTDC[i][j]);
				} else
					for (j=0;j<number_of_hits[i];j++) {
						if (!time_conversion_flag) FPRINT" %5i",iTDC[i][j]);
						if ( time_conversion_flag) FPRINT" %.3lf",iTDC[i][j]*LMF->tdcresolution);
					}
				FPRINT"\n");
			}
                        //FPRINT"Levelinfo: %I64x\n",LMF->GetLastLevelInfo());
                        FPRINT"Levelinfo: %i\n", (int)LMF->GetLastLevelInfo());
			FPRINT"\n");
		}

/*
                if (LMF->GetEventNumber()%50000 == 0) {
			__int8 c;
			//while (_kbhit()) c = _getch();
			if (c == 'q') break;
 		}
*/

		if (LMF->errorflag) break;
		if (LMF->GetEventNumber() > 100	&& only_100_events_flag) break;
	}

	//LMF->CloseOutputLMF();


	if (LMF) delete LMF;

	fclose(ascii_outputfile_handle);
        printf("Saved file: %s\n", ascii_output_filename.c_str());

	return 0;
}

