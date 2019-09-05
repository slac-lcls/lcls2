#include <stdio.h>  // for  sprintf, printf( "%lf\n", accum );
#include <iostream> // for cout, puts etc.
#include "psalg/hexanode/LMF_IO.hh"

using namespace std;

#define NUM_CHANNELS 32
#define NUM_IONS 16

char DEFAULT_FILE_NAME[] = "/reg/g/psdm/detector/data_test/lmf/hexanode-example-CO_4.lmf";

int main(int argc, char* argv[])
{
	if(argc>2) {
		printf("wrong number of arguments.");
	        printf("syntax: ex02_read_lmf_data filename\n");
		return 0;
	}

	char LMF_Filename[500];
	sprintf(LMF_Filename, (argc<2) ? DEFAULT_FILE_NAME : (char*)argv[1]);

	//std::cout << "File name:" << LMF_Filename << '\n';
	printf("File name: %s\n",LMF_Filename);
	

	LMF_IO* LMF = new LMF_IO(NUM_CHANNELS,NUM_IONS);

	if (!LMF->OpenInputLMF(LMF_Filename)) {	  
	   //std::cout << "Can't open file: " << LMF_Filename << '\n';
	   printf("Can't open file: %s\n",LMF_Filename);
	   return false;
	}

	printf("Starttime: %s",ctime(&LMF->Starttime));
	printf("Stoptime : %s",ctime(&LMF->Stoptime));

	unsigned int number_of_hits[NUM_CHANNELS];
	memset(number_of_hits,0,NUM_CHANNELS*4);
	int    iTDC[NUM_CHANNELS][NUM_IONS];
	double dTDC[NUM_CHANNELS][NUM_IONS];

	bool only_100_events_flag = true;
	bool time_conversion_flag = false;

	char error_text[512];
	//double first_timestamp = 0.;
	//double last_timestamp = 0.;

	unsigned int i, j;

// Start reading event data:
// ---------------------------------------
	while(true) {
		if (LMF->ReadNextEvent()) {
		  printf("------- #%i -------\n", (int)LMF->GetEventNumber());
			//double new_timestamp = LMF->GetDoubleTimeStamp();
			//if (first_timestamp == 0.) first_timestamp = new_timestamp;
			//new_timestamp -= first_timestamp;
			//if (LMF->timestamp_format != 0) FPRINT"T  = %.3lf ns = \t%lf s\n",new_timestamp*1.e9,new_timestamp);
			//if (LMF->timestamp_format != 0) FPRINT"dT = %.3lf ns = \t%lf s\n",(new_timestamp-last_timestamp)*1.e9,new_timestamp-last_timestamp);

			//last_timestamp = new_timestamp;

			LMF->GetNumberOfHitsArray(number_of_hits);
			if (LMF->errorflag) {
				LMF->GetErrorText(LMF->errorflag, error_text);
				printf("%s",error_text);
				return false;
			}
			if (LMF->data_format_in_userheader==5) {
				LMF->GetTDCDataArray(&dTDC[0][0]);
			} else LMF->GetTDCDataArray(&iTDC[0][0]);

			if (LMF->errorflag) {
				LMF->GetErrorText(LMF->errorflag, error_text);
				printf("%s",error_text);
				return false;
			}

			for (i=0; i<LMF->GetNumberOfChannels(); i++) {
				printf("chan %5i",i+1);
				printf(" %5i", number_of_hits[i]);
				if (LMF->data_format_in_userheader==5) {
					for (j=0; j<number_of_hits[i]; j++) printf(" %lf",dTDC[i][j]);
				} else
					for (j=0; j<number_of_hits[i]; j++) {
						if (!time_conversion_flag) printf(" %5i", iTDC[i][j]);
						if ( time_conversion_flag) printf(" %.3lf", iTDC[i][j]*LMF->tdcresolution);
					}
				printf("\n");
			}
			printf("Levelinfo: %i\n\n", (int)LMF->GetLastLevelInfo());
		}

		//if (LMF->GetEventNumber()%50000 == 0) {
		//  __int8 c;
		//  while (_kbhit()) c = _getch();
		//  if (c == 'q') break;
 		//}

		if (LMF->errorflag) break;
		if (LMF->GetEventNumber() > 100	&& only_100_events_flag) break;
	}

	//LMF->CloseOutputLMF();


	if (LMF) delete LMF;

	return 0;
}
