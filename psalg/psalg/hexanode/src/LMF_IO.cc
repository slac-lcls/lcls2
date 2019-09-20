#define WINVER 0x0501
// #pragma warning(disable : 4996)

#include "psalg/hexanode/LMF_IO.hh"

#define LMF_IO_CLASS_VERSION (2014)


#define DAQ_SOURCE_CODE		0x80000000
#define DAN_SOURCE_CODE		0x40000000
#define CCF_HISTORY_CODE	0x20000000


void MyFILE::seek(unsigned __int64 pos)
{
	__int32 rval = _fseeki64(file,  pos,SEEK_SET);
	if (rval == 0) this->position = pos; else error = 1;


/*
	if (pos < 1000000000) {
		f_stream.seekg((unsigned int)pos, std::ios::beg);
		this->position = pos;
		return;
	}

	unsigned __int64 real_pos = 0;
	f_stream.seekg(0, std::ios::beg);
	while (true) {
		
//		f_stream.seekg(1000000000,std::ios::cur);
		real_pos += 1000000000;
		__int64 diff = pos - real_pos;
		if (diff < 1000000000) break;
	}
	__int64 diff = pos - real_pos;
	__int32 diff1 = __int32(diff);
	f_stream.seekg(diff1,std::ios::cur);*/

}




/*
void MyFILE::seek_to_end()
{
	f_stream.seekg(0, std::ios::end);
}
*/





#define INT_MIN_ (-2147483647-1)




void LMF_IO::write_times(MyFILE * out_file,time_t _Starttime,time_t _Stoptime) {
	unsigned __int32 dummy;
	if (CTime_version_output > 2003) {
		dummy = INT_MIN_+10; *out_file << dummy;
		dummy = (unsigned __int32)_Starttime; *out_file << dummy;
		dummy =0;			*out_file << dummy;
		dummy =INT_MIN_+10;	*out_file << dummy;
		dummy = (unsigned __int32)_Stoptime;	*out_file << dummy;
		dummy = 0;			*out_file << dummy;
	} else {
		dummy = (unsigned __int32)_Starttime;	*out_file << dummy;
		dummy = (unsigned __int32)_Stoptime;	*out_file << dummy;
	}
}









unsigned __int32 ReadCStringLength(MyFILE &in_file)
{
	unsigned __int64 qwLength;
	unsigned __int32 dwLength;
	unsigned __int16 wLength;
	unsigned __int8  bLength;

	//__int32 nCharSize = sizeof(__int8);

	// First, try to read a one-byte length
	in_file >> bLength;

	if (bLength < 0xff) 
		return bLength;

	// Try a two-byte length
	in_file >> wLength;
	if (wLength == 0xfffe)
	{
		// Unicode string.  Start over at 1-byte length
		//nCharSize = sizeof(wchar_t);

		in_file >> bLength;
		if (bLength < 0xff)
			return bLength;

		// Two-byte length
		in_file >> wLength;
		// Fall through to continue on same branch as ANSI string
	}
	if (wLength < 0xffff)
		return wLength;

	// 4-byte length
	in_file >> dwLength;
	if (dwLength < 0xffffffff)
		return dwLength;

	// 8-byte length
	in_file >> qwLength;

	return (unsigned __int32)qwLength;
}




void Read_CString_as_StdString(MyFILE &in_file,std::string &stdstring)
{
	__int32 length = ReadCStringLength(in_file);
	__int8 * temp_string = new __int8[length+1];
	in_file.read(temp_string,length);
	temp_string[length] = 0;
	stdstring = temp_string;
}



void WriteCStringLength(MyFILE &out_file, unsigned __int32 nLength)
{
	unsigned __int8 dummy_uint8;
	unsigned __int16 dummy_uint16;
	unsigned __int32 dummy_uint32;
	unsigned __int64 dummy_uint64;

	if (nLength < 255)
	{
		dummy_uint8 = nLength; out_file << dummy_uint8;
	}
	else if (nLength < 0xfffe)
	{
		dummy_uint8 = 0xff; out_file << dummy_uint8;
		dummy_uint16 = nLength; out_file << dummy_uint16;
	}
	else if (nLength < 0xffffffff)
	{
		dummy_uint8 = 0xff; out_file << dummy_uint8;
		dummy_uint16 = 0xffff; out_file << dummy_uint16;
		dummy_uint32 = nLength; out_file << dummy_uint32;
	}
	else
	{
		dummy_uint8 = 0xff; out_file << dummy_uint8;
		dummy_uint16 = 0xffff; out_file << dummy_uint16;
		dummy_uint32 = 0xffffffff; out_file << dummy_uint32;
		dummy_uint64 = nLength; out_file << dummy_uint64;
	}
}



void Write_StdString_as_CString(MyFILE &out_file,std::string &stdstring)
{
	unsigned __int32 length = (unsigned __int32)(stdstring.length());
	WriteCStringLength(out_file,length);
	out_file.write(stdstring.c_str(),length);
}







/////////////////////////////////////////////////////////////////
__int32 LMF_IO::GetVersionNumber()
/////////////////////////////////////////////////////////////////
{
	return LMF_IO_CLASS_VERSION;
}











/////////////////////////////////////////////////////////////////
LMF_IO::LMF_IO(__int32 _num_channels, __int32 _num_ions)
/////////////////////////////////////////////////////////////////
{
	num_channels = _num_channels;
	num_ions = _num_ions;
	Initialize();
}






/////////////////////////////////////////////////////////////////
LMF_IO::~LMF_IO()
/////////////////////////////////////////////////////////////////
{
	if (InputFileIsOpen)  CloseInputLMF();
	if (OutputFileIsOpen) CloseOutputLMF();
	if (CAMAC_Data)		{delete[] CAMAC_Data; CAMAC_Data = 0;}
	if (i32TDC)			{delete[] i32TDC; i32TDC = 0;}
	if (us16TDC)		{delete[] us16TDC; us16TDC = 0;}
	if (dTDC)			{delete[] dTDC; dTDC = 0;}
	if (number_of_hits) {delete[] number_of_hits; number_of_hits = 0;}

	for (__int32 i=0;i<3;++i) {
		if (TDC8HP.TDC_info[i]) {
			if (TDC8HP.TDC_info[i]->INLCorrection)	{delete[] TDC8HP.TDC_info[i]->INLCorrection;	TDC8HP.TDC_info[i]->INLCorrection = 0;}
			if (TDC8HP.TDC_info[i]->DNLData)		{delete[] TDC8HP.TDC_info[i]->DNLData;			TDC8HP.TDC_info[i]->DNLData = 0;}
			if (TDC8HP.TDC_info[i]) {delete TDC8HP.TDC_info[i]; TDC8HP.TDC_info[i] = 0;}
		}
	}
	if (ui32buffer) {delete[] ui32buffer; ui32buffer = 0; this->ui32buffer_size = 0;}

	

	if (CCFHistory_strings) {
		for (__int32 i=0;i<number_of_CCFHistory_strings;++i) {
			if (CCFHistory_strings[i]) {delete CCFHistory_strings[i]; CCFHistory_strings[i] = 0;}
		}
		delete[] CCFHistory_strings; CCFHistory_strings = 0;
	}
	if (DAQ_source_strings) {
		for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {
			if (DAQ_source_strings[i]) {delete DAQ_source_strings[i]; DAQ_source_strings[i] = 0;}
		}
		delete[] DAQ_source_strings; DAQ_source_strings = 0;
	}
	if (DAN_source_strings) {
		for (__int32 i=0;i<number_of_DAN_source_strings;++i) {
			if (DAN_source_strings[i]) {delete DAN_source_strings[i]; DAN_source_strings[i] = 0;}
		}
		delete[] DAN_source_strings; DAN_source_strings = 0;
	}

	if (CCFHistory_strings_output) {
		for (__int32 i=0;i<number_of_CCFHistory_strings_output;++i) {
			if (CCFHistory_strings_output[i]) {
				delete CCFHistory_strings_output[i]; 
				CCFHistory_strings_output[i] = 0;
			}
		}
		delete[] CCFHistory_strings_output; CCFHistory_strings_output = 0;
	}
	if (DAQ_source_strings_output) {
		for (__int32 i=0;i<number_of_DAQ_source_strings_output;++i) {
			if (DAQ_source_strings_output[i]) {delete DAQ_source_strings_output[i]; DAQ_source_strings_output[i] = 0;}
		}
		delete[] DAQ_source_strings_output; DAQ_source_strings_output = 0;
	}
	if (DAN_source_strings_output) {
		for (__int32 i=0;i<number_of_DAN_source_strings_output;++i) {
			if (DAN_source_strings_output[i]) {delete DAN_source_strings_output[i]; DAN_source_strings_output[i] = 0;}
		}
		delete[] DAN_source_strings_output; DAN_source_strings_output = 0;
	}

}





/////////////////////////////////////////////////////////////////
void LMF_IO::Initialize()
/////////////////////////////////////////////////////////////////
{
	uint64_LMF_EventCounter = -1;
	not_Cobold_LMF = false;
	errorflag = 0;
	number_of_channels2 = 0;
	InputFileIsOpen = false;
	OutputFileIsOpen = false;
	SIMPLE_DAQ_ID_Orignial = 0;
	input_lmf = 0;
	output_lmf = 0;
	DAQ_ID = 0;
	frequency = 1.;
	common_mode = 0;
	timerange = 0;
	must_read_first = true;
	uint64_number_of_read_events = 0;
	uint64_number_of_written_events = 0;
	number_of_channels_output = 0;
	number_of_channels2_output = -1;
	number_of_channels = 0;
	max_number_of_hits = 0;
	max_number_of_hits_output = 0;
	max_number_of_hits2_output = -1;
	data_format_in_userheader_output = -2;
	output_byte_counter = 0;
	timestamp_format = 0;
	timestamp_format_output = -1;
	DOUBLE_timestamp = 0.;
	ui64_timestamp = 0;
	skip_header = false;
	time_reference_output = 0;
	system_timeout_output = -1;
	Numberofcoordinates = -2;
	Numberofcoordinates_output = -2;
	DAQ_ID_output = 0;
	User_header_size_output = 0;
	CAMAC_Data = 0;
	tdcresolution_output = -1.;
	tdcresolution = 0.;
	TDC8HP.SyncValidationChannel = 0;
	User_header_size = 0;
	TDC8HP.UserHeaderVersion = 6; // 4 = Cobold 2008 first release in 2008
	                              // 5 = Cobold 2008 R2 in August 2009
								  // 6 = Cobold 2009?

	HM1.use_normal_method = true;
	TDC8PCI2.use_normal_method_2nd_card = true;
	TDC8PCI2.use_normal_method = true;
	TDC8HP.VHR_25ps = true;
	common_mode_output = -1;
	DAQ_info = "";
	DAQ_info_Length = 0;
	Versionstring = "";
	FilePathName = "";
	OutputFilePathName = "";
	Comment = "";
	Comment_output = "";
	DAQ_info = "";
	Camac_CIF = "";
	TDC8HP.variable_event_length = 0;
	DAQVersion_output = -1;
	LMF_Version = -1;
	LMF_Version_output = -1;
	TDC8HP.csConfigFile = "";
	TDC8HP.csINLFile = "";
	TDC8HP.csDNLFile = "";
	TDC8HP.GroupingEnable_p66		 = false;
	TDC8HP.GroupingEnable_p66_output = false;

	DAQ_SOURCE_CODE_bitmasked  = 0;
	DAN_SOURCE_CODE_bitmasked  = 0;
	CCF_HISTORY_CODE_bitmasked = 0;
	number_of_CCFHistory_strings = 0;
	number_of_DAN_source_strings = 0;
	number_of_DAQ_source_strings = 0;
	number_of_CCFHistory_strings_output = 0;
	number_of_DAN_source_strings_output = 0;
	number_of_DAQ_source_strings_output = 0;
	CCFHistory_strings = 0;
	DAN_source_strings = 0;
	DAQ_source_strings = 0;
	CCFHistory_strings_output = 0;
	DAN_source_strings_output = 0;
	DAQ_source_strings_output = 0;

	TDC8HP.number_of_bools   = 0;
	TDC8HP.number_of_doubles = 0;
	TDC8HP.number_of_doubles = 0;

	TDC8PCI2.variable_event_length = 0;

	i32TDC =	new __int32[num_channels*num_ions];
	us16TDC =	new unsigned __int16[num_channels*num_ions];
	dTDC =		new double [num_channels*num_ions];
	number_of_hits = new unsigned __int32[num_channels];

	TDC8HP.DMAEnable = true;
	TDC8HP.SSEEnable = false;
	TDC8HP.MMXEnable = true;
	TDC8HP.GroupTimeOut = 0.1;

	TDC8HP.channel_offset_for_rising_transitions = 0;

//	TDC8HP.bdummy = false;
//	TDC8HP.idummy = 0;
//	TDC8HP.ddummy = 0.;

	TDC8HP.i32NumberOfDAQLoops  = 1;
	TDC8HP.TDC8HP_DriverVersion = 0x00000000;
	TDC8HP.iTriggerChannelMask  = 0;
	TDC8HP.iTime_zero_channel   = 0;

	TDC8HP.Number_of_TDCs = 0;

	time_t osBinaryTime;
	time( &osBinaryTime );
	time_t time_dummy(osBinaryTime);
	Starttime = time_dummy;
	Stoptime = time_dummy;
	Starttime_output = 0;
	Stoptime_output  = 0;
	CTime_version    = 0;
	CTime_version_output = 0;
	number_of_DAQ_source_strings_output = -1;
	number_of_CCFHistory_strings_output = -1;
	number_of_DAN_source_strings_output = -1;

	LMF_Header_version = 476759;
	ui64LevelInfo = 0;

	Cobold_Header_version = 2002;
	Cobold_Header_version_output = 0;

	TDC8HP.OffsetTimeZeroChannel_s = 0.;
	TDC8HP.BinsizeType = 0;

	for (__int32 i=0;i<3;++i) {
		TDC8HP.TDC_info[i] = 0;
		TDC8HP.TDC_info[i] = new TDC8HP_info_struct;
		TDC8HP.TDC_info[i]->INLCorrection = 0;
		TDC8HP.TDC_info[i]->INLCorrection	= new __int32[8*1024];
		TDC8HP.TDC_info[i]->DNLData = 0;
		TDC8HP.TDC_info[i]->DNLData			= new unsigned __int16[8*1024];
	}

	ui32buffer_size = 0;
	ui32buffer = 0;

	error_text[0] = (char*)"no error";
	error_text[1] = (char*)"error reading timestamp";
	error_text[2] = (char*)"error reading data";
	error_text[3] = (char*)"input file is already open";
	error_text[4] = (char*)"could not open input file";
	error_text[5] = (char*)"could not connect CAchrive to input file";
	error_text[6] = (char*)"error reading header";
	error_text[7] = (char*)"LMF not data of TDC8PCI2 or 2TDC8PCI2 or TDC8HP or CAMAC";
	error_text[8] = (char*)"file format not supported (only unsigned __int16 16bit and signed integer 32)";
	error_text[9] = (char*)"intput file not open";
	error_text[10] = (char*)"output file not open";
	error_text[11] = (char*)"could not open output file";
	error_text[12] = (char*)"ouput file is already open";
	error_text[13] = (char*)"could not connect CAchrive to output file";
	error_text[14] = (char*)"some parameters are not initialized";
	error_text[15] = (char*)"CAMAC data tried to read with wrong function";
	error_text[16] = (char*)"seek does not work with non-fixed event lengths";
	error_text[17] = (char*)"writing file with non-fixed event length dan DAQVersion < 2008 no possible";
	error_text[18] = (char*)"end of input file";
	error_text[19] = (char*)"more channels in file than specified at new LMF_IO()";
	error_text[20] = (char*)"more hits per channel in file than specified at new LMF_IO()";
}






/////////////////////////////////////////////////////////////////
void LMF_IO::CloseInputLMF()
/////////////////////////////////////////////////////////////////
{
	if (input_lmf)	{input_lmf->close(); delete input_lmf; input_lmf = 0;}
	InputFileIsOpen = false;
}







/////////////////////////////////////////////////////////////////
unsigned __int64 LMF_IO::GetLastLevelInfo()
/////////////////////////////////////////////////////////////////
{
	return ui64LevelInfo;
}








/////////////////////////////////////////////////////////////////
bool LMF_IO::OpenNonCoboldFile(void)
/////////////////////////////////////////////////////////////////
{
	input_lmf->seek(0);

	if (skip_header) {
		DAQ_ID = DAQ_ID_RAW32BIT;
		data_format_in_userheader = 10;
		return true;
	}

	if (DAQ_ID == DAQ_ID_RAW32BIT) return true;

	DAQ_ID = DAQ_ID_SIMPLE;

	*input_lmf >> SIMPLE_DAQ_ID_Orignial;
	unsigned tempi;
	*input_lmf >> tempi; uint64_Numberofevents = tempi;
	*input_lmf >> data_format_in_userheader;

	User_header_size = 3*sizeof(__int32);
	Headersize = 0;

	return true;
}








/////////////////////////////////////////////////////////////////
bool LMF_IO::OpenInputLMF(std::string Filename)
/////////////////////////////////////////////////////////////////
{
	return OpenInputLMF((__int8*)Filename.c_str());
}







/////////////////////////////////////////////////////////////////
bool LMF_IO::OpenInputLMF(__int8 * LMF_Filename)
/////////////////////////////////////////////////////////////////
{
	unsigned __int32	unsigned_int_Dummy;
	__int32				data_format_in_header;
	__int8				byte_Dummy;
	__int32				byte_counter;

	if (InputFileIsOpen) {
		errorflag = 3; // file is already open
		return false;
	}
	input_lmf = new MyFILE(true);

	TDC8HP.UserHeaderVersion = 0; // yes, 0 is ok here and 2 in LMF_IO::initialization is also ok

	input_lmf->open(LMF_Filename);

	if (input_lmf->error) {
		errorflag = 4; // could not open file
		input_lmf = 0;
		return false;
	}

//L10:

//  READ LMF-HEADER
	ArchiveFlag = 0;

	*input_lmf >> ArchiveFlag;

	DAQ_SOURCE_CODE_bitmasked  =  ArchiveFlag & DAQ_SOURCE_CODE;
	DAN_SOURCE_CODE_bitmasked  =  ArchiveFlag & DAN_SOURCE_CODE;
	CCF_HISTORY_CODE_bitmasked =  ArchiveFlag & CCF_HISTORY_CODE;

	ArchiveFlag = ArchiveFlag & 0x1fffffff;
	
	if (ArchiveFlag != 476758 && ArchiveFlag != 476759) { // is this not a Cobold list mode file?
		not_Cobold_LMF = true;
		if (!OpenNonCoboldFile()) return false;
		errorflag = 0; // no error
		InputFileIsOpen = true;
		return true;
	}

	if (ArchiveFlag == 476758) Cobold_Header_version = 2002;
	if (ArchiveFlag == 476759) Cobold_Header_version = 2008;

	*input_lmf >> data_format_in_header;

	data_format_in_userheader = data_format_in_header;
	if (data_format_in_header != LM_USERDEF && data_format_in_header != LM_SHORT && data_format_in_header != LM_SLONG && data_format_in_header != LM_DOUBLE && data_format_in_header != LM_CAMAC) {
		errorflag = 8;
		CloseInputLMF();
		return false;
	}

	if (Cobold_Header_version <= 2002)	*input_lmf >> Numberofcoordinates;
	if (Cobold_Header_version >= 2008)	{
		unsigned __int64 temp;
		*input_lmf >> temp;
		Numberofcoordinates =__int32(temp);
	}
	
	if (Numberofcoordinates >= 0) CAMAC_Data = new unsigned __int32[Numberofcoordinates];

	if (Cobold_Header_version <= 2002)	*input_lmf >> Headersize;
	if (Cobold_Header_version >= 2008)	{
		unsigned __int64 temp;
		*input_lmf >> temp;
		Headersize =__int32(temp);
	}


	if (Cobold_Header_version <= 2002)	*input_lmf >> User_header_size;
	if (Cobold_Header_version >= 2008)	{
		unsigned __int64 temp;
		*input_lmf >> temp;
		User_header_size =__int32(temp);
	}

	if (skip_header) {
		__int32 backstep;
		if (Cobold_Header_version <= 2002) backstep = sizeof(__int32)*5;
		if (Cobold_Header_version >= 2008) backstep = sizeof(__int32)*2 + sizeof(__int64)*3;
		for (unsigned __int32 i=0;i<Headersize-backstep;++i) *input_lmf >> byte_Dummy;
		for (unsigned __int32 i=0;i<User_header_size;++i) *input_lmf >> byte_Dummy;
		errorflag = 0; // no error
		InputFileIsOpen = true;
		goto L666;
	}

	if (Cobold_Header_version >= 2008) *input_lmf >> uint64_Numberofevents;
	if (Cobold_Header_version <= 2002) {
		__int32 temp;
		*input_lmf >> temp;
		uint64_Numberofevents = temp;
	}

	// get CTime version:
	if (!CTime_version) {
		unsigned __int32 dummy_uint32;
		unsigned __int64 pos = input_lmf->tell(); 
		*input_lmf >> dummy_uint32;
		if ((int)dummy_uint32 == INT_MIN_+10) CTime_version = 2005; else CTime_version = 2003;
		input_lmf->seek(pos);
	}
	
	CTime_version_output = CTime_version;

	unsigned __int32 dummy_uint32;
	if(CTime_version >= 2005) {
		*input_lmf >> dummy_uint32;
		*input_lmf >> dummy_uint32; Starttime = dummy_uint32;
		*input_lmf >> dummy_uint32;
	} else {*input_lmf >> dummy_uint32; Starttime = dummy_uint32;}

	if(CTime_version >= 2005) {
		*input_lmf >> dummy_uint32;
		*input_lmf >> dummy_uint32; Stoptime = dummy_uint32;
		*input_lmf >> dummy_uint32;
	} else {
		*input_lmf >> dummy_uint32; Stoptime = dummy_uint32;
	}

	Read_CString_as_StdString(*input_lmf,Versionstring);
	Read_CString_as_StdString(*input_lmf,FilePathName);
	Read_CString_as_StdString(*input_lmf,Comment);
	Comment_output = Comment;

	byte_counter = 0;

	if (CCF_HISTORY_CODE_bitmasked) {
		*input_lmf >> number_of_CCFHistory_strings;
		CCFHistory_strings = new std::string*[number_of_CCFHistory_strings];
		memset(CCFHistory_strings,0,sizeof(std::string*)*number_of_CCFHistory_strings);
		for (__int32 i=0;i<number_of_CCFHistory_strings;++i) {
			__int32 string_len;
			*input_lmf >> string_len;					
			CCFHistory_strings[i] = new std::string();
			CCFHistory_strings[i]->reserve(string_len);
			while (string_len > 0) {
				__int8 c;
				*input_lmf >> c;	
				*CCFHistory_strings[i] += c;
				--string_len;
			}
		}
	}

	if (DAN_SOURCE_CODE_bitmasked) {
		*input_lmf >> number_of_DAN_source_strings;
		DAN_source_strings = new std::string*[number_of_DAN_source_strings];
		memset(DAN_source_strings,0,sizeof(std::string*)*number_of_DAN_source_strings);
		for (__int32 i=0;i<number_of_DAN_source_strings;++i) {
			__int32 string_len;
			*input_lmf >> string_len;
			DAN_source_strings[i] = new std::string();
			DAN_source_strings[i]->reserve(string_len);
			while (string_len > 0) {
				__int8 c;
				*input_lmf >> c; 
				*DAN_source_strings[i] += c;
				--string_len;
			}
		}
	}	



	if (User_header_size == 0) {
		errorflag = 6;
		InputFileIsOpen = false;
		goto L666;
	}

	if (__int32(input_lmf->tell()) != __int32(Headersize)) {
		errorflag = 6;
		InputFileIsOpen = false;
		goto L666;
	}



//  READ USER-HEADER
	if (Cobold_Header_version >= 2008) {
		*input_lmf >> LMF_Header_version;   byte_counter += sizeof(__int32);
	}

	if (Cobold_Header_version <= 2002) {*input_lmf >> unsigned_int_Dummy;	byte_counter += sizeof(__int32);}
	if (Cobold_Header_version >= 2008) {
		unsigned __int64 temp;
		*input_lmf >> temp;	unsigned_int_Dummy = (unsigned __int32)(temp);
		byte_counter += sizeof(unsigned __int64);
	}


	if (unsigned_int_Dummy != User_header_size) {
		errorflag = 6; // error reading header
		CloseInputLMF();
		return false;
	}

	*input_lmf >> DAQVersion;	byte_counter += sizeof(__int32);	// Version is always 2nd value

	*input_lmf >> DAQ_ID;		byte_counter += sizeof(__int32);	// DAQ_ID is always 3ed value

	if (DAQ_ID == DAQ_ID_TDC8)		goto L100;
	if (DAQ_ID == DAQ_ID_2TDC8)		goto L100;
	if (DAQ_ID == DAQ_ID_TDC8HP)	goto L100;
	if (DAQ_ID == DAQ_ID_TDC8HPRAW)	goto L100;
	if (DAQ_ID == DAQ_ID_HM1)		goto L100;
	if (DAQ_ID == DAQ_ID_CAMAC)		goto L100;
	if (DAQ_ID == DAQ_ID_HM1_ABM)	goto L100;
	if (DAQ_ID == DAQ_ID_TCPIP)		goto L100;

	errorflag = 7; // LMF not data of TDC8PCI2 or 2TDC8PCI2 or TDC8HP or CAMAC or HM1 or TCPIP
	CloseInputLMF();
	return false;

L100:
	if (DAQ_ID == DAQ_ID_TDC8)	 byte_counter += ReadTDC8PCI2Header();
	if (DAQ_ID == DAQ_ID_2TDC8)	 byte_counter += Read2TDC8PCI2Header();
	if (DAQ_ID == DAQ_ID_TDC8HP || DAQ_ID == DAQ_ID_TDC8HPRAW) byte_counter += ReadTDC8HPHeader_LMFV_1_to_7();
	if (DAQ_ID == DAQ_ID_HM1 || DAQ_ID == DAQ_ID_HM1_ABM)     byte_counter += ReadHM1Header();
	if (DAQ_ID == DAQ_ID_CAMAC)  byte_counter += ReadCAMACHeader();
	if (DAQ_ID == DAQ_ID_TCPIP)  byte_counter += ReadTCPIPHeader();

	if ((User_header_size != (unsigned __int32)byte_counter) || (data_format_in_userheader != data_format_in_header)) {
		if (!(DAQ_ID == DAQ_ID_TDC8 && this->LMF_Version == 0x8)) {
			errorflag = 6; // error reading header
			CloseInputLMF();
			return false;
		}
	}

	if (data_format_in_header != LM_USERDEF && data_format_in_userheader != LM_SHORT && data_format_in_userheader != LM_SLONG && data_format_in_userheader != LM_DOUBLE && data_format_in_userheader != LM_CAMAC) {
		errorflag = 6; // error reading header
		CloseInputLMF();
		return false;
	}

//	__int32 debug_pos = this->input_lmf->tell();

	errorflag = 0; // no error
	InputFileIsOpen = true;

L666:

	return true;
}

















/////////////////////////////////////////////////////////////////
__int32	LMF_IO::WriteTDC8PCI2Header()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;
	__int32 int_Dummy = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	int_Dummy = __int32(DAQ_info.length());
	*output_lmf << int_Dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += (unsigned __int32)(DAQ_info.length());

	if ((DAQVersion_output >= 20020408 && TDC8PCI2.use_normal_method)) {
		*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);
	}

	if (LMF_Version_output >= 0x9) {
		*output_lmf << number_of_DAQ_source_strings_output;  byte_counter += sizeof(__int32);

		if (DAQ_source_strings) {
			for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {
				unsigned __int32 unsigned_int_Dummy = DAQ_source_strings[i]->length();
				*output_lmf << unsigned_int_Dummy;   byte_counter += sizeof(__int32);
				output_lmf->write(DAQ_source_strings[i]->c_str(),DAQ_source_strings[i]->length());
				byte_counter += DAQ_source_strings[i]->length();
			}
		}
	}

	*output_lmf << system_timeout_output; byte_counter += sizeof(__int32);		//   system time-out
	*output_lmf << time_reference_output; byte_counter += sizeof(__int32);
	*output_lmf << common_mode_output; byte_counter += sizeof(__int32);		//   0 common start    1 common stop
	*output_lmf << tdcresolution_output; byte_counter += sizeof(double);		// tdc resolution in ns

	TDCDataType = 1;
	if (DAQVersion_output >= 20020408 && TDC8PCI2.use_normal_method) {*output_lmf << TDCDataType; byte_counter += sizeof(__int32);}
	*output_lmf << timerange; byte_counter += sizeof(__int32);	// time range of the tdc in microseconds

	if (DAQVersion_output < 20080507) {
		*output_lmf << number_of_channels_output; byte_counter += sizeof(__int32);			// number of channels
		*output_lmf << max_number_of_hits_output; byte_counter += sizeof(__int32);			// number of hits
	} else {
		__int64 i64_temp = number_of_channels_output;
		*output_lmf << i64_temp; byte_counter += sizeof(__int64);			// number of channels
		i64_temp = max_number_of_hits_output;
		*output_lmf << i64_temp; byte_counter += sizeof(__int64);			// number of hits
	}

	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	*output_lmf << module_2nd;	byte_counter += sizeof(__int32);	// indicator for 2nd module data
	
	if (DAQVersion_output >= 20020408 && TDC8PCI2.use_normal_method && (DAQ_ID_output == DAQ_ID_TDC8)) {
		*output_lmf << TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
		*output_lmf << TDC8PCI2.OpenTime_1st_card;			byte_counter += sizeof(__int32); // open time 1st card
		*output_lmf << TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
		*output_lmf << TDC8PCI2.TriggerFalling_1st_card;	byte_counter += sizeof(__int32); // trigger falling edge 1st card
		*output_lmf << TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card
		*output_lmf << TDC8PCI2.EmptyCounter_1st_card;		byte_counter += sizeof(__int32); // EmptyCounter 1st card
		*output_lmf << TDC8PCI2.EmptyCounter_since_last_Event_1st_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 1st card
	} 

	return byte_counter;
}




















/////////////////////////////////////////////////////////////////
void LMF_IO::CloseOutputLMF()
/////////////////////////////////////////////////////////////////
{
	if (!output_lmf) return;

	if (DAQ_ID_output == DAQ_ID_RAW32BIT) {
		output_lmf->close(); delete output_lmf; output_lmf = 0;
		OutputFileIsOpen = false;
		return;
	}

	if (Cobold_Header_version_output == 0) Cobold_Header_version_output = Cobold_Header_version;
	
	if (DAQ_ID_output == DAQ_ID_SIMPLE) {
		output_lmf->seek(0);

		*output_lmf << SIMPLE_DAQ_ID_Orignial;
		unsigned __int32 dummy = __int32(uint64_number_of_written_events);
		*output_lmf << dummy;
		*output_lmf << data_format_in_userheader_output;

		output_lmf->close(); delete output_lmf; output_lmf = 0;
		OutputFileIsOpen = false;
		return;
	}

//	if (out_ar) {
		output_lmf->seek(0);

		WriteFirstHeader();

		output_lmf->flush();
		Headersize_output = (unsigned __int32)(output_lmf->tell());
		unsigned __int64 seek_value;
		if (Cobold_Header_version_output <= 2002) seek_value = 3*sizeof(unsigned __int32);
		if (Cobold_Header_version_output >= 2008) seek_value = 2*sizeof(unsigned __int32) + sizeof(unsigned __int64);

		output_lmf->seek(seek_value);
		if (Cobold_Header_version_output <= 2002) *output_lmf << Headersize_output;
		if (Cobold_Header_version_output >= 2008) {
			unsigned __int64 temp = Headersize_output;
			*output_lmf << temp;
		}
		output_lmf->flush();
		output_lmf->seek(Headersize_output);

		if (Cobold_Header_version_output >= 2008 || DAQVersion_output >= 20080000) {
			*output_lmf << LMF_Header_version;
		}

		if (Cobold_Header_version_output <= 2002) *output_lmf << User_header_size_output;
		if (Cobold_Header_version_output >= 2008) {
			unsigned __int64 temp = User_header_size_output;
			*output_lmf << temp;
		}
		
		//out_ar->Close(); out_ar=0;
//	}
	
	if (output_lmf) {output_lmf->close(); delete output_lmf; output_lmf = 0;}
	OutputFileIsOpen = false;
}











/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadTDC8PCI2Header()
/////////////////////////////////////////////////////////////////
{
	__int32 byte_counter;
	byte_counter = 0;



	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit
	
	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;

//	input_lmf->flush();
	unsigned __int64 StartPosition = input_lmf->tell();
	__int32 old_byte_counter = byte_counter;
L50:
	byte_counter = old_byte_counter;

//	TRY
	if (DAQVersion >= 20020408 && TDC8PCI2.use_normal_method) {
		*input_lmf >> LMF_Version;
		byte_counter += sizeof(__int32);
	}

	if (DAQVersion >= 20080507) {
		if (LMF_Version >= 0x8 && data_format_in_userheader == -1) TDC8PCI2.variable_event_length = 1;
		if (LMF_Version >= 0x9) {
			*input_lmf >> number_of_DAQ_source_strings;    byte_counter += sizeof(__int32);
			DAQ_source_strings = new std::string*[number_of_DAQ_source_strings];
			memset(DAQ_source_strings,0,sizeof(std::string*)*number_of_DAQ_source_strings);
			for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {
				__int32 string_len;
				*input_lmf >> string_len;    byte_counter += sizeof(__int32);
				DAQ_source_strings[i] = new std::string();
				DAQ_source_strings[i]->reserve(string_len);
				while (string_len > 0) {
					__int8 c;
					*input_lmf >> c;     byte_counter += sizeof(__int8);
					*DAQ_source_strings[i] += c;
					--string_len;
				}
			}
		}
	}

	*input_lmf >> system_timeout;	byte_counter += sizeof(__int32);		//   system time-out
	*input_lmf >> time_reference;	byte_counter += sizeof(__int32);
	*input_lmf >> common_mode;		byte_counter += sizeof(__int32);			//   0 common start    1 common stop
	*input_lmf >> tdcresolution;	byte_counter += sizeof(double);	// tdc resolution in ns


	TDCDataType = 1;
	if (DAQVersion >= 20020408 && TDC8PCI2.use_normal_method) {*input_lmf >> TDCDataType; byte_counter += sizeof(__int32);}
	*input_lmf >> timerange; byte_counter += sizeof(__int32);			// time range of the tdc in microseconds

	if (this->DAQVersion < 20080507) {
		*input_lmf >> number_of_channels; byte_counter += sizeof(__int32);	// number of channels
		*input_lmf >> max_number_of_hits; byte_counter += sizeof(__int32);	// number of hits
	} else {
		__int64 i64_temp;
		*input_lmf >> i64_temp; number_of_channels = (unsigned __int32)(i64_temp); byte_counter += sizeof(__int64);	// number of channels
		*input_lmf >> i64_temp; max_number_of_hits = (unsigned __int32)(i64_temp); byte_counter += sizeof(__int64);	// number of hits
	}

	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}
	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	*input_lmf >> module_2nd;	byte_counter += sizeof(__int32);		// indicator for 2nd module data
	
	if (byte_counter == __int32(User_header_size) - 12) return byte_counter;

	if (DAQVersion >= 20020408 && TDC8PCI2.use_normal_method) {
		*input_lmf >> TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
		*input_lmf >> TDC8PCI2.OpenTime_1st_card;			byte_counter += sizeof(__int32); // open time 1st card
		*input_lmf >> TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
		*input_lmf >> TDC8PCI2.TriggerFalling_1st_card;		byte_counter += sizeof(__int32); // trigger falling edge 1st card
		*input_lmf >> TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card
		*input_lmf >> TDC8PCI2.EmptyCounter_1st_card;		byte_counter += sizeof(__int32); // EmptyCounter 1st card
		*input_lmf >> TDC8PCI2.EmptyCounter_since_last_Event_1st_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 1st card
	}

/*
	CATCH( CArchiveException, e )
		if (!TDC8PCI2.use_normal_method) return 0;
		TDC8PCI2.use_normal_method = false;
		in_ar->Close(); delete in_ar; in_ar = 0;
		input_lmf->seek(StartPosition);
		in_ar = new CArchive(input_lmf,CArchive::load);
		goto L50;
	END_CATCH
*/

	if (byte_counter != __int32(User_header_size) - 12 && DAQVersion < 20080507) {
		if (!TDC8PCI2.use_normal_method) return 0;
		TDC8PCI2.use_normal_method = false;
		input_lmf->seek(StartPosition);
		goto L50;
	}

	if (LMF_Version == 0x8) {
		input_lmf->flush();
		input_lmf->seek((unsigned __int64)(this->Headersize+this->User_header_size));
		input_lmf->flush();
	}

	return byte_counter;
}








/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadCAMACHeader()
/////////////////////////////////////////////////////////////////
{
	__int32 byte_counter;
	byte_counter = 0;

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;

	*input_lmf >> Camac_CIF_Length;	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[Camac_CIF_Length+1];
	input_lmf->read(__int8_temp,Camac_CIF_Length);
	__int8_temp[Camac_CIF_Length] = 0;
	Camac_CIF = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += Camac_CIF_Length;

	*input_lmf >> system_timeout; byte_counter += sizeof(__int32);		// system time-out
	*input_lmf >> time_reference; byte_counter += sizeof(__int32);
	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);		// data format (2=short integer)

	return byte_counter;
}





/////////////////////////////////////////////////////////////////
__int32	LMF_IO::Read2TDC8PCI2Header()
/////////////////////////////////////////////////////////////////
{
	unsigned __int64 StartPosition = 0;
	__int32 old_byte_counter = -1;
	bool desperate_mode = false;

	TDC8PCI2.variable_event_length = 0;
	__int32 byte_counter;
	byte_counter = 0;

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;

	*input_lmf >> LMF_Version; byte_counter += sizeof(__int32);
	*input_lmf >> system_timeout; byte_counter += sizeof(__int32);		// system time-out
	*input_lmf >> time_reference; byte_counter += sizeof(__int32);
	*input_lmf >> common_mode; byte_counter += sizeof(__int32);			// 0 common start    1 common stop
	*input_lmf >> tdcresolution; byte_counter += sizeof(double);	// tdc resolution in ns

	*input_lmf >> TDCDataType; byte_counter += sizeof(__int32);
	*input_lmf >> timerange; byte_counter += sizeof(__int32);			// time range of the tdc in microseconds
	if (DAQVersion >= 20080507) {
		TDC8PCI2.variable_event_length = 1;
		__int32 iDummy;
		unsigned __int64 i64temp;
		*input_lmf >> i64temp;		number_of_channels =__int32(i64temp);	byte_counter += sizeof(unsigned __int64);			// number of channels
		*input_lmf >> i64temp;		max_number_of_hits =__int32(i64temp);	byte_counter += sizeof(unsigned __int64);			// number of hits
		if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
		if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}

		*input_lmf >> i64temp;		number_of_channels2 =__int32(i64temp);	byte_counter += sizeof(unsigned __int64);			// number of channels2
		*input_lmf >> i64temp;		max_number_of_hits2 =__int32(i64temp);	byte_counter += sizeof(unsigned __int64);			// number of hits2
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);		// Sync Mode    (parameter 60)
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);		// IO address 2 (parameter 61)
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);
		*input_lmf >> iDummy;			byte_counter += sizeof(__int32);

		goto L200;
	}

	*input_lmf >> number_of_channels; byte_counter += sizeof(__int32);	// number of channels
	*input_lmf >> max_number_of_hits; byte_counter += sizeof(__int32);	// number of hits
	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}
	*input_lmf >> number_of_channels2; byte_counter += sizeof(__int32);	// number of channels2
	*input_lmf >> max_number_of_hits2; byte_counter += sizeof(__int32);	// number of hits2
	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);				// data format (2=short integer)

//	input_lmf->flush();
	StartPosition = input_lmf->tell();
	old_byte_counter = byte_counter;
	desperate_mode = false;
L50:
	byte_counter = old_byte_counter;

//	TRY

	if (TDC8PCI2.use_normal_method_2nd_card) {
		if (DAQVersion >= 20020408) {
			*input_lmf >> TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
			*input_lmf >> TDC8PCI2.OpenTime_1st_card ;			byte_counter += sizeof(__int32); // open time 1st card
			*input_lmf >> TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
			*input_lmf >> TDC8PCI2.TriggerFalling_1st_card;		byte_counter += sizeof(__int32); // trigger falling edge 1st card
			*input_lmf >> TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card
			*input_lmf >> TDC8PCI2.EmptyCounter_1st_card;		byte_counter += sizeof(__int32); // EmptyCounter 1st card
			*input_lmf >> TDC8PCI2.EmptyCounter_since_last_Event_1st_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 1st card
		}
		*input_lmf >> TDC8PCI2.sync_test_on_off;			byte_counter += sizeof(__int32); // sync test on/off
		*input_lmf >> TDC8PCI2.io_address_2nd_card;			byte_counter += sizeof(__int32); // io address 2nd card
		*input_lmf >> TDC8PCI2.GateDelay_2nd_card;			byte_counter += sizeof(__int32); // gate delay 2nd card
		*input_lmf >> TDC8PCI2.OpenTime_2nd_card;			byte_counter += sizeof(__int32); // open time 2nd card
		*input_lmf >> TDC8PCI2.WriteEmptyEvents_2nd_card;	byte_counter += sizeof(__int32); // write empty events 2nd card
		*input_lmf >> TDC8PCI2.TriggerFallingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger falling edge 2nd card
		*input_lmf >> TDC8PCI2.TriggerRisingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger rising edge 2nd card
		*input_lmf >> TDC8PCI2.EmptyCounter_2nd_card;		byte_counter += sizeof(__int32); // EmptyCounter 2nd card
		*input_lmf >> TDC8PCI2.EmptyCounter_since_last_Event_2nd_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 2nd card
	} else {
		*input_lmf >> module_2nd;							byte_counter += sizeof(__int32);	// indicator for 2nd module data
		*input_lmf >> TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
		*input_lmf >> TDC8PCI2.OpenTime_1st_card ;			byte_counter += sizeof(__int32); // open time 1st card
		*input_lmf >> TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
		*input_lmf >> TDC8PCI2.TriggerFalling_1st_card;		byte_counter += sizeof(__int32); // trigger falling edge 1st card
		*input_lmf >> TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card

		*input_lmf >> TDC8PCI2.GateDelay_2nd_card;			byte_counter += sizeof(__int32); // gate delay 2nd card
		if (!desperate_mode) { // this is only a quick fix.
			*input_lmf >> TDC8PCI2.OpenTime_2nd_card;			byte_counter += sizeof(__int32); // open time 2nd card
			*input_lmf >> TDC8PCI2.WriteEmptyEvents_2nd_card;	byte_counter += sizeof(__int32); // write empty events 2nd card
			*input_lmf >> TDC8PCI2.TriggerFallingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger falling edge 2nd card
			*input_lmf >> TDC8PCI2.TriggerRisingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger rising edge 2nd card
		}
	}

/*
	CATCH( CArchiveException, e )
		if (!TDC8PCI2.use_normal_method_2nd_card) return 0;
		TDC8PCI2.use_normal_method_2nd_card = false;
		in_ar->Close(); delete in_ar; in_ar = 0;
		input_lmf->seek(StartPosition);
		in_ar = new CArchive(input_lmf,CArchive::load);
		goto L50;
	END_CATCH
*/

L200:

	if (DAQVersion < 20080507) {
	        if (byte_counter != __int32(User_header_size) - 12) {
			if (desperate_mode) return 0;
			if (!TDC8PCI2.use_normal_method_2nd_card) {
				desperate_mode = true;
			}
			TDC8PCI2.use_normal_method_2nd_card = false;
			input_lmf->seek(StartPosition);
			goto L50;
		}
	}

	if (DAQVersion >= 20080507) {
	        if (byte_counter != __int32(User_header_size) - 20) {
			byte_counter = -1000; // XXX (this line is okay. I have put the XXX just to bring it to attention)
		}
	}

	return byte_counter;
}









/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadTDC8HPHeader_LMFV_1_to_7()
/////////////////////////////////////////////////////////////////
{
	this->TDC8HP.ui32oldRollOver = 0;
	this->TDC8HP.ui64RollOvers = 0;
	this->TDC8HP.ui32AbsoluteTimeStamp = 0;
	this->TDC8HP.ui64TDC8HP_AbsoluteTimeStamp = 0;

	__int32		byte_counter = 0;

	unsigned __int64 StartPosition1 = input_lmf->tell();

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;

//	input_lmf->flush();
	unsigned __int64 StartPosition = input_lmf->tell();
	__int32 old_byte_counter = byte_counter;
L50:
	byte_counter = old_byte_counter;

	if (DAQVersion > 20080000) TDC8HP.UserHeaderVersion = 4;

	if (TDC8HP.UserHeaderVersion >= 1) {
		*input_lmf >> LMF_Version; byte_counter += sizeof(__int32);
		if (LMF_Version == 8) TDC8HP.UserHeaderVersion = 5;
		if (LMF_Version >= 9) TDC8HP.UserHeaderVersion = 6;
	}
	
	if (TDC8HP.UserHeaderVersion >= 5) {
		input_lmf->seek(StartPosition1);
		return ReadTDC8HPHeader_LMFV_8_to_9();
	}

	*input_lmf >> time_reference; byte_counter += sizeof(__int32);
	*input_lmf >> tdcresolution; byte_counter += sizeof(double);		// tdc resolution in ns
	if (tdcresolution < 0.0001 || tdcresolution > 100.) {
		if (TDC8HP.UserHeaderVersion != 0) return 0;
		TDC8HP.UserHeaderVersion = 1;

		input_lmf->seek(StartPosition);
		goto L50;
	}
	tdcresolution = double(__int32(tdcresolution*10000+0.01))/10000.;
	if (TDC8HP.UserHeaderVersion >= 1) {
		*input_lmf >> TDCDataType; byte_counter += sizeof(__int32);
	}

	if (DAQVersion < 20080000) {
		*input_lmf >> number_of_channels; byte_counter += sizeof(__int32);			// number of channels
	} else {
		unsigned __int64 temp;
		*input_lmf >> temp;
		number_of_channels =__int32(temp);
		byte_counter += sizeof(unsigned __int64);			// number of channels
	}

	if (number_of_channels < 1 || number_of_channels > 100) {
		if (TDC8HP.UserHeaderVersion != 0) return 0;
//		in_ar->Close(); delete in_ar; in_ar = 0;
		input_lmf->seek(StartPosition);
//		in_ar = new CArchive(input_lmf,CArchive::load);
		goto L50;
	}


	if (DAQVersion < 20080000) {
		*input_lmf >> max_number_of_hits; byte_counter += sizeof(__int32);			// number of hits
	} else {
		unsigned __int64 temp;
		*input_lmf >> temp;
		max_number_of_hits =__int32(temp);
		byte_counter += sizeof(unsigned __int64);			// number of hits
	}

	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}

	if (max_number_of_hits < 1 || max_number_of_hits > 100) {
		if (TDC8HP.UserHeaderVersion != 0) return 0;
//		in_ar->Close(); delete in_ar; in_ar = 0;
		input_lmf->seek(StartPosition);
//		in_ar = new CArchive(input_lmf,CArchive::load);
		goto L50;
	}

	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);	// data format (2=short integer)
	*input_lmf >> TDC8HP.no_config_file_read;	byte_counter += sizeof(__int32);	// parameter 60-1
	if (DAQVersion < 20080000) {
		unsigned __int32 temp;
		*input_lmf >> temp; TDC8HP.RisingEnable_p61 = temp;		byte_counter += sizeof(__int32);	// parameter 61-1
		*input_lmf >> temp; TDC8HP.FallingEnable_p62 = temp;		byte_counter += sizeof(__int32);	// parameter 62-1
	}
	if (DAQVersion >= 20080000) {
		*input_lmf >> TDC8HP.RisingEnable_p61;		byte_counter += sizeof(__int64);	// parameter 61-1
		*input_lmf >> TDC8HP.FallingEnable_p62;		byte_counter += sizeof(__int64);	// parameter 62-1
	}
	*input_lmf >> TDC8HP.TriggerEdge_p63;		byte_counter += sizeof(__int32);	// parameter 63-1
	*input_lmf >> TDC8HP.TriggerChannel_p64;	byte_counter += sizeof(__int32);	// parameter 64-1
	*input_lmf >> TDC8HP.OutputLevel_p65;		byte_counter += sizeof(__int32);	// parameter 65-1
	*input_lmf >> TDC8HP.GroupingEnable_p66;	byte_counter += sizeof(__int32);	// parameter 66-1
	*input_lmf >> TDC8HP.AllowOverlap_p67;		byte_counter += sizeof(__int32);	// parameter 67-1
	*input_lmf >> TDC8HP.TriggerDeadTime_p68;	byte_counter += sizeof(double);		// parameter 68-1
	*input_lmf >> TDC8HP.GroupRangeStart_p69;	byte_counter += sizeof(double);		// parameter 69-1
	*input_lmf >> TDC8HP.GroupRangeEnd_p70;		byte_counter += sizeof(double);		// parameter 70-1
	*input_lmf >> TDC8HP.ExternalClock_p71;		byte_counter += sizeof(__int32);	// parameter 71-1
	*input_lmf >> TDC8HP.OutputRollOvers_p72;	byte_counter += sizeof(__int32);	// parameter 72-1
	*input_lmf >> TDC8HP.DelayTap0_p73;			byte_counter += sizeof(__int32);	// parameter 73-1
	*input_lmf >> TDC8HP.DelayTap1_p74;			byte_counter += sizeof(__int32);	// parameter 74-1
	*input_lmf >> TDC8HP.DelayTap2_p75;			byte_counter += sizeof(__int32);	// parameter 75-1
	*input_lmf >> TDC8HP.DelayTap3_p76;			byte_counter += sizeof(__int32);	// parameter 76-1
	*input_lmf >> TDC8HP.INL_p80;				byte_counter += sizeof(__int32);	// parameter 80-1
	*input_lmf >> TDC8HP.DNL_p81;				byte_counter += sizeof(__int32);	// parameter 81-1

	*input_lmf >> TDC8HP.csConfigFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csConfigFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csConfigFile_Length);
	__int8_temp[TDC8HP.csConfigFile_Length] = 0;
	TDC8HP.csConfigFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csConfigFile_Length;

	*input_lmf >> TDC8HP.csINLFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csINLFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csINLFile_Length);
	__int8_temp[TDC8HP.csINLFile_Length] = 0;
	TDC8HP.csINLFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csINLFile_Length;

	*input_lmf >> TDC8HP.csDNLFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csDNLFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csDNLFile_Length);
	__int8_temp[TDC8HP.csDNLFile_Length] = 0;
	TDC8HP.csDNLFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csDNLFile_Length;

	if (DAQVersion < 20080000) {
	        if (byte_counter == __int32(User_header_size) - 12 - 4) {  // Cobold 2002 v11
			if (TDC8HP.UserHeaderVersion < 2) TDC8HP.UserHeaderVersion = 2;
			*input_lmf >> TDC8HP.SyncValidationChannel;  byte_counter += sizeof(__int32); 	// parameter 77-1
		}
	        if (byte_counter == __int32(User_header_size) - 12 - 4 - 4) { // never used in official Cobold releases
			TDC8HP.UserHeaderVersion = 3;
			*input_lmf >> TDC8HP.SyncValidationChannel;  byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.VHR_25ps;				 byte_counter += sizeof(bool);
		}
	}

	if (this->DAQVersion >= 20080000) {
		if (this->data_format_in_userheader == -1) TDC8HP.variable_event_length = 1;
		*input_lmf >> TDC8HP.SyncValidationChannel;  byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.VHR_25ps;  byte_counter += sizeof(bool);

		*input_lmf >> TDC8HP.GroupTimeOut;  byte_counter += sizeof(double);

		*input_lmf >> TDC8HP.SSEEnable;  byte_counter += sizeof(bool);
		*input_lmf >> TDC8HP.MMXEnable;  byte_counter += sizeof(bool);
		*input_lmf >> TDC8HP.DMAEnable;  byte_counter += sizeof(bool);

		//// read TDCInfo
		*input_lmf >> TDC8HP.Number_of_TDCs;  byte_counter += sizeof(__int32);
		for(__int32 iCount=0;iCount<TDC8HP.Number_of_TDCs;++iCount) {
			*input_lmf >> TDC8HP.TDC_info[iCount]->index;				byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->channelCount;			byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->channelStart;			byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->highResChannelCount;	byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->highResChannelStart;	byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->lowResChannelCount;	byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->lowResChannelStart;	byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->resolution;			byte_counter += sizeof(double);
			*input_lmf >> TDC8HP.TDC_info[iCount]->serialNumber;			byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->version;				byte_counter += sizeof(__int32);
			*input_lmf >> TDC8HP.TDC_info[iCount]->fifoSize;				byte_counter += sizeof(__int32);
			input_lmf->read((__int8*)TDC8HP.TDC_info[iCount]->INLCorrection,sizeof(__int32)*8*1024);		byte_counter += sizeof(__int32)*8*1024;
			input_lmf->read((__int8*)TDC8HP.TDC_info[iCount]->DNLData,sizeof(unsigned __int16)*8*1024);	byte_counter += sizeof(unsigned __int16)*8*1024;
			*input_lmf >> TDC8HP.TDC_info[iCount]->flashValid;			byte_counter += sizeof(bool);
		}

		bool bool_dummy;
		*input_lmf >> bool_dummy; byte_counter += sizeof(bool);
		*input_lmf >> bool_dummy; byte_counter += sizeof(bool);
		*input_lmf >> bool_dummy; byte_counter += sizeof(bool);
		*input_lmf >> bool_dummy; byte_counter += sizeof(bool);
		*input_lmf >> bool_dummy; byte_counter += sizeof(bool);

		*input_lmf >> TDC8HP.i32NumberOfDAQLoops;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC8HP_DriverVersion;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.iTriggerChannelMask;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.iTime_zero_channel;	byte_counter += sizeof(__int32);

		__int32 int_dummy;
		*input_lmf >> int_dummy; byte_counter += sizeof(__int32);

		double double_dummy;
		*input_lmf >> double_dummy; byte_counter += sizeof(double);
		*input_lmf >> double_dummy; byte_counter += sizeof(double);
		*input_lmf >> double_dummy; byte_counter += sizeof(double);
		*input_lmf >> double_dummy; byte_counter += sizeof(double);
		*input_lmf >> double_dummy; byte_counter += sizeof(double);
	}

	return byte_counter;
}











/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadTDC8HPHeader_LMFV_8_to_9()
// reads LMF headers from Cobold 2008 R2 (release August 2009)
/////////////////////////////////////////////////////////////////
{
	this->TDC8HP.ui32oldRollOver = 0;
	this->TDC8HP.ui64RollOvers = 0;
	this->TDC8HP.ui32AbsoluteTimeStamp = 0;
	this->TDC8HP.ui64TDC8HP_AbsoluteTimeStamp = 0;

	__int32		byte_counter = 0;

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;

//	input_lmf->flush();
	//unsigned __int64 StartPosition = input_lmf->tell();
	__int32 old_byte_counter = byte_counter;

	byte_counter = old_byte_counter;

	if (DAQVersion > 20080000) TDC8HP.UserHeaderVersion = 4;

	if (TDC8HP.UserHeaderVersion >= 1) {
		*input_lmf >> LMF_Version; byte_counter += sizeof(__int32);
		if (LMF_Version == 8) TDC8HP.UserHeaderVersion = 5;
		if (LMF_Version >= 9) TDC8HP.UserHeaderVersion = 6;
	}

	*input_lmf >> this->number_of_DAQ_source_strings;  byte_counter += sizeof(__int32);

	DAQ_source_strings = new std::string*[number_of_DAQ_source_strings];
	memset(DAQ_source_strings,0,sizeof(std::string*)*number_of_DAQ_source_strings);
	for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {
		__int32 string_len;
		*input_lmf >> string_len;    byte_counter += sizeof(__int32);
		DAQ_source_strings[i] = new std::string();
		DAQ_source_strings[i]->reserve(string_len);
		while (string_len > 0) {
			__int8 c;
			*input_lmf >> c;     byte_counter += sizeof(__int8);
			*DAQ_source_strings[i] += c;
			--string_len;
		}
	}
	
	*input_lmf >> time_reference; byte_counter += sizeof(__int32);
	*input_lmf >> tdcresolution; byte_counter += sizeof(double);		// tdc resolution in ns
	*input_lmf >> TDCDataType; byte_counter += sizeof(__int32);

	unsigned __int64 temp_uint64;
	*input_lmf >> temp_uint64;
	number_of_channels =__int32(temp_uint64);
	byte_counter += sizeof(unsigned __int64);			// number of channels

	*input_lmf >> temp_uint64;
	max_number_of_hits =__int32(temp_uint64);
	byte_counter += sizeof(unsigned __int64);			// number of hits

	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}

	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);	// data format (2=short integer)

	bool temp_bool;

	*input_lmf >> temp_bool; TDC8HP.no_config_file_read = temp_bool; byte_counter += sizeof(bool);	// parameter 60-1
		
	*input_lmf >> TDC8HP.RisingEnable_p61;		byte_counter += sizeof(__int64);	// parameter 61-1
	*input_lmf >> TDC8HP.FallingEnable_p62;		byte_counter += sizeof(__int64);	// parameter 62-1
	*input_lmf >> TDC8HP.TriggerEdge_p63;		byte_counter += sizeof(__int32);	// parameter 63-1
	*input_lmf >> TDC8HP.TriggerChannel_p64;	byte_counter += sizeof(__int32);	// parameter 64-1

	*input_lmf >> temp_bool; TDC8HP.OutputLevel_p65    = temp_bool;	byte_counter += sizeof(bool);	// parameter 65-1
	*input_lmf >> temp_bool; TDC8HP.GroupingEnable_p66 = temp_bool;	byte_counter += sizeof(bool);	// parameter 66-1
	*input_lmf >> temp_bool; TDC8HP.AllowOverlap_p67   = temp_bool;	byte_counter += sizeof(bool);	// parameter 67-1

	*input_lmf >> TDC8HP.TriggerDeadTime_p68;	byte_counter += sizeof(double);		// parameter 68-1
	*input_lmf >> TDC8HP.GroupRangeStart_p69;	byte_counter += sizeof(double);		// parameter 69-1
	*input_lmf >> TDC8HP.GroupRangeEnd_p70;		byte_counter += sizeof(double);		// parameter 70-1

	*input_lmf >> temp_bool; TDC8HP.ExternalClock_p71   = temp_bool;	byte_counter += sizeof(bool);	// parameter 71-1
	*input_lmf >> temp_bool; TDC8HP.OutputRollOvers_p72 = temp_bool;	byte_counter += sizeof(bool);	// parameter 72-1

	*input_lmf >> TDC8HP.DelayTap0_p73;			byte_counter += sizeof(__int32);	// parameter 73-1
	*input_lmf >> TDC8HP.DelayTap1_p74;			byte_counter += sizeof(__int32);	// parameter 74-1
	*input_lmf >> TDC8HP.DelayTap2_p75;			byte_counter += sizeof(__int32);	// parameter 75-1
	*input_lmf >> TDC8HP.DelayTap3_p76;			byte_counter += sizeof(__int32);	// parameter 76-1

	*input_lmf >> temp_bool; TDC8HP.INL_p80 = temp_bool;	byte_counter += sizeof(bool);	// parameter 80-1
	*input_lmf >> temp_bool; TDC8HP.DNL_p81 = temp_bool;	byte_counter += sizeof(bool);	// parameter 81-1

	*input_lmf >> TDC8HP.csConfigFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csConfigFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csConfigFile_Length);
	__int8_temp[TDC8HP.csConfigFile_Length] = 0;
	TDC8HP.csConfigFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csConfigFile_Length;

	*input_lmf >> TDC8HP.csINLFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csINLFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csINLFile_Length);
	__int8_temp[TDC8HP.csINLFile_Length] = 0;
	TDC8HP.csINLFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csINLFile_Length;

	*input_lmf >> TDC8HP.csDNLFile_Length;
	byte_counter += sizeof(__int32);
	__int8_temp = new __int8[TDC8HP.csDNLFile_Length+1];
	input_lmf->read(__int8_temp,TDC8HP.csDNLFile_Length);
	__int8_temp[TDC8HP.csDNLFile_Length] = 0;
	TDC8HP.csDNLFile = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += TDC8HP.csDNLFile_Length;

	if (this->data_format_in_userheader == -1) TDC8HP.variable_event_length = 1;
	*input_lmf >> TDC8HP.SyncValidationChannel;  byte_counter += sizeof(__int32);
	*input_lmf >> TDC8HP.VHR_25ps;  byte_counter += sizeof(bool);
	*input_lmf >> TDC8HP.GroupTimeOut;  byte_counter += sizeof(double);
	*input_lmf >> TDC8HP.SSEEnable;  byte_counter += sizeof(bool);
	*input_lmf >> TDC8HP.MMXEnable;  byte_counter += sizeof(bool);
	*input_lmf >> TDC8HP.DMAEnable;  byte_counter += sizeof(bool);

	//// read TDCInfo
	*input_lmf >> TDC8HP.Number_of_TDCs;  byte_counter += sizeof(__int32);
	for(__int32 iCount=0;iCount<TDC8HP.Number_of_TDCs;++iCount) {
		*input_lmf >> TDC8HP.TDC_info[iCount]->index;				byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->channelCount;			byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->channelStart;			byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->highResChannelCount;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->highResChannelStart;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->lowResChannelCount;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->lowResChannelStart;	byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->resolution;			byte_counter += sizeof(double);
		*input_lmf >> TDC8HP.TDC_info[iCount]->serialNumber;			byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->version;				byte_counter += sizeof(__int32);
		*input_lmf >> TDC8HP.TDC_info[iCount]->fifoSize;				byte_counter += sizeof(__int32);
		input_lmf->read((__int8*)TDC8HP.TDC_info[iCount]->INLCorrection,sizeof(__int32)*8*1024);		byte_counter += sizeof(__int32)*8*1024;
		input_lmf->read((__int8*)TDC8HP.TDC_info[iCount]->DNLData,sizeof(unsigned __int16)*8*1024);	byte_counter += sizeof(unsigned __int16)*8*1024;
		*input_lmf >> TDC8HP.TDC_info[iCount]->flashValid;			byte_counter += sizeof(bool);
	}

	*input_lmf >> TDC8HP.number_of_bools; byte_counter += sizeof(__int32);
	for (__int32 i=0;i<TDC8HP.number_of_bools;++i) {
		bool bool_dummy;
		*input_lmf >> bool_dummy;	byte_counter += sizeof(bool);
	}

	*input_lmf >> TDC8HP.number_of_int32s; byte_counter += sizeof(__int32);
	*input_lmf >> TDC8HP.i32NumberOfDAQLoops;	byte_counter += sizeof(__int32);
	*input_lmf >> TDC8HP.TDC8HP_DriverVersion;	byte_counter += sizeof(__int32);
	*input_lmf >> TDC8HP.iTriggerChannelMask;	byte_counter += sizeof(__int32);
	*input_lmf >> TDC8HP.iTime_zero_channel;	byte_counter += sizeof(__int32); // 1 is first channel
	
	if (TDC8HP.UserHeaderVersion >= 6) {
		*input_lmf >> TDC8HP.BinsizeType;	byte_counter += sizeof(__int32); // 1 is first channel
		for (__int32 i=5;i<TDC8HP.number_of_int32s;++i) {
			__int32 int_dummy;
			*input_lmf >> int_dummy;	byte_counter += sizeof(__int32);
		}
	} else {
		for (__int32 i=4;i<TDC8HP.number_of_int32s;++i) {
			__int32 int_dummy;
			*input_lmf >> int_dummy;	byte_counter += sizeof(__int32);
		}
	}
	

	

	*input_lmf >> TDC8HP.number_of_doubles; byte_counter += sizeof(__int32);
	if (TDC8HP.UserHeaderVersion == 5) {
		for (__int32 i=0;i<TDC8HP.number_of_doubles;++i) {
			double double_dummy;
			*input_lmf >> double_dummy; byte_counter += sizeof(double);
		}
	}
	if (TDC8HP.UserHeaderVersion >= 6) {
		*input_lmf >> TDC8HP.OffsetTimeZeroChannel_s;  byte_counter += sizeof(double);
		for (__int32 i=1;i<TDC8HP.number_of_doubles;++i) {
			double double_dummy;
			*input_lmf >> double_dummy; byte_counter += sizeof(double);
		}
	}

	return byte_counter;
}














/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadTCPIPHeader()
/////////////////////////////////////////////////////////////////
{
	__int32	byte_counter = 0;
 

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;
	*input_lmf >> LMF_Version; byte_counter += sizeof(__int32);
	*input_lmf >> time_reference; byte_counter += sizeof(__int32);

	*input_lmf >> data_format_in_userheader;	byte_counter   += sizeof(__int32);
	*input_lmf >> number_of_channels;			byte_counter   += sizeof(__int32);
	*input_lmf >> max_number_of_hits;			byte_counter   += sizeof(__int32);

	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}

	tdcresolution = 1.;

	return byte_counter;
}














/////////////////////////////////////////////////////////////////
__int32	LMF_IO::ReadHM1Header()
/////////////////////////////////////////////////////////////////
{
	__int32		byte_counter;
	byte_counter = 0;

	HM1.use_normal_method = false;

	*input_lmf >> frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*input_lmf >> IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*input_lmf >> timestamp_format;	byte_counter += sizeof(__int32);	// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	*input_lmf >> DAQ_info_Length;	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	__int8 * __int8_temp = new __int8[DAQ_info_Length+1];
	input_lmf->read(__int8_temp,DAQ_info_Length);	// DAQInfo always 8th value
	__int8_temp[DAQ_info_Length] = 0;
	DAQ_info = __int8_temp;
	delete[] __int8_temp; __int8_temp = 0;
	byte_counter += DAQ_info_Length;


	__int32 nominalHeaderLength;
	nominalHeaderLength = sizeof(__int32)*21 + sizeof(double)*2 + DAQ_info_Length;	// size of user defined header
	if (DAQ_ID == DAQ_ID_HM1_ABM) nominalHeaderLength += 24*sizeof(__int32);
	if (nominalHeaderLength == __int32(User_header_size)) HM1.use_normal_method = true;

	if (DAQVersion >= 20020408 && HM1.use_normal_method) {
		*input_lmf >> LMF_Version; byte_counter += sizeof(__int32);
	}

	*input_lmf >> system_timeout; byte_counter += sizeof(__int32);		//   system time-out
	*input_lmf >> time_reference; byte_counter += sizeof(__int32);
	*input_lmf >> HM1.FAK_DLL_Value; byte_counter += sizeof(__int32);
	*input_lmf >> HM1.Resolution_Flag; byte_counter += sizeof(__int32);
	*input_lmf >> HM1.trigger_mode_for_start; byte_counter += sizeof(__int32);
	*input_lmf >> HM1.trigger_mode_for_stop; byte_counter += sizeof(__int32);
	*input_lmf >> tdcresolution; byte_counter += sizeof(double);		// tdc resolution in ns

	if (DAQVersion >= 20020408 && HM1.use_normal_method) {
		*input_lmf >> TDCDataType; byte_counter += sizeof(__int32);
	}

	*input_lmf >> HM1.Even_open_time; byte_counter += sizeof(__int32);
	*input_lmf >> HM1.Auto_Trigger; byte_counter += sizeof(__int32);
	*input_lmf >> number_of_channels; byte_counter += sizeof(__int32);			// number of channels
	*input_lmf >> max_number_of_hits; byte_counter += sizeof(__int32);			// number of hits
	if (int(number_of_channels) > num_channels)	{errorflag = 19;	return -100000;}
	if (int(max_number_of_hits) > num_ions)		{errorflag = 20;	return -100000;}
	*input_lmf >> HM1.set_bits_for_GP1; byte_counter += sizeof(__int32);
	*input_lmf >> data_format_in_userheader;	byte_counter += sizeof(__int32);				// data format (2=short integer)
	*input_lmf >> module_2nd;	byte_counter += sizeof(__int32);		// indicator for 2nd module data

	if (DAQ_ID == DAQ_ID_2HM1) {
		*input_lmf >> DAQSubVersion;						byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_FAK_DLL_Value;				byte_counter += sizeof(__int32);  // parameter 10-1
		*input_lmf >> HM1.TWOHM1_Resolution_Flag;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_trigger_mode_for_start;	byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_trigger_mode_for_stop;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_res_adjust;				byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_tdcresolution;				byte_counter += sizeof(double);
		*input_lmf >> HM1.TWOHM1_test_overflow;				byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_number_of_channels;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_number_of_hits;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_set_bits_for_GP1;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_HM1_ID_1;					byte_counter += sizeof(__int32);
		*input_lmf >> HM1.TWOHM1_HM1_ID_2;					byte_counter += sizeof(__int32);
	}

	if (DAQ_ID == DAQ_ID_HM1_ABM) {
		max_number_of_hits = 1;
		*input_lmf >> HM1.ABM_m_xFrom;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_xTo;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_yFrom;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_yTo;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_xMin;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_xMax;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_yMin;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_yMax;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_xOffset;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_yOffset;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_m_zOffset;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_Mode;				byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_OsziDarkInvert;	byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_ErrorHisto;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_XShift;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_YShift;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_ZShift;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_ozShift;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_wdShift;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_ucLevelXY;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_ucLevelZ;			byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_uiABMXShift;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_uiABMYShift;		byte_counter += sizeof(__int32);
		*input_lmf >> HM1.ABM_uiABMZShift;		byte_counter += sizeof(__int32);
	}

	return byte_counter;
}





























/////////////////////////////////////////////////////////////////
__int32 LMF_IO::WriteTCPIPHeader()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = (unsigned __int32)(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value

	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += (unsigned __int32)(DAQ_info.length());

	*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);
	*output_lmf << time_reference; byte_counter += sizeof(__int32);


	*output_lmf << data_format_in_userheader_output;					byte_counter   += sizeof(__int32);
	*output_lmf << number_of_channels_output;			byte_counter   += sizeof(__int32);
	*output_lmf << max_number_of_hits_output;			byte_counter   += sizeof(__int32);

	return byte_counter;
}

















/////////////////////////////////////////////////////////////////
__int32	LMF_IO::Write2TDC8PCI2Header()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;
	//__int32 int_Dummy = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = (unsigned __int32)(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += __int32(DAQ_info.length());

	*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);

	*output_lmf << system_timeout_output; byte_counter += sizeof(__int32);		//   system time-out
	*output_lmf << time_reference_output; byte_counter += sizeof(__int32);
	*output_lmf << common_mode_output; byte_counter += sizeof(__int32);		//   0 common start    1 common stop
	*output_lmf << tdcresolution_output; byte_counter += sizeof(double);		// tdc resolution in ns

	TDCDataType = 1;
	*output_lmf << TDCDataType; byte_counter += sizeof(__int32);
	*output_lmf << timerange; byte_counter += sizeof(__int32);	// time range of the tdc in microseconds

	*output_lmf << number_of_channels_output; byte_counter += sizeof(__int32);			// number of channels
	*output_lmf << max_number_of_hits_output; byte_counter += sizeof(__int32);			// number of hits
	*output_lmf << number_of_channels2_output; byte_counter += sizeof(__int32);	// number of channels2
	*output_lmf << max_number_of_hits2_output; byte_counter += sizeof(__int32);	// number of hits2
	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	if (TDC8PCI2.use_normal_method_2nd_card) {
		if (DAQVersion_output >= 20020408) {
			*output_lmf << TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
			*output_lmf << TDC8PCI2.OpenTime_1st_card ;			byte_counter += sizeof(__int32); // open time 1st card
			*output_lmf << TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
			*output_lmf << TDC8PCI2.TriggerFalling_1st_card;	byte_counter += sizeof(__int32); // trigger falling edge 1st card
			*output_lmf << TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card
			*output_lmf << TDC8PCI2.EmptyCounter_1st_card;		byte_counter += sizeof(__int32); // EmptyCounter 1st card
			*output_lmf << TDC8PCI2.EmptyCounter_since_last_Event_1st_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 1st card
		}
		*output_lmf << TDC8PCI2.sync_test_on_off;			byte_counter += sizeof(__int32); // sync test on/off
		*output_lmf << TDC8PCI2.io_address_2nd_card;		byte_counter += sizeof(__int32); // io address 2nd card
		*output_lmf << TDC8PCI2.GateDelay_2nd_card;			byte_counter += sizeof(__int32); // gate delay 2nd card
		*output_lmf << TDC8PCI2.OpenTime_2nd_card;			byte_counter += sizeof(__int32); // open time 2nd card
		*output_lmf << TDC8PCI2.WriteEmptyEvents_2nd_card;	byte_counter += sizeof(__int32); // write empty events 2nd card
		*output_lmf << TDC8PCI2.TriggerFallingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger falling edge 2nd card
		*output_lmf << TDC8PCI2.TriggerRisingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger rising edge 2nd card
		*output_lmf << TDC8PCI2.EmptyCounter_2nd_card;		byte_counter += sizeof(__int32); // EmptyCounter 2nd card
		*output_lmf << TDC8PCI2.EmptyCounter_since_last_Event_2nd_card;	byte_counter += sizeof(__int32); // Empty Counter since last event 2nd card
	} else {
		*output_lmf << module_2nd;							byte_counter += sizeof(__int32); // indicator for 2nd module data
		*output_lmf << TDC8PCI2.GateDelay_1st_card;			byte_counter += sizeof(__int32); // gate delay 1st card
		*output_lmf << TDC8PCI2.OpenTime_1st_card ;			byte_counter += sizeof(__int32); // open time 1st card
		*output_lmf << TDC8PCI2.WriteEmptyEvents_1st_card;	byte_counter += sizeof(__int32); // write empty events 1st card
		*output_lmf << TDC8PCI2.TriggerFalling_1st_card;	byte_counter += sizeof(__int32); // trigger falling edge 1st card
		*output_lmf << TDC8PCI2.TriggerRising_1st_card;		byte_counter += sizeof(__int32); // trigger rising edge 1st card

		*output_lmf << TDC8PCI2.GateDelay_2nd_card;			byte_counter += sizeof(__int32); // gate delay 2nd card
		*output_lmf << TDC8PCI2.OpenTime_2nd_card;			byte_counter += sizeof(__int32); // open time 2nd card
		*output_lmf << TDC8PCI2.WriteEmptyEvents_2nd_card;	byte_counter += sizeof(__int32); // write empty events 2nd card
		*output_lmf << TDC8PCI2.TriggerFallingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger falling edge 2nd card
		*output_lmf << TDC8PCI2.TriggerRisingEdge_2nd_card;	byte_counter += sizeof(__int32); // trigger rising edge 2nd card
	}
	return byte_counter;
}






/////////////////////////////////////////////////////////////////
__int32	LMF_IO::WriteTDC8HPHeader_LMFV_1_to_7()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;
	double	double_Dummy = 0.;
	__int32		int_Dummy = 0;
	//unsigned __int32 unsigned_int_Dummy = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = (unsigned __int32)(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value

	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += (unsigned __int32)(DAQ_info.length());

	if ((DAQVersion_output >= 20020408 && TDC8HP.UserHeaderVersion >=1) || TDC8HP.UserHeaderVersion >= 4) {
		*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);
	}

	*output_lmf << time_reference; byte_counter += sizeof(__int32);
	*output_lmf << tdcresolution_output; byte_counter += sizeof(double);		// tdc resolution in ns

	TDCDataType = 1;
	if ((DAQVersion_output >= 20020408 && TDC8HP.UserHeaderVersion >= 1) || TDC8HP.UserHeaderVersion >= 4) {*output_lmf << TDCDataType; byte_counter += sizeof(__int32);}

	if (DAQVersion_output < 20080000) {
		*output_lmf << number_of_channels_output; byte_counter += sizeof(__int32);			// number of channels
		*output_lmf << max_number_of_hits_output; byte_counter += sizeof(__int32);			// number of hits
	}
	if (DAQVersion_output >= 20080000) {
		unsigned __int64 temp;
		temp = number_of_channels_output;	*output_lmf << temp; byte_counter += sizeof(unsigned __int64);			// number of channels
		temp = max_number_of_hits_output;	*output_lmf << temp; byte_counter += sizeof(unsigned __int64);			// number of hits
	}
	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	*output_lmf << TDC8HP.no_config_file_read;	byte_counter += sizeof(__int32);	// parameter 60-1
	if (DAQVersion_output < 20080000) {
		unsigned __int32 temp;
		temp =__int32(TDC8HP.RisingEnable_p61);  *output_lmf << temp;	byte_counter += sizeof(__int32);	// parameter 61-1
		temp =__int32(TDC8HP.FallingEnable_p62); *output_lmf << temp;	byte_counter += sizeof(__int32);	// parameter 62-1
	}
	if (DAQVersion_output >= 20080000) {
		*output_lmf << TDC8HP.RisingEnable_p61;		byte_counter += sizeof(__int64);	// parameter 61-1
		*output_lmf << TDC8HP.FallingEnable_p62;	byte_counter += sizeof(__int64);	// parameter 62-1
	}
	*output_lmf << TDC8HP.TriggerEdge_p63;		byte_counter += sizeof(__int32);	// parameter 63-1
	*output_lmf << TDC8HP.TriggerChannel_p64;	byte_counter += sizeof(__int32);	// parameter 64-1
	*output_lmf << TDC8HP.OutputLevel_p65;		byte_counter += sizeof(__int32);	// parameter 65-1
	*output_lmf << TDC8HP.GroupingEnable_p66_output;	byte_counter += sizeof(__int32);	// parameter 66-1
	*output_lmf << TDC8HP.AllowOverlap_p67;		byte_counter += sizeof(__int32);	// parameter 67-1
	*output_lmf << TDC8HP.TriggerDeadTime_p68;	byte_counter += sizeof(double);	// parameter 68-1
	*output_lmf << TDC8HP.GroupRangeStart_p69;	byte_counter += sizeof(double);	// parameter 69-1
	*output_lmf << TDC8HP.GroupRangeEnd_p70;	byte_counter += sizeof(double);	// parameter 70-1
	*output_lmf << TDC8HP.ExternalClock_p71;	byte_counter += sizeof(__int32);	// parameter 71-1
	*output_lmf << TDC8HP.OutputRollOvers_p72;	byte_counter += sizeof(__int32);	// parameter 72-1
	*output_lmf << TDC8HP.DelayTap0_p73;		byte_counter += sizeof(__int32);	// parameter 73-1
	*output_lmf << TDC8HP.DelayTap1_p74;		byte_counter += sizeof(__int32);	// parameter 74-1
	*output_lmf << TDC8HP.DelayTap2_p75;		byte_counter += sizeof(__int32);	// parameter 75-1
	*output_lmf << TDC8HP.DelayTap3_p76;		byte_counter += sizeof(__int32);	// parameter 76-1
	*output_lmf << TDC8HP.INL_p80;				byte_counter += sizeof(__int32);	// parameter 80-1
	*output_lmf << TDC8HP.DNL_p81;				byte_counter += sizeof(__int32);	// parameter 81-1

	dummy = __int32(TDC8HP.csConfigFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csConfigFile.c_str(), __int32(TDC8HP.csConfigFile.length()));
	byte_counter += __int32(TDC8HP.csConfigFile.length());

	dummy = __int32(TDC8HP.csINLFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csINLFile.c_str(), __int32(TDC8HP.csINLFile.length()));
	byte_counter += __int32(TDC8HP.csINLFile.length());

	dummy = __int32(TDC8HP.csDNLFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csDNLFile.c_str(), __int32(TDC8HP.csDNLFile.length()));
	byte_counter += __int32(TDC8HP.csDNLFile.length());

	if (TDC8HP.UserHeaderVersion >= 2 ) {
		*output_lmf << TDC8HP.SyncValidationChannel; byte_counter += sizeof(__int32);
	}
	if (TDC8HP.UserHeaderVersion >= 3 ) {
		*output_lmf << TDC8HP.VHR_25ps; byte_counter += sizeof(bool);
	}

	if (TDC8HP.UserHeaderVersion >= 4) {

		*output_lmf << TDC8HP.GroupTimeOut;  byte_counter += sizeof(double);

		*output_lmf << TDC8HP.SSEEnable;  byte_counter += sizeof(bool);
		*output_lmf << TDC8HP.MMXEnable;  byte_counter += sizeof(bool);
		*output_lmf << TDC8HP.DMAEnable;  byte_counter += sizeof(bool);

		//// write TDCInfo
		*output_lmf << TDC8HP.Number_of_TDCs;	byte_counter += sizeof(__int32);
		for(__int32 iCount=0;iCount<TDC8HP.Number_of_TDCs;++iCount) {
			*output_lmf << TDC8HP.TDC_info[iCount]->index;				byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->channelCount;		byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->channelStart;		byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->highResChannelCount;	byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->highResChannelStart;	byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->lowResChannelCount;	byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->lowResChannelStart;	byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->resolution;			byte_counter += sizeof(double);
			*output_lmf << TDC8HP.TDC_info[iCount]->serialNumber;		byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->version;			byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.TDC_info[iCount]->fifoSize;			byte_counter += sizeof(__int32);
			output_lmf->write((__int8*)TDC8HP.TDC_info[iCount]->INLCorrection,sizeof(__int32)*8*1024);		byte_counter += sizeof(__int32)*8*1024;
			output_lmf->write((__int8*)TDC8HP.TDC_info[iCount]->DNLData,sizeof(unsigned __int16)*8*1024);	byte_counter += sizeof(unsigned __int16)*8*1024;
			*output_lmf << TDC8HP.TDC_info[iCount]->flashValid;			byte_counter += sizeof(bool);
		}

		bool bool_dummy = false;
		*output_lmf << bool_dummy; byte_counter += sizeof(bool);
		*output_lmf << bool_dummy; byte_counter += sizeof(bool);
		*output_lmf << bool_dummy; byte_counter += sizeof(bool);
		*output_lmf << bool_dummy; byte_counter += sizeof(bool);
		*output_lmf << bool_dummy; byte_counter += sizeof(bool);

		*output_lmf << TDC8HP.i32NumberOfDAQLoops;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC8HP_DriverVersion; byte_counter += sizeof(__int32);	
		*output_lmf << TDC8HP.iTriggerChannelMask;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.iTime_zero_channel;	byte_counter += sizeof(__int32);
		*output_lmf << int_Dummy;				byte_counter += sizeof(__int32);

		*output_lmf << double_Dummy; byte_counter += sizeof(double);
		*output_lmf << double_Dummy; byte_counter += sizeof(double);
		*output_lmf << double_Dummy; byte_counter += sizeof(double);
		*output_lmf << double_Dummy; byte_counter += sizeof(double);
		*output_lmf << double_Dummy; byte_counter += sizeof(double);
	}
	
	return byte_counter;
}












/////////////////////////////////////////////////////////////////
__int32	LMF_IO::WriteTDC8HPHeader_LMFV_8_to_9()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter	= 0;

	bool			 bool_dummy = false;
	//double			 double_Dummy	= 0.;
	//__int32			 int_Dummy		= 0;
	unsigned __int32 unsigned_int_Dummy = 0;
	__int64			 int64_dummy	= 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = (unsigned __int32)(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value

	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += (unsigned __int32)(DAQ_info.length());

	*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);


	*output_lmf << number_of_DAQ_source_strings_output;  byte_counter += sizeof(__int32);

	if (DAQ_source_strings) {
		for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {
			unsigned_int_Dummy = DAQ_source_strings[i]->length();
			*output_lmf << unsigned_int_Dummy;   byte_counter += sizeof(__int32);
			output_lmf->write(DAQ_source_strings[i]->c_str(),DAQ_source_strings[i]->length());
			byte_counter += DAQ_source_strings[i]->length();
		}
	}

	*output_lmf << time_reference; byte_counter += sizeof(__int32);
	*output_lmf << tdcresolution_output; byte_counter += sizeof(double);		// tdc resolution in ns

	TDCDataType = 1;
	*output_lmf << TDCDataType; byte_counter += sizeof(__int32);

	int64_dummy = number_of_channels_output;
	*output_lmf << int64_dummy; byte_counter += sizeof(__int64);			// number of channels
	int64_dummy = max_number_of_hits_output;
	*output_lmf << int64_dummy; byte_counter += sizeof(__int64);			// number of hits
	
	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	bool_dummy = TDC8HP.no_config_file_read ? true: false;
	*output_lmf << bool_dummy;	byte_counter += sizeof(bool);	// parameter 60-1
	*output_lmf << TDC8HP.RisingEnable_p61;		byte_counter += sizeof(__int64);	// parameter 61-1
	*output_lmf << TDC8HP.FallingEnable_p62;	byte_counter += sizeof(__int64);	// parameter 62-1
	
	*output_lmf << TDC8HP.TriggerEdge_p63;		byte_counter += sizeof(__int32);	// parameter 63-1
	*output_lmf << TDC8HP.TriggerChannel_p64;	byte_counter += sizeof(__int32);	// parameter 64-1
	bool_dummy = TDC8HP.OutputLevel_p65 ? true:false;
	*output_lmf << bool_dummy;		byte_counter += sizeof(bool);	// parameter 65-1
	bool_dummy = TDC8HP.GroupingEnable_p66_output ? true : false;
	*output_lmf << bool_dummy;	byte_counter += sizeof(bool);	// parameter 66-1
	bool_dummy = TDC8HP.AllowOverlap_p67 ? true: false;
	*output_lmf << bool_dummy;		byte_counter += sizeof(bool);	// parameter 67-1
	*output_lmf << TDC8HP.TriggerDeadTime_p68;	byte_counter += sizeof(double);	// parameter 68-1
	*output_lmf << TDC8HP.GroupRangeStart_p69;	byte_counter += sizeof(double);	// parameter 69-1
	*output_lmf << TDC8HP.GroupRangeEnd_p70;	byte_counter += sizeof(double);	// parameter 70-1
	bool_dummy = TDC8HP.ExternalClock_p71 ? true:false;
	*output_lmf << bool_dummy;	byte_counter += sizeof(bool);	// parameter 71-1
	bool_dummy = TDC8HP.OutputRollOvers_p72 ? true:false;
	*output_lmf << bool_dummy;	byte_counter += sizeof(bool);	// parameter 72-1
	*output_lmf << TDC8HP.DelayTap0_p73;		byte_counter += sizeof(__int32);	// parameter 73-1
	*output_lmf << TDC8HP.DelayTap1_p74;		byte_counter += sizeof(__int32);	// parameter 74-1
	*output_lmf << TDC8HP.DelayTap2_p75;		byte_counter += sizeof(__int32);	// parameter 75-1
	*output_lmf << TDC8HP.DelayTap3_p76;		byte_counter += sizeof(__int32);	// parameter 76-1
	bool_dummy = TDC8HP.INL_p80 ? true : false;
	*output_lmf << bool_dummy;				byte_counter += sizeof(bool);	// parameter 80-1
	bool_dummy = TDC8HP.DNL_p81 ? true:false;
	*output_lmf << bool_dummy;				byte_counter += sizeof(bool);	// parameter 81-1

	dummy = __int32(TDC8HP.csConfigFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csConfigFile.c_str(), __int32(TDC8HP.csConfigFile.length()));
	byte_counter += __int32(TDC8HP.csConfigFile.length());

	dummy = __int32(TDC8HP.csINLFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csINLFile.c_str(), __int32(TDC8HP.csINLFile.length()));
	byte_counter += __int32(TDC8HP.csINLFile.length());

	dummy = __int32(TDC8HP.csDNLFile.length());
	*output_lmf << dummy;	byte_counter += sizeof(__int32);
	output_lmf->write(TDC8HP.csDNLFile.c_str(), __int32(TDC8HP.csDNLFile.length()));
	byte_counter += __int32(TDC8HP.csDNLFile.length());

	*output_lmf << TDC8HP.SyncValidationChannel; byte_counter += sizeof(__int32);
	*output_lmf << TDC8HP.VHR_25ps; byte_counter += sizeof(bool);
	
	*output_lmf << TDC8HP.GroupTimeOut;  byte_counter += sizeof(double);

	*output_lmf << TDC8HP.SSEEnable;  byte_counter += sizeof(bool);
	*output_lmf << TDC8HP.MMXEnable;  byte_counter += sizeof(bool);
	*output_lmf << TDC8HP.DMAEnable;  byte_counter += sizeof(bool);

	//// write TDCInfo
	*output_lmf << TDC8HP.Number_of_TDCs;	byte_counter += sizeof(__int32);
	for(__int32 iCount=0;iCount<TDC8HP.Number_of_TDCs;++iCount) {
		*output_lmf << TDC8HP.TDC_info[iCount]->index;				byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->channelCount;		byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->channelStart;		byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->highResChannelCount;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->highResChannelStart;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->lowResChannelCount;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->lowResChannelStart;	byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->resolution;			byte_counter += sizeof(double);
		*output_lmf << TDC8HP.TDC_info[iCount]->serialNumber;		byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->version;			byte_counter += sizeof(__int32);
		*output_lmf << TDC8HP.TDC_info[iCount]->fifoSize;			byte_counter += sizeof(__int32);
		output_lmf->write((__int8*)TDC8HP.TDC_info[iCount]->INLCorrection,sizeof(__int32)*8*1024);		byte_counter += sizeof(__int32)*8*1024;
		output_lmf->write((__int8*)TDC8HP.TDC_info[iCount]->DNLData,sizeof(unsigned __int16)*8*1024);	byte_counter += sizeof(unsigned __int16)*8*1024;
		*output_lmf << TDC8HP.TDC_info[iCount]->flashValid;			byte_counter += sizeof(bool);
	}


	*output_lmf << TDC8HP.number_of_bools; byte_counter += sizeof(__int32);
	for (__int32 i=4; i<TDC8HP.number_of_bools; ++i) {
		bool bdummy = false;
		*output_lmf << bdummy; byte_counter += sizeof(bool);
	}

	if (TDC8HP.number_of_int32s < 4) TDC8HP.number_of_int32s = 4;
	if (TDC8HP.UserHeaderVersion >= 6) TDC8HP.number_of_int32s = 5;
	*output_lmf << TDC8HP.number_of_int32s; byte_counter += sizeof(__int32);
	*output_lmf << TDC8HP.i32NumberOfDAQLoops;	byte_counter += sizeof(__int32);
	*output_lmf << TDC8HP.TDC8HP_DriverVersion; byte_counter += sizeof(__int32);	
	*output_lmf << TDC8HP.iTriggerChannelMask;	byte_counter += sizeof(__int32);
	*output_lmf << TDC8HP.iTime_zero_channel;	byte_counter += sizeof(__int32);

	if (TDC8HP.UserHeaderVersion == 5) {
		for (__int32 i=4; i<TDC8HP.number_of_int32s; ++i) {
			__int32 idummy = 0;
			*output_lmf << idummy; byte_counter += sizeof(__int32);
		}
	} else {
		*output_lmf << TDC8HP.BinsizeType;	byte_counter += sizeof(__int32);
		for (__int32 i=5; i<TDC8HP.number_of_int32s; ++i) {
			__int32 idummy = 0;
			*output_lmf << idummy; byte_counter += sizeof(__int32);
		}
	}

	if (TDC8HP.UserHeaderVersion == 5) {
		*output_lmf << TDC8HP.number_of_doubles; byte_counter += sizeof(__int32);
		for (__int32 i=0; i<TDC8HP.number_of_doubles; ++i) {
			 double ddummy = 0.;
			*output_lmf << ddummy; byte_counter += sizeof(double);
		}
	}
	if (TDC8HP.UserHeaderVersion >= 6) {
		if (TDC8HP.number_of_doubles == 0) {
			*output_lmf << __int32(1); byte_counter += sizeof(__int32);
			*output_lmf << double(0.); byte_counter += sizeof(double);
		} else {
			*output_lmf << TDC8HP.number_of_doubles; byte_counter += sizeof(__int32);
			*output_lmf << TDC8HP.OffsetTimeZeroChannel_s; byte_counter += sizeof(double);
			for (__int32 i=1; i<TDC8HP.number_of_doubles; ++i) {
				 double ddummy = 0.;
				*output_lmf << ddummy; byte_counter += sizeof(double);
			}
		}
	}
	
	return byte_counter;
}












/////////////////////////////////////////////////////////////////
__int32	LMF_IO::WriteHM1Header()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;
	//__int32 int_Dummy = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = (unsigned __int32)(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += __int32(DAQ_info.length());

	if (DAQVersion_output >= 20020408 && HM1.use_normal_method) {*output_lmf << LMF_Version_output; byte_counter += sizeof(__int32);}

	*output_lmf << system_timeout_output; byte_counter += sizeof(__int32);		//   system time-out
	*output_lmf << time_reference_output; byte_counter += sizeof(__int32);
	if (DAQ_ID_output == DAQ_ID_HM1 || DAQ_ID_output == DAQ_ID_HM1_ABM) {
		*output_lmf << HM1.FAK_DLL_Value; byte_counter += sizeof(__int32);
		*output_lmf << HM1.Resolution_Flag; byte_counter += sizeof(__int32);
		*output_lmf << HM1.trigger_mode_for_start; byte_counter += sizeof(__int32);
		*output_lmf << HM1.trigger_mode_for_stop; byte_counter += sizeof(__int32);
	}
	*output_lmf << tdcresolution_output; byte_counter += sizeof(double);		// tdc resolution in ns

	TDCDataType = 1;
	if (DAQVersion_output >= 20020408 && HM1.use_normal_method) {*output_lmf << TDCDataType; byte_counter += sizeof(__int32);}
	
	if (DAQ_ID_output == DAQ_ID_HM1 || DAQ_ID_output == DAQ_ID_HM1_ABM) {
		*output_lmf << HM1.Even_open_time; byte_counter += sizeof(__int32);
		*output_lmf << HM1.Auto_Trigger; byte_counter += sizeof(__int32);
	}

	*output_lmf << number_of_channels_output; byte_counter += sizeof(__int32);			// number of channels
	if (DAQ_ID_output == DAQ_ID_HM1_ABM) {
		unsigned __int32 dummy = 0;
		*output_lmf << dummy; byte_counter += sizeof(__int32);
	} else { 
		*output_lmf << max_number_of_hits_output; byte_counter += sizeof(__int32);			// number of hits
	}
	if (DAQ_ID_output == DAQ_ID_HM1 || DAQ_ID_output == DAQ_ID_HM1_ABM) {
		*output_lmf << HM1.set_bits_for_GP1; byte_counter += sizeof(__int32);
	}
	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);				// data format (2=short integer)

	if (DAQ_ID_output == DAQ_ID_HM1 || DAQ_ID_output == DAQ_ID_HM1_ABM) {*output_lmf << module_2nd;	byte_counter += sizeof(__int32);}	// indicator for 2nd module data

	if (DAQ_ID_output == DAQ_ID_2HM1) {
		*output_lmf << DAQSubVersion;						byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_FAK_DLL_Value;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_Resolution_Flag;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_trigger_mode_for_start;	byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_trigger_mode_for_stop;	byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_res_adjust;				byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_tdcresolution;			byte_counter += sizeof(double);
		*output_lmf << HM1.TWOHM1_test_overflow;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_number_of_channels;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_number_of_hits;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_set_bits_for_GP1;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_HM1_ID_1;					byte_counter += sizeof(__int32);
		*output_lmf << HM1.TWOHM1_HM1_ID_2;					byte_counter += sizeof(__int32);
	}

	if (DAQ_ID_output == DAQ_ID_HM1_ABM) {
		*output_lmf << HM1.ABM_m_xFrom;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_xTo;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_yFrom;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_yTo;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_xMin;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_xMax;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_yMin;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_yMax;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_xOffset;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_yOffset;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_m_zOffset;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_Mode;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_OsziDarkInvert;	byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_ErrorHisto;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_XShift;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_YShift;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_ZShift;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_ozShift;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_wdShift;			byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_ucLevelXY;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_ucLevelZ;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_uiABMXShift;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_uiABMYShift;		byte_counter += sizeof(__int32);
		*output_lmf << HM1.ABM_uiABMZShift;		byte_counter += sizeof(__int32);
	}

	return byte_counter;
}







/////////////////////////////////////////////////////////////////
__int32	LMF_IO::WriteCAMACHeader()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32 byte_counter;
	byte_counter = 0;

	*output_lmf << frequency;	byte_counter += sizeof(double);		// frequency is always 4th value
	*output_lmf << IOaddress;	byte_counter += sizeof(__int32);		// IO address (parameter 1) always 5th value
	*output_lmf << timestamp_format_output;	byte_counter += sizeof(__int32);		// TimeInfo (parameter 2) always 6th value  (0,1,2)*32Bit

	unsigned __int32 dummy = __int32(DAQ_info.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);		// Length of DAQInfo always 7th value
	output_lmf->write(DAQ_info.c_str(), __int32(DAQ_info.length()));	// DAQInfo always 8th value
	byte_counter += __int32(DAQ_info.length());

	dummy = __int32(Camac_CIF.length());
	*output_lmf << dummy;
	byte_counter += sizeof(__int32);
	output_lmf->write(Camac_CIF.c_str(), __int32(Camac_CIF.length()));
	byte_counter += __int32(Camac_CIF.length());

	*output_lmf << system_timeout_output; byte_counter += sizeof(__int32);		// system time-out
	*output_lmf << time_reference_output; byte_counter += sizeof(__int32);
	*output_lmf << data_format_in_userheader_output;	byte_counter += sizeof(__int32);		// data format (2=short integer)

	return byte_counter;
}





bool LMF_IO::OpenOutputLMF(std::string LMF_Filename)
{
	return OpenOutputLMF((__int8*)LMF_Filename.c_str());
}



/////////////////////////////////////////////////////////////////
bool LMF_IO::OpenOutputLMF(__int8 * LMF_Filename)
/////////////////////////////////////////////////////////////////
{
  //double				double_Dummy = 0.;
  //unsigned __int32	unsigned_int_Dummy = 0;
  //__int32				int_Dummy = 0;

	if (OutputFileIsOpen) {
		errorflag = 12; // file is already open
		return false;
	}
	output_lmf = new MyFILE(false);

	output_lmf->open(LMF_Filename);

	if (output_lmf->error) {
		errorflag = 11; // could not open output file
		return false;
	}

//	out_ar = new CArchive(output_lmf,CArchive::store);
	
/*	if (!out_ar) {
		errorflag = 13; // could not connect CAchrive to output file
		output_lmf->Close(); output_lmf = 0;
		return false;
	}
*/

	if (number_of_DAQ_source_strings_output == -1 && number_of_DAQ_source_strings > 0) {
		number_of_DAQ_source_strings_output = number_of_DAQ_source_strings;
		DAQ_source_strings_output = new std::string*[number_of_DAQ_source_strings];
		memset(DAQ_source_strings_output,0,sizeof(std::string*)*number_of_DAQ_source_strings_output);
		for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {DAQ_source_strings_output[i] = new std::string(); *DAQ_source_strings_output[i] = *DAQ_source_strings[i];}
	} else number_of_DAQ_source_strings_output = 0;

	if (number_of_CCFHistory_strings_output == -1 && number_of_CCFHistory_strings > 0) {
		number_of_CCFHistory_strings_output = number_of_CCFHistory_strings;
		CCFHistory_strings_output = new std::string*[number_of_CCFHistory_strings];
		memset(CCFHistory_strings_output,0,sizeof(std::string*)*number_of_CCFHistory_strings_output);
		for (__int32 i=0;i<number_of_CCFHistory_strings;++i)  {CCFHistory_strings_output[i] = new std::string(); *CCFHistory_strings_output[i] = *CCFHistory_strings[i];}
	} else number_of_CCFHistory_strings_output = 0;

	if (number_of_DAN_source_strings_output == -1 && number_of_DAN_source_strings > 0) {
		number_of_DAN_source_strings_output = number_of_DAN_source_strings;
		DAN_source_strings_output = new std::string*[number_of_DAN_source_strings];
		memset(DAN_source_strings_output,0,sizeof(std::string*)*number_of_DAN_source_strings_output);
		for (__int32 i=0;i<number_of_DAN_source_strings;++i) {DAN_source_strings_output[i] = new std::string(); *DAN_source_strings_output[i] = *DAN_source_strings[i];}
	} else number_of_DAN_source_strings_output = 0;

	if (DAQ_ID_output == DAQ_ID_RAW32BIT) {
		if (number_of_channels_output  == 0) number_of_channels_output  = number_of_channels;
		if (max_number_of_hits_output  == 0) max_number_of_hits_output  = max_number_of_hits;
		if (number_of_channels_output == 0 || max_number_of_hits_output == 0) {errorflag =14; return false;}
		Numberofcoordinates_output = number_of_channels_output*(1+max_number_of_hits_output);
		data_format_in_userheader_output = 10;

		errorflag = 0; // no error
		OutputFileIsOpen = true;
		return true;
	}

	if (DAQ_ID_output == DAQ_ID_SIMPLE) {
		if (number_of_channels_output  == 0) number_of_channels_output  = number_of_channels;
		if (max_number_of_hits_output  == 0) max_number_of_hits_output  = max_number_of_hits;
		if (number_of_channels_output == 0 || max_number_of_hits_output == 0) {errorflag =14; return false;}
		Numberofcoordinates_output = number_of_channels_output*(1+max_number_of_hits_output);
		if (SIMPLE_DAQ_ID_Orignial == 0) SIMPLE_DAQ_ID_Orignial = DAQ_ID;
		*output_lmf << SIMPLE_DAQ_ID_Orignial;
		unsigned __int32 dummy = (unsigned __int32)(uint64_number_of_written_events);
		*output_lmf << dummy;
		*output_lmf << data_format_in_userheader_output;

		errorflag = 0; // no error
		OutputFileIsOpen = true;
		return true;
	}


	


// Preparing to write LMF-header:
// -----------------------------------
	OutputFilePathName = LMF_Filename;

	Headersize_output = 0;

	if (int(DAQVersion_output) == -1) DAQVersion_output = DAQVersion;

	if (DAQVersion_output >= 20080000 && Cobold_Header_version_output == 0) Cobold_Header_version_output = 2008;
	if (Cobold_Header_version_output == 0) Cobold_Header_version_output = Cobold_Header_version;

	if (tdcresolution_output < 0.) tdcresolution_output = tdcresolution;
	if (Starttime_output == 0)	{
		Starttime_output = Starttime;
	}
	if (Stoptime_output == 0)	{
		Stoptime_output = Stoptime;
	}
	if (system_timeout_output == -1) system_timeout_output = system_timeout;
	if (time_reference_output ==  0) time_reference_output = time_reference;
	if (common_mode_output    == -1) common_mode_output    = common_mode;

	if (number_of_channels_output  == 0) number_of_channels_output  = number_of_channels;
	if (max_number_of_hits_output  == 0) max_number_of_hits_output  = max_number_of_hits;
	
	if (data_format_in_userheader_output == -2) data_format_in_userheader_output = data_format_in_userheader;
	if (DAQ_ID_output == 0) DAQ_ID_output = DAQ_ID;
	if (DAQ_ID_output == DAQ_ID_TDC8HPRAW) DAQ_ID_output = DAQ_ID_TDC8HP;

	if (DAQ_ID_output == DAQ_ID_2TDC8) {
		if (number_of_channels2_output == -1) number_of_channels2_output = number_of_channels2;
		if (max_number_of_hits2_output == -1) max_number_of_hits2_output = max_number_of_hits2;
	}

	if (int(timestamp_format_output) == -1) timestamp_format_output = timestamp_format;

	if (Numberofcoordinates_output == -2) {
		if (data_format_in_userheader_output == LM_SHORT)  Numberofcoordinates_output = timestamp_format_output * 2;
		if (data_format_in_userheader_output == LM_DOUBLE) Numberofcoordinates_output = timestamp_format_output == 0 ? 0 : 1;
		if (data_format_in_userheader_output == LM_SLONG)  Numberofcoordinates_output = timestamp_format_output;
		if (data_format_in_userheader_output == LM_SHORT || data_format_in_userheader_output == LM_DOUBLE || data_format_in_userheader_output == LM_SLONG) {
			Numberofcoordinates_output += number_of_channels_output  * (max_number_of_hits_output +1);
			Numberofcoordinates_output += number_of_channels2_output * (max_number_of_hits2_output+1);
		}
		if (data_format_in_userheader_output == LM_CAMAC) Numberofcoordinates_output = Numberofcoordinates;

		if (data_format_in_userheader_output == LM_USERDEF) {
			if (DAQVersion_output < 20080000) {this->errorflag = 17; return false;}
			if (DAQ_ID_output == DAQ_ID_TDC8HP) {
				//Numberofcoordinates_output = 2 + timestamp_format_output + number_of_channels_output*(1+max_number_of_hits_output); // old
				Numberofcoordinates_output = number_of_channels_output*(1+max_number_of_hits_output);
			}
			if (DAQ_ID_output == DAQ_ID_TDC8 || DAQ_ID_output == DAQ_ID_2TDC8) {
				//Numberofcoordinates_output = 2 + timestamp_format_output*2 + (number_of_channels_output + number_of_channels2_output)*(1+max_number_of_hits_output); // old
				Numberofcoordinates_output = (number_of_channels_output + number_of_channels2_output)*(1+max_number_of_hits_output);
			}
			if (DAQ_ID_output == DAQ_ID_HM1) {
				//Numberofcoordinates_output = 2 + timestamp_format_output*2 + (number_of_channels_output + number_of_channels2_output)*(1+max_number_of_hits_output); // old
				Numberofcoordinates_output = (number_of_channels_output + number_of_channels2_output)*(1+max_number_of_hits_output);
			}
		}
	}



//  WRITE LMF-HEADER:


	WriteFirstHeader();

	output_lmf->flush();

	Headersize_output = (unsigned __int32)(output_lmf->tell());

	unsigned __int64 seek_value;
	if (Cobold_Header_version_output <= 2002) seek_value = 3*sizeof(unsigned __int32);
	if (Cobold_Header_version_output >= 2008) seek_value = 2*sizeof(unsigned __int32) + sizeof(unsigned __int64);
	output_lmf->seek(seek_value);


	if (Cobold_Header_version_output <= 2002) *output_lmf << Headersize_output;
	if (Cobold_Header_version_output >= 2008) {
		unsigned __int64 temp = Headersize_output;
		*output_lmf << temp;
	}

	output_lmf->flush();
	output_lmf->seek(Headersize_output);

	if (LMF_Version_output == -1) {
		LMF_Version_output = LMF_Version;
		if (LMF_Version_output == -1) LMF_Version_output = 8; // XXX if necessary: modify to latest LMF version number
	}

	output_byte_counter = 0;

//  WRITE USER-HEADER
	if (Cobold_Header_version_output >= 2008 || DAQVersion_output >= 20080000) {
		*output_lmf << LMF_Header_version;
		output_byte_counter += sizeof(__int32);
	}
	if (Cobold_Header_version_output <= 2002) {
		*output_lmf << User_header_size_output;
		output_byte_counter += sizeof(__int32);
	}
	if (Cobold_Header_version_output >= 2008) {
		unsigned __int64 temp = User_header_size_output;
		*output_lmf << temp;
		output_byte_counter += sizeof(unsigned __int64);
	}

	*output_lmf << DAQVersion_output;			output_byte_counter += sizeof(__int32);	// Version is always 2nd value
	*output_lmf << DAQ_ID_output;		output_byte_counter += sizeof(__int32);	// DAQ_ID is always 3ed value

	if (DAQ_ID_output == DAQ_ID_TDC8)	 output_byte_counter += WriteTDC8PCI2Header();
	if (DAQ_ID_output == DAQ_ID_2TDC8)	 output_byte_counter += Write2TDC8PCI2Header();
	if (DAQ_ID_output == DAQ_ID_TDC8HP || DAQ_ID_output == DAQ_ID_TDC8HPRAW)	 {
		if (this->LMF_Version_output <  8) output_byte_counter += WriteTDC8HPHeader_LMFV_1_to_7();
		if (this->LMF_Version_output >= 8) output_byte_counter += WriteTDC8HPHeader_LMFV_8_to_9();
	}

	if (DAQ_ID_output == DAQ_ID_HM1)	 output_byte_counter += WriteHM1Header();
	if (DAQ_ID_output == DAQ_ID_HM1_ABM) output_byte_counter += WriteHM1Header();
	if (DAQ_ID_output == DAQ_ID_CAMAC)   output_byte_counter += WriteCAMACHeader();
	if (DAQ_ID_output == DAQ_ID_TCPIP)   output_byte_counter += WriteTCPIPHeader();

	User_header_size_output = output_byte_counter;

	errorflag = 0; // no error
	OutputFileIsOpen = true;
	return true;
}











/////////////////////////////////////////////////////////////////
void LMF_IO::WriteFirstHeader()
/////////////////////////////////////////////////////////////////
{
	if (Cobold_Header_version_output >= 2008) {
		unsigned __int32 ArchiveFlagtemp = 476759;

		if (number_of_DAN_source_strings_output > 0) ArchiveFlagtemp = ArchiveFlagtemp | DAN_SOURCE_CODE;
		if (number_of_DAQ_source_strings_output > 0) ArchiveFlagtemp = ArchiveFlagtemp | DAQ_SOURCE_CODE; 
		if (number_of_CCFHistory_strings_output > 0) ArchiveFlagtemp = ArchiveFlagtemp | CCF_HISTORY_CODE; 

		*output_lmf << ArchiveFlagtemp;
		*output_lmf << data_format_in_userheader_output;

		unsigned __int64 temp;
		temp = Numberofcoordinates_output;  *output_lmf << temp;
		temp = Headersize_output;			*output_lmf << temp;
		temp = User_header_size_output;		*output_lmf << temp;
		*output_lmf << uint64_number_of_written_events;
	}

	if (Cobold_Header_version_output <= 2002) {
		unsigned __int32 ArchiveFlagtemp = 476758;
		*output_lmf << ArchiveFlagtemp;
		*output_lmf << data_format_in_userheader_output;

		*output_lmf << Numberofcoordinates_output;
		*output_lmf << Headersize_output;
		*output_lmf << User_header_size_output;
		unsigned __int32 dummy = (unsigned __int32)(uint64_number_of_written_events);
		*output_lmf << dummy;
	}

	write_times(output_lmf,Starttime_output,Stoptime_output);

	Write_StdString_as_CString(*output_lmf,Versionstring);
	Write_StdString_as_CString(*output_lmf,OutputFilePathName);
	Write_StdString_as_CString(*output_lmf,Comment_output);


	if (number_of_CCFHistory_strings_output > 0) {
		*output_lmf << number_of_CCFHistory_strings_output;
		for (__int32 i=0;i<number_of_CCFHistory_strings_output;++i) {
			unsigned __int32 unsigned_int_Dummy = CCFHistory_strings_output[i]->length();
			*output_lmf << unsigned_int_Dummy;					
			output_lmf->write(CCFHistory_strings_output[i]->c_str(),CCFHistory_strings_output[i]->length());
		}
	}

	if (number_of_DAN_source_strings_output > 0) {
		*output_lmf << number_of_DAN_source_strings_output;
		for (__int32 i=0;i<number_of_DAN_source_strings_output;++i) {
			unsigned __int32 unsigned_int_Dummy = DAN_source_strings_output[i]->length();
			*output_lmf << unsigned_int_Dummy;
			output_lmf->write(DAN_source_strings_output[i]->c_str(),DAN_source_strings_output[i]->length());
		}
	}
}














/////////////////////////////////////////////////////////////////
void LMF_IO::WriteEventHeader(unsigned __int64 timestamp, unsigned __int32 cnt[])
/////////////////////////////////////////////////////////////////
{
	unsigned __int64 HeaderLength = 0;
	// EventLength information in 64Bit
	if (DAQVersion_output >= 20080000 || data_format_in_userheader_output == LM_USERDEF) {
		HeaderLength = timestamp_format_output + 1 + number_of_channels_output;	// +1 for HeaderLength itself

		HeaderLength = timestamp_format_output*sizeof(__int32) + 2*sizeof(__int64);	// +2 for HeaderLength itself, +2 for EventCounter (size in __int32)
		for(__int32 iCount=0;iCount<number_of_channels_output;++iCount) HeaderLength += cnt[iCount]*sizeof(__int32);
		HeaderLength += number_of_channels_output*sizeof(__int16);
#ifdef LINUX
		HeaderLength = (HeaderLength & 0x00ffffffffffffffLL) | 0xff00000000000000LL;	// set 0xff in bits 63..56 as EventMarker
#else
		HeaderLength = (HeaderLength & 0x00ffffffffffffff) | 0xff00000000000000;	// set 0xff in bits 63..56 as EventMarker
#endif
		*output_lmf << HeaderLength;
		*output_lmf << uint64_number_of_written_events; 
	}

	myLARGE_INTEGER LARGE_Timestamp;
	LARGE_Timestamp.QuadPart = timestamp;

	if (timestamp_format_output > 0) {
		*output_lmf << LARGE_Timestamp.LowPart;
		if (timestamp_format_output == 2) *output_lmf << LARGE_Timestamp.HighPart;
	}
}













/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(unsigned __int64 timestamp, unsigned __int32 cnt[], __int32 * i32TDC)
/////////////////////////////////////////////////////////////////
{
	unsigned __int16 dummy_uint16;
	__int32 dummy_int32;
	double dummy_double;

	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	
	++uint64_number_of_written_events;

	WriteEventHeader(timestamp,cnt);

	__int32 i,j;
	if (DAQ_ID_output != DAQ_ID_SIMPLE) {
		for (i=0;i<number_of_channels_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits_output) hits = max_number_of_hits_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {
					dummy_uint16 = (unsigned __int16)(i32TDC[i*num_ions+j]);
					*output_lmf << dummy_uint16;
				}
				dummy_uint16 = (unsigned __int16)(0);
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);
				*output_lmf << dummy_double;
				for (j=0;j<hits;++j) {
					dummy_double = double(i32TDC[i*num_ions+j]);
					*output_lmf << dummy_double;
				}
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				*output_lmf << __int32(hits);
				for (j=0;j<hits;++j) *output_lmf << i32TDC[i*num_ions+j];
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_int32;
			}
			if (data_format_in_userheader_output == LM_USERDEF) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) *output_lmf << i32TDC[i*num_ions+j];
				if (DAQ_ID_output == DAQ_ID_TDC8HP && this->LMF_Version_output >= 9) {
					*output_lmf >> ui64LevelInfo;
				}
			}
		}
	}
	if (DAQ_ID_output == DAQ_ID_2TDC8) {
		for (i=number_of_channels_output;i<number_of_channels2_output+number_of_channels_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits2_output) hits = max_number_of_hits2_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);  *output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {
					dummy_uint16 = (unsigned __int16)(i32TDC[i*num_ions+j]);
					*output_lmf << dummy_uint16;
				}
				dummy_uint16 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);		*output_lmf << dummy_double;
				for (j=0;j<hits;++j) {
					dummy_double = double(i32TDC[i*num_ions+j]);
					*output_lmf << dummy_double;
			}
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				dummy_int32 = __int32(hits);		*output_lmf << dummy_int32;
				for (j=0;j<hits;++j) *output_lmf << i32TDC[i*num_ions+j];
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_int32;
			}
		}
	}

	if (DAQ_ID_output == DAQ_ID_SIMPLE) {
		unsigned __int32 channel;
		unsigned __int32 i;
		i = 0;
		for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) i = i + cnt[channel] + (cnt[channel]>0 ? 1 : 0);
		if (data_format_in_userheader_output == 2)  {dummy_uint16 = (unsigned __int16)i; *output_lmf << dummy_uint16;}
		if (data_format_in_userheader_output == 10) *output_lmf << i;
		if (data_format_in_userheader_output == 2) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_uint16 = (unsigned __int16)((channel << 8) + cnt[channel]);
					*output_lmf << dummy_uint16;
					for (i=0;i<cnt[channel];++i) {
						dummy_uint16 = (unsigned __int16)(i32TDC[channel*num_ions+i]);
						*output_lmf << dummy_uint16;
					}
				}
			}
		}
		if (data_format_in_userheader_output == 10) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_int32 = __int32((channel << 24) + cnt[channel]);
					*output_lmf << dummy_int32;
					for (i=0;i<cnt[channel];++i) *output_lmf << i32TDC[channel*num_ions+i];
				}
			}
		}
	} // end if (DAQ_ID_output == DAQ_ID_SIMPLE)

	return;
}



/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(double timestamp, unsigned __int32 cnt[], __int32 *i32TDC)
/////////////////////////////////////////////////////////////////
{
	if (!output_lmf|| !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	unsigned __int64 new_timestamp = (unsigned __int64)(timestamp * frequency);

	WriteTDCData(new_timestamp, cnt, i32TDC);

	return;
}












/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(unsigned __int64 timestamp, unsigned __int32 cnt[], double * d64TDC)
/////////////////////////////////////////////////////////////////
{
	unsigned __int16 dummy_uint16;
	double dummy_double;
	__int32 dummy_int32;

	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	
	++uint64_number_of_written_events;

	WriteEventHeader(timestamp,cnt);

	__int32 i,j;
	__int32 ii=0;
	if (DAQ_ID_output != DAQ_ID_SIMPLE) {
		for (i=0;i<number_of_channels_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits_output) hits = max_number_of_hits_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {
					dummy_uint16 = (unsigned __int16)(d64TDC[i*num_ions+j]+1.e-6);
					*output_lmf << dummy_uint16;
				}
				dummy_uint16 = 0;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);
				*output_lmf << dummy_double;
				for (j=0;j<hits;++j) 	*output_lmf << d64TDC[i*num_ions+j];
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				dummy_int32 = __int32(hits);
				*output_lmf << dummy_int32;
				for (j=0;j<hits;++j) {
					if (d64TDC[i*num_ions+j] >= 0.) ii =__int32(d64TDC[i*num_ions+j]+1.e-6);
					if (d64TDC[i*num_ions+j] <  0.) ii =__int32(d64TDC[i*num_ions+j]-1.e-6);
					*output_lmf << ii;
				}
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_int32;
			}
			if (data_format_in_userheader_output == LM_USERDEF) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {
					if (d64TDC[i*num_ions+j] >= 0.) ii =__int32(d64TDC[i*num_ions+j]+1.e-6);
					if (d64TDC[i*num_ions+j] <  0.) ii =__int32(d64TDC[i*num_ions+j]-1.e-6);
					*output_lmf << ii;
				}
				if (DAQ_ID_output == DAQ_ID_TDC8HP && this->LMF_Version_output >= 9) {
					*output_lmf >> ui64LevelInfo;
				}
			}
		}
	}
	if (DAQ_ID_output == DAQ_ID_2TDC8) {
		for (i=0;i<number_of_channels2_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits2_output) hits = max_number_of_hits2_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {
					dummy_uint16 = (unsigned __int16)(d64TDC[i*num_ions+j]+1.e-6);
					*output_lmf << dummy_uint16;
				}
				dummy_uint16 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);
				*output_lmf << dummy_double;
				for (j=0;j<hits;++j) *output_lmf << d64TDC[i*num_ions+j];
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				*output_lmf <<__int32(hits);
				for (j=0;j<hits;++j) {
					if (d64TDC[i*num_ions+j] >= 0.) ii =__int32(d64TDC[i*num_ions+j]+1.e-6);
					if (d64TDC[i*num_ions+j] <  0.) ii =__int32(d64TDC[i*num_ions+j]-1.e-6);
					*output_lmf << ii;
				}
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_int32;
			}
		}
	}

	if (DAQ_ID_output == DAQ_ID_SIMPLE) {
		unsigned __int32 channel;
		unsigned __int32 i;
		i = 0;
		for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) i = i + cnt[channel] + (cnt[channel]>0 ? 1 : 0);
		if (data_format_in_userheader_output == 2)  {
			dummy_uint16 = (unsigned __int16)i;
			*output_lmf << dummy_uint16;
		}
		if (data_format_in_userheader_output == 10) *output_lmf << i;
		if (data_format_in_userheader_output == 2) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_uint16 = (unsigned __int16)((channel << 8) + cnt[channel]);
					*output_lmf << dummy_uint16;
					for (i=0;i < cnt[channel]; ++i) {
						dummy_uint16 = (unsigned __int16)(d64TDC[channel*num_ions+i]);
						*output_lmf << dummy_uint16;
					}
				}
			}
		}
		if (data_format_in_userheader_output == 10) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_int32 = __int32((channel << 24) + cnt[channel]);
					*output_lmf << dummy_int32;
					for (i=0;i<cnt[channel];++i) {
						dummy_int32 = __int32(d64TDC[channel*num_ions+i]);
						*output_lmf << dummy_int32;
					}
				}
			}
		}
	} // end if (DAQ_ID_output == DAQ_ID_SIMPLE)

	return;
}

/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(double timestamp, unsigned __int32 cnt[], double *dtdc)
/////////////////////////////////////////////////////////////////
{
	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	unsigned __int64 new_timestamp = (unsigned __int64)(timestamp * frequency);

	WriteTDCData(new_timestamp, cnt, dtdc);

	return;
}











/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(unsigned __int64 timestamp, unsigned __int32 cnt[], unsigned __int16 * us16TDC)
/////////////////////////////////////////////////////////////////
{
	unsigned __int16 dummy_uint16;
	double dummy_double;
	__int32 dummy_int32;


	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	
	++uint64_number_of_written_events;

	WriteEventHeader(timestamp,cnt);

	__int32 i,j;

	if (DAQ_ID_output == DAQ_ID_HM1_ABM) {
		for (i=0;i<number_of_channels_output;++i) {
			if (data_format_in_userheader_output == 2)  *output_lmf << us16TDC[i*num_ions];
			if (data_format_in_userheader_output == 5)  {dummy_double = double(us16TDC[i*num_ions]); *output_lmf << dummy_double;}
			if (data_format_in_userheader_output == 10) {dummy_int32 = __int32(us16TDC[i*num_ions]); *output_lmf << dummy_int32;}
		}
	}
	if (DAQ_ID_output != DAQ_ID_SIMPLE && DAQ_ID_output != DAQ_ID_HM1_ABM) {
		for (i=0;i<number_of_channels_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits_output) hits = max_number_of_hits_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) *output_lmf << us16TDC[i*num_ions+j];
				dummy_uint16 = 0;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);
				*output_lmf << dummy_double;
				for (j=0;j<hits;++j) {dummy_double = double(us16TDC[i*num_ions+j]); *output_lmf << dummy_double;}
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				*output_lmf <<__int32(hits);
				for (j=0;j<hits;++j) {dummy_int32 = __int32(us16TDC[i*num_ions+j]); *output_lmf << dummy_int32;}
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits_output;++j) *output_lmf << dummy_int32;
			}
			if (data_format_in_userheader_output == LM_USERDEF) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) {dummy_int32 = __int32(us16TDC[i*num_ions+j]); *output_lmf << dummy_int32;}
				if (DAQ_ID_output == DAQ_ID_TDC8HP && this->LMF_Version_output >= 9) {
					*output_lmf >> ui64LevelInfo;
				}
			}
		}
	}

	if (DAQ_ID_output == DAQ_ID_2TDC8) {
		for (i=0;i<number_of_channels2_output;++i) {
			__int32 hits = cnt[i];
			if (hits > max_number_of_hits2_output) hits = max_number_of_hits2_output;
			if (data_format_in_userheader_output == 2) {
				dummy_uint16 = (unsigned __int16)(hits);
				*output_lmf << dummy_uint16;
				for (j=0;j<hits;++j) *output_lmf << us16TDC[i*num_ions+j];
				dummy_uint16 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_uint16;
			}
			if (data_format_in_userheader_output == 5) {
				dummy_double = double(hits);
				*output_lmf << dummy_double;
				for (j=0;j<hits;++j) {dummy_double = double(us16TDC[i*num_ions+j]); *output_lmf << dummy_double;}
				dummy_double = 0.;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_double;
			}
			if (data_format_in_userheader_output == 10) {
				dummy_int32 = __int32(hits);
				*output_lmf << dummy_int32;
				for (j=0;j<hits;++j) {dummy_int32 = __int32(us16TDC[i*num_ions+j]); *output_lmf << dummy_int32;}
				dummy_int32 = 0;
				for (j=hits;j<max_number_of_hits2_output;++j) *output_lmf << dummy_int32;
			}
		}
	}

	if (DAQ_ID_output == DAQ_ID_SIMPLE) {
		unsigned __int32 channel;
		unsigned __int32 i = 0;

		for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) i = i + cnt[channel] + (cnt[channel]>0 ? 1 : 0);
		if (data_format_in_userheader_output == 2)  {dummy_uint16 = (unsigned __int16)i; *output_lmf << dummy_uint16;}
		if (data_format_in_userheader_output == 10) *output_lmf << i;
		if (data_format_in_userheader_output == 2) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_uint16 = (unsigned __int16)((channel << 8) + cnt[channel]);
					*output_lmf << dummy_uint16;
					for (i=0;i<cnt[channel];++i) *output_lmf << us16TDC[channel*num_ions+i];
				}
			}
		}
		if (data_format_in_userheader_output == 10) {
			for (channel=0;channel < (unsigned __int32)number_of_channels_output;++channel) {
				if (cnt[channel]>0) {
					dummy_int32 = __int32((channel << 24) + cnt[channel]);
					*output_lmf << dummy_int32;
					for (i=0;i<cnt[channel];++i) {dummy_int32 = __int32(us16TDC[channel*num_ions+i]); *output_lmf << dummy_int32;}
				}
			}
		}
	} // end if (DAQ_ID_output == DAQ_ID_SIMPLE)

	return;
}







/////////////////////////////////////////////////////////////////
void LMF_IO::WriteTDCData(double timestamp, unsigned __int32 cnt[], unsigned __int16 * us16TDC)
/////////////////////////////////////////////////////////////////
{
	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	unsigned __int64 new_timestamp = (unsigned __int64)(timestamp * frequency);

	WriteTDCData(new_timestamp, cnt, us16TDC);

	return;
}








/////////////////////////////////////////////////////////////////
__int32 LMF_IO::GetErrorStatus()
/////////////////////////////////////////////////////////////////
{
	return errorflag;
}







/////////////////////////////////////////////////////////////////
bool LMF_IO::SeekToEventNumber(unsigned __int64 target_number)
/////////////////////////////////////////////////////////////////
{
	if (DAQ_ID == DAQ_ID_SIMPLE) return false;

	if (target_number == 0) {
		input_lmf->seek((unsigned __int64)(Headersize + User_header_size));
		must_read_first = true;
		errorflag = 0;
		input_lmf->error = 0;
		return true;
	}

	if (data_format_in_userheader == LM_USERDEF) {errorflag = 16; return false;}

	if (!input_lmf) {
		errorflag = 9;
		return false;
	}


/*	unsigned __int64 filesize;
	unsigned __int64 pos = input_lmf->tell();
	input_lmf->seek_to_end();
	filesize = input_lmf->tell();
	input_lmf->seek(pos);
*/

	

	if (target_number < 0) return false;
	if (target_number > uint64_Numberofevents) return false;
	__int32 eventsize = 0;
	if (data_format_in_userheader == 2 ) eventsize = 2 * Numberofcoordinates;
	if (data_format_in_userheader == 5 ) eventsize = 8 * Numberofcoordinates;
	if (data_format_in_userheader == 10) eventsize = 4 * Numberofcoordinates;

	if (DAQ_ID == DAQ_ID_RAW32BIT) eventsize = 4 * number_of_channels * (max_number_of_hits+1);

	if (input_lmf->filesize < (unsigned __int64)(eventsize)*target_number + (unsigned  __int64)(Headersize + User_header_size)) return false;

	unsigned __int64 new_position = (unsigned __int64)(eventsize)*target_number + (unsigned __int64)(Headersize + User_header_size);

	input_lmf->seek(new_position);



	uint64_number_of_read_events = target_number;
	must_read_first = true;
	errorflag = 0;
	input_lmf->error = 0;
	return true;
}







///////////////////////////////////////////////////////////////////////////////////
__int32 LMF_IO::PCIGetTDC_TDC8HP_25psGroupMode(unsigned __int64 &ref_ui64TDC8HPAbsoluteTimeStamp, __int32 count, unsigned __int32 * Buffer)
///////////////////////////////////////////////////////////////////////////////////
{
	memset(number_of_hits,0,num_channels*sizeof(__int32));		// clear the hit-counts values in _TDC array

	unsigned __int32 ui32DataWord;
	bool bOKFlag = false;
	unsigned __int8 ucTDCChannel;
	bool bFirstLevelWord = true;

	for(__int32 i = 0; i < count ; ++i)
	{
		ui32DataWord = Buffer[i];
		if ((ui32DataWord & 0xf8000000) == 0x18000000) // handle output level info
		{
			if (!bFirstLevelWord) continue;
			bFirstLevelWord = false;
			unsigned __int32 n = ui32DataWord & 0x7e00000;
			n >>= 21;
			unsigned __int32 ui32LevelInfo = ui32DataWord & 0x1fffff;
			if (n > 20) continue;
			if (n < 9) ui32LevelInfo >>= (9-n); else ui32LevelInfo <<= (n-9);
			unsigned __int64 ui64_temp_LevelInfo = ui32LevelInfo;
			ui64LevelInfo |= ui64_temp_LevelInfo;
			continue;
		}
		if( (ui32DataWord&0xC0000000)>0x40000000)		// valid data only if rising or falling trigger indicated
		{
			unsigned __int32 lTDCData = (ui32DataWord&0x00FFFFFF);
			if(lTDCData & 0x00800000)				// detect 24 bit signed flag
				lTDCData |= 0xff000000;				// if detected extend negative value to 32 bit
			if(!this->TDC8HP.VHR_25ps)								// correct for 100ps if nessesary
				lTDCData >>= 2;

			ucTDCChannel = (unsigned __int8)((ui32DataWord&0x3F000000)>>24);		// extract channel information
			// calculate TDC channel to _TDC channel
			if((ucTDCChannel >= 42) && (ucTDCChannel <= 49))
				ucTDCChannel -= 25;
			if((ucTDCChannel >= 21) && (ucTDCChannel <= 28))
				ucTDCChannel -= 12;
			
			bool bIsFalling = true;
			if ((ui32DataWord&0xC0000000) == 0xC0000000) bIsFalling = false;

			if (!bIsFalling) {
				ucTDCChannel += TDC8HP.channel_offset_for_rising_transitions;
			}

			if(ucTDCChannel < num_channels)	// if detected channel fits into TDC array then sort
			{
				++number_of_hits[ucTDCChannel];
				__int32 cnt = number_of_hits[ucTDCChannel];
				// increase Hit Counter;
				
				// test for oversized Hits
				if(cnt > num_ions) {
					--number_of_hits[ucTDCChannel];
					--cnt;
				}
				else			
					// if Hit # ok then store it
					i32TDC[ucTDCChannel*num_ions+cnt-1] = lTDCData;

				bOKFlag = true;
			}
		} 
		else
		{
			if ((ui32DataWord & 0xf0000000) == 0x00000000) {			// GroupWord detected
				this->TDC8HP.ui32AbsoluteTimeStamp = ui32DataWord & 0x00ffffff;
		}
			else if ((ui32DataWord & 0x10000000) == 0x10000000) {			// RollOverWord detected ?
				unsigned __int32 ui32newRollOver = (ui32DataWord & 0x00ffffff);
				if (ui32newRollOver > this->TDC8HP.ui32oldRollOver) {
					this->TDC8HP.ui64RollOvers += ui32newRollOver - this->TDC8HP.ui32oldRollOver;
				} else if (ui32newRollOver < this->TDC8HP.ui32oldRollOver) {
					this->TDC8HP.ui64RollOvers += ui32newRollOver;
					this->TDC8HP.ui64RollOvers += 1;
					this->TDC8HP.ui64RollOvers += (unsigned __int32)(0x00ffffff) - this->TDC8HP.ui32oldRollOver;
				}
				this->TDC8HP.ui32oldRollOver = ui32newRollOver;
			}
			//	only for debugging:
#ifdef _DEBUG
			else if (((ui32DataWord & 0xc0000000)>>30) == 0x00000001)			// ErrorWord detected ?
			{
				__int32 channel = (ui32DataWord & 0x3f000000)>>24;
				__int32 error = (ui32DataWord   & 0x00ff0000)>>16;
				__int32 count = ui32DataWord    & 0x0000ffff;
			}
#endif

		}
	}

	if (bOKFlag)
	{
		ref_ui64TDC8HPAbsoluteTimeStamp  = this->TDC8HP.ui64RollOvers * (unsigned __int64)(0x0000000001000000);
		ref_ui64TDC8HPAbsoluteTimeStamp += (unsigned __int64)(this->TDC8HP.ui32AbsoluteTimeStamp);
		this->TDC8HP.ui64TDC8HP_AbsoluteTimeStamp = ref_ui64TDC8HPAbsoluteTimeStamp;
	}
	
	return bOKFlag;
}







/////////////////////////////////////////////////////////////////
bool LMF_IO::Read_TDC8HP_raw_format(unsigned __int64 &ui64TDC8HP_AbsoluteTimeStamp_)
/////////////////////////////////////////////////////////////////
{
	__int32 count;
	*input_lmf >> count;
	if (input_lmf->error) return false;
	if (!count) return false;
	if (ui32buffer_size < count) {
		if (ui32buffer) {delete[] ui32buffer; ui32buffer = 0;}
		ui32buffer_size = count + 5000;
		ui32buffer = new unsigned __int32[ui32buffer_size];
	}
	input_lmf->read(this->ui32buffer,count*sizeof(__int32));
	if (input_lmf->error) return false;
	if (!PCIGetTDC_TDC8HP_25psGroupMode(ui64TDC8HP_AbsoluteTimeStamp_, count, this->ui32buffer)) return false;
	return true;
}












/////////////////////////////////////////////////////////////////
bool LMF_IO::ReadNextEvent()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32	i,j;
	__int32				int_Dummy;
	unsigned __int16	unsigned_short_Dummy;
	double			double_Dummy;

	if (!input_lmf) {
		errorflag = 9;
		return false;
	}
	if (DAQ_ID != DAQ_ID_SIMPLE) {
		if (max_number_of_hits == 0 || number_of_channels == 0) {
			errorflag = 14;
			return false;
		}
		if (data_format_in_userheader == LM_CAMAC) {
			errorflag = 15;
			return false;
		}
	}

	if (input_lmf->error) {
		if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1;
		return false;
	}

	if (data_format_in_userheader ==  2) memset(us16TDC,0,num_channels*num_ions*2);
	if (data_format_in_userheader ==  5) memset(dTDC,0,num_channels*num_ions*8);
	if (data_format_in_userheader == 10) memset(i32TDC,0,num_channels*num_ions*4);
	if (TDC8HP.variable_event_length == 1) memset(i32TDC,0,num_channels*num_ions*4);
	if (TDC8PCI2.variable_event_length == 1) memset(us16TDC,0,num_channels*num_ions*2);

	unsigned __int64 HPTDC_event_length = 0;
	unsigned __int64 TDC8PCI2_event_length = 0;

	DOUBLE_timestamp = 0.;
	ui64_timestamp = 0;

	if (TDC8HP.variable_event_length == 1) {
		if (this->TDC8HP.UserHeaderVersion >= 5 && this->TDC8HP.GroupingEnable_p66) {
			
			while (!Read_TDC8HP_raw_format(ui64_timestamp)) {
				if (input_lmf->error) break;
			}
			
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
			++uint64_number_of_read_events;
			DOUBLE_timestamp = double(ui64_timestamp)/frequency;  // time stamp in seconds.
			must_read_first = false;
			return true;
		} else {
			*input_lmf >> HPTDC_event_length;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
#ifdef LINUX
			HPTDC_event_length = HPTDC_event_length & 0x00ffffffffffffffLL;
#else
			HPTDC_event_length = HPTDC_event_length & 0x00ffffffffffffff;
#endif
			*input_lmf >> uint64_LMF_EventCounter;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
		}
	}

	if (TDC8PCI2.variable_event_length == 1) {
			*input_lmf >> TDC8PCI2_event_length;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
#ifdef LINUX
			TDC8PCI2_event_length = TDC8PCI2_event_length & 0x00ffffffffffffffLL;
#else
			TDC8PCI2_event_length = TDC8PCI2_event_length & 0x00ffffffffffffff;
#endif
			*input_lmf >> uint64_LMF_EventCounter;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
	}

	++uint64_number_of_read_events;

	//-------------------------------
	//  Read Time Stamp
	
	if (timestamp_format > 0) {

		if (timestamp_format > 0 ) {
			myLARGE_INTEGER aaa;
			aaa.QuadPart = 0;
			*input_lmf >> aaa.LowPart;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
			if (timestamp_format == 2 ) *input_lmf >> aaa.HighPart;
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 1; return false;}
			ui64_timestamp = aaa.QuadPart;
		}
		DOUBLE_timestamp = double(ui64_timestamp)/frequency;  // time stamp in seconds.
	}


	if (DAQ_ID != DAQ_ID_SIMPLE && TDC8HP.variable_event_length == 0 && TDC8PCI2.variable_event_length == 0) {
		for (i=0;i<number_of_channels+number_of_channels2;++i) {
				if (DAQ_ID == DAQ_ID_HM1_ABM) {
					number_of_hits[i] = 1;
					if (data_format_in_userheader ==  2) *input_lmf >> us16TDC[i*num_ions];
					if (data_format_in_userheader ==  5) *input_lmf >> dTDC[i*num_ions];
					if (data_format_in_userheader == 10) *input_lmf >> i32TDC[i*num_ions];
					if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
				}
				if (DAQ_ID != DAQ_ID_HM1_ABM) {
					if (data_format_in_userheader ==  2) {
						*input_lmf >> unsigned_short_Dummy;
						if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
						if (DAQ_ID == DAQ_ID_HM1) unsigned_short_Dummy = (unsigned_short_Dummy & 0x0007) - 1;
						number_of_hits[i] = (__int32 )unsigned_short_Dummy;
						for (j=0;j<max_number_of_hits;++j)  *input_lmf >> us16TDC[i*num_ions+j];
					}
					if (data_format_in_userheader ==  5) {
						*input_lmf >> double_Dummy;
						if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
						number_of_hits[i] =__int32(double_Dummy+0.1);
						for (j=0;j<max_number_of_hits;++j)  *input_lmf >> dTDC[i*num_ions+j];
					}
					if (data_format_in_userheader == 10) {
						*input_lmf >> int_Dummy;
						if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
						number_of_hits[i] = (__int32 )int_Dummy;
						for (j=0;j<max_number_of_hits;++j)  *input_lmf >> i32TDC[i*num_ions+j];
					}
					if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
				}
		} // for i
	}

	if ((DAQ_ID == DAQ_ID_TDC8 || DAQ_ID == DAQ_ID_2TDC8) && TDC8PCI2.variable_event_length == 1) {
		for (unsigned __int32 channel=0;channel<number_of_channels+number_of_channels2;++channel) {
			unsigned  __int16 n;
			*input_lmf >> n;		// store hits for this channel
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
			number_of_hits[channel] = n;
			for(__int32 i=0;i<(__int32 )n;++i) {	// transfer selected hits
				unsigned __int16 us16data;
				*input_lmf >> us16data;
				us16TDC[channel*num_ions+i] = us16data;
			}
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
		}
	}

	if ((DAQ_ID == DAQ_ID_TDC8HP || DAQ_ID == DAQ_ID_TDC8HPRAW) && TDC8HP.variable_event_length == 1) {
		for (unsigned __int32 channel=0;channel<number_of_channels;++channel) {
			unsigned  __int16 n;
			*input_lmf >> n;		// store hits for this channel
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
			number_of_hits[channel] = n;
			for(__int32 i=0;i<(__int32 )n;++i) {	// transfer selected hits
				__int32 i32data;
				*input_lmf >> i32data;
				i32TDC[channel*num_ions+i] = i32data;
			}
			if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
		}
		if (DAQ_ID == DAQ_ID_TDC8HP && this->LMF_Version >= 9) {
			*input_lmf >> ui64LevelInfo;
		}
	}


	if (DAQ_ID == DAQ_ID_SIMPLE) {
		unsigned __int16 us16_Dummy;
		__int32 i32_Dummy;
		__int32 number_of_words;
		__int32 channel;
		number_of_words = 0;
		if (data_format_in_userheader == 2 ) {*input_lmf >> us16_Dummy; number_of_words = us16_Dummy;}
		if (data_format_in_userheader == 10) *input_lmf >> number_of_words;
		if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
		for (i=0;i<number_of_channels;++i) number_of_hits[i] = 0;

		bool read_channel_marker;
		read_channel_marker = true;
		if (data_format_in_userheader == 2) {
			while (number_of_words > 0) {
				number_of_words--;
				*input_lmf >> us16_Dummy;
				if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
				if (read_channel_marker) {
					read_channel_marker = false;
					channel = 0;
					channel =__int32((us16_Dummy & 0xff00)  >> 8);
					number_of_hits[channel] =__int32(us16_Dummy & 0x00ff);
					i=0;
				} else {
					us16TDC[channel*num_ions+i] = us16_Dummy;
					++i;
					if (i == number_of_hits[channel]) read_channel_marker = true;
				}
			}
		}
		if (data_format_in_userheader == 10) {
			while (number_of_words > 0) {
				number_of_words--;
				*input_lmf >> i32_Dummy;
				if (input_lmf->error) {if (input_lmf->eof) this->errorflag = 18; else this->errorflag = 2; return false;}
				if (read_channel_marker) {
					read_channel_marker = false;
					channel = 0;
					channel =__int32((i32_Dummy & 0xff000000)  >> 24);
					number_of_hits[channel] =__int32(i32_Dummy & 0x000000ff);
					i=0;
				} else {
					i32TDC[channel*num_ions+i] = i32_Dummy;
					++i;
					if (i == number_of_hits[channel]) read_channel_marker = true;
				}
			}
		}
	}

	must_read_first = false;
	return true;
}








/////////////////////////////////////////////////////////////////
void LMF_IO::WriteCAMACArray(double timestamp, unsigned __int32 data[])
/////////////////////////////////////////////////////////////////
{
	unsigned __int16 dummy_uint16;
	unsigned __int8  dummy_uint8;


	if (!output_lmf || !OutputFileIsOpen) {
		errorflag = 10;
		return;
	}
	
	++uint64_number_of_written_events;

	myLARGE_INTEGER LARGE_Timestamp;
	LARGE_Timestamp.QuadPart = (__int64)(timestamp * frequency);

	if (timestamp_format_output >= 1) {
		if (data_format_in_userheader_output == 2) {
			dummy_uint16 = (unsigned __int16)(LARGE_Timestamp.LowPart & 0x0000ffff); *output_lmf << dummy_uint16;			// 32 Bit Low part, lower  16 Bit
			dummy_uint16 = (unsigned __int16)(LARGE_Timestamp.LowPart & 0xffff0000); *output_lmf << dummy_uint16;			// 32 Bit Low part, higher 16 Bit
		}
		if (data_format_in_userheader_output == 6) {
			dummy_uint8 = (unsigned __int8)(LARGE_Timestamp.LowPart & 0x000000ff); *output_lmf << dummy_uint8;			// 32 Bit Low part, lower 16 Bit, lower 8 bit
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.LowPart >> 8) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit Low part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)0; *output_lmf << dummy_uint8;	
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.LowPart >> 16) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit Low part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.LowPart >> 24) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit Low part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)0; *output_lmf << dummy_uint8;
		}
	}
	if (timestamp_format_output == 2) {
		if (data_format_in_userheader_output == 2) {
			dummy_uint16 = (unsigned __int16)(LARGE_Timestamp.HighPart & 0x0000ffff); *output_lmf << dummy_uint16;			// 32 Bit High part, lower  16 Bit
			dummy_uint16 = (unsigned __int16)(LARGE_Timestamp.HighPart & 0xffff0000); *output_lmf << dummy_uint16;			// 32 Bit High part, higher 16 Bit
		}
		if (data_format_in_userheader_output == 6) {
			dummy_uint8 = (unsigned __int8)(LARGE_Timestamp.HighPart & 0x000000ff); *output_lmf << dummy_uint8;			// 32 Bit High part, lower 16 Bit, lower 8 bit
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.HighPart >> 8) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit High part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)0; *output_lmf << dummy_uint8;	
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.HighPart >> 16) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit High part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)((LARGE_Timestamp.HighPart >> 24) & 0x000000ff); *output_lmf << dummy_uint8;	// 32 Bit High part, lower 16 Bit, high 8 bit
			dummy_uint8 = (unsigned __int8)0; *output_lmf << dummy_uint8;
		}
	}

	unsigned __int32 i;

	for (i=0;i<Numberofcoordinates - timestamp_format_output * 2;++i) {
		if (data_format_in_userheader_output == 2) {
			dummy_uint16 = (unsigned __int16)( data[i]        & 0x0000ffff); *output_lmf << dummy_uint16;
		}
		if (data_format_in_userheader_output == 6) {
			dummy_uint8 = (unsigned __int8)( data[i]        & 0x000000ff); *output_lmf << dummy_uint8;
			dummy_uint8 = (unsigned __int8)((data[i] >>  8) & 0x000000ff); *output_lmf << dummy_uint8;
			dummy_uint8 = (unsigned __int8)((data[i] >> 16) & 0x000000ff); *output_lmf << dummy_uint8;
		}
	}

	return;
}






/////////////////////////////////////////////////////////////////
bool LMF_IO::ReadNextCAMACEvent()
/////////////////////////////////////////////////////////////////
{
	unsigned __int32	i;

	if (!input_lmf) {
		errorflag = 9;
		return false;
	}

	++uint64_number_of_read_events;

	//-------------------------------
	//  Read Time Stamp
	DOUBLE_timestamp = 0.;
	
	unsigned __int32 time_temp;
	unsigned __int8 byte_1,byte_2,byte_3;
	unsigned __int16 unsigned_short;

	if (timestamp_format > 0) {
		//TRY
			myLARGE_INTEGER LARGE_timestamp;
			LARGE_timestamp.QuadPart = 0;
			if (timestamp_format >= 1 )
			{
				if (data_format_in_userheader == 2) {
					*input_lmf >> unsigned_short;
					time_temp = unsigned_short;
					*input_lmf >> unsigned_short;
					LARGE_timestamp.LowPart = time_temp + unsigned_short*256*256;
				}
				if (data_format_in_userheader == 6) {
					*input_lmf >> byte_1; *input_lmf >> byte_2; *input_lmf >> byte_3;
					time_temp = byte_1 + byte_2 * 256 + byte_3 * 256 * 256;
					LARGE_timestamp.LowPart = time_temp;

					*input_lmf >> byte_1; *input_lmf >> byte_2; *input_lmf >> byte_3;
					time_temp = byte_1 + byte_2 * 256 + byte_3 * 256 * 256;
					LARGE_timestamp.LowPart += time_temp * 256*256;
				}
			}
			if (timestamp_format == 2 ) {
				if (data_format_in_userheader == 2) {
					*input_lmf >> unsigned_short;
					time_temp = unsigned_short;
					*input_lmf >> unsigned_short;
					LARGE_timestamp.HighPart = time_temp + unsigned_short*256*256;
				}
				if (data_format_in_userheader == 6) {
					*input_lmf >> byte_1; *input_lmf >> byte_2; *input_lmf >> byte_3;
					time_temp = byte_1 + byte_2 * 256 + byte_3 * 256 * 256;
					LARGE_timestamp.HighPart = time_temp;

					*input_lmf >> byte_1; *input_lmf >> byte_2; *input_lmf >> byte_3;
					time_temp = byte_1 + byte_2 * 256 + byte_3 * 256 * 256;
					LARGE_timestamp.HighPart += time_temp * 256*256;
				}
			}
			ui64_timestamp = LARGE_timestamp.QuadPart;
/*		CATCH(CArchiveException,e)
			errorflag = 1;	// error reading timestamp
			return false;
		END_CATCH
		*/
		DOUBLE_timestamp = double(ui64_timestamp)/frequency;  // time stamp in seconds.
	}

//	TRY
		for (i=0;i<Numberofcoordinates - timestamp_format * 2;++i) {
			if (data_format_in_userheader == 6) {
				*input_lmf >> byte_1; *input_lmf >> byte_2; *input_lmf >> byte_3;
				CAMAC_Data[i] = byte_1 + byte_2 * 256 + byte_3 * 256 * 256;
			}
			if (data_format_in_userheader == 2) {
				*input_lmf >> unsigned_short;
				CAMAC_Data[i] = unsigned_short;
			}
		} // for i
/*	CATCH(CArchiveException,e)
		errorflag = 2; // error reading data
		return false;
	END_CATCH
	*/

	must_read_first = false;
	return true;
}





/////////////////////////////////////////////
void LMF_IO::GetCAMACArray(unsigned __int32 data[])
/////////////////////////////////////////////
{
	unsigned __int32 i;

	if (must_read_first) {
		if (!ReadNextCAMACEvent()) return;
	}
	for (i=0;i<Numberofcoordinates - timestamp_format * 2;++i) data[i] = CAMAC_Data[i];
}





/////////////////////////////////////////////////////////////////
void LMF_IO::GetTDCDataArray(__int32 *tdc)
/////////////////////////////////////////////////////////////////
{
	__int32 i,j;
	__int32 ii;

	if (must_read_first) {
		if (!ReadNextEvent()) return;
	}

	//__int32 max_channel = (number_of_channels+number_of_channels2 < num_channels) ? (number_of_channels+number_of_channels2) : num_channels;
	//__int32 max_hits = (max_number_of_hits < num_ions) ? max_number_of_hits : num_ions;
	__int32 max_channel = num_channels;
	__int32 max_hits = num_ions;

	if (data_format_in_userheader == LM_USERDEF) {
		if (DAQ_ID == DAQ_ID_TDC8HP || DAQ_ID == DAQ_ID_TDC8HPRAW) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = i32TDC[i*num_ions+j];
			}
		}
		if (DAQ_ID == DAQ_ID_TDC8 || DAQ_ID == DAQ_ID_2TDC8) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = us16TDC[i*num_ions+j];
			}
		}
	}
	if (data_format_in_userheader == 10) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = i32TDC[i*num_ions+j];
		}
	}
	if (data_format_in_userheader == 2) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] =__int32(us16TDC[i*num_ions+j]);
		}
	}
	if (data_format_in_userheader == 5) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) {
				if (dTDC[i*num_ions+j] >= 0.) ii =__int32(dTDC[i*num_ions+j]+1.e-19);
				if (dTDC[i*num_ions+j] <  0.) ii =__int32(dTDC[i*num_ions+j]-1.e-19);
				tdc[i*num_ions+j] = ii;
			}
		}
	}
}



/////////////////////////////////////////////////////////////////
void LMF_IO::GetTDCDataArray(unsigned __int16 *tdc)
/////////////////////////////////////////////////////////////////
{
	__int32 i,j;

	if (must_read_first) {
		if (!ReadNextEvent()) return;
	}

	//__int32 max_channel = (number_of_channels+number_of_channels2 < num_channels) ? (number_of_channels+number_of_channels2) : num_channels;
	//__int32 max_hits = (max_number_of_hits < num_ions) ? max_number_of_hits : num_ions;
	__int32 max_channel = num_channels;
	__int32 max_hits = num_ions;

	if (data_format_in_userheader == LM_USERDEF) {
		if (DAQ_ID == DAQ_ID_TDC8HP || DAQ_ID == DAQ_ID_TDC8HPRAW) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = (unsigned __int16)(i32TDC[i*num_ions+j]);
			}
		}
		if (DAQ_ID == DAQ_ID_TDC8 || DAQ_ID == DAQ_ID_2TDC8) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = (unsigned __int16)(us16TDC[i*num_ions+j]);
			}
		}
	}
	if (data_format_in_userheader == 10) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = (unsigned __int16)(i32TDC[i*num_ions+j]);
		}
	}
	if (data_format_in_userheader == 2) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = us16TDC[i*num_ions+j];
		}
	}
	if (data_format_in_userheader == 5) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = (unsigned __int16)(dTDC[i*num_ions+j]+1e-7);
		}
	}
}



/////////////////////////////////////////////////////////////////
void LMF_IO::GetTDCDataArray(double *tdc)
/////////////////////////////////////////////////////////////////
{
	__int32 i,j;

	if (must_read_first) {
		if (!ReadNextEvent()) return;
	}

	//__int32 max_channel = (number_of_channels+number_of_channels2 < num_channels) ? (number_of_channels+number_of_channels2) : num_channels;
	//__int32 max_hits = (max_number_of_hits < num_ions) ? max_number_of_hits : num_ions;
	__int32 max_channel = num_channels;
	__int32 max_hits = num_ions;

	if (data_format_in_userheader == LM_USERDEF) {
		if (DAQ_ID == DAQ_ID_TDC8HP || DAQ_ID == DAQ_ID_TDC8HPRAW) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = double(i32TDC[i*num_ions+j]);
			}
		}
		if (DAQ_ID == DAQ_ID_TDC8 || DAQ_ID == DAQ_ID_2TDC8) {
			for (i=0;i<max_channel;++i) {
				for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = double(us16TDC[i*num_ions+j]);
			}
		}
	}
	if (data_format_in_userheader == 10) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = double(i32TDC[i*num_ions+j]);
		}
	}
	if (data_format_in_userheader == 2) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = double(us16TDC[i*num_ions+j]);
		}
	}
	if (data_format_in_userheader == 5) {
		for (i=0;i<max_channel;++i) {
			for (j=0;j<max_hits;++j) tdc[i*num_ions+j] = dTDC[i*num_ions+j]+1e-7;
		}
	}
}





/////////////////////////////////////////////////////////////////
void LMF_IO::GetNumberOfHitsArray(unsigned __int32 cnt[]) {
/////////////////////////////////////////////////////////////////
	__int32 i;

	if (must_read_first) {
		if (!ReadNextEvent()) return;
	}

	for (i=0;i<num_channels;++i) cnt[i] = (number_of_hits[i] < (unsigned __int32)(num_ions)) ? number_of_hits[i] : num_ions;
}




/////////////////////////////////////////////////////////////////
void LMF_IO::GetNumberOfHitsArray(__int32 cnt[]) {
/////////////////////////////////////////////////////////////////
	__int32 i;

	if (must_read_first) {
		if (!ReadNextEvent()) return;
	}

	for (i=0;i<num_channels;++i) cnt[i] = (number_of_hits[i] < (unsigned __int32)(num_ions)) ? number_of_hits[i] : num_ions;
}




/////////////////////////////////////////////////////////////////
const char* LMF_IO::GetErrorText(__int32 error_id)
/////////////////////////////////////////////////////////////////
{
	return error_text[error_id];
}


/////////////////////////////////////////////////////////////////
void LMF_IO::GetErrorText(__int32 error_code, __int8 text[])
/////////////////////////////////////////////////////////////////
{
	sprintf(text,"%s",error_text[error_code]);
	return;
}


/////////////////////////////////////////////////////////////////
void LMF_IO::GetErrorText(__int8 text[])
/////////////////////////////////////////////////////////////////
{
	GetErrorText(errorflag,text);
	return;
}


/////////////////////////////////////////////////////////////////
unsigned __int64 LMF_IO::GetEventNumber()
/////////////////////////////////////////////////////////////////
{
	return uint64_number_of_read_events;
}

/////////////////////////////////////////////////////////////////
unsigned __int32 LMF_IO::GetNumberOfChannels()
/////////////////////////////////////////////////////////////////
{
	return number_of_channels+number_of_channels2;
}

/////////////////////////////////////////////////////////////////
unsigned __int32 LMF_IO::GetMaxNumberOfHits()
/////////////////////////////////////////////////////////////////
{
	return max_number_of_hits;
}


/////////////////////////////////////////////////////////////////
double LMF_IO::GetDoubleTimeStamp()
/////////////////////////////////////////////////////////////////
{
	if (must_read_first) {
		if (!ReadNextEvent()) return 0.;
	}
	return DOUBLE_timestamp;
}

/////////////////////////////////////////////////////////////////
unsigned __int64 LMF_IO::Getuint64TimeStamp()
/////////////////////////////////////////////////////////////////
{
	if (must_read_first) {
		if (!ReadNextEvent()) return ui64_timestamp;
	}
	return ui64_timestamp;
}




/////////////////////////////////////////////////////////////////
LMF_IO * LMF_IO::Clone()
/////////////////////////////////////////////////////////////////
{
	LMF_IO * clone = new LMF_IO(num_channels,num_ions);
	if (!clone) return 0;

	clone->Versionstring			= this->Versionstring;
	clone->FilePathName				= this->FilePathName;
	clone->OutputFilePathName		= this->OutputFilePathName;
	clone->Comment					= this->Comment;
	clone->Comment_output			= this->Comment_output;
	clone->DAQ_info					= this->DAQ_info;
	clone->Camac_CIF				= this->Camac_CIF;
	
	clone->Starttime				= this->Starttime;
	clone->Stoptime					= this->Stoptime;
	clone->Starttime_output			= this->Starttime_output;
	clone->Stoptime_output			= this->Stoptime_output;
	
	clone->time_reference			= this->time_reference;
	clone->time_reference_output	= this->time_reference_output;
	
	clone->ArchiveFlag				= this->ArchiveFlag;
	clone->Cobold_Header_version	= this->Cobold_Header_version;
	clone->Cobold_Header_version_output	= this->Cobold_Header_version_output;
	
	clone->uint64_LMF_EventCounter	= this->uint64_LMF_EventCounter;
	clone->uint64_number_of_read_events	= this->uint64_number_of_read_events;
	clone->uint64_Numberofevents	= this->uint64_Numberofevents;
	
	clone->Numberofcoordinates		= this->Numberofcoordinates;
	clone->CTime_version			= this->CTime_version;
	clone->CTime_version_output		= this->CTime_version_output;
	clone->CTime_version_output		= this->CTime_version_output;
	clone->SIMPLE_DAQ_ID_Orignial	= this->SIMPLE_DAQ_ID_Orignial;
	clone->DAQVersion				= this->DAQVersion;
	clone->DAQVersion_output		= this->DAQVersion_output;
	clone->DAQ_ID					= this->DAQ_ID;
	clone->DAQ_ID_output			= this->DAQ_ID_output;
	clone->data_format_in_userheader	= this->data_format_in_userheader;
	clone->data_format_in_userheader_output	= this->data_format_in_userheader_output;
	
	clone->Headersize				= this->Headersize;
	clone->User_header_size			= this->User_header_size;
	clone->User_header_size_output	= this->User_header_size_output;
	
	clone->IOaddress				= this->IOaddress;
	clone->timestamp_format			= this->timestamp_format;
	clone->timestamp_format_output	= this->timestamp_format_output;
	clone->timerange				= this->timerange;
	
	clone->number_of_channels		= this->number_of_channels;
	clone->number_of_channels2		= this->number_of_channels2;
	clone->max_number_of_hits		= this->max_number_of_hits;
	clone->max_number_of_hits2		= this->max_number_of_hits2;
	
	clone->number_of_channels_output	= this->number_of_channels_output;
	clone->number_of_channels2_output	= this->number_of_channels2_output;
	clone->max_number_of_hits_output	= this->max_number_of_hits_output;
	clone->max_number_of_hits2_output	= this->max_number_of_hits2_output;
	
	clone->DAQSubVersion			= this->DAQSubVersion;
	clone->module_2nd				= this->module_2nd;
	clone->system_timeout			= this->system_timeout;
	clone->system_timeout_output	= this->system_timeout_output;
	clone->common_mode				= this->common_mode;
	clone->common_mode_output		= this->common_mode_output;
	clone->DAQ_info_Length			= this->DAQ_info_Length;
	clone->Camac_CIF_Length			= this->Camac_CIF_Length;
	clone->LMF_Version				= this->LMF_Version;
	clone->LMF_Version_output		= this->LMF_Version_output;
	clone->TDCDataType				= this->TDCDataType;
	
	clone->LMF_Header_version		= this->LMF_Header_version;
	
	clone->tdcresolution			= this->tdcresolution;
	clone->tdcresolution_output		= this->tdcresolution_output;
	clone->frequency				= this->frequency;
	clone->DOUBLE_timestamp			= this->DOUBLE_timestamp;
	clone->ui64_timestamp			= this->ui64_timestamp;

	clone->number_of_CCFHistory_strings = this->number_of_CCFHistory_strings;
	clone->number_of_DAN_source_strings = this->number_of_DAN_source_strings;
	clone->number_of_DAQ_source_strings = this->number_of_DAQ_source_strings;
	clone->number_of_CCFHistory_strings_output = this->number_of_CCFHistory_strings_output;
	clone->number_of_DAN_source_strings_output = this->number_of_DAN_source_strings_output;
	clone->number_of_DAQ_source_strings_output = this->number_of_DAQ_source_strings_output;

	if (number_of_CCFHistory_strings >= 0) {
		clone->CCFHistory_strings = new std::string*[number_of_CCFHistory_strings];
		memset(clone->CCFHistory_strings,0,sizeof(std::string*)*number_of_CCFHistory_strings);
	}
	if (number_of_DAN_source_strings >= 0) {
		clone->DAN_source_strings = new std::string*[number_of_DAN_source_strings];
		memset(clone->DAN_source_strings,0,sizeof(std::string*)*number_of_DAN_source_strings);
	}
	if (number_of_DAQ_source_strings >= 0) {
		clone->DAQ_source_strings = new std::string*[number_of_DAQ_source_strings];
		memset(clone->DAQ_source_strings,0,sizeof(std::string*)*number_of_DAQ_source_strings);
	}
	if (this->CCFHistory_strings) {
		for (__int32 i=0;i<number_of_CCFHistory_strings;++i) {clone->CCFHistory_strings[i] = new std::string(); *clone->CCFHistory_strings[i] = *this->CCFHistory_strings[i];}
	}
	if (this->DAN_source_strings) {
		for (__int32 i=0;i<number_of_DAN_source_strings;++i) {clone->DAN_source_strings[i] = new std::string(); *clone->DAN_source_strings[i] = *this->DAN_source_strings[i];}
	}
	if (this->DAQ_source_strings) {
		for (__int32 i=0;i<number_of_DAQ_source_strings;++i) {clone->DAQ_source_strings[i] = new std::string(); *clone->DAQ_source_strings[i] = *this->DAQ_source_strings[i];}
	}

	if (number_of_CCFHistory_strings_output>=0) {
		clone->CCFHistory_strings_output = new std::string*[number_of_CCFHistory_strings_output];
		memset(clone->CCFHistory_strings_output,0,sizeof(std::string*)*number_of_CCFHistory_strings_output);
	}
	if (number_of_DAN_source_strings_output>=0) {
		clone->DAN_source_strings_output = new std::string*[number_of_DAN_source_strings_output];
		memset(clone->DAN_source_strings_output,0,sizeof(std::string*)*number_of_DAN_source_strings_output);
	}
	if (number_of_DAQ_source_strings_output>=0) {
		clone->DAQ_source_strings_output = new std::string*[number_of_DAQ_source_strings_output];
		memset(clone->DAQ_source_strings_output,0,sizeof(std::string*)*number_of_DAQ_source_strings_output);
	}
	if (this->CCFHistory_strings_output) {
		for (__int32 i=0;i<number_of_CCFHistory_strings_output;++i) {clone->CCFHistory_strings_output[i] = new std::string(); *clone->CCFHistory_strings_output[i] = *this->CCFHistory_strings_output[i];}
	}
	if (this->DAN_source_strings_output) {
		for (__int32 i=0;i<number_of_DAN_source_strings_output;++i) {clone->DAN_source_strings_output[i] = new std::string(); *clone->DAN_source_strings_output[i] = *this->DAN_source_strings_output[i];}
	}
	if (this->DAQ_source_strings_output) {
		for (__int32 i=0;i<number_of_DAQ_source_strings_output;++i) {clone->DAQ_source_strings_output[i] = new std::string(); *clone->DAQ_source_strings_output[i] = *this->DAQ_source_strings_output[i];}
	}

	clone->CCF_HISTORY_CODE_bitmasked = this->CCF_HISTORY_CODE_bitmasked;
	clone->DAN_SOURCE_CODE_bitmasked = this->DAN_SOURCE_CODE_bitmasked;
	clone->DAQ_SOURCE_CODE_bitmasked = this->DAQ_SOURCE_CODE_bitmasked;
	
	clone->errorflag				= this->errorflag;
	clone->skip_header				= this->skip_header;
	
	clone->uint64_number_of_written_events	= this->uint64_number_of_written_events;
	
	clone->not_Cobold_LMF			= this->not_Cobold_LMF;
	clone->Headersize_output		= this->Headersize_output;
	clone->output_byte_counter		= this->output_byte_counter;
	clone->Numberofcoordinates_output	= this->Numberofcoordinates_output;
	clone->must_read_first			= this->must_read_first;

	clone->num_channels				= this->num_channels;
	clone->num_ions					= this->num_ions;



// pointers and other special stuff:


//clone->InputFileIsOpen		= this->InputFileIsOpen;
//clone->in_ar					= this->in_ar;
//clone->input_lmf				= this->input_lmf;

//clone->OutputFileIsOpen			= this->OutputFileIsOpen;
//clone->out_ar					= this->out_ar;
//clone->output_lmf				= this->output_lmf;


	// TDC8PCI2
	clone->TDC8PCI2.GateDelay_1st_card			= this->TDC8PCI2.GateDelay_1st_card;
	clone->TDC8PCI2.OpenTime_1st_card			= this->TDC8PCI2.OpenTime_1st_card;
	clone->TDC8PCI2.WriteEmptyEvents_1st_card	= this->TDC8PCI2.WriteEmptyEvents_1st_card;
	clone->TDC8PCI2.TriggerFalling_1st_card		= this->TDC8PCI2.TriggerFalling_1st_card;
	clone->TDC8PCI2.TriggerRising_1st_card		= this->TDC8PCI2.TriggerRising_1st_card;
	clone->TDC8PCI2.EmptyCounter_1st_card		= this->TDC8PCI2.EmptyCounter_1st_card;
	clone->TDC8PCI2.EmptyCounter_since_last_Event_1st_card = this->TDC8PCI2.EmptyCounter_since_last_Event_1st_card;
	clone->TDC8PCI2.use_normal_method			= this->TDC8PCI2.use_normal_method;
	clone->TDC8PCI2.use_normal_method_2nd_card	= this->TDC8PCI2.use_normal_method_2nd_card;
	clone->TDC8PCI2.sync_test_on_off			= this->TDC8PCI2.sync_test_on_off;
	clone->TDC8PCI2.io_address_2nd_card			= this->TDC8PCI2.io_address_2nd_card;
	clone->TDC8PCI2.GateDelay_2nd_card			= this->TDC8PCI2.GateDelay_2nd_card;
	clone->TDC8PCI2.OpenTime_2nd_card			= this->TDC8PCI2.OpenTime_2nd_card;
	clone->TDC8PCI2.WriteEmptyEvents_2nd_card	= this->TDC8PCI2.WriteEmptyEvents_2nd_card;
	clone->TDC8PCI2.TriggerFallingEdge_2nd_card = this->TDC8PCI2.TriggerFallingEdge_2nd_card;
	clone->TDC8PCI2.TriggerRisingEdge_2nd_card	= this->TDC8PCI2.TriggerRisingEdge_2nd_card;
	clone->TDC8PCI2.EmptyCounter_2nd_card		= this->TDC8PCI2.EmptyCounter_2nd_card;
	clone->TDC8PCI2.EmptyCounter_since_last_Event_2nd_card = this->TDC8PCI2.EmptyCounter_since_last_Event_2nd_card;
	clone->TDC8PCI2.variable_event_length		= this->TDC8PCI2.variable_event_length;


	// HM1
	clone->HM1.FAK_DLL_Value			= this->HM1.FAK_DLL_Value;
	clone->HM1.Resolution_Flag			= this->HM1.Resolution_Flag;
	clone->HM1.trigger_mode_for_start	= this->HM1.trigger_mode_for_start;
	clone->HM1.trigger_mode_for_stop	= this->HM1.trigger_mode_for_stop;
	clone->HM1.Even_open_time			= this->HM1.Even_open_time;
	clone->HM1.Auto_Trigger				= this->HM1.Auto_Trigger;
	clone->HM1.set_bits_for_GP1			= this->HM1.set_bits_for_GP1;
	clone->HM1.ABM_m_xFrom				= this->HM1.ABM_m_xFrom;
	clone->HM1.ABM_m_xTo				= this->HM1.ABM_m_xTo;
	clone->HM1.ABM_m_yFrom				= this->HM1.ABM_m_yFrom;
	clone->HM1.ABM_m_yTo				= this->HM1.ABM_m_yTo;
	clone->HM1.ABM_m_xMin				= this->HM1.ABM_m_xMin;
	clone->HM1.ABM_m_xMax				= this->HM1.ABM_m_xMax;
	clone->HM1.ABM_m_yMin				= this->HM1.ABM_m_yMin;
	clone->HM1.ABM_m_yMax				= this->HM1.ABM_m_yMax;
	clone->HM1.ABM_m_xOffset			= this->HM1.ABM_m_xOffset;
	clone->HM1.ABM_m_yOffset			= this->HM1.ABM_m_yOffset;
	clone->HM1.ABM_m_zOffset			= this->HM1.ABM_m_zOffset;
	clone->HM1.ABM_Mode					= this->HM1.ABM_Mode;
	clone->HM1.ABM_OsziDarkInvert		= this->HM1.ABM_OsziDarkInvert;
	clone->HM1.ABM_ErrorHisto			= this->HM1.ABM_ErrorHisto;
	clone->HM1.ABM_XShift				= this->HM1.ABM_XShift;
	clone->HM1.ABM_YShift				= this->HM1.ABM_YShift;
	clone->HM1.ABM_ZShift				= this->HM1.ABM_ZShift;
	clone->HM1.ABM_ozShift				= this->HM1.ABM_ozShift;
	clone->HM1.ABM_wdShift				= this->HM1.ABM_wdShift;
	clone->HM1.ABM_ucLevelXY			= this->HM1.ABM_ucLevelXY;
	clone->HM1.ABM_ucLevelZ				= this->HM1.ABM_ucLevelZ;
	clone->HM1.ABM_uiABMXShift			= this->HM1.ABM_uiABMXShift;
	clone->HM1.ABM_uiABMYShift			= this->HM1.ABM_uiABMYShift;
	clone->HM1.ABM_uiABMZShift			= this->HM1.ABM_uiABMZShift;
	clone->HM1.use_normal_method		= this->HM1.use_normal_method;

	clone->HM1.TWOHM1_FAK_DLL_Value		= this->HM1.TWOHM1_FAK_DLL_Value;
	clone->HM1.TWOHM1_Resolution_Flag	= this->HM1.TWOHM1_Resolution_Flag;
	clone->HM1.TWOHM1_trigger_mode_for_start	= this->HM1.TWOHM1_trigger_mode_for_start;
	clone->HM1.TWOHM1_trigger_mode_for_stop		= this->HM1.TWOHM1_trigger_mode_for_stop;
	clone->HM1.TWOHM1_res_adjust		= this->HM1.TWOHM1_res_adjust;
	clone->HM1.TWOHM1_tdcresolution		= this->HM1.TWOHM1_tdcresolution;
	clone->HM1.TWOHM1_test_overflow		= this->HM1.TWOHM1_test_overflow;
	clone->HM1.TWOHM1_number_of_channels	= this->HM1.TWOHM1_number_of_channels;
	clone->HM1.TWOHM1_number_of_hits	= this->HM1.TWOHM1_number_of_hits;
	clone->HM1.TWOHM1_set_bits_for_GP1	= this->HM1.TWOHM1_set_bits_for_GP1;
	clone->HM1.TWOHM1_HM1_ID_1			= this->HM1.TWOHM1_HM1_ID_1;
	clone->HM1.TWOHM1_HM1_ID_2			= this->HM1.TWOHM1_HM1_ID_2;

	clone->TDC8HP.no_config_file_read	= this->TDC8HP.no_config_file_read;
	clone->TDC8HP.RisingEnable_p61		= this->TDC8HP.RisingEnable_p61;
	clone->TDC8HP.FallingEnable_p62		= this->TDC8HP.FallingEnable_p62;
	clone->TDC8HP.TriggerEdge_p63		= this->TDC8HP.TriggerEdge_p63;
	clone->TDC8HP.TriggerChannel_p64		= this->TDC8HP.TriggerChannel_p64;
	clone->TDC8HP.OutputLevel_p65		= this->TDC8HP.OutputLevel_p65;
	clone->TDC8HP.GroupingEnable_p66		= this->TDC8HP.GroupingEnable_p66;
	clone->TDC8HP.GroupingEnable_p66_output	= this->TDC8HP.GroupingEnable_p66_output;
	clone->TDC8HP.AllowOverlap_p67		= this->TDC8HP.AllowOverlap_p67;
	clone->TDC8HP.TriggerDeadTime_p68	= this->TDC8HP.TriggerDeadTime_p68;
	clone->TDC8HP.GroupRangeStart_p69	= this->TDC8HP.GroupRangeStart_p69;
	clone->TDC8HP.GroupRangeEnd_p70		= this->TDC8HP.GroupRangeEnd_p70;
	clone->TDC8HP.ExternalClock_p71		= this->TDC8HP.ExternalClock_p71;
	clone->TDC8HP.OutputRollOvers_p72	= this->TDC8HP.OutputRollOvers_p72;
	clone->TDC8HP.DelayTap0_p73			= this->TDC8HP.DelayTap0_p73;
	clone->TDC8HP.DelayTap1_p74			= this->TDC8HP.DelayTap1_p74;
	clone->TDC8HP.DelayTap2_p75			= this->TDC8HP.DelayTap2_p75;
	clone->TDC8HP.DelayTap3_p76			= this->TDC8HP.DelayTap3_p76;
	clone->TDC8HP.INL_p80				= this->TDC8HP.INL_p80;
	clone->TDC8HP.DNL_p81				= this->TDC8HP.DNL_p81;
	clone->TDC8HP.csConfigFile			= this->TDC8HP.csConfigFile;
	clone->TDC8HP.csINLFile				= this->TDC8HP.csINLFile;
	clone->TDC8HP.csDNLFile				= this->TDC8HP.csDNLFile;
	clone->TDC8HP.csConfigFile_Length	= this->TDC8HP.csConfigFile_Length;
	clone->TDC8HP.csINLFile_Length		= this->TDC8HP.csINLFile_Length;
	clone->TDC8HP.csDNLFile_Length		= this->TDC8HP.csDNLFile_Length;
	clone->TDC8HP.UserHeaderVersion		= this->TDC8HP.UserHeaderVersion;
	clone->TDC8HP.VHR_25ps				= this->TDC8HP.VHR_25ps;
	clone->TDC8HP.SyncValidationChannel	= this->TDC8HP.SyncValidationChannel;
	clone->TDC8HP.variable_event_length	= this->TDC8HP.variable_event_length;
	clone->TDC8HP.SSEEnable				= this->TDC8HP.SSEEnable;
	clone->TDC8HP.MMXEnable				= this->TDC8HP.MMXEnable;
	clone->TDC8HP.DMAEnable				= this->TDC8HP.DMAEnable;
	clone->TDC8HP.GroupTimeOut			= this->TDC8HP.GroupTimeOut;

	clone->TDC8HP.i32NumberOfDAQLoops   = this->TDC8HP.i32NumberOfDAQLoops;   
	clone->TDC8HP.TDC8HP_DriverVersion 	= this->TDC8HP.TDC8HP_DriverVersion; 	
	clone->TDC8HP.iTriggerChannelMask	= this->TDC8HP.iTriggerChannelMask;	
	clone->TDC8HP.iTime_zero_channel	= this->TDC8HP.iTime_zero_channel;	

	clone->TDC8HP.Number_of_TDCs = this->TDC8HP.Number_of_TDCs;
	for (__int32 i = 0;i<3;++i) {
		if (this->TDC8HP.TDC_info[i]) {
			clone->TDC8HP.TDC_info[i]->index				= this->TDC8HP.TDC_info[i]->index;
			clone->TDC8HP.TDC_info[i]->channelCount			= this->TDC8HP.TDC_info[i]->channelCount;
			clone->TDC8HP.TDC_info[i]->channelStart			= this->TDC8HP.TDC_info[i]->channelStart;
			clone->TDC8HP.TDC_info[i]->highResChannelCount	= this->TDC8HP.TDC_info[i]->highResChannelCount;
			clone->TDC8HP.TDC_info[i]->highResChannelStart	= this->TDC8HP.TDC_info[i]->highResChannelStart;
			clone->TDC8HP.TDC_info[i]->lowResChannelCount	= this->TDC8HP.TDC_info[i]->lowResChannelCount;
			clone->TDC8HP.TDC_info[i]->lowResChannelStart	= this->TDC8HP.TDC_info[i]->lowResChannelStart;
			clone->TDC8HP.TDC_info[i]->resolution			= this->TDC8HP.TDC_info[i]->resolution;
			clone->TDC8HP.TDC_info[i]->serialNumber			= this->TDC8HP.TDC_info[i]->serialNumber;
			clone->TDC8HP.TDC_info[i]->version				= this->TDC8HP.TDC_info[i]->version;
			clone->TDC8HP.TDC_info[i]->fifoSize				= this->TDC8HP.TDC_info[i]->fifoSize;
			clone->TDC8HP.TDC_info[i]->flashValid			= this->TDC8HP.TDC_info[i]->flashValid;
			memcpy(clone->TDC8HP.TDC_info[i]->INLCorrection,this->TDC8HP.TDC_info[i]->INLCorrection,sizeof(__int32)*8*1024);
			memcpy(clone->TDC8HP.TDC_info[i]->DNLData,this->TDC8HP.TDC_info[i]->DNLData,sizeof(__int16)*8*1024);
		}
	}

	return clone;
}
