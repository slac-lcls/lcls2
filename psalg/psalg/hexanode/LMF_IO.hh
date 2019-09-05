#ifndef _LMF_IO_
	#define _LMF_IO_
#include "fstream"
//#include "stdio.h"
#include "time.h"

#define LINUX

#ifdef LINUX
	#include "string.h"
        #define _fseeki64 fseeko
	#define _ftelli64 ftello

	#ifndef __int32_IS_DEFINED
		#define __int32_IS_DEFINED
		#define __int32 int
		#define __int16 short
		#define __int64 long long
		#define __int8 char
	#endif
#endif

#ifndef LINUX
#pragma warning(disable : 4996)
#endif


class MyFILE
{
public:
	MyFILE(bool mode_reading_) {error = 0; eof = false; mode_reading = mode_reading_; file = 0; position = 0; filesize = 0;}
	~MyFILE() {close(); error = 0; eof = false;}

	FILE * file;

	bool open(__int8* name) {
		if (file) {error = 1; return false;}
		eof = false;
		if (mode_reading) {
			file = fopen(name,"rb");
			if (!file) {error = 1; return false;}
			_fseeki64(file,  (unsigned __int64)0, SEEK_END);
			filesize = _ftelli64(file);
			_fseeki64(file,  (unsigned __int64)0, SEEK_SET);
		} else file = fopen(name,"wb");
		if (file) return true; else {error = 1; return false;}
	}

	void close() {
		if (file) {fclose(file);  file = 0;} else error = 1;
		position = 0; filesize = 0; eof = false;
	}

	unsigned __int64 tell() {return position;}
	void seek(unsigned __int64 pos);

	void read(__int8* string,__int32 length_bytes) {
		unsigned __int32 read_bytes = (unsigned __int32)(fread(string,1,length_bytes,file));
		if (read_bytes != (unsigned __int32)length_bytes) {
			error = 1;
			if (feof(file)) eof = true;
		}
		position += length_bytes;
	}

	void read(unsigned __int32 * dest,__int32 length_bytes) {
		unsigned __int32 read_bytes = (unsigned __int32)(fread(dest,1,length_bytes,file));
		if (read_bytes != (unsigned __int32)length_bytes) {
			error = 1;
			if (feof(file)) eof = true;
		}
		position += length_bytes;
	}

	void write(const __int8* string,__int32 length) {
		fwrite(string,1,length,file);
		position += length;
		if (filesize < position) filesize = position;
	}

	void flush() {fflush(file);}

	MyFILE & operator>>(unsigned __int8 &c)		{read((__int8*)&c,sizeof(unsigned __int8));		return *this;}
	MyFILE & operator>>(__int8 &c)				{read((__int8*)&c,sizeof(__int8));				return *this;}
	MyFILE & operator>>(unsigned __int16 &l)	{read((__int8*)&l,sizeof(unsigned __int16));	return *this;}
	MyFILE & operator>>(unsigned __int32 &l)	{read((__int8*)&l,sizeof(unsigned __int32));	return *this;}
	MyFILE & operator>>(unsigned __int64 &l)	{read((__int8*)&l,sizeof(unsigned __int64));	return *this;}
	MyFILE & operator>>(__int16 &s)				{read((__int8*)&s,sizeof(__int16));				return *this;}
	MyFILE & operator>>(__int32 &l)				{read((__int8*)&l,sizeof(__int32));				return *this;}
	MyFILE & operator>>(__int64 &l)				{read((__int8*)&l,sizeof(__int64));				return *this;}
	MyFILE & operator>>(double &d)				{read((__int8*)&d,sizeof(double));				return *this;}
	MyFILE & operator>>(bool &d)				{read((__int8*)&d,sizeof(bool));				return *this;}


	MyFILE & operator<<(unsigned __int8 c)		{write((__int8*)&c,sizeof(unsigned __int8));	return *this;}
	MyFILE & operator<<(__int8 c)				{write((__int8*)&c,sizeof(__int8));				return *this;}
	MyFILE & operator<<(unsigned __int16 l)		{write((__int8*)&l,sizeof(unsigned __int16));	return *this;}
	MyFILE & operator<<(unsigned __int32 l)		{write((__int8*)&l,sizeof(unsigned __int32));	return *this;}
	MyFILE & operator<<(unsigned __int64 l)		{write((__int8*)&l,sizeof(unsigned __int64));	return *this;}
	MyFILE & operator<<(__int16 s)				{write((__int8*)&s,sizeof(__int16));			return *this;}
	MyFILE & operator<<(__int32 l)				{write((__int8*)&l,sizeof(__int32));			return *this;}
	MyFILE & operator<<(__int64 l)				{write((__int8*)&l,sizeof(__int64));			return *this;}
	MyFILE & operator<<(double d)				{write((__int8*)&d,sizeof(double));				return *this;}
	MyFILE & operator<<(bool d)					{write((__int8*)&d,sizeof(bool));				return *this;}

	__int32 error;
	bool eof;
	unsigned __int64 filesize;

private:
	bool mode_reading;
	unsigned __int64 position;
};






#ifndef _WINNT_
typedef union _myLARGE_INTEGER {
	struct {
		unsigned __int32 LowPart;
		__int32 HighPart;
	}; 
	struct {
		unsigned __int32 LowPart;
		__int32 HighPart;
	} u;
	__int64 QuadPart;
} myLARGE_INTEGER,  *PmyLARGE_INTEGER;
#endif


#define DAQ_ID_HM1     0x000001
#define DAQ_ID_TDC8	   0x000002
#define DAQ_ID_CAMAC   0x000003
#define DAQ_ID_2HM1	   0x000004
#define DAQ_ID_2TDC8   0x000005
#define DAQ_ID_HM1_ABM 0x000006
#define DAQ_ID_TDC8HP  0x000008
#define DAQ_ID_TCPIP   0x000009
#define DAQ_ID_TDC8HPRAW 0x000010

#define DAQ_ID_RAW32BIT 100
#define DAQ_ID_SIMPLE 101


#define LM_BYTE					1	//  8bit integer
#define LM_SHORT				2	// 16bit integer
#define LM_LONG					3	// 32bit integer
#define	LM_FLOAT				4   // 32bit IEEE float
#define LM_DOUBLE				5	// 64bit IEEE float
#define LM_CAMAC				6	// 24bit integer
#define LM_DOUBLELONG			7	// 64bit integer
#define LM_SBYTE				8	// signed 8bit integer
#define LM_SSHORT				9	// signed 16bit integer
#define LM_SLONG				10	// signed 32bit integer
#define LM_SDOUBLELONG			11	// signed 64bit integer
#define LM_LASTKNOWNDATAFORMAT	LM_SDOUBLELONG
#define LM_USERDEF				-1	// user will handle the reading 




struct TDC8PCI2_struct
{
	__int32 GateDelay_1st_card;
	__int32 OpenTime_1st_card;
	__int32 WriteEmptyEvents_1st_card;
	__int32 TriggerFalling_1st_card;
	__int32 TriggerRising_1st_card;
	__int32 EmptyCounter_1st_card;
	__int32 EmptyCounter_since_last_Event_1st_card;
	bool use_normal_method;
	bool use_normal_method_2nd_card;
	__int32 sync_test_on_off;
	__int32 io_address_2nd_card;
	__int32 GateDelay_2nd_card;
	__int32 OpenTime_2nd_card;
	__int32 WriteEmptyEvents_2nd_card;
	__int32 TriggerFallingEdge_2nd_card;
	__int32 TriggerRisingEdge_2nd_card;
	__int32 EmptyCounter_2nd_card;
	__int32 EmptyCounter_since_last_Event_2nd_card;
	__int32 variable_event_length;
};






struct TDC8HP_info_struct
{
	__int32 index;
	__int32 channelCount;
	__int32 channelStart;
	__int32 highResChannelCount;
	__int32 highResChannelStart;
	__int32 lowResChannelCount;
	__int32 lowResChannelStart;
	double resolution;
	__int32 serialNumber;
	__int32 version;
	__int32 fifoSize;
	__int32 *INLCorrection;
	unsigned __int16 *DNLData;
	bool flashValid;
};





struct TDC8HP_struct
{
	__int32 no_config_file_read;
	unsigned __int64 RisingEnable_p61;
	unsigned __int64 FallingEnable_p62;
	__int32 TriggerEdge_p63;
	__int32 TriggerChannel_p64;
	__int32 OutputLevel_p65;
	__int32 GroupingEnable_p66;
	__int32 GroupingEnable_p66_output;
	__int32 AllowOverlap_p67;
	double TriggerDeadTime_p68;
	double GroupRangeStart_p69;
	double GroupRangeEnd_p70;
	__int32 ExternalClock_p71;
	__int32 OutputRollOvers_p72;
	__int32 DelayTap0_p73;
	__int32 DelayTap1_p74;
	__int32 DelayTap2_p75;
	__int32 DelayTap3_p76;
	__int32 INL_p80;
	__int32 DNL_p81;
	double  OffsetTimeZeroChannel_s;
	__int32 BinsizeType;

	std::string	csConfigFile, csINLFile, csDNLFile;

	__int32	csConfigFile_Length, csINLFile_Length, csDNLFile_Length;
	__int32	UserHeaderVersion;
	bool VHR_25ps;
	__int32	SyncValidationChannel;
	__int32 variable_event_length;
	bool SSEEnable, MMXEnable, DMAEnable;
	double GroupTimeOut;

	__int32		Number_of_TDCs;
	TDC8HP_info_struct * TDC_info[3];

//	bool	bdummy;
//	__int32	idummy;
//	double	ddummy;


	__int32 i32NumberOfDAQLoops;
	unsigned __int32 TDC8HP_DriverVersion;
	__int32 iTriggerChannelMask;
	__int32 iTime_zero_channel;

	
	__int32 number_of_bools;
	__int32 number_of_int32s;
	__int32 number_of_doubles;

	unsigned __int32 ui32oldRollOver;
	unsigned __int64 ui64RollOvers;
	unsigned __int32 ui32AbsoluteTimeStamp;
//	unsigned __int64 ui64OldTimeStamp;
	unsigned __int64 ui64TDC8HP_AbsoluteTimeStamp;

	__int32 channel_offset_for_rising_transitions;
};









struct HM1_struct
{
	__int32 FAK_DLL_Value;
	__int32 Resolution_Flag;
	__int32 trigger_mode_for_start;
	__int32 trigger_mode_for_stop;
	__int32 Even_open_time;
	__int32 Auto_Trigger;
	__int32 set_bits_for_GP1;
	__int32 ABM_m_xFrom;	
	__int32 ABM_m_xTo;
	__int32 ABM_m_yFrom;
	__int32 ABM_m_yTo;
	__int32 ABM_m_xMin;
	__int32 ABM_m_xMax;
	__int32 ABM_m_yMin;
	__int32 ABM_m_yMax;
	__int32 ABM_m_xOffset;
	__int32 ABM_m_yOffset;
	__int32 ABM_m_zOffset;
	__int32 ABM_Mode;
	__int32 ABM_OsziDarkInvert;
	__int32 ABM_ErrorHisto;
	__int32 ABM_XShift;
	__int32 ABM_YShift;
	__int32 ABM_ZShift;
	__int32 ABM_ozShift;
	__int32 ABM_wdShift;
	__int32 ABM_ucLevelXY;
	__int32 ABM_ucLevelZ;
	__int32 ABM_uiABMXShift;
	__int32 ABM_uiABMYShift;
	__int32 ABM_uiABMZShift;
	bool use_normal_method;

	__int32 TWOHM1_FAK_DLL_Value;
	__int32 TWOHM1_Resolution_Flag;
	__int32 TWOHM1_trigger_mode_for_start;
	__int32 TWOHM1_trigger_mode_for_stop;
	__int32 TWOHM1_res_adjust;
	double TWOHM1_tdcresolution;
	__int32 TWOHM1_test_overflow;
	__int32 TWOHM1_number_of_channels;
	__int32 TWOHM1_number_of_hits;
	__int32 TWOHM1_set_bits_for_GP1;
	__int32 TWOHM1_HM1_ID_1;
	__int32 TWOHM1_HM1_ID_2;
};


















class LMF_IO
{
public:

	LMF_IO(__int32 Number_of_Channels, __int32 Number_of_Hits);
    ~LMF_IO();

	static __int32			GetVersionNumber();
	bool			OpenInputLMF(__int8* Filename);
	bool			OpenInputLMF(std::string Filename);
	void			CloseInputLMF();

	bool			OpenOutputLMF(__int8* Filename);
	bool			OpenOutputLMF(std::string Filename);
	void			CloseOutputLMF();

	unsigned __int64 GetLastLevelInfo();

	LMF_IO *		Clone();

	void			WriteTDCData(double timestamp,unsigned __int32 cnt[],__int32 *tdc);
	void			WriteTDCData(unsigned __int64 timestamp,unsigned __int32 cnt[],__int32 *tdc);
	void			WriteTDCData(double timestamp,unsigned __int32 cnt[],double *tdc);
	void			WriteTDCData(unsigned __int64 timestamp,unsigned __int32 cnt[],double *tdc);
	void			WriteTDCData(double timestamp,unsigned __int32 cnt[],unsigned __int16 *tdc);
	void			WriteTDCData(unsigned __int64 timestamp,unsigned __int32 cnt[],unsigned __int16 *tdc);

	void			WriteFirstHeader();
	bool			ReadNextEvent();
	bool			ReadNextCAMACEvent();
	void			GetNumberOfHitsArray(unsigned __int32 cnt[]);
	void			GetNumberOfHitsArray(__int32 cnt[]);
	void			GetTDCDataArray(__int32 *tdc);
	void			GetTDCDataArray(double *tdc);
	void			GetTDCDataArray(unsigned __int16 * tdc);

	void			GetCAMACArray(unsigned __int32 []);
	void			WriteCAMACArray(double, unsigned int[]);

	unsigned __int64		GetEventNumber();
	double			GetDoubleTimeStamp();
	unsigned __int64 Getuint64TimeStamp();
	unsigned __int32	GetNumberOfChannels();
	unsigned __int32	GetMaxNumberOfHits(); 
	bool			SeekToEventNumber(unsigned __int64 Eventnumber);

	const char *	GetErrorText(__int32 error_id);
	void			GetErrorText(__int32 error_id, __int8 char_buffer[]);
	void			GetErrorText(__int8 char_buffer[]);

	__int32				GetErrorStatus();


	void prepare_Cobold2008b_TDC8HP_header_output() { // Cobold2008 release August 2009
		//TDC8HP.TriggerDeadTime_p68;	
		//TDC8HP.GroupRangeStart_p69;	
		//TDC8HP.GroupRangeEnd_p70;
		//Starttime_output;
		//Stoptime_output;
		TDC8HP.FallingEnable_p62 = 1044991; // all 9 channels on 2 cards
		TDC8HP.TriggerChannel_p64 = 1;
		frequency = 1.e12; // 1ps
		tdcresolution_output = 0.025; // or 0.016 ?
		TDC8HP.VHR_25ps = true;
		TDC8HP.UserHeaderVersion = 5;
		DAQVersion_output = 20080507;
		Cobold_Header_version_output = 2008;
		system_timeout_output = 5; // obsolete
		time_reference_output = (__int32)int(Starttime_output);
		common_mode_output = 0; // 0 = common start
		data_format_in_userheader_output = -1;
		DAQ_ID_output = 8; // 8 = TDC8HP
		timestamp_format_output = 2;
		LMF_Version_output = 8;
		IOaddress = 0;
		number_of_DAQ_source_strings_output = 0;
		TDC8HP.OutputRollOvers_p72 = 1;
		TDC8HP.Number_of_TDCs = 0;
		TDC8HP.number_of_int32s = 4;
		TDC8HP.GroupingEnable_p66_output = false;
	}

private:
	void			Initialize();

	void			write_times(MyFILE *,time_t,time_t);

	bool			OpenNonCoboldFile(void);

	__int32			ReadCAMACHeader();
	__int32			ReadTDC8PCI2Header();
	__int32			Read2TDC8PCI2Header();
	__int32			ReadTDC8HPHeader_LMFV_1_to_7();
	__int32			ReadTDC8HPHeader_LMFV_8_to_9();
	__int32			ReadHM1Header();
	__int32			ReadTCPIPHeader();

	void			WriteEventHeader(unsigned __int64 timestamp, unsigned __int32 cnt[]);
	__int32			WriteCAMACHeader();
	__int32			WriteTDC8PCI2Header();
	__int32			Write2TDC8PCI2Header();
	__int32			WriteTDC8HPHeader_LMFV_1_to_7();
	__int32			WriteTDC8HPHeader_LMFV_8_to_9();
	__int32			WriteHM1Header();
	__int32			WriteTCPIPHeader();
	bool			Read_TDC8HP_raw_format(unsigned __int64 &ui64TDC8HP_AbsoluteTimeStamp);
	__int32			PCIGetTDC_TDC8HP_25psGroupMode(unsigned __int64 &ui64TDC8HPAbsoluteTimeStamp, __int32 count, unsigned __int32 * Buffer);


public:
	std::string			Versionstring;
	std::string			FilePathName;
	std::string			OutputFilePathName;
	std::string			Comment;
	std::string			Comment_output;
	std::string			DAQ_info;
	std::string			Camac_CIF;

	char *error_text[21];

	time_t			Starttime;
	time_t			Stoptime;
	time_t			Starttime_output;
	time_t			Stoptime_output;

	__int32				time_reference;
	__int32				time_reference_output;
	
	unsigned __int32	ArchiveFlag;
	__int32				Cobold_Header_version;
	__int32				Cobold_Header_version_output;

	unsigned __int64	uint64_LMF_EventCounter;
	unsigned __int64	uint64_number_of_read_events;
	unsigned __int64	uint64_Numberofevents;

	__int32				Numberofcoordinates;
	__int32				CTime_version,CTime_version_output;
	unsigned __int32	SIMPLE_DAQ_ID_Orignial;
	unsigned __int32	DAQVersion;
	unsigned __int32	DAQVersion_output;
	unsigned __int32	DAQ_ID;
	unsigned __int32	DAQ_ID_output;
	__int32				data_format_in_userheader;
	__int32				data_format_in_userheader_output;

	unsigned __int32	Headersize;
	unsigned __int32	User_header_size;
	unsigned __int32	User_header_size_output;

	__int32				IOaddress;
	unsigned __int32	timestamp_format;
	unsigned __int32	timestamp_format_output;
	__int32				timerange;

	unsigned __int32	number_of_channels;
	unsigned __int32	number_of_channels2;
	unsigned __int32	max_number_of_hits;
	unsigned __int32	max_number_of_hits2;

	__int32				number_of_channels_output;
	__int32				number_of_channels2_output;
	__int32				max_number_of_hits_output;
	__int32				max_number_of_hits2_output;

	unsigned __int64	ui64LevelInfo;

	__int32				DAQSubVersion;
	__int32				module_2nd;
	__int32				system_timeout;
	__int32				system_timeout_output;
	__int32				common_mode;
	__int32				common_mode_output;
	__int32				DAQ_info_Length;
	__int32				Camac_CIF_Length;
	__int32				LMF_Version;
	__int32				LMF_Version_output;
	__int32				TDCDataType;
	
	unsigned __int32	LMF_Header_version;

	double			tdcresolution;
	double			tdcresolution_output;
	double			frequency;
	double			DOUBLE_timestamp;
	unsigned __int64	ui64_timestamp;

	__int32				errorflag;
	bool			skip_header;

	unsigned __int32 DAQ_SOURCE_CODE_bitmasked;
	unsigned __int32 DAN_SOURCE_CODE_bitmasked;
	unsigned __int32 CCF_HISTORY_CODE_bitmasked;

	__int32	number_of_CCFHistory_strings;
	__int32	number_of_CCFHistory_strings_output;
	std::string ** CCFHistory_strings;
	std::string ** CCFHistory_strings_output;
	__int32	number_of_DAN_source_strings;
	__int32	number_of_DAN_source_strings_output;
	std::string ** DAN_source_strings;
	std::string ** DAN_source_strings_output;
	__int32	number_of_DAQ_source_strings;
	__int32 number_of_DAQ_source_strings_output;
	std::string ** DAQ_source_strings;
	std::string ** DAQ_source_strings_output;

	HM1_struct		HM1;
	TDC8HP_struct	TDC8HP;
	TDC8PCI2_struct TDC8PCI2;

	unsigned __int64 uint64_number_of_written_events;

	MyFILE *	input_lmf;
	MyFILE *	output_lmf;

	bool			InputFileIsOpen;
	bool			OutputFileIsOpen;

private:
	unsigned __int32 *	ui32buffer;
	__int32				ui32buffer_size;
	bool				not_Cobold_LMF;
	unsigned __int32	Headersize_output;
	__int32				output_byte_counter;
	__int32				Numberofcoordinates_output;
	bool				must_read_first;
	unsigned __int32	* number_of_hits;
	__int32				*i32TDC;
	unsigned __int16	*us16TDC;
	double				*dTDC;

	__int32				num_channels;
	__int32				num_ions;


	unsigned __int32	* CAMAC_Data;
};
#endif
