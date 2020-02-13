"""
Detector Names
"""
SRC = 'xtcav' # 'XrayTransportDiagnostic.0:Opal1000.0'
CALIB_GROUP = 'Xtcav::CalibV1' 
ANALYSIS_VERSION = 'XTCAV_Analysis_Version'

EBEAM = 'EBeam'
GAS_DETECTOR = 'FEEGasDetEnergy'

ROI_SIZE_X_names = ['XTCAV_ROI_sizeX', 'ROI_X_Length', 'OTRS:DMP1:695:SizeX']
ROI_SIZE_Y_names = ['XTCAV_ROI_sizeY', 'ROI_Y_Length', 'OTRS:DMP1:695:SizeY']
ROI_START_X_names = ['XTCAV_ROI_startX', 'ROI_X_Offset', 'OTRS:DMP1:695:MinX']
ROI_START_Y_names = ['XTCAV_ROI_startY', 'ROI_Y_Offset', 'OTRS:DMP1:695:MinY']

ROI_SIZE_X=1024
ROI_SIZE_Y=1024
ROI_START_X=0
ROI_START_Y=0

UM_PER_PIX_names = ['XTCAV_calib_umPerPx','OTRS:DMP1:695:RESOLUTION']
STR_STRENGTH_names = ['XTCAV_strength_par_S','Streak_Strength','OTRS:DMP1:695:TCAL_X']
RF_AMP_CALIB_names = ['XTCAV_Amp_Des_calib_MV','XTCAV_Cal_Amp','SIOC:SYS0:ML01:AO214']
RF_PHASE_CALIB_names = ['XTCAV_Phas_Des_calib_deg','XTCAV_Cal_Phase','SIOC:SYS0:ML01:AO215']
DUMP_E_names = ['XTCAV_Beam_energy_dump_GeV','Dump_Energy','REFS:DMP1:400:EDES']
DUMP_DISP_names = ['XTCAV_calib_disp_posToEnergy','Dump_Disp','SIOC:SYS0:ML01:AO216']
"""
End Detector Names
"""

#Electron charge in coulombs
E_CHARGE=1.60217657e-19 

E_BEAM_CHARGE=5
XTCAV_RFAMP=20
XTCAV_RFPHASE=90
ENERGY_DETECTOR=0.2
DUMP_E_CHARGE=175E-12 #IN C
FS_TO_S = 1e-15

SNR_BORDER=100 #number of pixels near the border that can be considered to contain just noise
MIN_ROI_SIZE=3 #minimum number of pixels defining region of interest
ROI_PIXEL_FRACTION=0.001 #fraction of pixels that must be non-zero in roi(s) of image for analysis

DEFAULT_SPLIT_METHOD='scipyLabel'

DB_FILE_NAME = 'pedestals'
LOR_FILE_NAME = 'lasingoffreference'
