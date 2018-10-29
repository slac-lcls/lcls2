"""
    Dictionary for detector name conversion from LCLS to LCLS2 format.

    from psana.pscalib.calib.MDBConversionMap import DETECTOR_NAME_CONVERSION_DICT

    Generator command: Detector/app/find_detector_names
    Created on 2018-08-09 by Mikhail Dubrovin
"""
#------------------------------

DETECTOR_NAME_CONVERSION_DICT = {\
#== type 0000 tm6740
                 'CxiDg1.0:Tm6740.0': 'tm6740_0000',\
                 'CxiDg2.0:Tm6740.0': 'tm6740_0001',\
                 'CxiDg3.0:Tm6740.0': 'tm6740_0002',\
                 'CxiDg4.0:Tm6740.0': 'tm6740_0003',\
                 'CxiDsd.0:Tm6740.0': 'tm6740_0004',\
                 'CxiDsu.0:Tm6740.0': 'tm6740_0005',\
                 'CxiKb1.0:Tm6740.0': 'tm6740_0006',\
                 'CxiSc1.0:Tm6740.0': 'tm6740_0007',\
                 'CxiSc2.0:Tm6740.0': 'tm6740_0008',\
                 'CxiSc2.0:Tm6740.1': 'tm6740_0009',\
             'NoDetector.0:Tm6740.0': 'tm6740_0010',\
             'NoDetector.0:Tm6740.1': 'tm6740_0011',\
             'NoDetector.0:Tm6740.2': 'tm6740_0012',\
             'NoDetector.0:Tm6740.9': 'tm6740_0013',\
            'XcsBeamline.1:Tm6740.4': 'tm6740_0014',\
            'XcsBeamline.1:Tm6740.5': 'tm6740_0015',\
          'XcsEndstation.0:Tm6740.1': 'tm6740_0016',\
          'XppEndstation.1:Tm6740.1': 'tm6740_0017',\
              'XppMonPim.1:Tm6740.1': 'tm6740_0018',\
              'XppSb3Pim.1:Tm6740.1': 'tm6740_0019',\
              'XppSb4Pim.1:Tm6740.1': 'tm6740_0020',\
#== type 0001 acqiris
         'AmoEndstation.0:Acqiris.1': 'acqiris_0000',\
         'AmoEndstation.0:Acqiris.2': 'acqiris_0001',\
         'AmoEndstation.0:Acqiris.3': 'acqiris_0002',\
         'AmoEndstation.0:Acqiris.4': 'acqiris_0003',\
               'AmoETOF.0:Acqiris.0': 'acqiris_0004',\
               'AmoITOF.0:Acqiris.0': 'acqiris_0005',\
                  'Camp.0:Acqiris.0': 'acqiris_0006',\
         'CxiEndstation.0:Acqiris.0': 'acqiris_0007',\
         'CxiEndstation.0:Acqiris.1': 'acqiris_0008',\
                'CxiSc1.0:Acqiris.0': 'acqiris_0009',\
      'MecTargetChamber.0:Acqiris.0': 'acqiris_0010',\
         'MfxEndstation.0:Acqiris.0': 'acqiris_0011',\
         'SxrEndstation.0:Acqiris.0': 'acqiris_0012',\
         'SxrEndstation.0:Acqiris.1': 'acqiris_0013',\
         'SxrEndstation.0:Acqiris.2': 'acqiris_0014',\
         'SxrEndstation.0:Acqiris.3': 'acqiris_0015',\
         'SxrEndstation.0:Acqiris.4': 'acqiris_0016',\
           'XcsBeamline.0:Acqiris.0': 'acqiris_0017',\
                'XppLas.0:Acqiris.0': 'acqiris_0018',\
#== type 0002 andor
           'AmoEndstation.0:Andor.0': 'andor_0000',\
        'MecTargetChamber.0:Andor.1': 'andor_0001',\
        'MecTargetChamber.0:Andor.2': 'andor_0002',\
           'SxrEndstation.0:Andor.0': 'andor_0003',\
           'SxrEndstation.0:Andor.1': 'andor_0004',\
           'SxrEndstation.0:Andor.2': 'andor_0005',\
#== type 0003 cspad2x2
               'CxiDg2.0:Cspad2x2.0': 'cspad2x2_0000',\
               'CxiDg2.0:Cspad2x2.1': 'cspad2x2_0001',\
               'CxiDg3.0:Cspad2x2.0': 'cspad2x2_0002',\
               'CxiSc1.0:Cspad2x2.0': 'cspad2x2_0003',\
               'CxiSc2.0:Cspad2x2.0': 'cspad2x2_0004',\
               'CxiSc2.0:Cspad2x2.1': 'cspad2x2_0005',\
               'CxiSc2.0:Cspad2x2.2': 'cspad2x2_0006',\
               'CxiSc2.0:Cspad2x2.3': 'cspad2x2_0007',\
               'CxiSc2.0:Cspad2x2.4': 'cspad2x2_0008',\
               'CxiSc2.0:Cspad2x2.5': 'cspad2x2_0009',\
               'CxiSc2.0:Cspad2x2.6': 'cspad2x2_0010',\
               'CxiSc2.0:Cspad2x2.7': 'cspad2x2_0011',\
               'CxiSc2.0:Cspad2x2.8': 'cspad2x2_0012',\
               'CxiSc2.0:Cspad2x2.9': 'cspad2x2_0013',\
               'DetLab.0:Cspad2x2.0': 'cspad2x2_0014',\
        'MecEndstation.0:Cspad2x2.6': 'cspad2x2_0015',\
     'MecTargetChamber.0:Cspad2x2.0': 'cspad2x2_0016',\
     'MecTargetChamber.0:Cspad2x2.1': 'cspad2x2_0017',\
     'MecTargetChamber.0:Cspad2x2.2': 'cspad2x2_0018',\
     'MecTargetChamber.0:Cspad2x2.3': 'cspad2x2_0019',\
     'MecTargetChamber.0:Cspad2x2.4': 'cspad2x2_0020',\
     'MecTargetChamber.0:Cspad2x2.5': 'cspad2x2_0021',\
        'MfxEndstation.0:Cspad2x2.0': 'cspad2x2_0022',\
          'SxrBeamline.0:Cspad2x2.2': 'cspad2x2_0023',\
          'SxrBeamline.0:Cspad2x2.3': 'cspad2x2_0024',\
        'XcsEndstation.0:Cspad2x2.0': 'cspad2x2_0025',\
        'XcsEndstation.0:Cspad2x2.1': 'cspad2x2_0026',\
        'XcsEndstation.0:Cspad2x2.2': 'cspad2x2_0027',\
        'XcsEndstation.0:Cspad2x2.3': 'cspad2x2_0028',\
        'XcsEndstation.0:Cspad2x2.4': 'cspad2x2_0029',\
               'XppGon.0:Cspad2x2.0': 'cspad2x2_0030',\
               'XppGon.0:Cspad2x2.1': 'cspad2x2_0031',\
               'XppGon.0:Cspad2x2.2': 'cspad2x2_0032',\
               'XppGon.0:Cspad2x2.3': 'cspad2x2_0033',\
               'XppGon.0:Cspad2x2.4': 'cspad2x2_0034',\
#== type 0004 nodevice
                'BldEb.0:NoDevice.0': 'nodevice_0000',\
            'EpicsArch.0:NoDevice.0': 'nodevice_0001',\
            'EpicsArch.0:NoDevice.1': 'nodevice_0002',\
#== type 0005 opal2000
               'CxiDg4.0:Opal2000.0': 'opal2000_0000',\
        'CxiEndstation.0:Opal2000.1': 'opal2000_0001',\
        'CxiEndstation.0:Opal2000.2': 'opal2000_0002',\
        'CxiEndstation.0:Opal2000.3': 'opal2000_0003',\
               'CxiSc1.0:Opal2000.1': 'opal2000_0004',\
     'MecTargetChamber.0:Opal2000.0': 'opal2000_0005',\
     'MecTargetChamber.0:Opal2000.1': 'opal2000_0006',\
     'MecTargetChamber.0:Opal2000.2': 'opal2000_0007',\
        'MfxEndstation.0:Opal2000.0': 'opal2000_0008',\
           'NoDetector.0:Opal2000.0': 'opal2000_0009',\
#== type 0006 acqtdc
          'SxrEndstation.0:AcqTDC.2': 'acqtdc_0000',\
#== type 0007 epix10k
            'NoDetector.0:Epix10k.0': 'epix10k_0000',\
         'XcsEndstation.0:Epix10k.0': 'epix10k_0001',\
#== type 0008 pnccd
                    'Camp.0:pnCCD.0': 'pnccd_0000',\
                    'Camp.0:pnCCD.1': 'pnccd_0001',\
           'SxrEndstation.0:pnCCD.0': 'pnccd_0002',\
           'XcsEndstation.0:pnCCD.0': 'pnccd_0003',\
#== type 0009 wave8
           'CxiEndstation.1:Wave8.0': 'wave8_0000',\
                  'DetLab.0:Wave8.0': 'wave8_0001',\
           'MfxEndstation.0:Wave8.0': 'wave8_0002',\
           'SxrEndstation.0:Wave8.0': 'wave8_0003',\
           'XcsEndstation.0:Wave8.0': 'wave8_0004',\
           'XppEndstation.0:Wave8.0': 'wave8_0005',\
#== type 0010 gsc16ai
      'MecTargetChamber.0:Gsc16ai.0': 'gsc16ai_0000',\
         'SxrEndstation.0:Gsc16ai.0': 'gsc16ai_0001',\
         'XppEndstation.0:Gsc16ai.0': 'gsc16ai_0002',\
#== type 0011 zyla
            'AmoEndstation.0:Zyla.0': 'zyla_0000',\
            'CxiEndstation.0:Zyla.0': 'zyla_0001',\
         'MecTargetChamber.0:Zyla.1': 'zyla_0002',\
            'XcsEndstation.0:Zyla.0': 'zyla_0003',\
            'XcsEndstation.0:Zyla.1': 'zyla_0004',\
            'XppEndstation.0:Zyla.0': 'zyla_0005',\
            'XppEndstation.0:Zyla.1': 'zyla_0006',\
#== type 0012 epix
               'NoDetector.0:Epix.0': 'epix_0000',\
            'XcsEndstation.0:Epix.0': 'epix_0001',\
            'XcsEndstation.0:Epix.1': 'epix_0002',\
#== type 0013 opal1000
               'AmoBPS.0:Opal1000.0': 'opal1000_0000',\
               'AmoBPS.0:Opal1000.1': 'opal1000_0001',\
               'AmoBPS.1:Opal1000.0': 'opal1000_0002',\
        'AmoEndstation.0:Opal1000.0': 'opal1000_0003',\
        'AmoEndstation.0:Opal1000.1': 'opal1000_0004',\
        'AmoEndstation.0:Opal1000.2': 'opal1000_0005',\
        'AmoEndstation.0:Opal1000.3': 'opal1000_0006',\
        'AmoEndstation.0:Opal1000.4': 'opal1000_0007',\
        'AmoEndstation.1:Opal1000.0': 'opal1000_0008',\
        'AmoEndstation.2:Opal1000.0': 'opal1000_0009',\
               'AmoVMI.0:Opal1000.0': 'opal1000_0010',\
               'CxiDg3.0:Opal1000.0': 'opal1000_0011',\
               'CxiDsu.0:Opal1000.0': 'opal1000_0012',\
        'CxiEndstation.0:Opal1000.0': 'opal1000_0013',\
        'CxiEndstation.0:Opal1000.1': 'opal1000_0014',\
        'CxiEndstation.0:Opal1000.2': 'opal1000_0015',\
               'CxiSc1.0:Opal1000.0': 'opal1000_0016',\
               'CxiSc1.0:Opal1000.1': 'opal1000_0017',\
               'CxiSc2.0:Opal1000.0': 'opal1000_0018',\
     'MecTargetChamber.0:Opal1000.1': 'opal1000_0019',\
     'MecTargetChamber.0:Opal1000.2': 'opal1000_0020',\
        'MfxEndstation.0:Opal1000.0': 'opal1000_0021',\
        'MfxEndstation.0:Opal1000.1': 'opal1000_0022',\
           'NoDetector.0:Opal1000.0': 'opal1000_0023',\
           'NoDetector.0:Opal1000.1': 'opal1000_0024',\
           'NoDetector.0:Opal1000.2': 'opal1000_0025',\
           'NoDetector.1:Opal1000.0': 'opal1000_0026',\
           'NoDetector.1:Opal1000.1': 'opal1000_0027',\
           'NoDetector.1:Opal1000.2': 'opal1000_0028',\
           'NoDetector.2:Opal1000.0': 'opal1000_0029',\
           'NoDetector.2:Opal1000.1': 'opal1000_0030',\
           'NoDetector.2:Opal1000.2': 'opal1000_0031',\
           'NoDetector.3:Opal1000.0': 'opal1000_0032',\
           'NoDetector.3:Opal1000.1': 'opal1000_0033',\
           'NoDetector.3:Opal1000.2': 'opal1000_0034',\
           'NoDetector.4:Opal1000.0': 'opal1000_0035',\
           'NoDetector.4:Opal1000.1': 'opal1000_0036',\
           'NoDetector.4:Opal1000.2': 'opal1000_0037',\
           'NoDetector.5:Opal1000.0': 'opal1000_0038',\
           'NoDetector.5:Opal1000.1': 'opal1000_0039',\
           'NoDetector.5:Opal1000.2': 'opal1000_0040',\
           'NoDetector.6:Opal1000.0': 'opal1000_0041',\
           'NoDetector.6:Opal1000.1': 'opal1000_0042',\
           'NoDetector.6:Opal1000.2': 'opal1000_0043',\
          'SxrBeamline.0:Opal1000.0': 'opal1000_0044',\
          'SxrBeamline.0:Opal1000.1': 'opal1000_0045',\
        'SxrBeamline.0:Opal1000.100': 'opal1000_0046',\
        'SxrEndstation.0:Opal1000.0': 'opal1000_0047',\
        'SxrEndstation.0:Opal1000.1': 'opal1000_0048',\
        'SxrEndstation.0:Opal1000.2': 'opal1000_0049',\
        'SxrEndstation.0:Opal1000.3': 'opal1000_0050',\
        'XcsEndstation.0:Opal1000.0': 'opal1000_0051',\
        'XcsEndstation.0:Opal1000.1': 'opal1000_0052',\
        'XcsEndstation.1:Opal1000.1': 'opal1000_0053',\
        'XcsEndstation.1:Opal1000.2': 'opal1000_0054',\
        'XcsEndstation.1:Opal1000.3': 'opal1000_0055',\
        'XppEndstation.0:Opal1000.0': 'opal1000_0056',\
        'XppEndstation.0:Opal1000.1': 'opal1000_0057',\
        'XppEndstation.0:Opal1000.2': 'opal1000_0058',\
'XrayTransportDiagnostic.0:Opal1000.0': 'opal1000_0059',\
    'FeeHxSpectrometer.0:Opal1000.1': 'opal1000_0060',\
#== type 0014 rayonix
         'CxiEndstation.0:Rayonix.0': 'rayonix_0000',\
         'MfxEndstation.0:Rayonix.0': 'rayonix_0001',\
         'XppEndstation.0:Rayonix.0': 'rayonix_0002',\
             'XppSb1Pim.0:Rayonix.0': 'rayonix_0003',\
#== type 0015 controlscamera
    'MecTargetChamber.0:ControlsCamera.1': 'controlscamera_0000',\
    'XcsEndstation.0:ControlsCamera.0': 'controlscamera_0001',\
#== type 0016 fccd960
         'SxrEndstation.0:Fccd960.0': 'fccd960_0000',\
         'XcsEndstation.0:Fccd960.0': 'fccd960_0001',\
#== type 0017 beammonitor
     'XcsEndstation.0:BeamMonitor.0': 'beammonitor_0000',\
#== type 0018 cspad
                  'CxiDg4.0:Cspad.0': 'cspad_0000',\
                  'CxiDs1.0:Cspad.0': 'cspad_0001',\
                  'CxiDs2.0:Cspad.0': 'cspad_0002',\
                  'CxiDsd.0:Cspad.0': 'cspad_0003',\
        'MecTargetChamber.0:Cspad.0': 'cspad_0004',\
           'MfxEndstation.0:Cspad.0': 'cspad_0005',\
              'NoDetector.0:Cspad.0': 'cspad_0006',\
             'NoDetector.10:Cspad.0': 'cspad_0007',\
             'NoDetector.11:Cspad.0': 'cspad_0008',\
             'NoDetector.12:Cspad.0': 'cspad_0009',\
              'NoDetector.1:Cspad.0': 'cspad_0010',\
              'NoDetector.2:Cspad.0': 'cspad_0011',\
              'NoDetector.3:Cspad.0': 'cspad_0012',\
              'NoDetector.4:Cspad.0': 'cspad_0013',\
              'NoDetector.5:Cspad.0': 'cspad_0014',\
              'NoDetector.6:Cspad.0': 'cspad_0015',\
              'NoDetector.7:Cspad.0': 'cspad_0016',\
              'NoDetector.8:Cspad.0': 'cspad_0017',\
              'NoDetector.9:Cspad.0': 'cspad_0018',\
             'SxrBeamline.0:Cspad.1': 'cspad_0019',\
           'XcsEndstation.0:Cspad.0': 'cspad_0020',\
                  'XppGon.0:Cspad.0': 'cspad_0021',\
#== type 0019 imp
             'CxiEndstation.0:Imp.0': 'imp_0000',\
             'CxiEndstation.0:Imp.1': 'imp_0001',\
               'SxrBeamline.0:Imp.0': 'imp_0002',\
             'SxrEndstation.0:Imp.0': 'imp_0003',\
             'XcsEndstation.0:Imp.0': 'imp_0004',\
             'XcsEndstation.0:Imp.1': 'imp_0005',\
             'XppEndstation.0:Imp.0': 'imp_0006',\
             'XppEndstation.0:Imp.1': 'imp_0007',\
#== type 0020 epix10ka
               'DetLab.0:Epix10ka.0': 'epix10ka_0000',\
     'MecTargetChamber.0:Epix10ka.0': 'epix10ka_0001',\
        'MfxEndstation.0:Epix10ka.0': 'epix10ka_0002',\
        'MfxEndstation.0:Epix10ka.1': 'epix10ka_0003',\
        'MfxEndstation.0:Epix10ka.2': 'epix10ka_0004',\
#== type 0021 dualandor
       'SxrEndstation.0:DualAndor.0': 'dualandor_0000',\
#== type 0022 opal8000
     'MecTargetChamber.0:Opal8000.0': 'opal8000_0000',\
     'MecTargetChamber.0:Opal8000.1': 'opal8000_0001',\
     'MecTargetChamber.0:Opal8000.2': 'opal8000_0002',\
#== type 0023 usdusb
          'AmoEndstation.0:USDUSB.0': 'usdusb_0000',\
          'CxiEndstation.0:USDUSB.0': 'usdusb_0001',\
          'CxiEndstation.0:USDUSB.1': 'usdusb_0002',\
          'SxrEndstation.0:USDUSB.0': 'usdusb_0003',\
          'SxrEndstation.0:USDUSB.1': 'usdusb_0004',\
          'XppEndstation.0:USDUSB.0': 'usdusb_0005',\
          'XppEndstation.0:USDUSB.1': 'usdusb_0006',\
#== type 0024 timepix
         'XcsEndstation.0:Timepix.0': 'timepix_0000',\
#== type 0025 encoder
           'SxrBeamline.0:Encoder.0': 'encoder_0000',\
                'XppGon.0:Encoder.0': 'encoder_0001',\
#== type 0026 oceanoptics
     'AmoEndstation.0:OceanOptics.0': 'oceanoptics_0000',\
    'MecTargetChamber.0:OceanOptics.': 'oceanoptics_0001',\
    'MecTargetChamber.0:OceanOptics.1': 'oceanoptics_0002',\
     'SxrEndstation.0:OceanOptics.0': 'oceanoptics_0003',\
            'XppLas.0:OceanOptics.0': 'oceanoptics_0004',\
            'XppLas.0:OceanOptics.1': 'oceanoptics_0005',\
            'XppLas.0:OceanOptics.2': 'oceanoptics_0006',\
#== type 0027 lecroy
       'MecTargetChamber.0:LeCroy.0': 'lecroy_0000',\
          'XcsEndstation.0:LeCroy.0': 'lecroy_0001',\
#== type 0028 orcafl40
        'XcsEndstation.0:OrcaFl40.0': 'orcafl40_0000',\
        'XppEndstation.0:OrcaFl40.0': 'orcafl40_0001',\
#== type 0029 quartz4a150
     'CxiEndstation.0:Quartz4A150.0': 'quartz4a150_0000',\
     'XppEndstation.0:Quartz4A150.0': 'quartz4a150_0001',\
#== type 0030 fccd
            'SxrEndstation.0:Fccd.0': 'fccd_0000',\
          'SxrEndstation.0:Fccd.100': 'fccd_0001',\
#== type 0031 pimax
           'AmoEndstation.0:Pimax.0': 'pimax_0000',\
#== type 0032 epix100a
               'CxiSc1.0:Epix100a.0': 'epix100a_0000',\
               'CxiSc2.0:Epix100a.0': 'epix100a_0001',\
               'CxiSc2.0:Epix100a.1': 'epix100a_0002',\
               'DetLab.0:Epix100a.0': 'epix100a_0003',\
     'MecTargetChamber.0:Epix100a.0': 'epix100a_0004',\
        'MfxEndstation.0:Epix100a.0': 'epix100a_0005',\
           'NoDetector.0:Epix100a.0': 'epix100a_0006',\
           'NoDetector.0:Epix100a.1': 'epix100a_0007',\
        'XcsEndstation.0:Epix100a.0': 'epix100a_0008',\
        'XcsEndstation.0:Epix100a.1': 'epix100a_0009',\
        'XcsEndstation.0:Epix100a.2': 'epix100a_0010',\
        'XcsEndstation.0:Epix100a.3': 'epix100a_0011',\
        'XcsEndstation.0:Epix100a.4': 'epix100a_0012',\
        'XcsEndstation.0:Epix100a.5': 'epix100a_0013',\
        'XcsEndstation.0:Epix100a.6': 'epix100a_0014',\
               'XppGon.0:Epix100a.0': 'epix100a_0015',\
               'XppGon.0:Epix100a.1': 'epix100a_0016',\
               'XppGon.0:Epix100a.2': 'epix100a_0017',\
#== type 0033 princeton
       'CxiEndstation.0:Princeton.0': 'princeton_0000',\
    'MecTargetChamber.0:Princeton.0': 'princeton_0001',\
    'MecTargetChamber.0:Princeton.1': 'princeton_0002',\
    'MecTargetChamber.0:Princeton.2': 'princeton_0003',\
    'MecTargetChamber.0:Princeton.3': 'princeton_0004',\
    'MecTargetChamber.0:Princeton.4': 'princeton_0005',\
    'MecTargetChamber.0:Princeton.5': 'princeton_0006',\
       'SxrEndstation.0:Princeton.0': 'princeton_0007',\
         'XcsBeamline.0:Princeton.0': 'princeton_0008',\
       'XppEndstation.0:Princeton.0': 'princeton_0009',\
#== type 0034 epixsampler
            'CxiSc2.0:EpixSampler.0': 'epixsampler_0000',\
#== type 0035 jungfrau
        'CxiEndstation.0:Jungfrau.0': 'jungfrau_0000',\
               'DetLab.0:Jungfrau.0': 'jungfrau_0001',\
               'DetLab.0:Jungfrau.1': 'jungfrau_0002',\
     'MecTargetChamber.0:Jungfrau.0': 'jungfrau_0003',\
        'MfxEndstation.0:Jungfrau.0': 'jungfrau_0004',\
        'MfxEndstation.0:Jungfrau.1': 'jungfrau_0005',\
        'XcsEndstation.0:Jungfrau.0': 'jungfrau_0006',\
        'XcsEndstation.0:Jungfrau.1': 'jungfrau_0007',\
        'XppEndstation.0:Jungfrau.0': 'jungfrau_0008',\
        'XppEndstation.0:Jungfrau.1': 'jungfrau_0009',\
#== type 0036 fli
          'MecTargetChamber.0:Fli.0': 'fli_0000',\
             'XcsEndstation.0:Fli.0': 'fli_0001',\
#== type 0037 opal4000
        'CxiEndstation.0:Opal4000.0': 'opal4000_0000',\
        'CxiEndstation.0:Opal4000.1': 'opal4000_0001',\
        'CxiEndstation.0:Opal4000.3': 'opal4000_0002',\
               'CxiSc2.0:Opal4000.0': 'opal4000_0003',\
               'CxiSc2.0:Opal4000.1': 'opal4000_0004',\
     'MecTargetChamber.0:Opal4000.0': 'opal4000_0005',\
     'MecTargetChamber.0:Opal4000.1': 'opal4000_0006',\
     'MecTargetChamber.0:Opal4000.2': 'opal4000_0007',\
     'MecTargetChamber.0:Opal4000.3': 'opal4000_0008',\
        'MfxEndstation.0:Opal4000.0': 'opal4000_0009',\
#== type 0038 ipimb
                  'CxiDg1.0:Ipimb.0': 'ipimb_0000',\
                  'CxiDg2.0:Ipimb.0': 'ipimb_0001',\
                  'CxiDg2.0:Ipimb.1': 'ipimb_0002',\
                  'CxiDg3.0:Ipimb.0': 'ipimb_0003',\
                  'CxiDg4.0:Ipimb.0': 'ipimb_0004',\
           'CxiEndstation.0:Ipimb.0': 'ipimb_0005',\
             'SxrBeamline.0:Ipimb.1': 'ipimb_0006',\
           'SxrEndstation.0:Ipimb.0': 'ipimb_0007',\
             'XcsBeamline.1:Ipimb.4': 'ipimb_0008',\
             'XcsBeamline.1:Ipimb.5': 'ipimb_0009',\
             'XcsBeamline.2:Ipimb.4': 'ipimb_0010',\
             'XcsBeamline.2:Ipimb.5': 'ipimb_0011',\
           'XcsEndstation.0:Ipimb.0': 'ipimb_0012',\
           'XcsEndstation.0:Ipimb.1': 'ipimb_0013',\
           'XppEndstation.1:Ipimb.0': 'ipimb_0014',\
               'XppMonPim.1:Ipimb.0': 'ipimb_0015',\
               'XppMonPim.1:Ipimb.1': 'ipimb_0016',\
               'XppSb2Ipm.1:Ipimb.0': 'ipimb_0017',\
               'XppSb3Ipm.1:Ipimb.0': 'ipimb_0018',\
               'XppSb3Pim.1:Ipimb.0': 'ipimb_0019',\
               'XppSb4Pim.1:Ipimb.0': 'ipimb_0020',\
#== type 0039 evr
                'NoDetector.0:Evr.0': 'evr_0000',\
                'NoDetector.0:Evr.1': 'evr_0001',\
                'NoDetector.0:Evr.2': 'evr_0002',\
}

#------------------------------

if __name__ == "__main__" :
    dic = DETECTOR_NAME_CONVERSION_DICT
    dnames = ['CxiSc2.0:Epix100a.1', 'MecTargetChamber.0:Andor.1', 'CxiSc2.0:Cspad2x2.3', 'CxiDs2.0:Cspad.0']
    for name in dnames :
        print('%30s ---> %s' % (name,dic[name]))

#------------------------------
