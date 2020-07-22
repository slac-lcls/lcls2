daqConfig = {'readoutGroup':('i', 0),
             'full_event'  :('i', 8),
             'full_size'   :('i', 2048),
             'test_pattern':('i', 0),
             'fs_range_vpp':('i', 0xa000),
             'trig_shift'  :('i', 0),
             'sync_ph_even':('i', 11250),
             'sync_ph_odd' :('i',  1450),
             'enable'      :('i', 0),
             'raw_start'   :('i', 4),
             'raw_gate'    :('i', 54),
             'raw_prescale':('i', 0),
             'fex_start'   :('i', 4),
             'fex_gate'    :('i', 40),
             'fex_prescale':('i', 1),
             'fex_ymin'    :('i', 2040),
             'fex_ymax'    :('i', 2056),
             'fex_xpre'    :('i', 1),
             'fex_xpost'   :('i', 1) }

daqReset   = {'timrxrst'    :('i', 0),
              'timpllrst'   :('i', 0),
              'pgploopback' :('i', 0),
              'reset'       :('i', 0),
              'jesdsetup'   :('i', 0),
              'jesdinit'    :('i', 0),
              'jesdadcinit' :('i', 0),
              'jesdclear'   :('i', 0)}

monTiming = {'timframecnt':('i', 0),
             'timpausecnt':('i', 0),
             'timerrcntsum':('i', 0),
             'timrstcntsum':('i', 0),
             'trigcnt'    :('i', 0),
             'trigcntsum' :('i', 0),
             'readcntsum' :('i', 0),
             'startcntsum':('i', 0),
             'queuecntsum':('i', 0),
             'msgdelayset':('i', 0),
             'msgdelayget':('i', 0),
             'headercntl0':('i', 0),
             'headercntof':('i', 0),
             'headerfifow':('i', 0),
             'headerfifor':('i', 0),
             'fulltotrig' :('i', 0),
             'nfulltotrig':('i', 0) }

monPgp    = {'loclinkrdy' :('ai', [0]*4),
             'remlinkrdy' :('ai', [0]*4),
             'txclkfreq'  :('af', [0.]*4),
             'rxclkfreq'  :('af', [0.]*4),
             'txcnt'      :('ai', [0]*4),
             'txcntsum'   :('ai', [0]*4),
             'txerrcntsum':('ai', [0]*4),
             'rxcnt'      :('ai', [0]*4),
             'rxlast'     :('ai', [0]*4, '{:x}'),
             'rempause'   :('ai', [0]*4),
             'remlinkid'  :('ai', [0]*4, '{:08x}') }

monBuf    = {'freesz'  :('i', 0),
             'freeevt' :('i', 0),
             'fifoof'  :('i', 0) }

monBufDetail = {'bufstate' : ('ai',[0]*16),
                'trgstate' : ('ai',[0]*16),
                'bufbeg'   : ('af',[0]*16),
                'bufend'   : ('af',[0]*16) }

monFlow   = {'fmask'   :('i', 0, 'remaining streams'),
             'fcurr'   :('i', 0, 'current stream'),
             'frdy'    :('i', 0, 'ready streams'),
             'srdy'    :('i', 0, 'subordinate ready'),
             'mrdy'    :('i', 0, 'main ready'),
             'raddr'   :('i', 0),
             'npend'   :('i', 0),
             'ntrig'   :('i', 0),
             'nread'   :('i', 0),
             'pkoflow' :('i', 0),
             'oflow'   :('i', 0) }
 
monEnv    = {'local12v' :('f',0),
             'edge12v'  :('f',0),
             'aux12v'   :('f',0),
             'fmc12v'   :('f',0),
             'boardtemp':('f',0),
             'local3v3' :('f',0),
             'local2v5' :('f',0),
             'totalpwr' :('f',0),
             'fmcpwr'   :('f',0),
             'sync_even':('i',0),
             'sync_odd' :('i',0) }


monJesd   = {'stat' :('ai',[0]*112),
             'clks' :('af',[0.]*5) }

monAdc   = {'oor_ina_0' :('i',0),
            'oor_ina_1' :('i',0),
            'oor_inb_0' :('i',0),
            'oor_inb_1' :('i',0),
            'alarm'     :('i',0) }

monJesdTtl = {'ttl' :('as',['GTResetDone','RxDataValid','RxDataNAlign','SyncStatus','RxBufferOF','RxBufferUF',
                            'CommaPos','ModuleEn','SysRefDet','CommaDet','DispErr','DecErr','BuffLatency','CdrStatus'])}
