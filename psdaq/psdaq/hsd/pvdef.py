daqConfig = {'readoutGroup'     :('i', 0, 'ROG'),
             'test_pattern'     :('i', 0),
             'fs_range_vpp'     :('i', 0xa000),
             'enable'           :('i', 0),
             'raw_start'        :('i', 4),
             'raw_gate'         :('i', 54),
             'raw_prescale'     :('i', 0),
             'raw_keep'         :('i', 0),
             'fex_start'        :('i', 4),
             'fex_gate'         :('i', 40),
             'fex_prescale'     :('i', 1),
             'fex_ymin'         :('i', 2040),
             'fex_ymax'         :('i', 2056),
             'fex_xpre'         :('i', 1),
             'fex_xpost'        :('i', 1),
             'fex_corr_baseline':('i',16384),
             'fex_corr_accum'   :('i',12),
             'full_event'       :('i', 8, 'Min buffers thr'),
             'full_size_raw'    :('i', 2048, 'Min rows thr (40sa)'),
             'full_size_fex'    :('i', 2048, 'Min rows thr (40sa)')}

pgpConfig  = {'diffctrl'    :('ai', [15]*4),
              'precursor'   :('ai', [7]*4),
              'postcursor'  :('ai', [7]*4)}

daqReset   = {'timrxrst'    :('i', 0),
              'timpllrst'   :('i', 0),
              'pgploopback' :('i', 0),
              'pgpnofull'   :('i', 0),
              'reset'       :('i', 0),
              'jesdsetup'   :('i', 0, 'ADC input selection'),
              'jesdinit'    :('i', 0),
              'jesdadcinit' :('i', 0),
              'jesdclear'   :('i', 0),
              'cfgdump'     :('i', 0)}

monTiming = {'timframecnt':('i', 0, '929kH nominal'),
             'timpausecnt':('i', 0, 'out of 148M'),
             'timerrcntsum':('i', 0, 'integ bit errs'),
             'timrstcntsum':('i', 0, 'integ link rsts'),
             'trigcntsum' :('i', 0, 'integ HDR counts'),
             'trigcntrate':('i', 0, 'rate HDR counts'),
             'group'      :('i', 0, 'ROG'),
             'l0delay'    :('i', 0, 'ROG L0Delay+1'),
             'hdrcount'   :('i', 0, 'ROG HDR Count'),
             'chndatapaus':('i', 0, 'ADC Data paused'),
             'hdrfifopaus':('i', 0, 'ROG HDR FIFO paused'),
             'hdrfifoof'  :('i', 0, 'ROG HDR FIFO OF'),
             'hdrfifoofl' :('i', 0, 'ROG HDR OF latched'),
             'hdrfifow'   :('i', 0, 'ROG HDR FIFO fill count'),
             'hdrfifor'   :('i', 0, 'ROG HDR FIFO pause thr'),
             'fulltotrig' :('i', 0, 'Max full to L0 (timing clks)'),
             'nfulltotrig':('i', 0, 'Min nfull to L0 (timing clks)') }

monPgp    = {'loclinkrdy' :('ai', [0]*4),
             'remlinkrdy' :('ai', [0]*4),
             'txclkfreq'  :('af', [0.]*4),
             'rxclkfreq'  :('af', [0.]*4),
             'txcnt'      :('ai', [0]*4),
             'txcntsum'   :('ai', [0]*4),
             'txerrcntsum':('ai', [0]*4),
             'rxcnt'      :('ai', [0]*4),
             'rxcntsum'   :('ai', [0]*4),
             'rxerrcntsum':('ai', [0]*4),
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

monFlow   = {'fmask'   :('i', 0, 'remaining streams (mask)'),
             'fcurr'   :('i', 0, 'current stream (<<1)'),
             'frdy'    :('i', 0, 'ready streams (mask)'),
             'srdy'    :('i', 0, '--'),
             'mrdy'    :('i', 0, 'master/slave/master ready'),
             'raddr'   :('i', 0, 'read address (supersamples)'),
             'npend'   :('i', 0, 'buffer index'),
             'ntrig'   :('i', 0, 'buffer index'),
             'nread'   :('i', 0, 'buffer index'),
             'pkoflow' :('i', 0, 'cntOflow'),
             'oflow'   :('i', 0, 'write address'),
             'bstat'   :('i', 0, 'build st (1=idle,rdhdr,wrhdr,rdchn)'), 
             'dumps'   :('i', 0, 'events dumped'),
             'bhdrv'   :('i', 0, 'event header valid'),
             'bval'    :('i', 0, 'dma master valid'),
             'brdy'    :('i', 0, 'dma slave ready')}
 
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
            'alarm'     :('i',0),
            'oor_fex'   :('i',0) }

monJesdTtl = {'ttl' :('as',['GTResetDone','RxDataValid','RxDataNAlign','SyncStatus','RxBufferOF','RxBufferUF',
                            'CommaPos','ModuleEn','SysRefDet','CommaDet','DispErr','DecErr','BuffLatency','CdrStatus'])}
