
XPM Rx serial link formats

    From DTI    :          (Encapsulated on Bp link [SALT])
         Full(15 downto 0)
    From XPM    :          (AMC Link)
         D_215_C & K_EOS_C
         Full(15 downto 0) 
         MBZ(15:8) trigsrc(7:4) partition(3:0) 
         MBZ(15)   strobe(14)   trigword(13:5) tag(4:0)
         (last two words repeat)
         D_215_C & K_EOF_C
    From Sensor :          (AMC Link)
         D_215_C & K_SOF_C
         Full(15)  strobe(14)   trigword(13:5) tag(4:0)
         D_215_C & K_EOF_C
         (partition#, trigsrc must be configured)

DTI

    From DS (DRP):
         
