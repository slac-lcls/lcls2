-------------------------------------------------------------------------------
-- Title      : TimingPkg
-------------------------------------------------------------------------------
-- File       : TimingPkg.vhd
-- Author     : Benjamin Reese  <bareese@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-01
-- Last update: 2018-07-20
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.TimingExtnPkg.all;

package TimingPkg is

   constant D_102_C : slv(7 downto 0) := "01001010";  -- D10.2, 0x4A
   constant D_215_C : slv(7 downto 0) := "10110101";  -- D21.5, 0xB5
   constant K_COM_C : slv(7 downto 0) := "10111100";  -- K28.5, 0xBC
   constant K_SOF_C : slv(7 downto 0) := "11110111";  -- K23.7, 0xF7
   constant K_EOF_C : slv(7 downto 0) := "11111101";  -- K29.7, 0xFD
   constant K_280_C : slv(7 downto 0) := "00011100";  -- K28.0, 0x1C
   constant K_281_C : slv(7 downto 0) := "00111100";  -- K28.1, 0x3C
   constant K_EOS_C : slv(7 downto 0) := K_280_C;

   constant TIMING_MESSAGE_BITS_C  : integer := 944;
   --  Frame without BSA, beamEnergy, version
   constant TIMING_MESSAGE_BITS_NO_BSA_C  : integer := TIMING_MESSAGE_BITS_C-256-16;
   constant TIMING_MESSAGE_WORDS_C : integer := TIMING_MESSAGE_BITS_C/16;

--   constant TIMING_MESSAGE_VERSION_C : slv(15 downto 0) := x"0000";
   --  Added photon wavelen meta data
   constant TIMING_MESSAGE_VERSION_C : slv(15 downto 0) := x"0001";

   constant TIMING_STREAM_ID  : slv(3 downto 0) := x"0";
   
   type TimingRxType is record
      data       : slv(15 downto 0);
      dataK      : slv( 1 downto 0);
      decErr     : slv( 1 downto 0);
      dspErr     : slv( 1 downto 0);
   end record;
   constant TIMING_RX_INIT_C : TimingRxType := (
      data      => x"0000",
      dataK     => "00",
      decErr    => "00",
      dspErr    => "00" );
   type TimingRxArray is array (natural range<>) of TimingRxType;

   type TimingPhyControlType is record
      reset        : sl;
      inhibit      : sl;
      polarity     : sl;
      bufferByRst  : sl;
      pllReset     : sl;
   end record;
   constant TIMING_PHY_CONTROL_INIT_C : TimingPhyControlType := (
      reset       => '0',
      inhibit     => '0',
      polarity    => '0',
      bufferByRst => '0',
      pllReset    => '0' );
   constant TIMING_PHY_CONTROL_INHIBIT_C : TimingPhyControlType := (
      reset       => '0',
      inhibit     => '1',
      polarity    => '0',
      bufferByRst => '0',
      pllReset    => '0' );
   type TimingPhyControlArray is array (natural range<>) of TimingPhyControlType;

   type TimingPhyStatusType is record
      locked       : sl;
      resetDone    : sl;
      bufferByDone : sl;
      bufferByErr  : sl;
   end record;
   constant TIMING_PHY_STATUS_INIT_C : TimingPhyStatusType := (
      locked       => '0',
      resetDone    => '0',
      bufferByDone => '0',
      bufferByErr  => '0' );
   constant TIMING_PHY_STATUS_FORCE_C : TimingPhyStatusType := (
      locked       => '1',
      resetDone    => '1',
      bufferByDone => '0',
      bufferByErr  => '0' );      
   type TimingPhyStatusArray is array (natural range<>) of TimingPhyStatusType;

   type TimingSerialType is record
      ready      : sl;                -- tx: new segment ready,
                                      -- rx: last segment valid
      data       : slv(15 downto 0);  -- 
      offset     : slv( 6 downto 0);  -- segment index
      last       : sl;                -- last segment
   end record;
   constant TIMING_SERIAL_INIT_C : TimingSerialType := (
      ready      => '0',
      data       => x"0000",
      offset     => (others=>'0'),
      last       => '0' );
   type TimingSerialArray is array (natural range<>) of TimingSerialType;
   
--   type TimingMessageSlv is slv(TIMING_MESSAGE_BITS_C-1 downto 0);
   type TimingMessageType is record
      version         : slv(15 downto 0);
      pulseId         : slv(63 downto 0);
      timeStamp       : slv(63 downto 0);
      fixedRates      : slv(9 downto 0);
      acRates         : slv(5 downto 0);
      acTimeSlot      : slv(2 downto 0);
      acTimeSlotPhase : slv(11 downto 0);
      resync          : sl;
      beamRequest     : slv(31 downto 0);  -- [31:16] charge(pC),
                                           -- [7:4] destn,
                                           -- [0] beam present
      beamEnergy      : Slv16Array(0 to 3);
      photonWavelen   : Slv16Array(0 to 1);
      syncStatus      : sl;
      mpsValid        : sl;
      bcsFault        : slv(0 downto 0);
      mpsLimit        : slv(15 downto 0);
      mpsClass        : slv4Array(0 to 15);
      bsaInit         : slv(63 downto 0);
      bsaActive       : slv(63 downto 0);
      bsaAvgDone      : slv(63 downto 0);
      bsaDone         : slv(63 downto 0);
      control         : slv16Array(0 to 17);
   end record;
   constant TIMING_MESSAGE_INIT_C : TimingMessageType := (
      version         => TIMING_MESSAGE_VERSION_C,
      pulseId         => (others => '0'),
      timeStamp       => (others => '0'),
      fixedRates      => (others => '0'),
      acRates         => (others => '0'),
      acTimeSlot      => (others => '0'),
      acTimeSlotPhase => (others => '0'),
      resync          => '0',
      beamRequest     => (others => '0'),
      beamEnergy      => (others => (others => '0')),
      photonWavelen   => (others => (others => '0')),
      syncStatus      => '0',
      mpsValid        => '0',
      bcsFault        => (others => '0'),
      mpsLimit        => (others => '0'),
      mpsClass        => (others => (others => '1')),
      bsaInit         => (others => '0'),
      bsaActive       => (others => '0'),
      bsaAvgDone      => (others => '0'),
      bsaDone         => (others => '0'),
      control         => (others => (others => '0')) );
   type TimingMessageArray is array (natural range<>) of TimingMessageType;      

   function toSlv  (message              : TimingMessageType) return slv;
   function toSlv32(vector               : slv)               return Slv32Array;
   function toSlvNoBsa(message           : TimingMessageType) return slv;
   function toTimingMessageType(vector : slv)                 return TimingMessageType;

   constant TIMING_DATABUFF_BITS_C  : integer := 416;
   constant TIMING_STREAM_BITS_C  : integer := 704;

   type TimingDataBuffType is record
      dtype      : slv(15 downto 0);
      version    : slv(15 downto 0);
      dmod       : slv(191 downto 0);
      epicsTime  : slv(63 downto 0);
      edefAvgDn  : slv(31 downto 0);
      edefMinor  : slv(31 downto 0);
      edefMajor  : slv(31 downto 0);
      edefInit   : slv(31 downto 0);
   end record;
   constant TIMING_DATA_BUFF_INIT_C : TimingDataBuffType := (
      dtype      => (others=>'0'),
      version    => (others=>'0'),
      dmod       => (others=>'0'),
      epicsTime  => (others=>'0'),
      edefAvgDn  => (others=>'0'),
      edefMinor  => (others=>'0'),
      edefMajor  => (others=>'0'),
      edefInit   => (others=>'0') );
   type TimingDataBuffArray is array (natural range<>) of TimingDataBuffType;
   
   type TimingStreamType is record
      pulseId         : slv(31 downto 0);
      eventCodes      : slv(255 downto 0);
      dbuff           : TimingDataBuffType;
   end record;
   constant TIMING_STREAM_INIT_C : TimingStreamType := (
      pulseId         => (others=>'0'),
      eventCodes      => (others=>'0'),
      dbuff           => TIMING_DATA_BUFF_INIT_C );
   type TimingStreamArray is array (natural range<>) of TimingStreamType;

   function toTimingDataBuffType(vector : slv) return TimingDataBuffType;
   function toTimingStreamType(vector : slv) return TimingStreamType;
   function toSlv(stream : TimingStreamType) return slv;
   function toSlv(dbuff  : TimingDataBuffType) return slv;

   -- LCLS-I Timing Data Type
   type LclsV1TimingDataType is record
      linkUp        : sl;
      gtRxData      : slv(15 downto 0);-- gtRxRecClk domain
      gtRxDataK     : slv(1 downto 0);-- gtRxRecClk domain
      gtRxDispErr   : slv(1 downto 0);-- gtRxRecClk domain
      gtRxDecErr    : slv(1 downto 0);-- gtRxRecClk domain
   end record;
   constant LCLS_V1_TIMING_DATA_INIT_C : LclsV1TimingDataType := (
      linkUp        => '0',
      gtRxData      => (others=>'0'),
      gtRxDataK     => (others=>'0'),
      gtRxDispErr   => (others=>'0'),
      gtRxDecErr    => (others=>'0'));
   type LclsV1TimingDataArray is array (natural range<>) of LclsV1TimingDataType;

   -- LCLS-II Timing Data Type
   type LclsV2TimingDataType is record
      linkUp     : sl;
   end record;
   constant LCLS_V2_TIMING_DATA_INIT_C : LclsV2TimingDataType := (
      linkUp     => '0');
   type LclsV2TimingDataArray is array (natural range<>) of LclsV2TimingDataType;

   type TimingBusType is record
      strobe  : sl;                     -- 1 MHz timing strobe
      valid   : sl;
      message : TimingMessageType;
      stream  : TimingStreamType;
      v1      : LclsV1TimingDataType;
      v2      : LclsV2TimingDataType;
      modesel : sl;  -- LCLS-II selected
      extn    : TimingExtnType;
      extnValid : sl;
   end record;
   constant TIMING_BUS_INIT_C : TimingBusType := (
      strobe  => '0',
      valid   => '0',
      message => TIMING_MESSAGE_INIT_C,
      stream  => TIMING_STREAM_INIT_C,
      v1      => LCLS_V1_TIMING_DATA_INIT_C,
      v2      => LCLS_V2_TIMING_DATA_INIT_C,
      modesel => '0',
      extn    => TIMING_EXTN_INIT_C,
      extnValid => '0' );

   type TimingBusArray is array (integer range<>) of TimingBusType;

   type TimingPhyType is record
      dataK      : slv(1 downto 0);
      data       : slv(15 downto 0);
      control    : TimingPhyControlType;
   end record;
   constant TIMING_PHY_INIT_C : TimingPhyType := (
      dataK      => "00",
      data       => x"0000",
      control    => TIMING_PHY_CONTROL_INIT_C );
   type TimingPhyArray is array (integer range<>) of TimingPhyType;

   type TimingTrigType is record
      trigPulse  : slv(15 downto 0);
      timeStamp  : slv(63 downto 0);
      bsa        : slv(127 downto 0);  -- LCLS-I control info
      dmod       : slv(191 downto 0);  --
   end record;
   constant TIMING_TRIG_INIT_C : TimingTrigType := (
      trigPulse  => (others=>'0'),
      timeStamp  => (others=>'0'),
      bsa        => (others=>'0'),
      dmod       => (others=>'0') );
 
end package TimingPkg;

package body TimingPkg is


   -------------------------------------------------------------------------------------------------
   -- Convert a timing message record into a big long SLV
   -------------------------------------------------------------------------------------------------
   function toSlv (message : TimingMessageType) return slv
   is
      variable vector : slv(TIMING_MESSAGE_BITS_C-1 downto 0) := (others => '0');
      variable i      : integer                               := 0;
   begin
      assignSlv(i, vector, message.version);           -- 1 word
      assignSlv(i, vector, message.pulseId);           -- 4 words
      assignSlv(i, vector, message.timeStamp);         -- 4 words
      assignSlv(i, vector, message.fixedRates);        
      assignSlv(i, vector, message.acRates);           -- 1 word
      assignSlv(i, vector, message.acTimeSlot);
      assignSlv(i, vector, message.acTimeSlotPhase);
      assignSlv(i, vector, message.resync);            -- 1 word
      assignSlv(i, vector, message.beamRequest);       -- 2 words
      for j in message.beamEnergy'range loop
        assignSlv(i, vector, message.beamEnergy(j));
      end loop;                                        -- 4 words
      for j in message.photonWavelen'range loop
        assignSlv(i, vector, message.photonWavelen(j));
      end loop;                                        -- 2 words
      assignSlv(i, vector, "0000000000000");           -- 13 unused bits
      assignSlv(i, vector, message.syncStatus);
      assignSlv(i, vector, message.mpsValid);
      assignSlv(i, vector, message.bcsFault);          -- 1 bit
      assignSlv(i, vector, message.mpsLimit);          -- 1 word
      for j in message.mpsClass'range loop
         assignSlv(i, vector, message.mpsClass(j));
      end loop;                                        -- 4 words
      assignSlv(i, vector, message.bsaInit);           -- 4 words
      assignSlv(i, vector, message.bsaActive);         -- 4 words
      assignSlv(i, vector, message.bsaAvgDone);        -- 4 words
      assignSlv(i, vector, message.bsaDone);           -- 4 words
      for j in message.control'range loop
         assignSlv(i, vector, message.control(j));
      end loop;                                        -- 18 words
      return vector;
   end function;

   -------------------------------------------------------------------------------------------------
   -- Convert a timing message record into a big long SLV
   -------------------------------------------------------------------------------------------------
   function toSlv32 (vector : slv) return Slv32Array
   is
      variable vec32  : Slv32Array(vector'length/32-1 downto 0) := (others => x"00000000");
      variable i      : integer := vector'right;
   begin
     for j in 0 to vector'length/32-1 loop
       vec32(j) := vector(i+31 downto i);
       i        := i+32;
     end loop;  -- j
     return vec32;
   end function;

   -------------------------------------------------------------------------------------------------
   -- Convert a timing message record into a big long SLV with no BSA
   -------------------------------------------------------------------------------------------------
   function toSlvNoBsa (message : TimingMessageType) return slv
   is
      variable vector : slv(TIMING_MESSAGE_BITS_NO_BSA_C-1 downto 0) := (others => '0');
      variable i      : integer := 0;
   begin
--      assignSlv(i, vector, message.version);
      assignSlv(i, vector, message.pulseId);
      assignSlv(i, vector, message.timeStamp);
      assignSlv(i, vector, message.fixedRates);
      assignSlv(i, vector, message.acRates);
      assignSlv(i, vector, message.acTimeSlot);
      assignSlv(i, vector, message.acTimeSlotPhase);
      assignSlv(i, vector, message.resync);
      assignSlv(i, vector, message.beamRequest);
      for j in message.beamEnergy'range loop
        assignSlv(i, vector, message.beamEnergy(j));
      end loop;                                        -- 4 words
      for j in message.photonWavelen'range loop
        assignSlv(i, vector, message.photonWavelen(j));
      end loop;                                        -- 2 words
      assignSlv(i, vector, message.control(16));    -- use this field to complete
                                                    -- modifier word encoding
      --assignSlv(i, vector, "0000000000000");        -- 13 unused bits
      --assignSlv(i, vector, message.syncStatus);
      --assignSlv(i, vector, message.mpsValid);
      --assignSlv(i, vector, message.bcsFault);
      assignSlv(i, vector, message.mpsLimit);
      for j in message.mpsClass'range loop
         assignSlv(i, vector, message.mpsClass(j));
      end loop;
      for j in message.control'range loop
         assignSlv(i, vector, message.control(j));
      end loop;
      return vector;
   end function;

   -------------------------------------------------------------------------------------------------
   -- Convert an SLV into a timing record
   -------------------------------------------------------------------------------------------------
   function toTimingMessageType (vector : slv) return TimingMessageType
   is
      variable message : TimingMessageType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.version);
      assignRecord(i, vector, message.pulseId);
      assignRecord(i, vector, message.timeStamp);
      assignRecord(i, vector, message.fixedRates);
      assignRecord(i, vector, message.acRates);
      assignRecord(i, vector, message.acTimeSlot);
      assignRecord(i, vector, message.acTimeSlotPhase);
      assignRecord(i, vector, message.resync);
      assignRecord(i, vector, message.beamRequest);
      for j in message.beamEnergy'range loop
        assignRecord(i, vector, message.beamEnergy(j));
      end loop;                                        -- 4 words
      for j in message.photonWavelen'range loop
        assignRecord(i, vector, message.photonWavelen(j));
      end loop;                                        -- 2 words
      i := i+ 13;                        -- 13 unused bits
      assignRecord(i, vector, message.syncStatus);
      assignRecord(i, vector, message.mpsValid);
      assignRecord(i, vector, message.bcsFault);
      assignRecord(i, vector, message.mpsLimit);
      for j in message.mpsClass'range loop
         assignRecord(i, vector, message.mpsClass(j));
      end loop;
      assignRecord(i, vector, message.bsaInit);
      assignRecord(i, vector, message.bsaActive);
      assignRecord(i, vector, message.bsaAvgDone);
      assignRecord(i, vector, message.bsaDone);
      for j in message.control'range loop
         assignRecord(i, vector, message.control(j));
      end loop;
      return message;
   end function;

   function toTimingDataBuffType(vector : slv) return TimingDataBuffType
   is
      variable message : TimingDataBuffType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.dtype);
      assignRecord(i, vector, message.version);
      assignRecord(i, vector, message.dmod);
      assignRecord(i, vector, message.epicsTime);
      assignRecord(i, vector, message.edefAvgDn);
      assignRecord(i, vector, message.edefMinor);
      assignRecord(i, vector, message.edefMajor);
      assignRecord(i, vector, message.edefInit);
      return message;
   end function;

   function toTimingStreamType(vector : slv) return TimingStreamType
   is
      variable message : TimingStreamType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.pulseId);
      assignRecord(i, vector, message.eventCodes);
      assignRecord(i, vector, message.dbuff.dtype);
      assignRecord(i, vector, message.dbuff.version);
      assignRecord(i, vector, message.dbuff.dmod);
      assignRecord(i, vector, message.dbuff.epicsTime);
      assignRecord(i, vector, message.dbuff.edefAvgDn);
      assignRecord(i, vector, message.dbuff.edefMinor);
      assignRecord(i, vector, message.dbuff.edefMajor);
      assignRecord(i, vector, message.dbuff.edefInit);
      return message;
   end function;

   function toSlv (stream : TimingStreamType) return slv
   is
      variable vector : slv(TIMING_STREAM_BITS_C-1 downto 0) := (others => '0');
      variable i      : integer                              := 0;
   begin
      assignSlv(i, vector, stream.pulseId);
      assignSlv(i, vector, stream.eventCodes);
      assignSlv(i, vector, stream.dbuff.dtype);
      assignSlv(i, vector, stream.dbuff.version);
      assignSlv(i, vector, stream.dbuff.dmod);
      assignSlv(i, vector, stream.dbuff.epicsTime);
      assignSlv(i, vector, stream.dbuff.edefAvgDn);
      assignSlv(i, vector, stream.dbuff.edefMinor);
      assignSlv(i, vector, stream.dbuff.edefMajor);
      assignSlv(i, vector, stream.dbuff.edefInit);
      return vector;
   end function;

   function toSlv (dbuff : TimingDataBuffType) return slv
   is
      variable vector : slv(TIMING_DATABUFF_BITS_C-1 downto 0) := (others => '0');
      variable i      : integer                              := 0;
   begin
      assignSlv(i, vector, dbuff.dtype);
      assignSlv(i, vector, dbuff.version);
      assignSlv(i, vector, dbuff.dmod);
      assignSlv(i, vector, dbuff.epicsTime);
      assignSlv(i, vector, dbuff.edefAvgDn);
      assignSlv(i, vector, dbuff.edefMinor);
      assignSlv(i, vector, dbuff.edefMajor);
      assignSlv(i, vector, dbuff.edefInit);
      return vector;
   end function;

end package body TimingPkg;
