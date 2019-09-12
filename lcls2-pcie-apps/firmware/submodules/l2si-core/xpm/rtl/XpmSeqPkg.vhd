-------------------------------------------------------------------------------
-- Title      : XpmSeqPkg
-------------------------------------------------------------------------------
-- File       : XpmSeqPkg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2018-03-24
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Package of constants and record definitions for the Timing Geneartor.
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use ieee.math_real.all;
use work.StdRtlPkg.all;
use work.TPGPkg.all;

package XpmSeqPkg is

  constant XPMSEQDEPTH : integer := 1;
  
  type XpmSeqStatusType is record
                          -- implemented resources
                          nexptseq      : slv (7 downto 0);
                          seqaddrlen    : slv (3 downto 0);
                          --
                          countRequest  : Slv32Array(XPMSEQDEPTH-1 downto 0);
                          countInvalid  : Slv32Array(XPMSEQDEPTH-1 downto 0);
                          countUpdate   : sl;  -- single sysclk pulse
                          seqRdData     : Slv32Array(XPMSEQDEPTH-1 downto 0);
                          seqState      : SequencerStateArray(XPMSEQDEPTH-1 downto 0);
                        end record;

  constant XPM_SEQ_STATUS_INIT_C : XpmSeqStatusType := (
    nexptseq      => (others=>'0'),
    seqaddrlen    => (others=>'0'),
    countRequest  => (others=>(others=>'0')),
    countInvalid  => (others=>(others=>'0')),
    countUpdate   => '0',
    seqRdData     => (others=>(others=>'0')),
    seqState      => (others=>SEQUENCER_STATE_INIT_C) );

  type XpmSeqConfigType is record
                          seqEnable     : slv(XPMSEQDEPTH-1 downto 0);
                          seqRestart    : slv(XPMSEQDEPTH-1 downto 0);
                          diagSeq       : slv( 6 downto 0);
                          seqAddr       : SeqAddrType;
                          seqWrData     : slv(31 downto 0);
                          seqWrEn       : slv(XPMSEQDEPTH-1 downto 0);
                          seqJumpConfig : TPGJumpConfigArray(XPMSEQDEPTH-1 downto 0);
                        end record;

  constant XPM_SEQ_CONFIG_INIT_C : XpmSeqConfigType := (
    seqEnable         => (others=>'0'),
    seqRestart        => (others=>'0'),
    diagSeq           => (others=>'1'),
    seqAddr           => (others=>'0'),
    seqWrData         => (others=>'0'),
    seqWrEn           => (others=>'0'),
    seqJumpConfig     => (others=>TPG_JUMPCONFIG_INIT_C)
    );

  type XpmSeqConfigArray is array(natural range<>) of XpmSeqConfigType;
  
end XpmSeqPkg;

package body XpmSeqPkg is
end package body XpmSeqPkg;
