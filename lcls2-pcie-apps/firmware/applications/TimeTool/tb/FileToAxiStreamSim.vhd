-------------------------------------------------------------------------------
-- File       : AxiStreamBytePackerTbTx.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description:
-- AxiStream data packer tester, tx module
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Firmware Standard Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Firmware Standard Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.StdRtlPkg.all;

use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPkg.all;
use work.SsiPkg.all;
use work.AxiPciePkg.all;
use work.TestingPkg.all;

use STD.textio.all;
use ieee.std_logic_textio.all;

entity FileToAxiStreamSim is
   generic (
      TPD_G              : time                := 1 ns;
      BYTE_SIZE_C        : positive            := 1;
      BITS_PER_TRANSFER  : natural             := 128;
      AXIS_CONFIG_G      : AxiStreamConfigType := AXI_STREAM_CONFIG_INIT_C);
   port (
      -- System clock and reset
      axiClk       : in  sl;
      axiRst       : in  sl;
      -- Outbound frame
      mAxisMaster  : out AxiStreamMasterType;
      mAxisSlave   : in  AxiStreamSlaveType);

end FileToAxiStreamSim;

architecture rtl of FileToAxiStreamSim is

   type StateType is (IDLE_S,MOVE_S);

   type RegType is record
      byteCount    : natural;
      frameCount   : natural;
      sleepCount   : natural;
      counter      : natural;
      master       : AxiStreamMasterType;
      slave        : AxiStreamSlaveType;
      state        : StateType;
   end record RegType;

   constant REG_INIT_C : RegType := (
      byteCount    => 0,
      frameCount   => 0,
      sleepCount   => 0,
      master       => AXI_STREAM_MASTER_INIT_C,
      slave        => AXI_STREAM_SLAVE_INIT_C,
      counter      => 100,
      state        => IDLE_S);

   constant TEST_INPUT_FILE_NAME : string := TEST_FILE_PATH & "/sim_input_data.dat";

   file file_VECTORS : text;
   file file_RESULTS : text;
 
   constant c1_WIDTH : natural := 8;
   constant c2_WIDTH : natural := 2;

   constant CLK_PERIOD_G : time      := 10 ns;

   signal r            : RegType := REG_INIT_C;
   signal rin          : RegType;

   signal sysClk       :  sl;
   signal sysRst       :  sl;

   signal outCtrl      : AxiStreamCtrlType;
   
   signal r_ADD_TERM1  : std_logic_vector(BITS_PER_TRANSFER-1 downto 0) := (others => '0');
   signal r_ADD_TERM2  : sl := '0';
   --signal w_SUM       : std_logic_vector(c_WIDTH downto 0);



begin
   
   outCtrl.pause <= not mAxisSlave.tready;

   U_axilClk_2 : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 0 ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => sysClk,
         rst  => sysRst);


   process 
      variable v           : RegType := REG_INIT_C;
      variable v_ILINE     : line;
      variable v_OLINE     : line;
      variable v_ADD_TERM1 : std_logic_vector(BITS_PER_TRANSFER-1 downto 0);
      variable v_ADD_TERM2 : sl := '0';
      variable v_SPACE     : character;

   begin 
      v := r;
      
      --w_SUM <= (others=>'1');

      --file_open(file_RESULTS, "output_results.txt", write_mode);
      file_open(file_VECTORS, TEST_INPUT_FILE_NAME,  read_mode);
      

    while not endfile(file_VECTORS) loop

    wait for CLK_PERIOD_G;

    v.slave.tReady    := not outCtrl.pause;
    v.master          := AXI_STREAM_MASTER_INIT_C;
    v.master.tValid   := '1';

    case r.state is

            when IDLE_S =>
            ------------------------------
            -- check which state
            ------------------------------
                  if v.slave.tReady = '1' then  

                        v.state     := MOVE_S;
                      
                  else

                        v.state     := IDLE_S;

                  end if;
 
            when MOVE_S =>
            ------------------------------
            -- move data
            ------------------------------
                  if v.slave.tReady = '1' then
                            readline(file_VECTORS, v_ILINE);
                            read(v_ILINE, v_ADD_TERM1);
                            read(v_ILINE, v_SPACE);           -- read in the space character
                            read(v_ILINE, v_ADD_TERM2);
             
                           -- Pass the variable to a signal to allow the ripple-carry to use it
                            r_ADD_TERM1 <= v_ADD_TERM1;
                            r_ADD_TERM2 <= v_ADD_TERM2;

                            v.master.tData(BITS_PER_TRANSFER-1 downto 0)        := v_ADD_TERM1;
                            --v.master.tValid                                     := '1';   <- this line is bad and violates axi protocol.  should always be true
                            v.master.tLast                                      := v_ADD_TERM2;
                            v.master.tKeep(BITS_PER_TRANSFER/8-1 downto 0)      := (others=>'1');
            
                            --v.state         := IDLE_S;
             
                        
                    else
                            --v.master.tValid := '0';   --message to downstream data processing that there's no valid data ready <- this line is bad and violates axi handshake protocol
                            v.slave.tReady  := '0';   --message to upstream that we're not ready
                            v.master.tLast  := '0';
                            v.state         := IDLE_S;



                   end if;

        end case;

        --if v.counter < 1 then
        --    v.master := AXI_STREAM_MASTER_INIT_C;
        --    v.master.tKeep  := (others=>'0');            

        --    readline(file_VECTORS, v_ILINE);
        --    read(v_ILINE, v_ADD_TERM1);
        --   read(v_ILINE, v_SPACE);           -- read in the space character
        --    read(v_ILINE, v_ADD_TERM2);
       
            -- Pass the variable to a signal to allow the ripple-carry to use it
        --    r_ADD_TERM1 <= v_ADD_TERM1;
        --    r_ADD_TERM2 <= v_ADD_TERM2;
       

            --need to modify the python code to write 128 bits to a single line so below can be used instead
            --for i in 0 to INT_CONFIG_C.TDATA_BYTES_C-1 loop
            --   v.master.tData(i*8+7 downto i*8) := inMaster.tData(i*8+7 downto i*8) + r.addValue;
            --end loop;
            
        --    v.master.tData(BITS_PER_TRANSFER-1 downto 0)        := v_ADD_TERM1;
        --    v.master.tValid                                     := '1';
        --    v.master.tLast                                      := v_ADD_TERM2;
        --    v.master.tKeep(BITS_PER_TRANSFER/8-1 downto 0)      := (others=>'1');
        --    v.counter                                           := 15;
       --else
        --    v.master.tKeep      := (others=>'0');
        --    v.master.tKeep(0)   := '1';
        --    v.master.tValid     := '0';
        --    v.master.tLast      := '0';
        --    v.counter           := v.counter-1;

       --end if;

      --v.master.tLast := '0';
      --if (v_ADD_TERM2(1)='1') then
      --       v.master.tLast := '1';
      --else
      --       v.master.tLast := '0';
      --end if;

 
      --write(v_OLINE, w_SUM, right, c_WIDTH);
      --writeline(file_RESULTS, v_OLINE);

    
      rin <= v;

      mAxisMaster <= v.master;

    end loop;
 
    file_close(file_VECTORS);
    file_close(file_RESULTS);
     
    wait;

   end process;

   seq : process (sysClk) is
   begin
      if (rising_edge(sysClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end architecture rtl;
