##############################################################################
## This file is part of 'Camera link gateway'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'Camera link gateway', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[0].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]] \ 
                               -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[0].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]]

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[1].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]] \ 
                               -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[1].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]]

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[2].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]] \ 
                               -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[2].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]]

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[3].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]] \ 
                               -group [get_clocks -of_objects [get_pins {U_Hardware/GEN_LANE[3].GEN_PGP2b.U_Lane/REAL_PGP.U_Pgp/PgpGthCoreWrapper_1/U_PgpGthCore/inst/gen_gtwizard_gthe3_top.PgpGthCore_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[2].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]]
