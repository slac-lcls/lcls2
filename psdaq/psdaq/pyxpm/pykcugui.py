#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# This file is part of the 'lcls2-pgp-pcie-apps'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'lcls2-pgp-pcie-apps', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os
import sys
import argparse
import importlib
import rogue
import pyrogue.pydm
import struct

if __name__ == "__main__":

#################################################################

    # Set the argument parser
    parser = argparse.ArgumentParser()

    # Convert str to bool
    argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

    # Add arguments
    parser.add_argument(
        "--dev",
        type     = str,
        required = False,
        default  = '/dev/datadev_0',
        help     = "path to device",
    )

    parser.add_argument(
        "--pollEn",
        type     = argBool,
        required = False,
        default  = True,
        help     = "Enable auto-polling",
    )

    parser.add_argument(
        "--initRead",
        type     = argBool,
        required = False,
        default  = True,
        help     = "Enable read all variables at start",
    )

    parser.add_argument(
        "--serverPort",
        type     = int,
        required = False,
        default  = 9099,
        help     = "Zeromq server port",
    )

    parser.add_argument(
        "--guiType",
        type     = str,
        required = False,
        default  = 'PyDM',
        help     = "Sets the GUI type (PyDM or PyQt)",
    )

    # Get the arguments
    args = parser.parse_args()

    import kcu

    #################################################################

    with kcu.DevRoot(
            dev            = args.dev,
            zmqSrvEn       = True,
            pollEn         = args.pollEn,
            initRead       = args.initRead,
        ) as root:

        def handle(msg):
            s = struct.Struct('H')
            siter = s.iter_unpack(msg)
            src = next(siter)[0]
            print(f'src is {src},  len is {len(msg)}')

        root.handle(handle)

        ######################
        # Development PyDM GUI
        ######################
        if (args.guiType == 'PyDM'):
            pyrogue.pydm.runPyDM(
                serverList  = root.zmqServer.address,
                sizeX = 800,
                sizeY = 1000,
            )

        #################
        # No GUI
        #################
        elif (args.guiType == 'None'):

            # Wait to be killed via Ctrl-C
            print('Running root server.  Hit Ctrl-C to exit')
            try:
                while True:
                    time.sleep(1)
            except:
                pass
            print('Stopping root server...')
            root.stop()

        ####################
        # Undefined GUI type
        ####################
        else:
            raise ValueError("Invalid GUI type (%s)" % (args.guiType) )


    #################################################################
