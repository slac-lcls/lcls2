#!/usr/bin/env python
import sys
import psana.pscalib.calib.MDBWebUtils as wu

def do_main():
    print('command jwt' + wu.info_ticket)
    sys.exit(0)

if __name__ == "__main__":
    do_main()

#EOF
