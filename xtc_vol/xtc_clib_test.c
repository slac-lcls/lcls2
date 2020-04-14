/*
 * xtc_clib_test.c
 *
 *  Created on: Jan 20, 2020
 *      Author: tonglin
 */


#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"xtc_io_api_c.h"

int main(int argc, char* argv[] ){
    xtc_object* helper = xtc_file_open(argv[1]);
    //xtc_it_open(helper);
    return 0;
}
