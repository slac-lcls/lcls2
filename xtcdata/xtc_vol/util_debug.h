/*
 * util_debug.h
 *
 *  Created on: Sep 5, 2019
 *      Author: tonglin
 */

#ifndef UTIL_DEBUG_H_
#define UTIL_DEBUG_H_
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int MY_RANK_DEBUG;
#define DEBUG_PRINT  //printf("%s:%u, rank = %d\n", __func__, __LINE__, MY_RANK_DEBUG);

#endif /* UTIL_DEBUG_H_ */
