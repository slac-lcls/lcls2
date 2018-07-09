HOME := $(shell pwd)
KVER := $(shell uname -r)

ccflags-y := -I$(HOME)/../include/
ccflags-y += -DDMA_IN_KERNEL=1

obj-m := pgpGen4Daq.o pgpGen4Test.o pgpGen4SimCam.o

all:
	make  -C /lib/modules/$(KVER)/build M=$(HOME) CONFIG_DEBUG_INFO=YES modules

clean:
	make -C /lib/modules/$(KVER)/build M=$(HOME) clean
