PGPCARD_HOME := $(shell pwd)
PGPCARD_KVER := $(shell uname -r)

obj-m := pgpcardG3.o

all:
	make  -C /lib/modules/$(PGPCARD_KVER)/build M=$(PGPCARD_HOME) CONFIG_DEBUG_INFO=YES modules

clean:
	make -C /lib/modules/$(PGPCARD_KVER)/build M=$(PGPCARD_HOME) clean
