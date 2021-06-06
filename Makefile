TARGET=test1

# need to set the corresponding path according to the local system
NVIDIA_CL_INCLUDE_DIR ?= /home/htcch/winsys/include/cuda
NVIDIA_CL_LIB_DIR ?= /home/htcch/winsys/lib

#
CC=gcc -std=gnu99 -Wall -D_DEFAULT_SOURCE -D_GNU_SOURCE
LINKER = $(CC)

#
CFLAGS = -g -D_DEBUG -I$(NVIDIA_CL_INCLUDE_DIR)
LIBS = -L/home/htcch/winsys/lib -lOpenCL

#
ifeq ($(OS),"Windows_NT")
CFLAGS += -DWIN32
endif

OBJECTS := opencl-context.o test1.o

all: $(TARGET)

test1: $(OBJECTS)
	$(LINKER) -o $@ $^ $(CFLAGS) $(LIBS)
	
$(OBJECTS): %.o : %.c
	$(CC) -o $@ -c $< $(CFLAGS)
	
.PHONY: clean
clean:
	rm -f test1.exe test1 *.o
