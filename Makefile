BIN_DIR=bin
TARGET=$(BIN_DIR)/test1

# set opencl target version: 
OPENCL_TARGE_VERSION ?= 120


# need to set the corresponding path according to the local system
ifeq ($(OS),Windows_NT)
NVIDIA_CL_INCLUDE_DIR = /home/htcch/winsys/include/cuda
NVIDIA_CL_LIB_DIR = /home/htcch/winsys/lib
endif

#
CC=gcc -std=gnu99 -Wall -D_DEFAULT_SOURCE -D_GNU_SOURCE
LINKER = $(CC)

#
CFLAGS = -g -D_DEBUG -Iinclude -DCL_TARGET_OPENCL_VERSION=$(OPENCL_TARGE_VERSION)
LIBS = -lm -lpthread

ifneq (,$(NVIDIA_CL_INCLUDE_DIR))
CFLAGS += -I$(NVIDIA_CL_INCLUDE_DIR)
endif

ifneq (,$(NVIDIA_CL_LIB_DIR))
LIBS += -L/home/htcch/winsys/lib -lOpenCL
endif

#
ifeq ($(OS),Windows_NT)
CFLAGS += -DWIN32
endif

SRC_DIR=src
OBJ_DIR=obj
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: do_init $(TARGET)

bin/test1: $(OBJECTS)
	$(LINKER) -o $@ $^ $(CFLAGS) $(LIBS)
	
$(OBJECTS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -o $@ -c $< $(CFLAGS)
	
.PHONY: do_init clean
do_init:
	mkdir -p $(BIN_DIR) $(OBJ_DIR)

clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o
