BIN_DIR=bin
TARGETS=$(BIN_DIR)/test1 $(BIN_DIR)/single-process-multi-tasks

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
LIBS = -lm -lpthread -ljson-c

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

BASE_SRC_DIR=base
BASE_OBJ_DIR=obj/base
BASE_SOURCES := $(wildcard $(BASE_SRC_DIR)/*.c)
BASE_OBJECTS := $(BASE_SOURCES:$(BASE_SRC_DIR)/%.c=$(BASE_OBJ_DIR)/%.o)

UTILS_SRC_DIR=utils
UTILS_OBJ_DIR=obj/utils
UTILS_SOURCES := $(wildcard $(UTILS_SRC_DIR)/*.c)
UTILS_OBJECTS := $(UTILS_SOURCES:$(UTILS_SRC_DIR)/%.c=$(UTILS_OBJ_DIR)/%.o)

all: do_init $(TARGETS)

$(BIN_DIR)/test1: $(OBJ_DIR)/test1.o $(BASE_OBJECTS) $(UTILS_OBJECTS)
	$(LINKER) -o $@ $^ $(CFLAGS) $(LIBS)

$(BIN_DIR)/single-process-multi-tasks: $(OBJ_DIR)/single-process-multi-tasks.o $(BASE_OBJECTS) $(UTILS_OBJECTS)
	$(LINKER) -o $@ $^ $(CFLAGS) $(LIBS)

$(OBJECTS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -o $@ -c $< $(CFLAGS)
	
$(BASE_OBJECTS): $(BASE_OBJ_DIR)/%.o : $(BASE_SRC_DIR)/%.c
	$(CC) -o $@ -c $< $(CFLAGS)
	
$(UTILS_OBJECTS): $(UTILS_OBJ_DIR)/%.o : $(UTILS_SRC_DIR)/%.c
	$(CC) -o $@ -c $< $(CFLAGS)
	
.PHONY: do_init clean
do_init:
	mkdir -p $(BIN_DIR) $(OBJ_DIR) $(BASE_OBJ_DIR) $(UTILS_OBJ_DIR)

clean:
	rm -f $(TARGETS) $(OBJ_DIR)/*.o $(BASE_OBJ_DIR)/*.o $(UTILS_OBJ_DIR)/*.o
