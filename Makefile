include make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE1 := $(CUDA_DIR)/include
INCLUDE2 := /usr/local/cuda/samples/common/inc

SRC = main.cu timespec_functions.cpp

EXE = main

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE1) -I$(INCLUDE2) -L$(CUDA_LIB_DIR) -lpthread

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE1) -I$(INCLUDE2) -L$(CUDA_LIB_DIR) -lpthread

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE1) -I$(INCLUDE2) -L$(CUDA_LIB_DIR) -lpthread

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE1) -I$(INCLUDE2) -L$(CUDA_LIB_DIR) -lpthread

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
