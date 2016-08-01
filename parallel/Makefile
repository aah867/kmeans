###########################################
# Makefile fo Kmeans by Abdullah Al Hasib #
###########################################

platform=$(shell uname -s)

ifeq ($(platform),Darwin)
    CC=g++
    CFLAGS = -g -O0 -Wall -msse4.1 -std=c++0x -DDEBUG
    INCLUDES = -I. -I../include
else
    CC = icpc
    
#   for debugging
#   CFLAGS = -g -Wall -O0 -msse3 -std=c++0x -fopenmp -DDEBUG
#   without vectorization    
    CFLAGS = -O3 -no-vec -std=c++0x -fopenmp -DPROFILE_L1_CACHE
#   with auto-vectorization
#   CFLAGS = -O3 -msse3 -std=c++0x -fopenmp 

    INCLUDES = -I. -I../include -I/usr/include -I/usr/local/include -I${HOME}/include -I/usr/include/x86_64-linux-gnu/c++/4.8
    LIBS = -L/usr/lib -L/usr/local/lib -L${HOME}/lib
endif

SCALAR_SRC = support.cpp scalar.cpp
SCALAR_OBJ = $(SCALAR_SRC:.cpp=.o)
SCALAR_EXEC = scalar 

SIMD_BASIC_SRC = support.cpp simd_basic.cpp
SIMD_BASIC_OBJ = $(SIMD_BASIC_SRC:.cpp=.o)
SIMD_BASIC_EXEC = simd_basic

SIMD_OPT_SRC = support.cpp simd_opt.cpp
SIMD_OPT_OBJ = $(SIMD_OPT_SRC:.cpp=.o)
SIMD_OPT_EXEC = simd_opt

#$(CC) -o $(EXEC) $(OBJ) $(LIBS) -lenergy

default: $(SCALAR_EXEC) $(SIMD_OPT_EXEC) $(SIMD_BASIC_EXEC)

.cpp.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(SCALAR_EXEC):$(SCALAR_OBJ)
	$(CC) $(CFLAGS) -o $(SCALAR_EXEC) $(SCALAR_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a

$(SIMD_BASIC_EXEC):$(SIMD_BASIC_OBJ)
	$(CC) $(CFLAGS) -o $(SIMD_BASIC_EXEC) $(SIMD_BASIC_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a

$(SIMD_OPT_EXEC):$(SIMD_OPT_OBJ)
	$(CC) $(CFLAGS) -o $(SIMD_OPT_EXEC) $(SIMD_OPT_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a

run:
	make clean
	make

clean:
	rm -f *.o $(SCALAR_EXEC)  $(SIMD_BASIC_EXEC) $(SIMD_OPT_EXEC)
