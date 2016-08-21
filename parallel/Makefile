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
#    CFLAGS = -g -O0 -std=c++0x -fopenmp -DPROFILE_L1_CACHE
#   without vectorization    
    CFLAGS = -O2 -no-vec -std=c++0x -fopenmp -DPROFILE_L1_CACHE
    AUTO_VEC_CFLAGS = -O2 -msse2 -std=c++0x -fopenmp -DPROFILE_L1_CACHE
#    CFLAGS = -O3 -std=c++0x -fopenmp -DPROFILE_L2_CACHE
#   with auto-vectorization
#   CFLAGS = -O3 -msse2 -std=c++0x -fopenmp 

    INCLUDES = -I. -I../include -I/usr/include -I/usr/local/include -I${HOME}/include -I/usr/include/x86_64-linux-gnu/c++/4.8
    LIBS = -L/usr/lib -L/usr/local/lib -L${HOME}/lib
endif

SCALAR_SRC = support.cpp scalar.cpp
SCALAR_OBJ = $(SCALAR_SRC:.cpp=.o)
SCALAR_EXEC = scalar 

AUTO_VEC_SRC = support.cpp scalar.cpp
AUTO_VEC_OBJ = $(SCALAR_SRC:.cpp=.o)
AUTO_VEC_EXEC = auto_vec 

SIMD_BASIC_SRC = support.cpp simd_basic.cpp
SIMD_BASIC_OBJ = $(SIMD_BASIC_SRC:.cpp=.o)
SIMD_BASIC_EXEC = simd_basic

SIMD_OPT_SRC = support.cpp simd_opt.cpp
SIMD_OPT_OBJ = $(SIMD_OPT_SRC:.cpp=.o)
SIMD_OPT_EXEC = simd_opt

SIMD_CMP_SRC = support.cpp simd_cmp.cpp
SIMD_CMP_OBJ = $(SIMD_CMP_SRC:.cpp=.o)
SIMD_CMP_EXEC = simd_cmp

default: $(SCALAR_EXEC) $(SIMD_OPT_EXEC) $(SIMD_BASIC_EXEC) $(SIMD_CMP_EXEC) $(AUTO_VEC_EXEC)

.cpp.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(SCALAR_EXEC):$(SCALAR_OBJ)
	$(CC) $(CFLAGS) -o $(SCALAR_EXEC) $(SCALAR_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a -lenergy

$(SIMD_BASIC_EXEC):$(SIMD_BASIC_OBJ)
	$(CC) $(CFLAGS) -o $(SIMD_BASIC_EXEC) $(SIMD_BASIC_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a -lenergy

$(SIMD_OPT_EXEC):$(SIMD_OPT_OBJ)
	$(CC) $(CFLAGS) -o $(SIMD_OPT_EXEC) $(SIMD_OPT_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a -lenergy

$(SIMD_CMP_EXEC):$(SIMD_CMP_OBJ)
	$(CC) $(CFLAGS) -o $(SIMD_CMP_EXEC) $(SIMD_CMP_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a -lenergy

$(AUTO_VEC_EXEC):$(AUTO_VEC_OBJ)
	rm -f scalar.o support.o auto_vec
	$(CC) $(INCLUDES) $(AUTO_VEC_CFLAGS) -c $(AUTO_VEC_SRC)
	$(CC) $(AUTO_VEC_CFLAGS) -o $(AUTO_VEC_EXEC) $(AUTO_VEC_OBJ) $(LIBS) -lrt /home/abdullah/lib/libpapi.a -lenergy

run:
	make clean
	make

clean:
	rm -f *.o $(SCALAR_EXEC)  $(SIMD_BASIC_EXEC) $(SIMD_OPT_EXEC) $(SIMD_CMP_EXEC) $(AUTO_VEC_EXEC)
