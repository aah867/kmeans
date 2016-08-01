#!/bin/bash

make clean
make

./scalar | tee log_scalar.txt
./simd_opt | tee log_simd_opt.txt
./simd_basic | tee log_simd_basic.txt

echo "******** SCALAR-SIMD_OPT *****"
diff log_scalar.txt log_simd_opt.txt

echo "******** SCALAR-SIMD_SCALE *****"
diff log_scalar.txt log_simd_basic.txt
