#!/bin/bash

MAX=20
freq=( 800 1200 1600 2100 2500 2900 3500 )
filename=output_scalar
threads=( 1 2 4 6 8 )

#MAX=2
#freq=( 800 1200 )
#filename=output_scalar
#threads=( 1 2 )

for t in ${threads[@]};
do
    export OMP_NUM_THREADS=$t
    sleep 1

    echo -n > "output_scalar_$t.txt"
    echo -n > "output_simd_basic_$t.txt"
    echo -n > "output_simd_opt_$t.txt"

    for i in ${freq[@]};
    do
        /home/abdullah/scripts/set_cpufreq $i
      #  sleep 2

        echo -e "\nFrequency ${i}" >> "output_scalar_$t.txt"
        for j in `seq 1 $MAX`; 
        do
            ./scalar >> output_scalar_$t.txt
        done
     #   sleep 2

        echo -e "\nFrequency ${i}" >> "output_simd_basic_$t.txt"
        for j in `seq 1 $MAX`; 
        do
            ./simd_basic >> output_simd_basic_$t.txt
        done
       # sleep 2

        echo -e "\nFrequency $i" >> "output_simd_opt_$t.txt"
        for j in `seq 1 $MAX`; 
        do
            ./simd_opt >> output_simd_opt_$t.txt
        done
        #sleep 2
    done
done
