/*
 * kmeans.h
 *
 *  Created on: Jul 23, 2016
 *      Author: hasib
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include <iomanip>
#include <cstdint>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <simd.h>
#include <time.h>
#ifdef __linux__
#include <omp.h>
#endif

using namespace std;


#define DEBUG_RESULT
#if 0
#define DEBUG_ERROR
#define DEBUG_DISTANCE
#define DEBUG_COUNTS
#define DEBUG_LABELS
#define DEBUG_MIN_DISTANCE
#endif

#ifdef __linux__
#include <papi.h>
double timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p);
#endif

//#define USE_LOOKUP
#define NUM_THREADS 2

#define DATASET_640

#ifdef DATASET_16
	#define DATA_SIZE 16 //16
	#define DIMENSION 4 //16
	#define CLUSTER_SIZE 8
#else
#ifdef DATASET_64
	#define DATA_SIZE 12
	#define DIMENSION 4
	#define CLUSTER_SIZE 8
#else // KDDCUP04Bio
	#define DATA_SIZE 1096	// 145744	(145751)
	#define DIMENSION 72		// 72		(74)
	#define CLUSTER_SIZE  256	// 2000		(2000)
#endif
#endif

#ifdef PROFILE_L1_CACHE

#define TOT_EVENTS 5
#define NATIVE_EVENTS 4
#define PAPI_EVENTS 1

#else

#define PAPI_EVENTS 4

#endif

#define KMEANS_T 0.01		// error tolerance

typedef int32_t data_t;
typedef int32_t simd_data_t;

void
load_initial_clusters(
		data_t * centroids,
		int num_clusters,
		const data_t* data);

void
load_simd_data(
		simd_data_t* simd_data,
		data_t* data,
		unsigned long N);

void
load_initial_simd_clusters(
		simd_data_t * centroids,
		int num_clusters,
		const data_t* data);

void
print_simd_reg(simd_int mm_reg);

void
print_simd_reg(simd_int mm_reg);

void
print_centroids(
		const data_t* __restrict__ centroids,
		int num_centroids);

void
print_simd_centroids(
		simd_data_t * centroids,
		int num_clusters);

void print_data(
		simd_data_t *data,
		int size);

unsigned long
loadData(string filename, data_t *data, int max_length);

void
get_clusterIndex(unsigned long ts_length, int num_clusters);

void
load_lookup(data_t *centroids, int num_clusters, int32_t lookup[][DIMENSION]);

void
load_simd_lookup(simd_data_t *simd_centroids, int num_clusters, int32_t *lookup);

simd_int horizontal_add(simd_int vec);

long add_simd_int(simd_int vec);

#ifdef __linux__
long long add_simd256_int(__m256i vec);
#endif

int
kmeans_scalar(
		const data_t * const ts_data,
		unsigned long ts_length,
		data_t* c,
		int num_clusters);

int
kmeans_simd_basic(
		const data_t * const ts_data,
		unsigned long ts_length,
		data_t* __restrict__ c,
		int num_clusters);

int
kmeans_simd_opt(
		const simd_data_t * const ts_data,
		unsigned long ts_length,
		simd_data_t* __restrict__ c,
		int num_clusters);

void warmup_cache();

#endif /* KMEANS_H_ */
