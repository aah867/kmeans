/*
 * simd_multithreaded.cpp
 *
 *  Created on: Jul 23, 2016
 *      Author: hasib
 */

#include <kmeans.h>

using namespace std;

#ifdef USE_LOOKUP
	static int32_t
	lookup[CLUSTER_SIZE][DIMENSION];
#endif

#ifdef DEBUG_MIN_DISTANCE
int32_t tmp_min_distance[DATA_SIZE];
#endif

int
kmeans_simd_basic(
		const data_t * const ts_data,
		unsigned long ts_length,
		data_t* __restrict__ c,
		int num_clusters)
{
	int ctr=0;
	int32_t p_err;
	long long old_error=0, error=0;
	int32_t *labels = (int32_t *) _mm_malloc(ts_length*sizeof(int32_t), ALIGN);
	int32_t *counts = (int32_t*) _mm_malloc(num_clusters*sizeof(int32_t), ALIGN);
	data_t *c1 = (data_t*) _mm_malloc(num_clusters*DIMENSION*sizeof(data_t), ALIGN);

#ifdef __linux__

	#ifdef PROFILE_L1_CACHE

		char event_name[NATIVE_EVENTS][BUFSIZ] =
		{
			"perf::PERF_COUNT_HW_CACHE_L1D:MISS",
			"perf::PERF_COUNT_HW_CACHE_L1D:ACCESS",
			"perf::PERF_COUNT_HW_CACHE_L1D:PREFETCH",
			//"perf::PERF_COUNT_HW_CACHE_LL:PREFETCH",
			"perf::PERF_COUNT_HW_CACHE_L1D:WRITE",
		};

		long long values[TOT_EVENTS] = {0};
		if (PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT))
			cout << "PAPI_library_init error.";

		int EventSet = PAPI_NULL;
		int native[TOT_EVENTS];

		if ( PAPI_create_eventset(&EventSet) != PAPI_OK )
		{
			cout << "Error in creating eventset !" << endl;
		}

		for(int i=0; i<NATIVE_EVENTS; i++)
		{
			if( PAPI_event_name_to_code(event_name[i], &native[i]) != PAPI_OK )
				cout << "Error in NAME to CODE" << endl;
		}

		native[NATIVE_EVENTS] = PAPI_TOT_INS;
		if( PAPI_add_events(EventSet, native, TOT_EVENTS) != PAPI_OK )
		{
			cout << "Error in Add events !" << endl;
		}

		warmup_cache();

		if ( PAPI_start(EventSet) != PAPI_OK )
		{
			cout << "PAPI error in start" << endl;
		}

	#else

		#ifdef PROFILE_L2_CACHE

			int events[PAPI_EVENTS] = {PAPI_L2_TCM, PAPI_L2_TCA, PAPI_L3_TCM, PAPI_L3_TCA};
			long long values[PAPI_EVENTS] = {0};

			PAPI_library_init(PAPI_VER_CURRENT);
			if (PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT))
				cout <<  "PAPI_library_init error.";

			warmup_cache();

			if ( PAPI_start_counters(events, PAPI_EVENTS) != PAPI_OK )
			{
				cout << "Error in creating eventset" << endl;
			}
		#else
			#ifdef PROFILE_ENERGY
				energy_counter *e = new energy_counter;
				FILE* outputf = fopen("profiles.dat", "w+");
				if (!outputf) fprintf(stderr, "Cannot open output file.\n");
				initialize(e, outputf);
				warmup_cache();
				int d = start_reading(e);

			#else
				char event_name[NATIVE_EVENTS][BUFSIZ] =
				{
					"CYCLE_ACTIVITY:STALLS_LDM_PENDING",
					//"perf::PERF_COUNT_HW_CACHE_LL:PREFETCH",
				};

				long long values[TOT_EVENTS] = {0};
				if (PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT))
					cout << "PAPI_library_init error.";

				int EventSet = PAPI_NULL;
				int native[TOT_EVENTS];

				if ( PAPI_create_eventset(&EventSet) != PAPI_OK )
				{
					cout << "Error in creating eventset !" << endl;
				}

				for(int i=0; i<NATIVE_EVENTS; i++)
				{
					if( PAPI_event_name_to_code(event_name[i], &native[i]) != PAPI_OK )
						cout << "Error in NAME to CODE" << endl;
				}

				native[NATIVE_EVENTS] = PAPI_RES_STL;
				native[NATIVE_EVENTS+1] = PAPI_TOT_CYC;

				if( PAPI_add_events(EventSet, native, TOT_EVENTS) != PAPI_OK )
				{
					cout << "Error in Add events !" << endl;
				}

				warmup_cache();

				if ( PAPI_start(EventSet) != PAPI_OK )
				{
					cout << "PAPI error in start" << endl;
				}
			#endif
		#endif
	#endif

	struct timespec start, end;     // Wall clock time
	double timeElapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);

#endif

	do
	{
		#ifdef USE_LOOKUP
		load_lookup(c, num_clusters, lookup);
		#endif
		memset(labels, 0, (ts_length)*sizeof(int32_t));
		memset(counts, 0, num_clusters*sizeof(int32_t));
		memset(c1, 0, num_clusters*DIMENSION*sizeof(data_t));
		old_error = error, error = 0;

#if 0
		#ifdef __linux__
		__m256i mm256_error = _mm256_setzero_si256();
		#else
		simd_int mm_error = simd_set_zero();
		#endif
#endif

		#pragma omp parallel for reduction(+:error)
		for(int j=0; j<ts_length; j++)
		{
			simd_int mm_min_distance = simd_set1_i(INT_MAX);
			int min_distance = INT_MAX;

			for(int k=0; k<num_clusters; k++)
			{
				simd_int mm_u;
				simd_int mm_distance = simd_set_zero();

				for(int l=0; l<DIMENSION; l = l+VECSIZE32)
				{
					simd_int mm_data = simd_loadu_i((simd_int const*) &ts_data[j*DIMENSION+l]);
					simd_int mm_c = simd_loadu_i((simd_int const*) &c[k*DIMENSION+l]);
					#ifdef USE_LOOKUP
					mm_u = simd_loadu_i((simd_int const*) &lookup[k][l]);
					#else
					mm_u = _mm_srli_epi32(simd_mul_lo(mm_c, mm_c), 0x01);
					#endif
					mm_distance = simd_add_i(mm_distance, simd_sub_i(mm_u, simd_mul_lo(mm_data, mm_c)));
				}

				int32_t distance = simd_horizontal_add(mm_distance);
				if(distance < min_distance)
				{
					min_distance = distance;
					labels[j] = k;
				}
			}

			error += min_distance;

			#ifdef DEBUG_MIN_DISTANCE
			cout << "i:" << j << "\t min_distance: " << min_distance << endl;
			#endif
		}

		for(int j=0; j<ts_length; j++)
		{
			for(int l=0; l<DIMENSION; l++)
			{
				c1[labels[j]*DIMENSION + l] += ts_data[j*DIMENSION + l];
			}
			counts[labels[j]]++;
		}

		for(int i=0; i<num_clusters; i++)
		{
			if (counts[i] > 0)
				for(int l=0; l<DIMENSION; l++)
					c[i*DIMENSION + l] = (int)(c1[i*DIMENSION + l]/counts[i]);
			else
				for(int l=0; l<DIMENSION; l++)
					c[i*DIMENSION + l] = c1[i*DIMENSION + l];
		}

		p_err = ((error-old_error)*100)/error;
		if(p_err < 0)
			p_err = p_err *(-1);

		#ifdef DEBUG_ERROR
		cout << "Error:" << "Loop:" << ctr << "\t p_err:: " << p_err << "% \t(" << old_error << "\t:" << error << "): " << error-old_error << endl;
		#endif

		ctr++;

	} while (p_err >= KMEANS_T);

#ifdef __linux__

clock_gettime(CLOCK_MONOTONIC, &end);
timeElapsed = timespecDiff(&end, &start);

#ifdef PROFILE_L1_CACHE
	if(PAPI_stop(EventSet, values) != PAPI_OK)
		fprintf(stderr, "PAPI error in stop\n");

	cout.precision(2);
	cout << "L1:" << fixed << setw(15) << timeElapsed << setw(10) << ((double) values[0]*100)/values[1] << setw(15) << values[0] << setw(15) << values[1] << setw(15) << values[2] << setw(15) << values[3] << setw(20) << values[4];
	cout << endl;
#else
	#ifdef PROFILE_L2_CACHE
		if(PAPI_stop_counters(values, PAPI_EVENTS) != PAPI_OK)
			fprintf(stderr, "PAPI error in stop\n");
		cout.precision(2);
		cout << "L2:" << fixed << setw(15) << timeElapsed << setw(8) << ((double)values[0]*100)/values[1] << setw(8) << ((double)values[2]*100)/values[3] << setw(15) << values[0] << setw(15) << values[1];
		cout<< setw(15) << values[2] << setw(25) <<  values[3];
		cout << endl;
	#else
		#ifdef PROFILE_ENERGY
			d = stop_reading(e);
			estimate_energy(e);
			cout.precision(2);
			cout << "E:" << fixed << setw(15) << timeElapsed << setw(15) << e->core_energy << setw(15) << e->package_energy;
			cout << endl;
			finalize(e);
		#else
			if(PAPI_stop(EventSet, values) != PAPI_OK)
				fprintf(stderr, "PAPI error in stop\n");
			cout.precision(2);
			cout << "I:" << fixed << setw(15) << timeElapsed << setw(15) << values[0] << setw(15) << values[1] << setw(20) << values[2];
			cout << endl;
		#endif
	#endif
#endif
#endif

	_mm_free(labels);
	_mm_free(counts);
	_mm_free(c1);

	return ctr;
}

int main()
{
	string filename;

	#ifdef DATASET_16
	filename = "../dataset/dim016.txt";
	#else
	#ifdef DATASET_64
	filename = "../dataset/dim064.txt";
	#else
	filename = "../dataset/KDDCUP04Bio.txt";
	#endif
	#endif

	data_t *data = (data_t*) malloc(sizeof(data_t)*DATA_SIZE*DIMENSION);
	unsigned long size = loadData(filename, data, DATA_SIZE);
	if(size < DATA_SIZE)
	{
		cerr << "Error in loading data." << endl;
		return 0;
	}

	get_clusterIndex(DATA_SIZE, CLUSTER_SIZE);
	data_t *centroids = (data_t*) malloc(sizeof(data_t)*CLUSTER_SIZE*DIMENSION);
	load_initial_clusters(centroids, CLUSTER_SIZE, data);

	#ifdef USE_LOOKUP
	load_lookup(centroids, CLUSTER_SIZE, lookup);
	#endif

	int ctr = kmeans_simd_basic(data, DATA_SIZE, centroids, CLUSTER_SIZE);

	#ifdef DEBUG_RESULT
	cout << "SIMD BASIC RESULT:" << ctr << endl;
	print_centroids(centroids, CLUSTER_SIZE);
	#else
	(void) ctr;
	#endif

	free(data);
	free(centroids);

	return 0;
}
