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

	static int32_t
	simd_lookup[CLUSTER_SIZE*DIMENSION];
#endif

#ifdef DEBUG_MIN_DISTANCE
int32_t tmp_min_distance[DATA_SIZE];
#endif

int
kmeans_simd_opt(
		const simd_data_t * const __restrict__ ts_data,
		unsigned long const ts_length,
		simd_data_t* __restrict__ c,
		int const num_clusters)
{
	int ctr=0;
	int32_t p_err;
	long long old_error=0, error=0;
	int32_t labels[ts_length];
	int32_t * const counts = (int32_t*) _mm_malloc(num_clusters*sizeof(int32_t), ALIGN);
	simd_data_t * const c1 = (simd_data_t*) _mm_malloc((num_clusters*DIMENSION)*sizeof(simd_data_t), ALIGN);

#ifdef __linux__

	#ifdef PROFILE_L1_CACHE

		char event_name[NATIVE_EVENTS][BUFSIZ] =
		{
			"perf::PERF_COUNT_HW_CACHE_L1D:MISS",
			"perf::PERF_COUNT_HW_CACHE_L1D:ACCESS",
			"perf::PERF_COUNT_HW_CACHE_L1D:PREFETCH",
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

//		#ifdef PROFILE_L3_CACHE

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

	#endif

	struct timespec start, end;     // Wall clock time
	double timeElapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);

	#endif


	do
	{
		#ifdef USE_LOOKUP
		load_simd_lookup(c, num_clusters, simd_lookup);
		#endif
		memset(labels, 0, (ts_length)*sizeof(int32_t));
		memset(counts, 0, num_clusters*sizeof(int32_t));
		memset(c1, 0, (num_clusters*DIMENSION)*sizeof(simd_data_t));
		old_error = error, error = 0;

		#ifdef __linux__
		__m256i mm256_error = _mm256_setzero_si256();
		#else
		simd_int mm_error = simd_set_zero();
		#endif

		int i;
		#pragma omp parallel for
		for(i=0; i<ts_length; i+=VECSIZE32)
		{
			simd_int mm_min_distance = simd_set1_i(INT_MAX);
			simd_int mm_minlabels = _mm_setr_epi32(0,1,2,3);

			for(int j=0; j<num_clusters; j+=VECSIZE32)
			{
				simd_int mm_d1 = simd_set_zero();
				simd_int mm_d2 = simd_set_zero();
				simd_int mm_d3 = simd_set_zero();
				simd_int mm_d4 = simd_set_zero();
				simd_int mm_l1 = _mm_setr_epi32(j, j+1, j+2, j+3);
				
				for(int k=0; k<DIMENSION; k++)
				{
					simd_int mm_data = simd_loadu_i((simd_int const*) &ts_data[i*DIMENSION+k*VECSIZE32]);

					simd_int mm_c = simd_loadu_i((simd_int const*) &c[j*DIMENSION+k*VECSIZE32]);
					#ifdef USE_LOOKUP
					simd_int mm_u = simd_loadu_i((simd_int const*) &simd_lookup[j*DIMENSION+k*VECSIZE32]);
					#else
					simd_int mm_u = _mm_srli_epi32(simd_mul_lo(mm_c, mm_c), 0x01);
					#endif
					mm_d1 = simd_add_i(mm_d1, simd_sub_i(mm_u, simd_mul_lo(mm_data, mm_c)));

					mm_c = _mm_shuffle_epi32(mm_c, 0x39);
					mm_u = _mm_shuffle_epi32(mm_u, 0x39);
					mm_d2 = simd_add_i(mm_d2, simd_sub_i(mm_u, simd_mul_lo(mm_data, mm_c)));

					mm_c = _mm_shuffle_epi32(mm_c, 0x39);
					mm_u = _mm_shuffle_epi32(mm_u, 0x39);
					mm_d3 = simd_add_i(mm_d3, simd_sub_i(mm_u, simd_mul_lo(mm_data, mm_c)));

					mm_c = _mm_shuffle_epi32(mm_c, 0x39);
					mm_u = _mm_shuffle_epi32(mm_u, 0x39);
					mm_d4 = simd_add_i(mm_d4, simd_sub_i(mm_u, simd_mul_lo(mm_data, mm_c)));

					//_mm_clevict((void const*) &ts_data[i*DIMENSION+k*VECSIZE32], 0);
				}

				#ifdef DEBUG_DISTANCE
				int32_t buf[4];
				simd_storeu_i((simd_int*) buf, mm_d1);
				cout << "Cluster: " << j << " :" << buf[2] << endl;
				simd_storeu_i((simd_int*) buf, mm_d2);
				cout << "Cluster: " << j << " :" << buf[2] << endl;
				simd_storeu_i((simd_int*) buf, mm_d3);
				cout << "Cluster: " << j << " :" << buf[2] << endl;
				simd_storeu_i((simd_int*) buf, mm_d4);
				cout << "Cluster: " << j << " :" << buf[2] << endl;
				#endif
				simd_int mm_l2 = _mm_shuffle_epi32(mm_l1, 0x39);
				simd_int mm_l3 = _mm_shuffle_epi32(mm_l2, 0x39);
				simd_int mm_l4 = _mm_shuffle_epi32(mm_l3, 0x39);

				simd_int mm_mask = _mm_cmplt_epi32(mm_d2, mm_d1);
				mm_d2 = _mm_or_si128(_mm_and_si128(mm_mask, mm_d2), _mm_andnot_si128(mm_mask, mm_d1));
				mm_l2 = _mm_or_si128(_mm_and_si128(mm_mask, mm_l2), _mm_andnot_si128(mm_mask, mm_l1));

				mm_mask = _mm_cmplt_epi32(mm_d4, mm_d3);
				mm_d4 = _mm_or_si128(_mm_and_si128(mm_mask, mm_d4), _mm_andnot_si128(mm_mask, mm_d3));
				mm_l4 = _mm_or_si128(_mm_and_si128(mm_mask, mm_l4), _mm_andnot_si128(mm_mask, mm_l3));

				mm_mask = _mm_cmplt_epi32(mm_d4, mm_d2);
				mm_d4 = _mm_or_si128(_mm_and_si128(mm_mask, mm_d4), _mm_andnot_si128(mm_mask, mm_d2));
				mm_l4 = _mm_or_si128(_mm_and_si128(mm_mask, mm_l4), _mm_andnot_si128(mm_mask, mm_l2));

				mm_mask = _mm_cmplt_epi32(mm_d4, mm_min_distance);
				mm_min_distance = _mm_or_si128(_mm_and_si128(mm_mask, mm_d4), _mm_andnot_si128(mm_mask, mm_min_distance));
				mm_minlabels = _mm_or_si128(_mm_and_si128(mm_mask, mm_l4), _mm_andnot_si128(mm_mask, mm_minlabels));

				_mm_prefetch((const char *) &ts_data[(i+1)*DIMENSION], _MM_HINT_NTA);
			}

			#pragma omp critical
			{
				#ifdef __linux__
				mm256_error = _mm256_add_epi64(mm256_error, _mm256_cvtepi32_epi64(mm_min_distance));
				#else
				mm_error = simd_add_i(mm_error, mm_min_distance);
				#endif
				simd_storeu_i((simd_int*) &labels[i], mm_minlabels);
			}

			#ifdef DEBUG_MIN_DISTANCE
			int32_t buf[4];
			simd_storeu_i((simd_int*) buf, mm_min_distance);
			for(int pp=0; pp<4; pp++)
				tmp_min_distance[i+pp] = buf[pp];
			#endif
		}
		
		for(int i=0; i<ts_length; i+=VECSIZE32)
		{
			for(int ll=0; ll <VECSIZE32; ll++)
			{
				counts[labels[i+ll]]++;
				int B=labels[i+ll]/VECSIZE32;
				int C=labels[i+ll]%VECSIZE32;

				for(int kk=0; kk<DIMENSION; kk++)
				{
					c1[B*VECSIZE32*DIMENSION + kk*VECSIZE32 + C] += ts_data[i*DIMENSION + kk*VECSIZE32 + ll];
				}
			}
		}

		int limit = num_clusters/VECSIZE32;
		for(int pp=0; pp<limit; pp++)
		{
			for(int ll=0, B=pp*VECSIZE32; ll <VECSIZE32; ll++)
			{
				if(counts[B+ll] > 0)
					for(int kk=0, C=pp*DIMENSION*VECSIZE32+ll; kk<DIMENSION; kk++)
						c[C + kk*VECSIZE32] = c1[C + kk*VECSIZE32]/counts[B+ll];
				else
					for(int kk=0, C=pp*DIMENSION*VECSIZE32+ll; kk<DIMENSION; kk++)
						c[C + kk*VECSIZE32] = c1[C + kk*VECSIZE32];
			}
		}

		#ifdef DEBUG_COUNTS
		for(int i=0; i<num_clusters; i++)
		{
			cout << " counts: (" <<  i << "):\t" << counts[i] << endl;
		}
		cout << endl;
		#endif

		#ifdef DEBUG_LABELS
		for(int i=0; i<ts_length; i++)
		{
			cout << " labels: (" <<  i << "):\t" << labels[i] << endl;
		}
		cout << endl;
		#endif

		#ifdef DEBUG_MIN_DISTANCE
		for(int i=0; i<ts_length; i++)
		{
			cout << "i:" << i << "\t min_distance: " << tmp_min_distance[i] << endl;
		}
		cout << endl;
		#endif

		#ifdef __linux__
		error = add_simd256_int(mm256_error);
		#else
		error = add_simd_int(mm_error);
		#endif
		
		p_err = ((error-old_error)*100)/error;
		if(p_err < 0)
			p_err = p_err *(-1);

		#ifdef DEBUG_ERROR
		cout << "Error:" << "Loop:" << ctr << "\t p_err:: " << p_err << "% \t(" << old_error << "\t:" << error << "): " << error-old_error << endl;
		#endif

		ctr++;

	} while(p_err >= KMEANS_T);

	#ifdef __linux__

	clock_gettime(CLOCK_MONOTONIC, &end);
	timeElapsed = timespecDiff(&end, &start);

	#ifdef PROFILE_L1_CACHE

		if(PAPI_stop(EventSet, values) != PAPI_OK)
			fprintf(stderr, "PAPI error in stop\n");

		cout.precision(2);
		cout << "L1:" << fixed << setw(15) << timeElapsed << setw(10) << ((double) values[0]*100)/values[1] << setw(10) << values[0] << setw(10) << values[1] << setw(10) << values[2] << setw(10) << values[3] << setw(10) << values[4];
		cout << endl;

	#else

		if(PAPI_stop_counters(values, PAPI_EVENTS) != PAPI_OK)
			fprintf(stderr, "PAPI error in stop\n");
		cout.precision(2);
		cout << "L2:" << fixed << setw(15) << timeElapsed << setw(8) << ((double)values[0]*100)/values[1] << setw(8) << ((double)values[2]*100)/values[3] << setw(15) << values[0] << setw(15) << values[1];
		cout<< setw(15) << values[2] << setw(15) <<  values[3];
		cout << endl;

	#endif

	#endif

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

	simd_data_t *simd_data = (simd_data_t*) _mm_malloc((DATA_SIZE*DIMENSION)*sizeof(simd_data_t), ALIGN);
	load_simd_data(simd_data, data, DATA_SIZE);

	simd_data_t *simd_centroids = (simd_data_t*) _mm_malloc((CLUSTER_SIZE*DIMENSION)*sizeof(simd_data_t), ALIGN);
	load_initial_simd_clusters(simd_centroids, CLUSTER_SIZE, data);
	#ifdef USE_LOOKUP
	load_simd_lookup(simd_centroids, CLUSTER_SIZE, simd_lookup);
	#endif

	int ctr = kmeans_simd_opt(simd_data, DATA_SIZE, simd_centroids, CLUSTER_SIZE);

	#ifdef DEBUG_RESULT
	cout << "SIMD RESULT:" << ctr << endl;
	print_simd_centroids(simd_centroids, CLUSTER_SIZE);
	#else
	(void) ctr;
	#endif

	_mm_free(simd_centroids);
	_mm_free(simd_data);
	free(data);
	free(centroids);

	return 0;
}
