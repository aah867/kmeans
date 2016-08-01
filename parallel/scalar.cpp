#include <kmeans.h>

using namespace std;

#ifdef USE_LOOKUP
	static int32_t
	lookup[CLUSTER_SIZE][DIMENSION];
#endif

int
kmeans_scalar(
		const data_t * const ts_data,
		unsigned long ts_length,
		data_t* c,
		int num_clusters)
{
	long long old_error=0, error=0;
	int32_t *labels = (int32_t*) calloc(ts_length, sizeof(int32_t));
	int32_t *counts = (int32_t*) calloc(num_clusters, sizeof(int32_t));
	data_t *c1 = (data_t*) calloc(num_clusters*DIMENSION, sizeof(data_t));

	int32_t p_err;
	int ctr=0;

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
		load_lookup(c, num_clusters, lookup);
		#endif

		memset(labels, 0, (ts_length)*sizeof(int32_t));
		memset(counts, 0, num_clusters*sizeof(int32_t));
		memset(c1, 0, num_clusters*DIMENSION*sizeof(data_t));
		old_error = error, error = 0;

		#pragma omp parallel for
		for(int j=0; j<ts_length; j++)
		{
			int32_t min_distance = INT_MAX;
			for(int k=0; k<num_clusters; k++)
			{
				int32_t distance = 0;
				int a = 0;
				for(int l=0; l<DIMENSION; l++)
				{
					a = (ts_data[j*DIMENSION + l] * c[k*DIMENSION + l]);	// x_i*k_j
					#ifdef USE_LOOKUP
					distance += lookup[k][l] - a;
					#else
					int32_t b = (int32_t) ((c[k*DIMENSION + l] * c[k*DIMENSION + l])/2);	// k_j*k_j
					distance += (b - a);
					#endif
				}
				if(distance < min_distance)
				{
					labels[j] = k;
					min_distance = distance;
				}
			}
			error += min_distance;
			for(int l=0; l<DIMENSION; l++)
			{
				c1[labels[j]*DIMENSION + l] += ts_data[j*DIMENSION + l];;
			}
			counts[labels[j]]++;

			#ifdef DEBUG_MIN_DISTANCE
				cout << "i:" << j << "\t min_distance: " << min_distance << endl;
			#endif
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

	free(labels);
	free(counts);
	free(c1);

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
		cerr << "Error in loading data: "<< size << "<" << DATA_SIZE << endl;
		return 0;
	}

	get_clusterIndex(DATA_SIZE, CLUSTER_SIZE);
	data_t *centroids = (data_t*) malloc(sizeof(data_t)*CLUSTER_SIZE*DIMENSION);
	load_initial_clusters(centroids, CLUSTER_SIZE, data);

	#ifdef USE_LOOKUP
	load_lookup(centroids, CLUSTER_SIZE, lookup);
	#endif

	int ctr = kmeans_scalar(data, DATA_SIZE, centroids, CLUSTER_SIZE);

	#ifdef DEBUG_RESULT
	cout << "SCALAR RESULT:" << ctr << endl;
	print_centroids(centroids, CLUSTER_SIZE);
	#else
	(void) ctr;
	#endif

	free(data);
	free(centroids);

	return 0;
}
