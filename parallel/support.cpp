#include <kmeans.h>

using namespace std;

unsigned int cluster_index[CLUSTER_SIZE];

#ifdef __linux__
double timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
        return ((timeA_p->tv_sec * 1000000) + timeA_p->tv_nsec/(double)1000) -
                        ((timeB_p->tv_sec * 1000000) + timeB_p->tv_nsec/(double)1000);
}
#endif

void
print_simd_reg(simd_int mm_reg)
{
	int buffer[VECSIZE32];
	simd_storeu_i((simd_int*) buffer, mm_reg);
	for(int i=0; i<VECSIZE32; i++)
	{
		cout << buffer[i] << " ";
	}
	cout << endl;
}

void
print_centroids(
		const data_t* __restrict__ centroids,
		int num_centroids)
{
	printf("Scalar centroids:\n");
	for(int i=0; i<num_centroids; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			cout << centroids[i*DIMENSION+j] << " ";
		}
		cout << endl;
	}
}

void
print_simd_centroids(
		simd_data_t * centroids,
		int num_clusters
		)
{
	printf("SIMD centroids:\n");
#if 0
	for(int i=0; i<num_clusters; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			cout << centroids[i*DIMENSION+j] << " ";
		}
		cout << endl;
	}
#else
	for(int i=0; i<num_clusters; i+=VECSIZE32)
	{
		for(int ll=0; ll <VECSIZE32; ll++)
		{
			for(int kk=0; kk<DIMENSION; kk++)
			{
				cout << centroids[i*DIMENSION + kk*VECSIZE32 + ll] << " ";
			}
			cout << endl;
		}
	}
#endif

}

void print_data(
		simd_data_t *data,
		int size)
{
	for(int i=0; i<size; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			cout << data[i*DIMENSION+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

unsigned long
loadData(string filename, data_t *data, int max_length)
{
	ifstream myfile (filename);
	if ( !myfile.good() )
	{
		cerr << "Cannot find file." << endl;
		return false;
	}

	string line;
	unsigned long ts_length=0;
	while ( !myfile.eof())
	{
		double point;

		if(ts_length >= max_length)
			break;

		getline (myfile,line);
		stringstream ss(line);
		for(int i=0; i<DIMENSION; i++)
		{
			ss >> point;
			data[ts_length*DIMENSION+i] = (int32_t)point;
//			cout <<ts_length+1 << ":" << data[ts_length*DIMENSION+i] << "\t";
		}
//		cout << endl;
		ts_length++;
	}

	myfile.close();
	return ts_length;
}

void
get_clusterIndex(unsigned long ts_length, int num_clusters)
{
#if 1
	//cout << "Cluster index:" << endl;
	for(int j=0; j<num_clusters; j++)
	{
		cluster_index[j] = rand()%ts_length;
		//cout << cluster_index[j] << " ";
	}
	//cout << endl;
#else
	cluster_index[0] = 7;
	cluster_index[1] = 1;
	cluster_index[2] = 9;
	cluster_index[3] = 12;
	cluster_index[4] = 3;
	cluster_index[5] = 6;
	cluster_index[6] = 14;
	cluster_index[7] = 15;
#endif
}

void
load_lookup(data_t *centroids, int num_clusters, int32_t lookup[][DIMENSION])
{
	//cout << "\n Lookup Table is building:" << endl;
	for(int i=0; i<num_clusters; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			lookup[i][j] = (centroids[i*DIMENSION+j] * centroids[i*DIMENSION+j]) >> 1;
			//cout << lookup[i][j] << " ";
		}
		//cout << endl;
	}
}

void
load_simd_lookup(simd_data_t *simd_centroids, int num_clusters, int32_t *simd_lookup)
{
	for(int i=0; i<num_clusters; i++)
	{
		for(int j=0; j<DIMENSION; j+=VECSIZE32)
		{
			simd_int mm_c = simd_loadu_i((simd_int const*) &simd_centroids[i*DIMENSION+j]);
			mm_c = _mm_srli_epi32(simd_mul_lo(mm_c, mm_c), 0x01);
			simd_storeu_i((simd_int*) &simd_lookup[i*DIMENSION+j], mm_c);
		}
	}
}

simd_int horizontal_add(simd_int vec)
{
    simd_int temp;

    temp = vec;
    temp = _mm_shuffle_epi32(temp, 0x39);
    vec = simd_add_i(temp, vec);

    temp = vec;
    temp = _mm_shuffle_epi32(temp, 0x72);
    vec = simd_add_i(vec, temp);

    return vec;
}

// TODO: Optimize this part by using SIMD horizontal add + SIMD extract
long add_simd_int(simd_int vec)
{
	long sum=0;
	int32_t tmp[VECSIZE32];

	simd_storeu_i((simd_int*) tmp, vec);
	for(int i=0; i<VECSIZE32; i++)
		sum += tmp[i];

	return sum;
}

int32_t
simd_horizontal_add(simd_int vec)
{
	int32_t sum[4];
	simd_int mm_a = vec;
	simd_int mm_b = _mm_shuffle_epi32(mm_a, 0x39);
	mm_b = simd_add_i(mm_a, mm_b);
	mm_a = _mm_shuffle_epi32(mm_b, 0x3E);
	mm_b = simd_add_i(mm_a, mm_b);

	simd_storeu_i((simd_int*) sum, mm_b);

	return sum[0];
}

#if __linux__
long long add_simd256_int(__m256i vec)
{
	long long sum=0;
	int64_t tmp[VECSIZE32];

	_mm256_storeu_si256((__m256i *) tmp, vec);
	for(int i=0; i<VECSIZE32; i++)
		sum += tmp[i];

	return sum;
}
#endif

void
load_initial_clusters(
		data_t * centroids,
		int num_clusters,
		const data_t* data
		)
{
//	cout << "\n Initial cluster values:" << endl;
	for(int i=0; i<num_clusters; i++)
	{
		int indx = cluster_index[i];
//		cout << indx << ":\t";
		for(int j=0; j<DIMENSION; j++) {
			centroids[i*DIMENSION+j] = data[indx*DIMENSION+j];
//			cout << centroids[i*DIMENSION+j] << "\t";
		}
//		cout << endl;
	}
//	cout << endl;
}

void
load_simd_data(
		simd_data_t* simd_data,
		data_t* data,
		unsigned long N)
{
	int x_indx = 0;
	int y_indx = 0;
	int ctr = 0;
	int x_base = 0;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			*(simd_data+i*DIMENSION+j) = data[x_indx*DIMENSION+y_indx];
			x_indx++;
			if(x_indx%VECSIZE32 == 0)
			{
				x_indx = x_base;
				y_indx++;
				if(y_indx%DIMENSION==0)
				{
					ctr++;
					y_indx = 0;
					x_base = ctr*VECSIZE32;
					x_indx = x_base;
				}
			}
		}
	}

#if 0
	cout << "SIMD Data:" << endl;
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			cout << *(simd_data+i*DIMENSION + j) << " ";
		}
		cout << endl;
	}
#endif
}

void
load_initial_simd_clusters(
		simd_data_t * centroids,
		int num_clusters,
		const data_t* data
		)
{
#if 0
	for(int i=0; i<num_clusters; i++)
		for(int j=0; j<DIMENSION; j++)
			centroids[j*num_clusters+i] = data->point[cluster_index[i]*DIMENSION+j];
#endif

	for(int i=0; i<num_clusters; i++)
	{
		int B=i/VECSIZE32;
		int C=i%VECSIZE32;

		for(int j=0; j<DIMENSION; j++)
		{
			centroids[B*DIMENSION*VECSIZE32+j*VECSIZE32+C]=data[cluster_index[i]*DIMENSION+j];
		}
	}

//	cout << "\n Initial SIMD cluster values:" << endl;
//	print_data(centroids, num_clusters);
#if 0
	for(int i=0; i<num_clusters; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			//cout << centroids[i*DIMENSION+j] << " ";
			////cout << centroids->points[j*num_clusters + i] << " "; // For comparison
		}
		//cout << endl;
	}
	//cout << endl;

	//cout << "????? VEC ?????" << endl;
	for(int i=0; i<num_clusters; i++)
	{
		for(int j=0; j<DIMENSION; j++)
		{
			//cout << *(centroids+j*VECSIZE32 + i) << " ";
			if( j== DIMENSION-1) {
				//cout << endl;
			}
		}
		//cout << endl;
	}
#endif
}


void warmup_cache()
{
    srand((unsigned)time(0));

    int *a = (int*) malloc(4096*4096*sizeof(int));
    int *b = (int*) malloc(4096*4096*sizeof(int));
    int *c = (int*) malloc(4096*4096*sizeof(int));

    for(int i=0; i<4096*4096; i++)
    {
        a[i] = (int) rand()/RAND_MAX;
        b[i] = (int) rand()/RAND_MAX;
    }
    for(int i=0; i<4096*4096; i++)
    {
        c[i] = a[i] + b[i];
    }
    free(a);
    free(b);
    free(c);
}
