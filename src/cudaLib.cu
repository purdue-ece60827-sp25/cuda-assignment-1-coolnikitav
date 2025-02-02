
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < size) {
		y[i] += scale * x[i];
	}
}

int runGpuSaxpy(int vectorSize) {
	// vectorSize = 268435456;

	// Find size in bytes
	int size = vectorSize * sizeof(float);

	// Create the scale
	float scale = rand() / (float)RAND_MAX * RAND_MAX;

	// Create host vectors
	float* h_X = (float*) malloc(size);
	float* h_Y = (float*) malloc(size);
	float* h_Y_prev = (float*) malloc(size);

	vectorInit(h_X, vectorSize);
	vectorInit(h_Y, vectorSize);
	std::memcpy(h_Y_prev, h_Y, size);

	// Create device vectors
	float *d_X, *d_Y;
	cudaMalloc((void**) &d_X, size);
	cudaMalloc((void**) &d_Y, size);

	// Transfer vectors to device
	cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

	// Run the kernel
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(d_X, d_Y, scale, vectorSize);

	// Transfer vector to host
	cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

	// Verify results
	// Commented out for profiling
	// int errorCount = verifyVector(h_X, h_Y_prev, h_Y, scale, vectorSize);
	// std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free the memory
	free(h_X);
	free(h_Y);
	free(h_Y_prev);
	cudaFree(d_X);
	cudaFree(d_Y);
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// Setup the generator
	curandState_t rng;
	curand_init(clock64(), threadId, 0, &rng);

	// Generator random numbers and check for hit
	uint64_t hitCount = 0;
	float x, y;
	for (uint64_t i = 0; i < sampleSize; i++) {
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);

		if (int(x * x + y * y) == 0) {
			++hitCount;
		}
	}

	// Update pSums with hitCount for threadId
	pSums[threadId] = hitCount;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t sum = 0;
	for (uint64_t i = 0; i < reduceSize; i++) {
		if ((reduceSize * threadId + i) < pSumSize) {
			sum += pSums[reduceSize * threadId + i];
		}
	}

	totals[threadId] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() * 1000 << " milliseconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	// generateThreadCount = 1024;
	// sampleSize = 1000000;
	// reduceSize = 512;
	// reduceThreadCount = generateThreadCount / reduceSize;

	// std::cout << "generateThreadCount: " << generateThreadCount << " sampleSize: " << sampleSize 
	//  		  << " reduceThreadCount: " << reduceThreadCount << " reduceSize: " << reduceSize << std::endl;

	double approxPi = 0;

	uint64_t pSumsSize = generateThreadCount * sizeof(uint64_t);
	uint64_t totalsSize = reduceThreadCount * sizeof(uint64_t);

	uint64_t *d_pSums, *d_totals;
	cudaMalloc((void**) &d_pSums, pSumsSize);
	cudaMalloc((void**) &d_totals, totalsSize);
	
	float threadsPerBlock = 64.0;
	generatePoints<<<ceil(generateThreadCount / threadsPerBlock), threadsPerBlock>>>(d_pSums, pSumsSize, sampleSize);
	reduceCounts<<<ceil(reduceThreadCount/threadsPerBlock),threadsPerBlock>>>(d_pSums, d_totals, generateThreadCount, reduceSize);

	uint64_t *h_totals = (uint64_t*) malloc(totalsSize);
	cudaMemcpy(h_totals, d_totals, totalsSize, cudaMemcpyDeviceToHost);

	uint64_t totalsSum = 0;
	for (uint64_t i = 0; i < reduceThreadCount; i++) {
		totalsSum += h_totals[i];
	}

	approxPi = ((double)totalsSum / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	return approxPi;
}
