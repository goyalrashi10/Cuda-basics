//============================================================================
// Name        : lab3.cu
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : vector addition using CUDA & C++,
// To Run      : nvcc lab3.cu -lcublas -o lab3.out
// Note        : Please see report to understand how to run the code to get
//               different outputs
//============================================================================

#include <stdio.h>
#include <time.h>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_runtime_api.h"


#define MAX_THREADS_PER_BLOCK 1024
#define MAX_NO_OF_BLOCKS 65536


// Adding two vectors
__global__ void add(double *a, double *b, double *c) {
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

__global__ void add_loop_unroll(double *a, double *b, double *c) {
    
    int index = 2*(threadIdx.x + blockIdx.x * blockDim.x);
    c[index] = a[index] + b[index];
    c[index+1] = a[index+1] + b[index+1];
}


// Adding two vectors using loop unrolling
__global__ void add_shared(double *a, double *b, double *d) {
    
    extern __shared__ double temp[];
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //c[index] = a[index] + b[index];
    int s_index = threadIdx.x;
    temp[s_index] = a[s_index] + b[s_index];
    __syncthreads();
    d[index] = temp[s_index];
}


// Getting a random number
void random_ints(double* x, int size)
{
	int i;
	for (i=0; i<size; i++) {
		x[i] = rand()%10;
	}
}

void printVector(double* x, int n)
{
    for (int i = 0; i < n; i++){
        printf("%.1f ", x[i]);
    }
    printf("\n");
}

void vectorVectorAddition(int N) {
	
	clock_t start_time, end_time, start_computation_time, end_computation_time, total, totalComputation, start_shared, end_shared,  total_shared, memory_start, memory_end, total_memory_time, total_memory_time1, loop_unroll_start, lopp_unroll_end, loop_unroll_total;
	double total_time, total_computation_time;
    
    double gflops =0.0;
    double gflops1 =0.0;
    double gflops2 =0.0;
    double gflops3 =0.0;
    double gflops4 =0.0;
	
	// Host copies of a, b, c
	double *a, *b, *c,*d,*e;

	// Device copies of a, b, c
	double *d_a, *d_b, *d_c, *d_d,*d_e;
	int size = N * sizeof(double);

	int noOfThreads = 1;
	int noOfBlocks = 1;	

	for (noOfBlocks = 2; noOfBlocks <= MAX_NO_OF_BLOCKS; noOfBlocks = noOfBlocks * 2)
	{
		for (noOfThreads = 4; noOfThreads <= MAX_THREADS_PER_BLOCK; noOfThreads = noOfThreads * 2)
		{
			if (noOfBlocks == MAX_NO_OF_BLOCKS)
			{
				noOfBlocks = noOfBlocks - 1;
			}
            //printf("%d, %d, \n", noOfBlocks, noOfThreads);

			if (noOfBlocks * noOfThreads < N)
            {
				continue;
			}
            if (noOfBlocks * noOfThreads > 2*N)
            {
                break;
            }
            memory_start = clock();
			//printf("%d, %d, \n", noOfBlocks, noOfThreads);
			// Allocate space for device copies of a, b , c	and d
			cudaMalloc((void **)&d_a, size);
			cudaMalloc((void **)&d_b, size);
			cudaMalloc((void **)&d_c, size);
            cudaMalloc((void **)&d_d, size);
            cudaMalloc((void **)&d_e, size);


			// Allocate space for host copies of a, b , c and d
			a = (double *)malloc(size);
			random_ints(a, N);	
			b = (double *)malloc(size);
			random_ints(b, N);
			c = (double *)malloc(size);
            d = (double *)malloc(size);
            e = (double *)malloc(size);

			// Started measuring time
			
			//printf("Start time: %ld\n", start_time);

			// Copy inputs to device
			cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
            
            memory_end = clock();
            
            total_memory_time = memory_end - memory_start;

			// Started measuring operation time
			start_computation_time = clock();
			//printf("Start computation time: %ld\n", start_time);

			// Launch add() kernel on GPU
			add<<<noOfBlocks,noOfThreads>>>(d_a, d_b, d_c);

			// Stopped measuring operation time
			end_computation_time = clock();
			//printf("Stop computation time: %ld\n", start_time);

			// Copy result back to host
			cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

			// Stop measuring time
            
            start_shared = clock();
            //printf("Start computation time: %ld\n", start_time);
            
            // Launch  shared memory add() kernel on GPU
            add_shared<<<noOfBlocks,noOfThreads,sizeof(double)*noOfThreads>>>(d_a, d_b, d_d);
            
            // Stopped measuring operation time
            end_shared = clock();
            
            // Copy result back to host
            cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
            
            loop_unroll_start = clock();
            //printf("Start computation time: %ld\n", start_time);
            
            // Launch  shared memory add() kernel on GPU
            add_loop_unroll<<<noOfBlocks,noOfThreads/2,sizeof(double)*noOfThreads>>>(d_a, d_b, d_e);
            
            // Stopped measuring operation time
            lopp_unroll_end = clock();
            
            // Copy result back to host
            cudaMemcpy(e, d_e, size, cudaMemcpyDeviceToHost);

            
            
			//printf("End time: %ld\n", end_time);

			//printVector(a, N);
			//printVector(b, N);
			//printVector(c, N);
            //printVector(d, N);
           // printVector(e, N);
			
			cublasStatus_t ret; 
			cublasHandle_t h;
			cublasCreate(&h);

			double alpha = 1.0;
        	int inc = 1;
            
            start_time = clock();

			ret = cublasDaxpy(h, N, &alpha, d_a, inc, d_b, inc);
            end_time = clock();
            
            
			cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
            
			//printf("CuBlas result\n");
			//printVector(b, N);

			cublasDestroy(h);
            
			// Computation Time for global
			totalComputation = end_computation_time - start_computation_time;
			//printf("Total computation clock cycles taken by GPU: %ld\n", total);
			total_computation_time = ((double) totalComputation) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
			//printf(" Global memory excecution time %f, ", gflops);
            
            //total time for global implementation
            total_memory_time1 = total_memory_time + totalComputation;
            total_computation_time= ((double) total_memory_time1) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            gflops1=gflops;
            //printf(" Total global memory excecution time %f, ", gflops);
            total_memory_time1 = 0;

            // Computation Time for shared memory kernel
            total_shared = end_shared - start_shared;
            //printf("Total computation clock cycles taken by GPU: %ld\n", total);
            total_computation_time = ((double) total_shared) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            //printf(" Shared memory Kernel excecution time %f, ", gflops);

            //total time for shared implementation
            total_memory_time1 = total_memory_time + total_shared;
            total_computation_time= ((double) total_memory_time1) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            gflops2=gflops;
            //printf(" Total Shared memory excecution time %f, ", gflops);
            total_memory_time1= 0 ;
            
            // Computation Time for loop unroll kernel
            loop_unroll_total =lopp_unroll_end -  loop_unroll_start;
            total_computation_time = ((double) loop_unroll_total) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            //printf(" Shared loop unroll excecution time %f, ", gflops);
            
            //total time for loop unroll implementation
            total_memory_time1 = total_memory_time + loop_unroll_total;
            total_computation_time= ((double) total_memory_time1) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            gflops3=gflops;
            //printf(" Total Loop unroll excecution time %f, ", gflops);
            total_memory_time1 = 0;

            
			// computation time for cuBLAS
			total = end_time - start_time;
			//printf("Total clock cycles taken by GPU: %ld\n", total);
			total_time = ((double) total) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_time;
			//printf("cuBLAS execution time %f\n", gflops);
            
            //total time for cuBLAS implementation
            total_memory_time1 = total_memory_time + total;
            total_computation_time= ((double) total_memory_time1) / ((double) (CLOCKS_PER_SEC));
            gflops = 1.0e-9 * N/total_computation_time;
            gflops4=gflops;
            //printf(" Total cuBLAS excecution time %f, ", gflops);

            printf(" Total excecution time Global, Shared, Loop, cublas Vector Size %d, (Blocks: %d, Threads: %d) %f, %f, %f, %f \n",size,noOfBlocks,noOfThreads, gflops1,gflops2,gflops3,gflops4);

			// Cleanup
            free(a); free(b); free(c); free(d); free(e);
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_e);

			break;
		}

		//break;
	}

	printf("\n\n");
}

int main(void)
{
	for (int i=1; i<=4; i++)
	{
		int size = pow(16, i);
		//printf("Vector size %d\n", size);
		vectorVectorAddition(size);
		//break;
	}

	return 0;
}
