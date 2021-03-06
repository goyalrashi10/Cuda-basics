//============================================================================
// Name        : MultiplyMatrix.cu
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : matrix Multiplication using CUDA & C++,
// To Run      : nvcc matrix_multiply.cu -lcublas -o matrix_multiply.out
// Note        : Please see report to understand how to run the code to get
//               different outputs
//============================================================================

#include <stdio.h>
#include <math.h>
using namespace std;

// CUDA and CUBLAS functions
// #include <helper_functions.h>
// #include <helper_cuda.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define TILE_WIDTH 2
//kernel implementation for Matrix Multiplication Naive (Non Shared)
__global__ void Matrix_Multiply_Cuda_1d( float *d_gpu_matrix_a , float *d_gpu_matrix_b , float *d_gpu_matrix_c , const int WIDTH ){
	
	//calculating matrix element value
	for (int k = 0 ; k<WIDTH ; k++ ){
	  		d_gpu_matrix_c[(blockIdx.x*blockDim.x)+threadIdx.x]+= d_gpu_matrix_a[threadIdx.x * WIDTH + k ] * d_gpu_matrix_b[ k * WIDTH + threadIdx.x] ;
	}	
}

//kernel implementation for Matrix Multiplication Naive (Non Shared)
__global__ void Matrix_Multiply_Cuda_2d( float *d_gpu_matrix_a , float *d_gpu_matrix_b , float *d_gpu_matrix_c , const int WIDTH ){
	
	// calculate row & col values for current thread
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
	
	//calculating matrix element value
	for (int k = 0 ; k<WIDTH ; k++ ){
	  d_gpu_matrix_c[row*WIDTH + col]+= d_gpu_matrix_a[row * WIDTH + k ] * d_gpu_matrix_b[ k * WIDTH + col] ;
	}
}

//kernel implementation for Matrix Multiplication using Shared Memory Tiling
__global__ void Matrix_Multiply_Tiled( float *d_gpu_matrix_a , float *d_gpu_matrix_b , float *d_gpu_matrix_c , const int WIDTH )
{
	//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
	__shared__ float gpu_tile_a [TILE_WIDTH][TILE_WIDTH] ;
	__shared__ float gpu_tile_b [TILE_WIDTH][TILE_WIDTH] ;

	// calculate row & col values for current thread
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

	// m indicate number of phase
	for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ){
		gpu_tile_a[threadIdx.y][threadIdx.x] =  d_gpu_matrix_a[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
		gpu_tile_b[threadIdx.y][threadIdx.x] =  d_gpu_matrix_b[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;

		// wait for threads to synchronize
	 	__syncthreads() ;

	 	// Do for tile
	   	for ( int k = 0; k<TILE_WIDTH ; k++ ){
			d_gpu_matrix_c[row*WIDTH + col]+= gpu_tile_a[threadIdx.x][k] * gpu_tile_b[k][threadIdx.y] ;
		}

		// wait for threads to synchronize
	 	__syncthreads() ; 
	 }
}
//kernel implementation for Matrix Multiplication using Shared Memory Tiling-Loop UnRolling
__global__ void Matrix_Multiply_Loop_Unroll( float *d_gpu_matrix_a , float *d_gpu_matrix_b , float *d_gpu_matrix_c , const int WIDTH )
{
	//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
	__shared__ float gpu_tile_a [TILE_WIDTH][TILE_WIDTH] ;
	__shared__ float gpu_tile_b [TILE_WIDTH][TILE_WIDTH] ;

	// calculate row & col values for current thread
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

	// m indicate number of phase
	for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ){
		gpu_tile_a[threadIdx.y][threadIdx.x] =  d_gpu_matrix_a[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
		gpu_tile_b[threadIdx.y][threadIdx.x] =  d_gpu_matrix_b[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;

		// wait for threads to synchronize
	 	__syncthreads() ;

	 	// Do for tile
	   	for ( int k = 0; k<TILE_WIDTH ; k+=4 ){
	   	
			d_gpu_matrix_c[row*WIDTH + col]+= gpu_tile_a[threadIdx.x][k] * gpu_tile_b[k][threadIdx.y] 
												+ gpu_tile_a[threadIdx.x][k+1] * gpu_tile_b[k+1][threadIdx.y]
												+ gpu_tile_a[threadIdx.x][k+2] * gpu_tile_b[k+2][threadIdx.y] 
												+ gpu_tile_a[threadIdx.x][k+3] * gpu_tile_b[k+3][threadIdx.y] ;
		}

		// wait for threads to synchronize
	 	__syncthreads() ; 
	 }
}

//kernel implementation for Matrix Multiplication using cublas
void Matrix_Multiply_cublas(float *d_gpu_matrix_a , float *d_gpu_matrix_b , float *d_gpu_matrix_c , const int WIDTH ){

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t ret =cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,WIDTH, WIDTH, WIDTH, &alpha, 
					d_gpu_matrix_b,WIDTH,d_gpu_matrix_a,WIDTH, &beta,  d_gpu_matrix_c, WIDTH);

	cublasDestroy_v2(handle);
}

int main ()
{

	// defining size of matrix
	const int WIDTH = 8 ;

    printf("\n Program 2 for Matrix Multiplication!! \n\n");

    printf("\n Matrix Size :%d \n",WIDTH);
    printf("\n Number of Elements :%d \n",WIDTH*WIDTH);
    printf("\n Tile Size :%d \n",TILE_WIDTH);
    printf("\n Loop Unrolled by factor of 4 \n");

	// defining matrix
	float h_matrix_a[WIDTH][WIDTH];
	float h_matrix_b[WIDTH][WIDTH];
	float h_matrix_output_1d[WIDTH][WIDTH];
	float h_matrix_output_2d[WIDTH][WIDTH];
	float h_matrix_output_tiled[WIDTH][WIDTH] ;
	float h_matrix_output_loop_Unroll[WIDTH][WIDTH] ;
	float h_matrix_output_cublas[WIDTH][WIDTH] ;
                     
	float *d_gpu_matrix_a; 
	float *d_gpu_matrix_b;
	float *d_gpu_matrix_output_1d;
	float *d_gpu_matrix_output_2d;
	float *d_gpu_matrix_output_tiled ;
	float *d_gpu_matrix_output_loop_Unroll ;
	float *d_gpu_matrix_cublas ; 
	
	int i , j ;

	//Calculations for gflops
	clock_t start;
    	clock_t end;
	double dSeconds =0.0;
	double gflops =0.0;
	double dNumOps =0.0;
	double totalSeconds  =0.0;


	//input in host array
	for ( i = 0 ; i<WIDTH ; i++ ){
		for (j = 0 ; j<WIDTH ; j++ ){
			h_matrix_a[i][j] = 1 ;
			h_matrix_b[i][j] = 2 ;
		}
	}	
	start=clock();
	//create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
	cudaMalloc((void **) &d_gpu_matrix_a , WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_gpu_matrix_b , WIDTH*WIDTH*sizeof (int) ) ;

	//copy host array to device array
	cudaMemcpy ( d_gpu_matrix_a , h_matrix_a , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;
	cudaMemcpy ( d_gpu_matrix_b , h_matrix_b , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;

	//allocating cuda memory for output arrays
	cudaMalloc((void **) &d_gpu_matrix_output_1d , 			WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_gpu_matrix_output_2d , 			WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_gpu_matrix_output_tiled , 		WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_gpu_matrix_output_loop_Unroll , WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_gpu_matrix_cublas , 			WIDTH*WIDTH*sizeof (int) ) ;
	end=clock();
	totalSeconds = (end-start)/1000.0;
	printf("Total operational time= %.2f(ms)\n\n",totalSeconds*1000.);


	int grid= WIDTH;
	int block=WIDTH;
	dNumOps = 2.0 * (double)WIDTH * (double)WIDTH * (double)WIDTH;
	
	//Calling kernel 1 D
	start=clock();
	Matrix_Multiply_Cuda_1d <<<grid,block>>> ( d_gpu_matrix_a , d_gpu_matrix_b ,d_gpu_matrix_output_1d , WIDTH) ;
	cudaMemcpy(h_matrix_output_1d , d_gpu_matrix_output_1d , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
	end=clock();

	dSeconds = (end-start)/1000.0;
	gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("\n1D = %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n\n",gflops, dSeconds*1000., dNumOps);

	//printf the result array after Generic Approach
// 	printf ("Matrix C (1D): \n") ;
 //	for ( i = 0 ; i<WIDTH ; i++ ){
 //		for ( j = 0 ; j < WIDTH ; j++ ){
 //			printf ("%f   ",h_matrix_output_1d[i][j] ) ;
 //		}
 //		printf ("\n") ;
 //	}

	//calling kernal 2D
	dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;

	start=clock();
	Matrix_Multiply_Cuda_2d <<<dimGrid,dimBlock>>> ( d_gpu_matrix_a , d_gpu_matrix_b ,d_gpu_matrix_output_2d , WIDTH) ;
	cudaMemcpy(h_matrix_output_2d , d_gpu_matrix_output_2d , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
	end=clock();

	dSeconds = (end-start)/1000.0;
	gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("\n2D = %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n\n",gflops, dSeconds*1000., dNumOps);

	//printf the result array after 2D Approach
// 	printf ("Matrix C (2D): \n") ;
// 	for ( i = 0 ; i<WIDTH ; i++ ){
 //		for ( j = 0 ; j < WIDTH ; j++ ){
 //			printf ("%f   ",h_matrix_output_2d[i][j] ) ;
 //		}
 //		printf ("\n") ;
 //	}


	//calling kernel for tiling
	start=clock();
  	Matrix_Multiply_Tiled<<<dimGrid,dimBlock>>> ( d_gpu_matrix_a , d_gpu_matrix_b ,d_gpu_matrix_output_tiled , WIDTH) ;
  	cudaMemcpy(h_matrix_output_tiled , d_gpu_matrix_output_tiled , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
	end=clock();

	dSeconds = (end-start)/1000.0;
	gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("\nShared Memory Tiled = %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n\n",gflops, dSeconds*1000., dNumOps);

	//printf the result array after tiling Approach
  //	printf ("Matrix C (Tiling): \n") ;
 //	for ( i = 0 ; i<WIDTH ; i++ ){
 //		for ( j = 0 ; j < WIDTH ; j++ ){
 //			printf ("%f   ",h_matrix_output_tiled[i][j] ) ;
 ///		}
  //		printf ("\n") ;
 //	}

	//calling kernel for tiling + loop unrolling
	start=clock();
 	Matrix_Multiply_Tiled<<<dimGrid,dimBlock>>> ( d_gpu_matrix_a , d_gpu_matrix_b ,d_gpu_matrix_output_loop_Unroll , WIDTH) ;
  	cudaMemcpy(h_matrix_output_loop_Unroll , d_gpu_matrix_output_loop_Unroll , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
	end=clock();

	dSeconds = (end-start)/1000.0;
	gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("\nShared Memory Tiled - Loop Unroll = %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n\n",gflops, dSeconds*1000., dNumOps);

 	//printf the result array after tiling + loop unrolling Approach
 //	printf ("Matrix C (Tiling): \n") ;
 //	for ( i = 0 ; i<WIDTH ; i++ ){
 //		for ( j = 0 ; j < WIDTH ; j++ ){
 //			printf ("%f   ",h_matrix_output_loop_Unroll[i][j] ) ;
 //		}
 //		printf ("\n") ;
 //	}


	//calling kernel for cuBLAS
	start=clock();
    Matrix_Multiply_cublas( d_gpu_matrix_a , d_gpu_matrix_b ,d_gpu_matrix_cublas , WIDTH) ;
  	cudaMemcpy(h_matrix_output_cublas , d_gpu_matrix_cublas , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
	end=clock();

	dSeconds = (end-start)/1000.0;
	gflops = 1.0e-9 * dNumOps/dSeconds;
    printf("\nCuBlas = %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n\n",gflops, dSeconds*1000., dNumOps);

	//printf the result array after cuBLAS Approach
//  	printf ("Matrix C (Cublas): \n") ;
// 	for ( i = 0 ; i<WIDTH ; i++ ){
 //		for ( j = 0 ; j < WIDTH ; j++ ){
 //			printf ("%f   ",h_matrix_output_cublas[i][j] ) ;
 //		}
  //		printf ("\n") ;
 //	}

	cudaFree( d_gpu_matrix_a); 
	cudaFree( d_gpu_matrix_b);
	cudaFree( d_gpu_matrix_output_1d);
	cudaFree( d_gpu_matrix_output_2d);
	cudaFree( d_gpu_matrix_output_tiled) ; // device array
	cudaFree( d_gpu_matrix_cublas) ; // device array

 	system("pause") ;
}
