//============================================================================
// Name        : AddingVectors.cu
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : Vector Addition using CUDA & C++,
//============================================================================

#include <iostream>
using namespace std;


// kernel GPU steps to add vectors
__global__ void add_vectors( int *gpu_vector_a, int *gpu_vector_b, int *gpu_vector_c, int vector_length) {
    
    // computing index from generic pattern
    int index = (blockIdx.x*blockDim.x)+threadIdx.x; // handle the data at this index
    
    // avoid out of bound Array size
    if (index < vector_length)
        gpu_vector_c[index] = gpu_vector_a[index] + gpu_vector_b[index];
}

int main() {
    cout << "!!!Program 0:Adding Vectors!!!" <<endl;// prints !!!Hello World!!!
    
    // Variable to store user choice of vector size
    int user_choice=0;
    int invalid_selected=1;
    
    //Declaring CPU vectors for storing values and results
    int *vector_a;
    int *vector_b;
    int *vector_c;
    
    //Declaring Device/GPU vectors for storing values & results
    int *gpu_vector_a;
    int *gpu_vector_b;
    int *gpu_vector_c;
    
    //Declaring varialble for blocks in a Grid for kernel processing
    int grid_n_blocks;
    
    //Declaring number of threads in a block for kernel processing
    int grid_n_threads;
    
    //variables to calculate processing time
    clock_t start;
    clock_t end;
    float time_elapsed;

    //variables to calculate vector size
    int vector_length=1;
    int vector_size=1;
    
    //Displaying various matrix sizes for users to select

    cout<<endl;
    cout<<"1. 2^8   = 256 elements "<<endl;
    cout<<"2. 2^9   = 512 elements"<<endl;
    cout<<"3. 2^12  = 4096 elements"<<endl;
    cout<<"4. 2^15  = 32768 elements"<<endl;
    cout<<"5. Exit"<<endl;

    while(invalid_selected==1){
        cout<<"please select a valid vector size: "<<endl;
        cin>>user_choice;
        
        if(user_choice==1){
            
            vector_length=256;
            grid_n_threads=256;
            invalid_selected=0;
        
        }else if(user_choice==2){
        
            vector_length=512;
            grid_n_threads=256;
            invalid_selected=0;
        
        }else if(user_choice==3){
        
            vector_length=4096;
            grid_n_threads=512;
            invalid_selected=0;
        
        }else if(user_choice==4){
            
            vector_length=32768;
            grid_n_threads=512;
            invalid_selected=0;
        
        }else if(user_choice==5){
            terminate();
        }
    }
    
    //Declaring variables to store vector length & size
    vector_size = vector_length*sizeof(int); //2^7
    
    //Allocating memory to CPU vectors
    vector_a = (int*)malloc(vector_size);
    vector_b = (int*)malloc(vector_size);
    vector_c = (int*)malloc(vector_size);
    
    //Allocating memory fo Device/GPU vectors for storing values & results
    cudaMalloc(&gpu_vector_a,vector_size);
    cudaMalloc(&gpu_vector_b,vector_size);
    cudaMalloc(&gpu_vector_c,vector_size);
    
    
    //Load CPU vectors with random values
    srand(time(0));
    for(int i = 0; i<vector_length; i++){
        //Loading vector_a & vector_b with random values
        vector_a[i]=rand()%10+1;
        vector_b[i]=rand()%20+10;
    }
    
    //Print vector A
    cout<<endl<<endl<<"Vector A :[";
    for(int i = 0; i<vector_length; i++){
        cout<<vector_a[i]<<",";
    }
    cout<<"]"<<endl;

    //Print vector B
    cout<<endl<<endl<<"Vector B :[";
    for(int i = 0; i<vector_length; i++){
        cout<<vector_b[i]<<",";
    }
    cout<<"]"<<endl;

    //Copying CPU Vector variables into GPU vector variables
    cudaMemcpy(gpu_vector_a,vector_a,vector_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_vector_b,vector_b,vector_size,cudaMemcpyHostToDevice);
    
    //call Kernel to perform addition
    grid_n_blocks=ceil(vector_length/grid_n_threads);
    
    
    start=clock(); //storing start time
    add_vectors<<<grid_n_blocks,grid_n_threads>>>( gpu_vector_a, gpu_vector_b, gpu_vector_c ,vector_length);
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time
    
    //Copying GPU Vector variable into CPU vector variable
    cudaMemcpy(vector_c,gpu_vector_c,vector_size,cudaMemcpyDeviceToHost);
    
    //Printing results for computed variable
    cout<<endl<<endl<<"Vector C :[";
    for(int i = 0; i<vector_length; i++){
        cout<<vector_c[i]<<",";
    }
    cout<<"]"<<endl;
    
    cout<<"Number of Blocks in each Grid    : "<<grid_n_blocks<<endl;
    cout<<"Number of Threads in each Block  : "<<grid_n_threads<<endl;
    cout<<"Total time taken is              : "<<time_elapsed<<" ms"<<endl;
    
    // releasing the memory allocated on the GPU
    cudaFree( gpu_vector_a );
    cudaFree( gpu_vector_b );
    cudaFree( gpu_vector_c );
    
    return 0;
}
