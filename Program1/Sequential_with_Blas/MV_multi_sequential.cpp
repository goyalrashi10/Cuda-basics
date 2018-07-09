//============================================================================
// Name        : MV_multi_sequential.cpp
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : Multiply Matrix A & Vector B in sequential order using C++,
//============================================================================

#include <iostream>
using namespace std;

extern "C"{
    // FORTRAN adds _ after all the function names
    // and all variables are called by reference
    void dgemv_(char* TRANS, const int* M, const int* N,
                double* alpha, double* A, const int* LDA, double* X,
                const int* INCX, double* beta, double* C, const int* INCY);
}

//void dgemv( char* TRANS, const int* M, const int* N,
//           double* alpha, double* A, const int* LDA, double* X,
//           const int* INCX, double* beta, double* C, const int* INCY ){
//    daxpy_( &N, &a, A, &inca, B, &incb );
//};


int main() {
    
    // Variable to store user choice of vector size
    int user_choice=0;
    int invalid_selected=1;
    
    //Declaring CPU matrix for storing values
    double **matrix_a;

    //Declaring CPU vectors for storing values and results
    double *vector_b;
    double *vector_c;
    double *blas_vector_c;
    double *vector_delta;
    
    //variables to calculate processing time
    clock_t start;
    clock_t end;
    float time_elapsed;
    
    //variables to store vector size
    int rows_in_matrix=4;
    int cols_in_matrix=4;
    
    cout<<endl;
    cout<<"#################### Program 1: Multiply Matrix to Vectors  ####################"<<endl<<endl;

    //Allocating memory for rows in CPU Matrix
    matrix_a= new double*[rows_in_matrix];
    double *blas_matrix_A=new double[rows_in_matrix*cols_in_matrix];
    
    //Allocating memory for columns in CPU Matrix
    for(int i=0;i<rows_in_matrix;i++){
        matrix_a[i]=new double[cols_in_matrix];
    }

    srand(time(0));
    int counter=0;

//    Load CPU Matrix with random values
    cout<<endl<<endl<<"Matrix :[[";
    for(int i=0;i<rows_in_matrix;i++){
        for(int j=0;j<cols_in_matrix;j++){
            matrix_a[i][j]=rand()%20+10;
            cout<<matrix_a[i][j]<<",";
            
            blas_matrix_A[counter]=matrix_a[i][j];
            counter=counter+1;
        }
        cout<<"],"<<endl;
    }
    cout<<"]"<<endl;
    
    //Allocating memory to CPU vectors
    vector_b = (double*)malloc(cols_in_matrix*sizeof(double));
    vector_c = (double*)malloc(rows_in_matrix*sizeof(double));
    blas_vector_c = (double*)malloc(rows_in_matrix*sizeof(double));
    vector_delta= (double*)malloc(rows_in_matrix*sizeof(double));
    
    //Load CPU vectors with random values
    cout<<endl<<endl<<"Vector B :[";
    for(int i = 0; i<cols_in_matrix; i++){
        vector_b[i]=rand()%10+1;
        cout<<vector_b[i]<<","<<endl;
    }
    cout<<"]"<<endl;
    
    start=clock(); //storing start time
//    Calculate Matrix A * Vector B
    for(int i = 0; i<rows_in_matrix; i++){
        int sum=0;
        for(int j = 0; j<cols_in_matrix; j++){
            sum=sum+matrix_a[i][j]*vector_b[j];
        }
        vector_c[i]=sum;
    }
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time

    cout<<"~~~~~~~~~~~~~~~~~~~ VECTOR C USING SEQUENTIAL ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    //Printing results for computed Vector
    cout<<endl<<endl<<"Vector C :[";
    for(int i = 0; i<rows_in_matrix; i++){
        cout<<vector_c[i]<<","<<endl;
    }
    cout<<"]"<<endl<<endl;

    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;

    char TRANS='T';
    int M=rows_in_matrix;
    int N=cols_in_matrix;
    double alpha=1.0;
    int LDA = ( rows_in_matrix >= cols_in_matrix ) ? rows_in_matrix : cols_in_matrix;
    double beta=0.0;
    int incA=1;
    int incB=1;
 
    cout<<"~~~~~~~~~~~~~~~~~~~ VECTOR C USING BLAS (DGEMV) ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    start=clock(); //storing start time
    dgemv_(&TRANS,&M,&N,&alpha,blas_matrix_A,&LDA,vector_b,&incA,&beta,blas_vector_c,&incB);
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time
    
    //Printing results for computed Vector
    cout<<endl<<endl<<"Vector C :[";
    for(int i = 0; i<rows_in_matrix; i++){
        cout<<blas_vector_c[i]<<","<<endl;
    }
    cout<<"]"<<endl<<endl;
    
    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;
    
    cout<<"~~~~~~~~~~~~~~~~~~~ DELTA ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    
    //    //Printing results for computed variable
    cout<<endl<<endl<<"Delta :[";
    for(int i = 0; i<rows_in_matrix; i++){
        vector_delta[i]=blas_vector_c[i]-vector_c[i];
        cout<<vector_delta[i]<<","<<endl;
    }
    cout<<"]"<<endl<<endl;
    
    cout<<"*********************** END OF PROGRAM ***********************"<<endl<<endl<<endl;

    return 0;
}
