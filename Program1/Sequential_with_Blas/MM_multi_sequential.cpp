//============================================================================
// Name        : MM_multi_sequential.cpp
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : Multiply Matrix A & B in sequential order using C++,
//============================================================================

#include <iostream>
using namespace std;

extern "C"{
    // FORTRAN adds _ after all the function names
    // and all variables are called by reference
    void dgemm_(char* TRANSA, char* TRANSB, const int* M,
                const int* N, const int* K, double* alpha, double* A,
                const int* LDA, double* B, const int* LDB, double* beta,
                double* C, const int* LDC);
}

int main() {
    
    // Variable to store user choice of vector size
    int user_choice=0;
    int invalid_selected=1;
    
    //Declaring CPU matrix for storing values
    double **matrix_a;
    double **matrix_b;
    double **matrix_c;
    
    //variables to calculate processing time
    clock_t start;
    clock_t end;
    float time_elapsed;
    
    //variables to store vector size
    int rows_in_matrixA=4;
    int cols_in_matrixA=4;
    int rows_in_matrixB=4;
    int cols_in_matrixB=4;
    
    cout<<endl;
    cout<<"#################### Program 1: Multiply Matrix to Matrix  ####################"<<endl<<endl;

    //Allocating memory for rows in CPU Matrix A
    matrix_a= new double*[rows_in_matrixA];
    matrix_b= new double*[rows_in_matrixB];
    matrix_c= new double*[rows_in_matrixA];

    //Allocating memory for rows in CPU Matrix A
    double *blas_matrix_A= new double[rows_in_matrixA*cols_in_matrixA];
    double *blas_matrix_B= new double[rows_in_matrixB*cols_in_matrixB];
    double *blas_matrix_C= new double[rows_in_matrixA*cols_in_matrixB];
    double *matrix_delta= new double[rows_in_matrixA*cols_in_matrixB];

    //Allocating memory for columns in CPU Matrix A
    for(int i=0;i<rows_in_matrixA;i++){
        matrix_a[i]=new double[cols_in_matrixA];
        matrix_c[i]=new double[cols_in_matrixB];
    }
    
    //Allocating memory for columns in CPU Matrix B
    for(int i=0;i<rows_in_matrixB;i++){
        matrix_b[i]=new double[cols_in_matrixB];
    }
    
    srand(time(0));
    int counter=0;
    //Load CPU Matrix with random values
    cout<<endl<<endl<<"Matrix A:[[";
    for(int i=0;i<rows_in_matrixA;i++){
        for(int j=0;j<cols_in_matrixA;j++){
            matrix_a[i][j]=rand()%20+10;
            cout<<matrix_a[i][j]<<",";
            
            blas_matrix_A[counter]=matrix_a[i][j];
            counter=counter+1;
        }
        cout<<"],"<<endl;
    }
    cout<<"]"<<endl;
    
    counter=0;
    //Load CPU Matrix with random values
    cout<<endl<<endl<<"Matrix B:[[";
    for(int i=0;i<rows_in_matrixB;i++){
        for(int j=0;j<cols_in_matrixB;j++){
            matrix_b[i][j]=rand()%10+10;
            cout<<matrix_b[i][j]<<",";
            
            blas_matrix_B[counter]=matrix_b[i][j];
            counter=counter+1;
        }
        cout<<"],"<<endl;
    }
    cout<<"]"<<endl;
    
    start=clock(); //storing start time
    //Calculate Matrix A * Matrix B
    for(int i=0;i<rows_in_matrixA; i++){
        for(int j=0;j<cols_in_matrixB; j++){
            int sum=0;
            for(int k = 0; k<rows_in_matrixB; k++){
                sum=sum+matrix_a[i][k]*matrix_b[k][j];
            }
            matrix_c[i][j]=sum;
        }
    }
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time

    cout<<"~~~~~~~~~~~~~~~~~~~ Matrix C USING SEQUENTIAL ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    //Printing results for computed Vector
    cout<<endl<<endl<<"Matrix C:[[";
    for(int i=0;i<rows_in_matrixA;i++){
        for(int j=0;j<cols_in_matrixB;j++){
            cout<<matrix_c[i][j]<<",";
        }
        cout<<"],"<<endl;
    }

    cout<<endl;
    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;

    cout<<"~~~~~~~~~~~~~~~~~~~ Matrix C USING BLAS (DGEMM) ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    char TRANSA='N';
    char TRANSB='N';
    int M=rows_in_matrixA;
    int N=cols_in_matrixB;
    int K=rows_in_matrixB;
    double alpha=1.0;
    int LDA = ( rows_in_matrixA >= cols_in_matrixA ) ? rows_in_matrixA : cols_in_matrixA;
    int LDB = ( rows_in_matrixB >= cols_in_matrixB ) ? rows_in_matrixB : cols_in_matrixB;
    int LDC = ( rows_in_matrixA >= cols_in_matrixB ) ? rows_in_matrixA : cols_in_matrixB;
    double beta=0.0;

    start=clock(); //storing start time
    dgemm_(&TRANSA, &TRANSB, &M,&N, &K, &alpha, blas_matrix_B,
           &LDA,blas_matrix_A , &LDB, &beta,blas_matrix_C, &LDC);
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time
    
    cout<<endl<<endl<<"using Blas Matrix C:[[";
    for(int i=0;i<(rows_in_matrixA*cols_in_matrixB);i++){
        cout<<blas_matrix_C[i]<<",";
        if((i+1)%(cols_in_matrixB)==0){
            cout<<"],"<<endl;
        }
    }
    cout<<"],"<<endl<<endl;

    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;
    
    cout<<"~~~~~~~~~~~~~~~~~~~ DELTA ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    
    //    //Printing results for computed variable
    cout<<endl<<endl<<"Delta :[";

    counter=0;
    for(int i=0;i<rows_in_matrixA;i++){
        for(int j=0;j<cols_in_matrixA;j++){
            matrix_delta[counter]=blas_matrix_C[counter]-matrix_c[i][j];
            cout<<matrix_delta[counter]<<",";
            counter=counter+1;
        }
        cout<<"],"<<endl;
    }
    cout<<"]"<<endl<<endl;

    cout<<"*********************** END OF PROGRAM ***********************"<<endl<<endl<<endl;

    return 0;
}
