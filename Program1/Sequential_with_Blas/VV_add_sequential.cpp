//============================================================================
// Name        : VV_add_sequential.cpp
// Author      : Rashi Goyal
// Version     :
// Copyright   : Your copyright notice
// Description : Add Vector A & B in sequential order using C++,
//============================================================================

#include <iostream>
//#include </usr/local/opt/openblas/include/cblas.h>
//#include <cblas.h>
using namespace std;

extern "C"{
    // FORTRAN adds _ after all the function names
    // and all variables are called by reference
    double daxpy_( const int *N, double *a, double *A, const int *inca, double *B, const int *incb );
}

void daxpy( int N, double a, double *A, int inca, double *B, int incb ){
    daxpy_( &N, &a, A, &inca, B, &incb );
};


int main() {
    
    //Declaring CPU vectors for storing values and results
    double *vector_a;
    double *vector_b;
    double *vector_c;
    double *vector_delta;
    
    //variables to calculate processing time
    clock_t start;
    clock_t end;
    float time_elapsed;

    //variables to store vector size
    int vector_length=1;
    int vector_size=1;
    
    vector_length=10;

    //Declaring variables to store vector length & size
    vector_size = vector_length*sizeof(int); //2^7
    
    //Allocating memory to CPU vectors
    vector_a = new double[vector_size];
    vector_b = new double[vector_size];
    vector_c = new double[vector_size];
    vector_delta= new double[vector_size];
    
    //Load CPU vectors with random values
    srand(time(0));
    for(int i = 0; i<vector_length; i++){
        //Loading vector_a & vector_b with random values
        vector_a[i]=rand()%10+1;
        vector_b[i]=rand()%20+10;
    }

    cout<<endl;
    cout<<"#################### Program 1: Adding Vectors  ####################"<<endl<<endl;

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
    cout<<"]"<<endl<<endl;
    
    start=clock(); //storing start time
    //Calculate vector A+B
    for(int i = 0; i<vector_size; i++){

        //Adding two vectors
        vector_c[i]=vector_a[i]+vector_b[i];
    }
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time
    
    cout<<"~~~~~~~~~~~~~~~~~~~ VECTOR C USING SEQUENTIAL ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    cout<<endl<<endl<<"Vector C :[";
    for(int i = 0; i<vector_length; i++){
        cout<<vector_c[i]<<",";
    }
    cout<<"]"<<endl<<endl;

    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;

    cout<<"~~~~~~~~~~~~~~~~~~~ VECTOR C USING BLAS (DAXPY) ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;

    start=clock(); //storing start time
    daxpy(vector_length,1,vector_a, 1, vector_b, 1 );
    end=clock(); //storing end time
    time_elapsed=(end - start)/(float) 1000;    //calculating elapsed time
    
//    //Printing results for computed variable
    cout<<endl<<endl<<"Vector C :[";
    for(int i = 0; i<vector_length; i++){
        cout<<vector_b[i]<<",";
    }
    cout<<"]"<<endl<<endl;

    cout<<"Total time taken is : "<<time_elapsed<<" ms"<<endl<<endl;

    cout<<"~~~~~~~~~~~~~~~~~~~ DELTA ~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    
    //    //Printing results for computed variable
    cout<<endl<<endl<<"Delta :[";
    for(int i = 0; i<vector_length; i++){
        vector_delta[i]=vector_b[i]-vector_c[i];
        cout<<vector_delta[i]<<",";
    }
    cout<<"]"<<endl<<endl;

    cout<<"*********************** END OF PROGRAM ***********************"<<endl<<endl<<endl;

    return 0;
}
