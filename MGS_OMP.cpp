//QR factorization by Modified Gram-Schmidt
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;

double *A,*A_copy,*Q,*R;                                    //Representative matrices for QR factorization  
double *qr;                                                 //Q*R
int n;                                                      //size of matrix

void initvector(int *vector, int m);

int main() 
{
	int i,j,k;
    int thread_id,num_threads;
	int error_eval=1,prt_mat=0;                             //Flags variables for evaluating error and printing matrix 
	double norm_error;                                      //Norm of error in evaluation i.e e=(QR-A)
    double exec_time;                                       //Execution time

    int *step;                                              //stores information on which column has been modified
   
	n=1000;                                               	//set matrix size
  
   	//Allocate memory  
   	A = (double *)malloc(n*n*sizeof(double));
   	A_copy = (double *)malloc(n*n*sizeof(double));
   	Q = (double *)malloc(n*n*sizeof(double));
   	R = (double *)malloc(n*n*sizeof(double));	   	
   
    step=(int *)malloc(n*sizeof(int));

    initvector(step,n);

	exec_time=omp_get_wtime();								//Start counting execution time
	double r_sq = 0;
    srand(time(0));

	#pragma omp parallel private(i,j,k,r_sq,thread_id)
	{
		thread_id=omp_get_thread_num();
        //Create A and its copy
	  	#pragma omp for 
	  	for (i = 0; i<n; i++)
	  	{
		  	for(j=0;j<n;j++)
		  	{
			  	A[i*n+j] = (rand()%10);
			  	A_copy[i*n+j] = A[i*n+j];			  	
		  	}
		  			  	
		}
	
		/* Implementation of Modified GS: A is iteratively modified by matrix transformations to yield a upper triangular matrix */
	
		/* Thread 0 calculates the first column of Q,R */    
		r_sq=0;
		if(thread_id==0)
		{
			for (i=0; i<n; i++){			
				r_sq = r_sq + A[0*n+i] * A[0*n+i]; 
			}
			R[0*n+0] = sqrt(r_sq);  			
			for (i=0; i<n; i++) {
				Q[0*n+i] = A[0*n+i]/R[0*n+0];							
	      	}
	      	step[0]=1;
		}

        /* Calculated k_th column of Q,R */  
	    for (k=1; k<n; k++)
	    {
	    	  	
			step[k-1]=1;									//Check if previous column is computed
	  		#pragma omp for 
		    for(j=0; j<n; j++)
		    {	
		    	if(j>=k)
		    	{	    	
			        R[(k-1)*n+j]=0;	
			        for(i=0; i<n; i++) 
					{			        	
			        	R[j*n+(k-1)] += Q[(k-1)*n+i] * A[j*n+i]; //update R			        	
			        } 
			        for (i=0; i<n; i++) 
					{				        	
			        	A[j*n+i] = A[j*n+i] - R[j*n+(k-1)]*Q[(k-1)*n+i];//update A
			        }			        
			       
					if(j==k)
					{
						thread_id=omp_get_thread_num();
						r_sq=0;
						for (i=0; i<n; i++){			
							r_sq = r_sq + A[k*n+i] * A[k*n+i]; 
						}
						R[k*n+k] = sqrt(r_sq);//update R  			

						for (i=0; i<n; i++) 
						{
							Q[k*n+i] = A[k*n+i]/R[k*n+k];//update Q			
				      	}
				      	step[k-1]=0;
                        step[k]=1;
			      	}
		      	}		        
			}			
			r_sq=0;	      	
		}
	}
	exec_time=omp_get_wtime()-exec_time;
   	printf("Execution time (in sec): %f \n",exec_time);
   	
   	// Print matrix

 	if(prt_mat==1) 
    {  		
        cout<<endl<<"Printing A..."<<endl;
		for (i = 0; i<n; i++){
		  	for(j=0;j<n;j++){	
                cout<<setprecision(3)<<fixed<<A_copy[j*n+i]<<"  ";		  	
			  	//printf("%.3f ", A_copy[j*n+i]);
		  	}
		  	cout<<endl;
		}

		cout<<endl<<"Printing Q..."<<endl;
		for (i = 0; i<n; i++)
        {
		  	for(j=0;j<n;j++)
            {			  	
			  	cout<<setprecision(3)<<fixed<<Q[j*n+i]<<"  ";	
		  	}
		  	cout<<endl;
		}

		cout<<endl<<"Printing R..."<<endl;
		for (i = 0; i<n; i++)
        {
		  	for(j=0;j<n;j++)
            {			  	
			  	cout<<setprecision(3)<<fixed<<R[j*n+i]<<"  ";	
		  	}
		  	cout<<endl;
		}
	}	
	
	/* Evaluate error */

    qr = (double *)malloc(n*n*sizeof(double));      
	double sum;

	if(error_eval==1)
    {
        
		for (i = 0; i < n; i++) 
        {
			for (j = 0; j < n; j++) 
            {
				sum = 0;
				for (k = 0; k < n; k++) 
                {
					sum = sum + Q[k*n+i] * R[j*n+k];
				}
				qr[j*n+i] = sum;
			}
		}
		if(prt_mat==1)
        {
        cout<<endl<<"QR is "<<endl;
			for (i = 0; i < n; i++) 
            {
				for (j = 0; j < n; j++) 
                {
					cout<<setprecision(3)<<fixed<<qr[j*n+i]<<"  ";
				}
				cout<<endl;
			}
		}
		norm_error=0;
		for (i = 0; i < n; i++) 
        {
			for (j = 0; j < n; j++) 
            {				
				norm_error+=fabs(qr[j*n+i]-A_copy[j*n+i]);
			}			
		}
		cout<<endl<<"Error is "<<norm_error<<endl;
	}

    cout<<endl<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Prog End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
	
    return 0;
}

void initvector(int *vector, int m)
{
    for(int iter=0;iter<m;iter++)
    {
        vector[iter]=0;

    }

}
