#include <iostream>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
using namespace std;
#define N  100

float ** identity(int n)                                        //Defining Identity matrix
{
    float **A = new float *[n];
    for (int i = 0; i < n; i++)
    {
        A[i] = new float[n];
        
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                A[i][j] = 1;
            else
                A[i][j] = 0;
        }
    }

    return A;
}

float *base(int row)                                            //Defining function for basis vectors
{
    float *v = new float[row];
    v[0] = 1;
    
    for(int i = 1; i < row; i++)
    {
        v[i] = 0;
    }
    
    return v; 
}

float norm_col(float *col, int n)                               //Calculating l-2 norm
{ 
    int sum =0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
    {
        sum += col[i]*col[i];
    }
    return sqrt(sum);
}

float **product(float **mat1, float **mat2, int n)              //Calculating matrix product
{
    float **prod = new float*[n];
    int B = 10;
    for (int i = 0; i < n; i++)
    {
        prod[i] = new float[n];
        
        for (int j = 0; j < n; j++)
            prod[i][j] = 0;
    }
    
    #pragma omp parallel for
    for (int ii = 0; ii < n; ii+=B)                             // Applying Loop Blocking
    {
        for (int jj = 0; jj < n; jj+=B) 
        {
            for (int kk = 0; kk < n; kk+=B) 
            {
                for (int i = ii; i < ii+B; i++) 
                {
                    for (int j = jj; j < jj+B; j++) 
                    {
                        for (int k = kk; k < kk+B; k++) 
                        {
                            prod[i][j] += mat1[i][k]*mat2[k][j];
                        }
                    }
                }
            }
        }
    }
    
    return prod;
}

float *ortho_vec(float **mat, int indx, int n)                  // Calculating orthogonal vector v
{
    float *u = new float[n - indx], *e, *v = new float[n - indx];
    for(int i = indx; i < n; i++)
    {
        u[i - indx] = mat[i][indx];
    }
    
    e = base(n-indx);
    *e = norm_col(u, n - indx);
    #pragma omp parallel for
    for(int j = 0; j < n - indx; j++)
    {
        u[j] = u[j] - e[j];
    }
    if (u[0] == 0)                                              // This if statement is checking if the orthogonal vector is zero or not 
    {                                                           // If yes then making sure that it's not zero
        #pragma omp parallel for
        for(int i = 0; i < n - indx; i++)
        {
            v[i] = u[i] + e[i];
        } 
    }
    else
    {
        #pragma omp parallel for
        for(int i = 0; i < n - indx; i++)
        {
            v[i] = u[i]/norm_col(u, n - indx);
        } 
    }
    return v;
}

float **householder_mat(float **mat, int n)                     // Householder transformation function                    
{
    float **H_iter_pre;
    for(int iter = 0; iter < n - 1; iter++)
    {
        float *v;
        v = ortho_vec(mat, iter, n);

        float **H_iter = new float*[n], **H_iter1 = new float*[n - iter], **I;
        
        for (int i = 0; i < n; i++)
            H_iter[i] = new float[n];
        for (int i = 0; i < n - iter; i++)
        	H_iter1[i] = new float[n];
        
        I = identity(n - iter);

        #pragma omp parallel for                                //finding Householder matrix
        for(int i = 0; i < n - iter; i++)
        {
            for(int j = 0; j < n - iter; j++)
            {
                H_iter1[i][j] = I[i][j] - 2*v[i]*v[j];
            }
        }
        #pragma omp parallel for                                //Add Identity matrix of suitable size to make the dimension of H same as A
        for(int i = 0; i < n ; i++)
        {
            for(int j = 0; j < n ; j++)
            {
                if(i < iter || j< iter)
                {
                    if(i == j)
                    {
                        H_iter[i][j] = 1;
                    }
                    else
                    {
                        H_iter[i][j] = 0;
                    }
                }
                else
                {
                    H_iter[i][j] = H_iter1[i - iter][j - iter];
                }
            }
        }
        
        if(iter == 0)
        {
            H_iter_pre = H_iter;
        }
        else
        {
            H_iter_pre = product(H_iter, H_iter_pre, n);         //Multiplying successive Householder matrix 
        }
        mat = product(H_iter, mat, n);

    }
    return H_iter_pre;
}

float **transpose(float **mat, int n)                           //Taking transpose 
{
    float **temp = new float *[n];
    
    for (int i = 0; i < n; i++)
    {
        temp[i] = new float[n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            temp[j][i] = mat[i][j];
        }
    }

    return temp;
}
void printPro(float **a, int n)                                //printing function
{
    if(n>=30)
    {  
        for(int j=0;j<n;j++)
        {
            cout<<setprecision(4)<<a[n-1][j]<<"     ";
        }
        cout<<"\n";       
    }   
    else
    {
        for(int i=0;i<n;i++)
        {    
          for(int j=0;j<n;j++)
          {
              cout<<setprecision(4)<<a[i][j]<<"     ";
          }
          cout<<"\n";       
        }        
    }
    cout<<"\n";
}


int main()
{ 
    omp_set_num_threads(16);
    float B[N][N];
    srand(time(0));
    #pragma omp parallel for
    for(int i =0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            B[i][j] = (int)rand()%10;
        }
    }

    float **Q, **R, **Q_T, **A = new float *[N];
    
    for (int i = 0; i < N; i++)
    {
        A[i] = new float[N];
    }
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = B[i][j];
        }
    }
    auto start = std::chrono::high_resolution_clock::now();

    Q_T = householder_mat(A, N);
    Q = transpose(Q_T, N);
    R = product(Q_T, A, N);

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start); 
    cout<<"Time taken for conversion: "<<setprecision(6)<<duration.count()/1000000000.0<<endl;

    // cout<<"---------- A ----------"<<"\n"; 
    // printPro(A, N);
    // cout<<"---------- R ----------"<<"\n"; 
    // printPro(R, N);
    // cout<<"---------- Q ----------"<<"\n"; 
    // printPro(Q, N);
    // cout<<"---------- Q_T ----------"<<"\n"; 
    // printPro(Q_T, N);
    
}