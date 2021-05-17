%%cu
// Remove the above "%%cu" if running on a local CUDA installation instead of Google colab notebook.
/*
    Parallelized Modified Gram Schmidt algorithm for QR decomposition of matrix.
    Authored-by: Abhijeet Prasad Bodas, Indian Institute of Technology Bombay
    Environment: Google Colab notebook
    CUDA version: v11.0.221
    GPU: Tesla T4
*/


#include <bits/stdc++.h>
using namespace std;

// Dimension of the matrix.
extern const int N = 3;

void print_matrix(double *m)
{
    // Prints a given N*N matrix.
    cout << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%5.4f ", m[i * N + j]);
        }
        cout << endl;
    }
}

void initialize_matrix(double *m)
{
    // Initialize a N*N matrix with random values.

    // **Note**: The QR method requires the matrix (A) to have linearly independent
    // columns. With a random initialization, there is a (pretty rare) chance that
    // this may not be satisfied.
    for (int q = 0; q < N * N; q++)
    {
        // m[q] = (double) rand() / RAND_MAX * 100;
        m[q] = (double) 10 / sqrt(q + 1);
    }
}

/*
    Common variable naming scheme used in all kernels:

    `column_index`:
    The index of the column in the array. The first column will have
    `column_index` = 0, and the last one will have `column_index` = N - 1.
    This is used when only a single column is used in the kernel.

    `previous_column_index` and `current_column_index`: "Previous" denotes the column vector
    **on** which the projections are being made, and "current" denotes the column
    vector **of** which the projections are being taken.

    `element_in_current_column_index`:
    The index of the element in the current column. This is also the index in the previous
    column, because all calculations are between corresponding elements!
    This will usually be the same as the thread ID.
*/

__global__ void innerprod_self(double *m, double *result, int column_index)
{
    // Calculate the 2-norm of column vector of a matrix (m), and store it in result.

    __shared__ double single_squares[N];
    int element_in_column_index = threadIdx.x;
    single_squares[element_in_column_index] = pow(m[column_index + element_in_column_index * N], 2);

    // Proceed only after all threads have completed above calculations.
    __syncthreads();

    if (element_in_column_index == 0)
    {
        // There is nothing special about `element_in_column_index` = 0. We only want to make
        // sure that we do the following calculation only once.

        double temp = 0;
        for (int e = 0; e < N; e++)
        {
            temp = temp + single_squares[e];
        }

        // Store the result.
        result[0] = sqrt(temp);
    }
}

__global__ void scale(double *m, double *val, int column_index)
{
    // Used to normalize a column vector, by passing the norm of it in `val`.
    // Will divide all elements of the column vector with `val`.

    int element_in_column_index = threadIdx.x;
    m[column_index + element_in_column_index * N] = m[column_index + element_in_column_index * N] / val[0];
}

__global__ void  calculate_coefficients(double *m, double *coefficients, int previous_column_index)
{
    /*
        Calculates the dot products of previous_column with each of the columns of the
        matrix after the current column, and stores them in the the `coefficients` array.
        Naturally, the `coefficients` array is of length (N - previous_column_index).

        Note that, because the `previous_column` has already been normalized, these
        dot products are also the lengths of the projections of the column vectors after
        the `previous_column` on the `previous_columns`. Hence, they will be used as coefficients
        to multiply the `previous_column` with, to get the projection vectors, which we will be
        later subtracting from the column vectors after the `previous_column`.
    */
    __shared__ double prod[N];


    // When the previous column index is `w`, we will need to calculate (N-w coefficients), and will have
    // assigned one block to calculation of each coefficient. The blockID is the index of the current column
    // vector **relative to the previous column vector**. So, the column just next to the `previous_column`
    // will have blockIdx = 0, and so on. So, we need to calculate the absolute index of the current column
    // separately.
    int current_column_relative_index = blockIdx.x;
    int current_column_absolute_index = current_column_relative_index + previous_column_index + 1;

    int element_in_current_column_index = threadIdx.x;
    prod[element_in_current_column_index] = m[previous_column_index + element_in_current_column_index * N] * m[current_column_absolute_index + element_in_current_column_index * N]; // (element of previous col vector) * (element of current column vector)
    __syncthreads();

    if (element_in_current_column_index == 0)
    {
        double temp;
        temp = 0;
        for (int e = 0; e < N; e++)
        {
            temp = temp + prod[e];
        }
        coefficients[current_column_relative_index] = temp;
    }
}

__global__ void subtract_projections(double *m, double *coefficients, int previous_column_index)
{
    // Given the projection lengths (coefficients) and the (normalized) previous column,
    // subtract the projections.
    int element_in_current_column_index = threadIdx.x;
    int current_column_relative_index = blockIdx.x;
    int current_column_absolute_index = current_column_relative_index + previous_column_index + 1;

    m[current_column_absolute_index + element_in_current_column_index * N] = m[current_column_absolute_index + element_in_current_column_index * N] - coefficients[current_column_relative_index] * m[previous_column_index + element_in_current_column_index * N];
}

__global__ void multiply_transpose(double *q, double *a, double *r)
{
    // Stores Q_transpose * A in R

    int block = blockIdx.x;
    int thread = threadIdx.x;

    r[block * N + thread] = 0;
    for (int i = 0; i < N; i++)
    {
        // Multiply column of Q with another column of A, which has the same effect
        // as multiplying Q_transpose and A.
        r[block * N + thread] = r[block * N + thread] + q[i * N + block] * a[i * N + thread];
    }
}

int main(void)
{
    // Allocate memory and initialize the matrix A on the host (CPU).
    double *A_host = (double *)malloc(N * N * sizeof(double));
    double *Q_host = (double *)malloc(N * N * sizeof(double));
    double *R_host = (double *)malloc(N * N * sizeof(double));
    initialize_matrix(A_host);

    // Store A on the device.
    double *A_device;
    cudaMalloc((void **)&A_device, (N * N) * sizeof(double));
    cudaMemcpy(A_device, A_host, (N * N) * sizeof(double), cudaMemcpyHostToDevice);


    // Allocate memory for Q in device (GPU).
    double *Q_device;
    cudaMalloc((void **)&Q_device, (N * N) * sizeof(double));
    // Initialize Q on the device to A. After all operations, it will be transformed
    // into a orthonormal matrix.
    cudaMemcpy(Q_device, A_host, (N * N) * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory for R on device.
    double *R_device;
    cudaMalloc((void **)&R_device, (N * N) * sizeof(double));

    // This is a temporary variable which will be used to store the norms of previous vectors.
    double *norm_device;
    cudaMalloc((void **)&norm_device, sizeof(double));

    // 3.. 2.. 1.. GO!
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int w = 1; w < N; w++)
    {
        int column_index = w - 1;

        // Calculate norm of previous column vector.
        innerprod_self<<<1, N>>>(Q_device, norm_device, column_index);

        // Normalize the previous column vector.
        scale<<<1, N>>>(Q_device, norm_device, column_index);

        double *coefficients_device;
        int number_of_coefficients = N - w;
        cudaMalloc((void **)&coefficients_device, (N - w) * sizeof(double));

        // Calculate coefficients (projection lengths) for columns w to N-1.
        calculate_coefficients<<<number_of_coefficients, N>>>(Q_device, coefficients_device, column_index);

        // Subtract projections of the previous column vectors from columns w to N-1.
        subtract_projections<<<number_of_coefficients, N>>>(Q_device, coefficients_device, column_index);

        // Free memory.
        cudaFree(coefficients_device);
    }

    // Normalize the last column, because the loop didn't do that.
    innerprod_self<<<1, N>>>(Q_device, norm_device, (N - 1));
    scale<<<1, N>>>(Q_device, norm_device, (N - 1));


    // Copy Q from device to host.
    cudaMemcpy(Q_host, Q_device, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate R
    multiply_transpose<<<N, N>>>(Q_device, A_device, R_device);
    // Copy R from device to host.
    cudaMemcpy(R_host, R_device, N * N * sizeof(double), cudaMemcpyDeviceToHost);


    // Time up!
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_delta = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    // Comment this out for large N's.
    cout<<endl<<"Input matrix (A)"<<endl;
    print_matrix(A_host);
    cout<<endl<<"Output matrix (Q)"<<endl;
    print_matrix(Q_host);
    cout<<endl<<"Output matrix (R)"<<endl;
    print_matrix(R_host);

    // Print execution time.
    cout<<endl<<endl<<"Time taken for decomposition (nanoseconds): "<<time_delta.count()<<"\n";

    // Free memory.
    cudaFree(A_device);
    cudaFree(Q_device);
    cudaFree(R_device);
    cudaFree(norm_device);
    free(A_host);
    free(Q_host);
    free(R_host);

    return 0;
}
