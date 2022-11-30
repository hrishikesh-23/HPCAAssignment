// Create other necessary functions here
#include <vector>
// Fill in this function

__global__ void redMatMultOnGpu(int *a, int *b, int *c, int N) {
    
    int rowIndexOfC = blockIdx.x * blockDim.x + threadIdx.x;
    int colIndexOfC = blockIdx.y * blockDim.y + threadIdx.y;

    //Each element in RMM is computed from two rows of A and two columns of B.
    //here we get the two rows for the current thread.
    int row1OfA = rowIndexOfC*2;
    int row2OfA = row1OfA + 1;

    //here we get two columns for the current thread.
    int col1OfB = colIndexOfC*2;
    int col2OfB = col1OfB+1;

    int sum = 0;

    for(int i=0;i<N;i++){
        sum += a[row1OfA*N + i]*b[i*N + col1OfB];
    }

    for(int i=0;i<N;i++){
        sum += a[row1OfA*N + i]*b[i*N + col2OfB];
    }

    for(int i=0;i<N;i++){
        sum += a[row2OfA*N + i]*b[i*N + col1OfB];
    }

    for(int i=0;i<N;i++){
        sum += a[row2OfA*N + i]*b[i*N + col2OfB];
    }

    c[rowIndexOfC*(N>>1) + colIndexOfC] = sum;
  }


void gpuThread(int N, int *matA, int *matB, int *output)
{
    int matrixSize = N*N*sizeof(int);
    int ouputSize = matrixSize>>2;

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, matrixSize);
    cudaMalloc(&d_c, ouputSize);
  
    cudaMemcpy(d_a, matA, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, matB, matrixSize, cudaMemcpyHostToDevice);

    int sizeOfBlock = 16;

    //With the following configuration we are going to create total of (N*N/4) threads.
    //each thread will ultimately compute on element of the ouput matrix.
    dim3 threadsPerBlock(sizeOfBlock,sizeOfBlock);

    int numberOfBlocks = (N>>1)/16;
    dim3 blocksPerGrid(numberOfBlocks,numberOfBlocks);
    
    redMatMultOnGpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(output, d_c, ouputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
