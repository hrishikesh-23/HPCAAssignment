// Create other necessary functions here
#include <vector>
// Fill in this function

__global__ void redMatMultOnGpu(int *a, int *b, int *c, int N) {
    
    int rowIndexOfC = blockIdx.x * blockDim.x + threadIdx.x;
    int colIndexOfC = blockIdx.y * blockDim.y + threadIdx.y;

    int row1OfA = rowIndexOfC*2;
    int row2OfA = row1OfA + 1;

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

    dim3 threadsPerBlock(sizeOfBlock,sizeOfBlock);

    int numberOfBlocks = (N>>1)/16;
    dim3 blocksPerGrid(numberOfBlocks,numberOfBlocks);
    
    redMatMultOnGpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(output, d_c, ouputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
// void gpuThread(int N, int *matA, int *matB, int *output)
// {
//     int matrixSize = N*N*sizeof(int);
//     int ouputSize = matrixSize>>2;

//     int *d_a, *d_b, *d_c;
//     cudaMalloc(&d_a, matrixSize);
//     cudaMalloc(&d_b, matrixSize);
//     cudaMalloc(&d_c, ouputSize);
  
//     cudaMemcpy(d_a, matA, matrixSize, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, matB, matrixSize, cudaMemcpyHostToDevice);

//     int NoOfThreadsPerBlock;
//     if(N<512){
//         NoOfThreadsPerBlock = (N>>1);
//     }else{
//         NoOfThreadsPerBlock = 256;
//     }


//     dim3 threadsPerBlock(NoOfThreadsPerBlock);
//     dim3 blocksPerGrid((N>>1)/NoOfThreadsPerBlock);
    
//     redMatMultOnGpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

//     cudaMemcpy(matC, d_c, ouputSize, cudaMemcpyDeviceToHost);

//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
// }

// __global__ void redMatMultOnGpu(int *a, int *b, int *c, int N) {
    
//     int rowIndexOfC = blockIdx.x * blockDim.x + threadId.x;

//     int row1OfA = rowIndexOfC*2;
//     int row2OfA = row1OfA + 1;

//     for(int i=0;i<N;i++){
//         int matASum = a[row1OfA*N + i] + a[row2OfA*N + i];
//         int rowOfB = i;
//         for(int j=0;j<N;j+=2){
//            c[rowIndexOfC*(N>>1) + (j>>1)] += (matASum * b[rowOfB*N + j]) + (matASum * b[rowOfB*N + j+1]);
//         }
//     }
//   }



// void gpuThread(int N, int *matA, int *matB, int *output)
// {
//     int matrixSize = N*N*sizeof(int);
//     int ouputSize = matrixSize>>2;

//     int *d_a, *d_b, *d_c;
//     cudaMalloc(&d_a, matrixSize);
//     cudaMalloc(&d_b, matrixSize);
//     cudaMalloc(&d_c, ouputSize);
  
//     cudaMemcpy(d_a, matA, matrixSize, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, matB, matrixSize, cudaMemcpyHostToDevice);

//     int NoOfThreadsPerBlock;
//     if(N<512){
//         NoOfThreadsPerBlock = N;
//     }else{
//         NoOfThreadsPerBlock = 512;
//     }


//     dim3 threadsPerBlock(NoOfThreadsPerBlock);
//     dim3 blocksPerGrid((N>>1), N/NoOfThreadsPerBlock);
    
//     redMatMultOnGpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

//     cudaMemcpy(matC, d_c, ouputSize, cudaMemcpyDeviceToHost);

//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
// }

// __global__ void redMatMultOnGpu(int *a, int *b, int *c, int N) {
//     //blockIdx.x represents rows of matA
//     int rowA = blockIdx.x *2;
//     int colA = (blockIdx.y*blockDim.y) + threadIdx.x;
//     int matAElement1Index = (rowA * N) + colA;
//     int matAElement2Index = ((rowA+1) * N) + colA;

//     //colA is the row of Mat B with which we want to multiply matAElements

//     int matAElement = a[matAElement1Index] + a[matAElement2Index];
//     for(int i=0;i<N;i+=2){
//         c[(blockIdx.x*(N>>1)+i/2] += matAElement * (b[(colA * N)+i] + b[(colA * N)+i+1]);
//     }
    

//   }
