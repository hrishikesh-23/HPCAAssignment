#include <pthread.h>

// Create other necessary functions here
#define MAX_THREAD 16  //number of thread to run
//params to pass in threads
struct single_thread_Input
{
    int start_row;
    int End_row;
};
int *matA;
int *matB;
int *output;
int N;

//multithread function

void *multiThread(void *args)
{ 
    struct single_thread_Input *In = (struct single_thread_Input *)args;
    
    int start, end;
    start = In->start_row;
    end = In->End_row;
    assert(N >= 4 and N == (N & ~(N - 1)));
    for (int rowA = start; rowA < end; rowA += 2)
    {
        for (int colB = 0; colB < N; colB += 2)
        {
            int sum = 0;
            for (int iter = 0; iter < N; iter++)
            {
                sum += matA[rowA * N + iter] * matB[iter * N + colB];
                sum += matA[(rowA + 1) * N + iter] * matB[iter * N + colB];
                sum += matA[rowA * N + iter] * matB[iter * N + (colB + 1)];
                sum += matA[(rowA + 1) * N + iter] * matB[iter * N + (colB + 1)];
            }

            // compute output indices
            int rowC = rowA >> 1;
            int colC = colB >> 1;
            int indexC = rowC * (N >> 1) + colC;
            output[indexC] = sum;
        }
    }
    return NULL;
}

// Fill in this function
void multiThread(int n, int *A, int *B, int *o)
{
    N = n;
    matA = A;
    matB = B;
    output = o;
     pthread_t threads[MAX_THREAD];

    int size ;
    size = N/MAX_THREAD; //size of a thread (no of rows it compute)
    
    //initialize input to thread
    struct single_thread_Input Input[MAX_THREAD];
    int beg=0;
    for(int i=0;i<MAX_THREAD;i++,beg+=size)
    {
      Input[i].start_row= beg;
      Input[i].End_row= beg+size;

    }

    for (int i = 0; i < MAX_THREAD; i++)
    {
        pthread_create(&threads[i], NULL, multiThread, (void*)&(Input[i]));
     }

    for (int i = 0; i < MAX_THREAD; i++)
        pthread_join(threads[i], NULL);

  }
