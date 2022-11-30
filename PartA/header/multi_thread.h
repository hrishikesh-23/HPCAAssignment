#include <pthread.h>

//Following structure is required to pass parameters to each instance of the thread.
struct single_thread_Input
{
    int start_row;// corresponds to the starting row where thread should start computing
    int End_row;// corresponds to the last row where thread should stop computing
};

//Defining the matrices to be global so that all threads can access and work on them in parallel.
int *globalMatA;
int *globalMatB;
int *globalOutput;
int N;

void *threadExecution(void *args)
{ 
    struct single_thread_Input *threadExecutionInput = (struct single_thread_Input *)args;
    
    int start, end;
    start = threadExecutionInput->start_row;
    end = threadExecutionInput->End_row;
    int startRowOfMatA = start*2;
    int endRowOfMatA = end*2;
    __m256i matAElement,rowiterOfMatB1,rowiterOfMatB2,resultRow,currentValue;
    for(int rowA = startRowOfMatA; rowA < endRowOfMatA; rowA +=2) {
        for(int iter = 0;iter<N ; iter++){
            //Store the adddition of element [rowA,iter] and [rowA+1,iter] from matA
            matAElement = _mm256_set1_epi32(globalMatA[rowA*N + iter] + globalMatA[(rowA+1)*N + iter]);
            for(int colB = 0;colB<N;colB+=16){
                rowiterOfMatB1 = _mm256_loadu_si256((__m256i *)&globalMatB[iter*N+colB]);
                rowiterOfMatB2 = _mm256_loadu_si256((__m256i *)&globalMatB[iter*N+colB+8]);
                resultRow = _mm256_hadd_epi32(_mm256_mullo_epi32(matAElement,rowiterOfMatB1),_mm256_mullo_epi32(matAElement,rowiterOfMatB2));
                int outputIndex = (rowA>>1)*(N>>1) + (colB>>1);
                currentValue = _mm256_loadu_si256((__m256i *)&globalOutput[outputIndex]);
                _mm256_storeu_si256((__m256i *)&globalOutput[outputIndex],_mm256_add_epi32(currentValue,resultRow));
            }

        }
    }

    /*
    It was observed that the function _mm256_hadd_epi32 adds elements of two arguments vectors horizontally, but in doing
    so it shuffle the result a bit
    consider two vectors A = [a1 a2 a3 a4 a5 a6 a7 a8]
    and B = [b1 b2 b3 b4 b5 b6 b7 b8]

    then output C is :- [a1+a2 a3+a4 b1+b2 b3+b4 a5+a6 a7+a8 b5+b6 b7+b8]

    To adjust this we are shuffling the elements of the row.
    */
    __m256i shuffleElements =_mm256_setr_epi32(0,1,4,5,2,3,6,7);
    int shuffleStart = start * (N>>1);
    int shuffleEnd = end * (N>>1);
    for(int eachOutputIndex=shuffleStart;eachOutputIndex<shuffleEnd;eachOutputIndex+=8){
        _mm256_storeu_si256((__m256i *)&globalOutput[eachOutputIndex],_mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i *)&globalOutput[eachOutputIndex]), shuffleElements));
    }

    return NULL;
}


// Fill in this function
void multiThread(int n, int *matA, int *matB, int *output)
{
    N = n;
    int NumberOfThreads;
    if(N<64){
        NumberOfThreads = (N>>1);
    }else{
        NumberOfThreads = 64;
    }
    cout<<"Executing multi threaded code for matrix size : "<<N<<". Number of threads : "<<NumberOfThreads<<"\n";
    globalMatA = matA;
    globalMatB = matB;
    globalOutput = output;
    pthread_t threads[NumberOfThreads];

    int rowsPerThread = (N>>1)/NumberOfThreads;
    
    //initialize input to thread
    //here we assign the rows to each thread on which that thread is going to work
    struct single_thread_Input Input[NumberOfThreads];
    int startRow=0;
    for(int i=0;i<NumberOfThreads;i++,startRow+=rowsPerThread)
    {
      Input[i].start_row= startRow;
      Input[i].End_row= startRow+rowsPerThread;
    }

    for (int i = 0; i < NumberOfThreads; i++)
    {
        pthread_create(&threads[i], NULL, threadExecution, (void*)&(Input[i]));
    }

    for (int i = 0; i < NumberOfThreads; i++)
        pthread_join(threads[i], NULL);

}


