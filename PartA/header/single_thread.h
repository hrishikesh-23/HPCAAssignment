// Optimize this function
#include <immintrin.h>
//First attempt

void singleThread(int N, int *matA, int *matB, int *output)
{
 __m256i matAElement,rowiterOfMatB1,rowiterOfMatB2,resultRow,currentValue;
 cout<<"Running the optimized code with AVX. Matrix size : "<<N<<"\n";
  assert( N>=4 and N == ( N &~ (N-1)));
  for(int rowA = 0; rowA < N; rowA +=2) {
    for(int iter = 0;iter<N ; iter++){
      //Store the adddition of element [rowA,iter] and [rowA+1,iter] from matA
      matAElement = _mm256_set1_epi32(matA[rowA*N + iter] + matA[(rowA+1)*N + iter]);
      for(int colB = 0;colB<N;colB+=16){
       //bring first 8 columns in the current row
       rowiterOfMatB1 = _mm256_loadu_si256((__m256i *)&matB[iter*N+colB]);
       //bring next 8 columns in the current row
       rowiterOfMatB2 = _mm256_loadu_si256((__m256i *)&matB[iter*N+colB+8]);
       resultRow = _mm256_hadd_epi32(_mm256_mullo_epi32(matAElement,rowiterOfMatB1),_mm256_mullo_epi32(matAElement,rowiterOfMatB2));
       int outputIndex = (rowA>>1)*(N>>1) + (colB>>1);

       //Fetch already stored results and add currently calculated results in it.
       currentValue = _mm256_loadu_si256((__m256i *)&output[outputIndex]);
       _mm256_storeu_si256((__m256i *)&output[outputIndex],_mm256_add_epi32(currentValue,resultRow));
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
  int outputSize = ((N*N)>>2);
  for(int eachOutputIndex=0;eachOutputIndex<outputSize;eachOutputIndex+=8){
      _mm256_storeu_si256((__m256i *)&output[eachOutputIndex],_mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i *)&output[eachOutputIndex]), shuffleElements));
    }

}

