#include <iostream>
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <chrono>

using namespace std;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()

int main(){
    int N=8;
    cout<<"Vector size : "<<N;

    int* A = new int[N];
    int* B = new int[N];
    int* C = new int[N];

    for(int i=0;i<N;i++){
        A[i] = rand()%256;
        B[i] = rand()%256;
    }

    __m256i rowiterOfMatB1,rowiterOfMatB2,result;
    rowiterOfMatB1 = _mm256_loadu_si256((__m256i *)&A[0]);
    rowiterOfMatB2 = _mm256_loadu_si256((__m256i *)&B[0]);
    result = _mm256_hadd_epi32(rowiterOfMatB1,rowiterOfMatB2);
    _mm256_storeu_si256((__m256i *)&C[0],result);
    
    cout<<"Vector A is : \n";
    for(int i=0;i<N;i++){
        cout<<A[i]<<" ";
    }
    cout<<"\n";

    cout<<"Vector B is : \n";
    for(int i=0;i<N;i++){
        cout<<B[i]<<" ";
    }
    cout<<"\n";

    cout<<"Vector C is : \n";
    for(int i=0;i<N;i++){
        cout<<C[i]<<" ";
    }
    cout<<"\n";


    // auto begin = TIME_NOW;
    // int* res = new int[N];
    // for(int i=0;i<N;i++){
    //     res[i] = A[i]+B[i];
    // }
    // auto end = TIME_NOW;
    // cout << "Reference execution time: " << 
    //     (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";

    // // __m256i* a = (__m256i*)A;
    // // __m256i* b = (__m256i*)B;
    // int* ans = new int[N];

    //  auto beginV = TIME_NOW;


    // for(int j=0;j<N;j+=8){
    //     __m256i tempa = _mm256_loadu_si256((__m256i *)&A[j]);
    //     __m256i tempb = _mm256_loadu_si256((__m256i *)&B[j]);
    //     __m256i temp = _mm256_add_epi32(tempa,tempb);
    //     _mm256_storeu_si256((__m256i *)&ans[j],temp); 
    // }

    // auto endV = TIME_NOW;
    // cout << "Reference execution time with AVX : " << 
    //     (double)TIME_DIFF(std::chrono::microseconds, beginV, endV) / 1000.0 << " ms\n";

    // for(int i=0;i<N;i++){
    //     if(res[i]!=ans[i]){
    //         cout<<"Mismatch at index : "<<i<<"Result = : "<<res[i]<<" Answer : "<<ans[i];
    //         break;
    //     }
    // }
    return 0;

}