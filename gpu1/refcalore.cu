#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../../common/book.h"
#include "../../common/cpu_anim.h"

#define DIM 1024
#define SPEED 0.05f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

// dichiariamo delle referenze a delle textures
// queste dichiarazioni sono globali allo scope del file
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;
// dovremo legare le referenze al buffer di memoria (non e' automatico)
// cudaBindTexture( offset, texture ref, devPtr, size)

struct DataBlock {
  unsigned char  *output_bitmap;
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;
  CPUAnimBitmap *bitmap; // definita in cpu_anim.h
  cudaEvent_t start, stop;
  float TotalTime;
  float frames;

};


__global__ void aggiornoSiti( float *dest,  bool dstOut    ){

  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x + y* blockDim.x *gridDim.x;

  //le coordinate dei siti vicini, lette dal vettore 1D
  int left = offset-1;// dovremo legare le referenze al buffer di memoria (non e' automatico)
// cudaBindTexture( offset, texture ref, devPtr, size)
  int right= offset+1;
  if(x==0) left++;
  if(x==DIM-1) right--;

  int top = offset - DIM;
  int bot = offset + DIM;
  if(y==0) top+=DIM;
  if(y==DIM-1) bot-=DIM;


  float c, t, b, l, r;
  // i valori della temperatura nei siti vicini sono letti tramite le textures

  // se dstOut allora si caricano i valori di input per aggiornare la cella
  // altrimenti si salvano nel buffer, prima della prossima iterazione

  //per utilizzare la memoria delle texture, non valgono le [], ma si ricorre a tex1Dfetch
  //non e' una funzione ma un operatore intrinseco
  //il compilatore deve sapere a compiling time quale array viene letto da tex1Dfetch
  //e quindi le due possibilita' vengono chiarite tramite il bool

  if(dstOut) {
    c=tex1Dfetch(texIn, offset);
    t=tex1Dfetch(texIn, top);
    b=tex1Dfetch(texIn, bot);
    l=tex1Dfetch(texIn, left);
    r=tex1Dfetch(texIn, right);

  } else{
    c=tex1Dfetch(texOut, offset);
    t=tex1Dfetch(texOut, top);
    b=tex1Dfetch(texOut, bot);
    l=tex1Dfetch(texOut, left);
    r=tex1Dfetch(texOut, right);

  }
  // il  valore del sito di riferimento viene aggiornato secondo la regola
  // del gradiente naif

  dest[offset] = c+SPEED * (t+b+l+r-4*c);


    }


// questo kernel ricopia a ogni iterazione il valore costante di quelle celle
// considerate come sorgenti costanti di calore

__global__ void copiaSorgenti( float *iptr ){

  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x + y* blockDim.x *gridDim.x;

  float c = tex1Dfetch(texConstSrc, offset);
  if (c!=0)
    iptr[offset] = c;

  //la texture con le sorgenti ha tutti zero eccetto le sorgenti appunto
  //non sovrascriviamo gli elementi che non sono sorgenti e che sono stati
  //calcolati con la procedura
};


///////////////////////////////////////////////////////////////////////////
// questo kernel induce una sorta di animazione, ovvero calcola diverse
// iterazioni della procedura di aggiornamento che vengono poi visualizzate

void animazione_gpu( DataBlock *d, int ticks){
  cudaEventRecord( d->start, 0);
  float elapsedTime;

  dim3 blocks(DIM/32, DIM/32);
  dim3 threads(32,32);
  CPUAnimBitmap *bitmap = d->bitmap;


  volatile bool dstOut=true;
  for (int i=0; i<90; i++){
    float *in, *out;
    if (dstOut){
      in = d->dev_inSrc;
      out= d->dev_outSrc;

    } else {
      out= d->dev_inSrc;
      in = d->dev_outSrc;

    };

    copiaSorgenti<<<blocks,threads>>>( in );
    aggiornoSiti<<<blocks, threads>>>(out, dstOut);
    dstOut=!dstOut;

  };

  float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc  );

  cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);


  // cudaEventRecord( d->stop, 0);
  // cudaEventSynchronize(d->stop);
  // cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
  // d->TotalTime += elapsedTime;

  // ++d->frames;

//  cout << "tempo per frame=" << d->TotalTime/d->frames;



};


void animazione_exit(DataBlock *d){


  cudaUnbindTexture(texConstSrc);
  cudaUnbindTexture(texIn);
  cudaUnbindTexture(texOut);

  cudaFree( d->dev_inSrc);
  cudaFree( d->dev_outSrc);
  cudaFree( d->dev_constSrc);

  cudaEventDestroy( d->start );
  cudaEventDestroy( d->stop );


};

//////////////////////////////////////////////////////////////////////
int main(void){
  DataBlock data;
  CPUAnimBitmap bitmap(DIM,DIM, &data);
  data.bitmap = &bitmap;
  data.TotalTime=0;
  data.frames=0;

  //  cudaEventCreate(&data.start);
  // cudaEventCreate(&data.stop);
  // float elapsedTime;

  int imageSize = bitmap.image_size();

  cudaMalloc((void **)&data.output_bitmap, imageSize );

  cudaMalloc((void **) &data.dev_inSrc   , imageSize );
  cudaMalloc((void **) &data.dev_outSrc  , imageSize );
  cudaMalloc((void **) &data.dev_constSrc, imageSize );

  cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, imageSize);
  cudaBindTexture(NULL, texIn,       data.dev_inSrc,    imageSize);
  cudaBindTexture(NULL, texOut,      data.dev_outSrc,   imageSize);


    // intialize the constant data
    float *temp = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    cudaMemcpy( data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    cudaMemcpy( data.dev_inSrc, temp, imageSize,cudaMemcpyHostToDevice ) ;
    free( temp );



    bitmap.anim_and_exit( (void (*)(void*,int))animazione_gpu,
                           (void (*)(void*))animazione_exit );

};
