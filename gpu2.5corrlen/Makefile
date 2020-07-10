LIBS=-L/usr/local/cuda/lib/ -lcurand
#-lcuda -lm 
#-lcudart
OBJS=
NVCC=nvcc -arch=sm_20 -lm -Wno-deprecated-gpu-targets

INCLUDE=
#-I/home/vicini/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/ -I/home/vicini/NVIDIA_GPU_Computing_SDK/shared/inc/\ -I/usr/local/cuda/include
ECHO=/bin/echo

OBJS=



default: main.x
%.x: %.o
	$(NVCC) $(OBJS) $(LIBS) -lcudart $< -o $@

main.o: main.cu
	$(NVCC) -c $(INCLUDE) -lcudart main.cu

clean:
	rm -f *.x *.o


