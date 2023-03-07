OBJ += fp32
OBJ += int8
OBJ += fp8


FLAGS = -std=c++11 -lstdc++ -lcublas -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -res-usage -lcublasLt -lcudart -lcufft -lineinfo -Xcompiler -fopenmp
#FLAGS = -std=c++11 -lcublas -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp
#ifdef DEBUG
FLAGS += -g -G -keep
#endif

all : $(OBJ)

fp32 : fp32.cpp cublasLt_Ltgemm_fp32.cu
	nvcc $^ -o $@ $(FLAGS)

int8 : int8.cpp cublasLt_Ltgemm_int8.cu
	nvcc $^ -o $@ $(FLAGS)

fp8 : fp8.cpp cublasLt_Ltgemm_fp8.cu
	nvcc $^ -o $@ $(FLAGS)

.PHONY : clean

clean :
	rm -f $(OBJ)
	rm -f *dlink*
	rm -f *compute_*
	rm -f *.ii
	rm -f *fatbin*
	rm -f *module*
	rm -f *.o
	rm -f *cudafe*
	rm -f *.ptx
	rm -f *.cubin

#make -j16