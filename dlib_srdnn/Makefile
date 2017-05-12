CXX=g++ -std=c++11
FILES=dnn_utils.cpp loss_layer.cpp main.cpp
CUDA=/usr/local/cuda
CFLAGS=-O2 -I. -I$(CUDA)/include/ -fpermissive -DDLIB_JPEG_SUPPORT -DDLIB_USE_CUDA -DDLIB_DISABLE_ASSERTS
LDFLAGS=-L. -L$(CUDA)/lib64/ -ldlib -lcudnn -lcuda -lcudart -lpthread -lcublas -lcurand

main: $(FILES)
	$(CXX) $(FILES) $(CFLAGS) $(LDFLAGS) -o main
