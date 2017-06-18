CXX=g++ -std=c++11
FILES=dnn_utils.cpp loss_layer.cpp main.cpp
CUDA=/usr/local/cuda
CFLAGS=-g -I. -I$(CUDA)/include/ -fpermissive -DDLIB_JPEG_SUPPORT -DDLIB_USE_CUDA -DDLIB_DISABLE_ASSERTS
LDFLAGS=-L. -L$(CUDA)/lib64/ -ldlib -lcudnn -lcuda -lcudart -lpthread -lcublas -lcurand

main: $(FILES) delete_srdnn
	$(CXX) $(FILES) $(CFLAGS) $(LDFLAGS) -o main

delete_srdnn:
	-rm srdnn_sync

