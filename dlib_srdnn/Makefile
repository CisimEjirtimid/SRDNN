CXX=g++ -std=c++11 -O2
FILES=dnn_utils.cpp loss_layer.cpp main.cpp
CUDA=/usr/local/cuda-8.0
CFLAGS=-I$(CUDA)/include/ -I../argagg/include/ -fpermissive -DDLIB_JPEG_SUPPORT -DDLIB_USE_CUDA -DDLIB_DISABLE_ASSERTS
LDFLAGS=-L$(CUDA)/lib64/ -L/usr/local/lib/ -ldlib -lcudnn -lcuda -lcudart -lpthread -lcublas -lcurand 

OBJS=$(FILES:.cpp=.o)

main: $(OBJS) 
	$(CXX) $(OBJS) $(LDFLAGS) -o main

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $^ -o $@

clean:
	-rm main
	-rm $(OBJS)
