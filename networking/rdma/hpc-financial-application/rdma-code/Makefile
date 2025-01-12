CPP = g++
CFLAGS = -g -std=c++23 -Wall -pedantic 
VFLAGS = -lrdmacm -libverbs

all: client server

clean:
	rm -rf ../bin

client: client.cpp
	$(CPP) $(CFLAGS) -o rdma_client client.cpp rdmaclient.cpp $(VFLAGS)
	mkdir -p ../bin
	mv rdma_client ../bin/

server: server.cpp
	$(CPP) $(CFLAGS) -o rdma_server server.cpp rdmaserver.cpp $(VFLAGS)
	mkdir -p ../bin
	mv rdma_server ../bin/
