CPP = g++
CFLAGS = -g -std=c++23 -Wall -pedantic 

all: client server

clean:
	rm -rf ../bin

client: client.cpp
	$(CPP) $(CFLAGS) -o standard_client client.cpp standardclient.cpp
	mkdir -p ../bin
	mv standard_client ../bin/

server: server.cpp
	$(CPP) $(CFLAGS) -o standard_server server.cpp standardserver.cpp
	mkdir -p ../bin
	mv standard_server ../bin/
