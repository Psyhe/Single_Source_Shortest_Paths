CC := CC     # use mpiCC when not on okeanos
LFLAGS := 
ALL :=  dijkstra

all : $(ALL)

sssp: dijkstra.cpp
	$(CC) -o $@ $<  # -lblas # uncomment this when not on okeanos

clean :
	rm -f $(ALL)