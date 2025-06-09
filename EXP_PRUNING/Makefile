CC := CC     # use mpiCC when not on okeanos
LFLAGS := 
ALL :=  sssp

all : $(ALL)

sssp: reading.cpp
	$(CC) -o $@ $<  # -lblas # uncomment this when not on okeanos

clean :
	rm -f $(ALL)