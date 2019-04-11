CC = mpicc

CFLAGS = -g -Wall

LDFLAGS =

OBJECTS = main.o fdiff.o utils.o

TARGET = cg_fd

all: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $^ $(LDFLAGS)

.PHONY: clean

clean:
	$(RM) $(OBJECTS) $(TARGET)
