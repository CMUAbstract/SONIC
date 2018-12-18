OBJECTS = main.o
EXEC = mnist

DEPS = libmspbuiltins libfixed libmat libdnn libalpaca libio libmsp

ifeq ($(BACKEND), tails)
DEPS += libdsp libmspdriver
endif

include tools/maker/Makefile
