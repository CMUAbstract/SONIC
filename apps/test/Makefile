OBJECTS = main.o
EXEC = test

DEPS = libmspbuiltins libfixed libmat libdnn libalpaca libio libmsp

ifeq ($(BACKEND), tails)
DEPS += libdsp libmspdriver
endif

include tools/maker/Makefile
