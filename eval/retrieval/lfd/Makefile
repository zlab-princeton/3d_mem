# Headless-first Makefile (GLUT kept for laptop builds)

CC      ?= gcc
CFLAGS  ?= -O2 -std=c11 -Wall
INCLUDES = -I. -I./fftw
MAIN    = 3DAlignment

SRCS := $(wildcard *.c) $(wildcard ./fftw/*.c)
OBJS := $(SRCS:.c=.o)

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))

# Use HEADLESS=1 on Slurm (default). Set HEADLESS=0 on your laptop.
HEADLESS ?= 1

ifeq ($(HEADLESS),1)
    CFLAGS += -DHEADLESS_OSMESA
    # Prefer pkg-config; fall back if not present
    LDLIBS := $(shell pkg-config --libs osmesa glu 2>/dev/null)
    ifeq ($(strip $(LDLIBS)),)
        LDLIBS := -lOSMesa -lGLU -lm
    else
        LDLIBS += -lm
    endif
else
    LDLIBS := -lglut -lGL -lGLU -lm
endif

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: release clean

release: $(MAIN)
	@mkdir -p $(MKFILE_DIR)/../Executable
	mv $(MKFILE_DIR)/$(MAIN) $(MKFILE_DIR)/../Executable

clean:
	$(RM) $(OBJS) $(MAIN)
