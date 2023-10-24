CC=gcc
MAIN=src/hash.c
CFLAGS=-Wall -Wextra -Werror -pedantic -std=c99 -g
BIN=hash

all: $(BIN)

$(BIN): $(MAIN)
	$(CC) $(CFLAGS) -o $(BIN) $(MAIN)