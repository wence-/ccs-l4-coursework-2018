CFLAGS := -O3 -march=native -D_GNU_SOURCE -Wall -Wextra
LDFLAGS = -lm
CC = gcc

OBJ = optimised-gemm.o basic-gemm.o

BENCH_OUTPUT = benchmark.dat
BENCH_MIN = 20
BENCH_MAX = 1000
BENCH_STEP = 20

.PHONY: check clean help

all: gemm

help:
	@echo "Available targets are"
	@echo "  clean: Remove all build artifacts"
	@echo "  gemm: Build the gemm binary"
	@echo "  check: Run a simple-minded check of your implementation"
	@echo "  bench: Run a simple benchmark for square matrices over a range of sizes"
	@echo "         WARNING: overwrites the specified output file."
	@echo ""
	@echo "The following make variables are supported"
	@echo "  BENCH_OUTPUT: The output file for benchmark results"
	@echo "  BENCH_MIN: The smallest size to benchmark"
	@echo "  BENCH_MAX: The largest size to benchmark"
	@echo "  BENCH_STEP: The increment when generating sizes"

clean:
	-rm -f gemm $(OBJ)

gemm: gemm.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

check: gemm
	./gemm 10 10 10 CHECK

bench: gemm
	for n in $$(seq $(BENCH_MIN) $(BENCH_STEP) $(BENCH_MAX)); do \
          ./gemm $$n $$n $$n BENCH; \
        done > $(BENCH_OUTPUT)
