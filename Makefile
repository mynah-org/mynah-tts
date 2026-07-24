CC ?= cc
AR ?= ar
BUILD_DIR ?= build/cpu
CPPFLAGS ?= -Isrc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
LDFLAGS ?=
LDLIBS ?= -lm
BLAS ?= auto
SIMD ?= auto
SPEAKER ?= 0
MAX_STEPS ?= 0

ifeq ($(SIMD),scalar)
CFLAGS += -DMYNAH_DISABLE_SIMD
SIMD_NAME := scalar
else ifeq ($(SIMD),avx2)
CFLAGS += -mavx2 -mfma
SIMD_NAME := avx2/fma
else ifeq ($(SIMD),neon)
SIMD_NAME := neon
else
SIMD_NAME := compiler-default
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ifeq ($(BLAS),scalar)
BLAS_NAME := scalar
else
CPPFLAGS += -DMYNAH_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
LDLIBS += -framework Accelerate
BLAS_NAME := Accelerate
endif
else ifeq ($(BLAS),openblas)
CPPFLAGS += -DMYNAH_USE_OPENBLAS
LDLIBS += -lopenblas
BLAS_NAME := OpenBLAS
else ifeq ($(BLAS),auto)
ifneq ($(shell pkg-config --exists openblas 2>/dev/null && echo yes),)
CPPFLAGS += -DMYNAH_USE_OPENBLAS
LDLIBS += $(shell pkg-config --libs openblas)
BLAS_NAME := OpenBLAS
else
BLAS_NAME := scalar
endif
else
BLAS_NAME := scalar
endif

CORE_SOURCES := src/mynah_tts.c src/safetensors.c src/graph.c src/kernels.c src/audio.c src/backend.c src/threads.c src/qmat.c
CLI_SOURCE := cli/main.c
CORE_OBJECTS := $(CORE_SOURCES:%.c=$(BUILD_DIR)/%.o)
CLI_OBJECT := $(CLI_SOURCE:%.c=$(BUILD_DIR)/%.o)
TARGET := $(BUILD_DIR)/mynah-tts
LIBRARY := $(BUILD_DIR)/libmynah_tts.a

.PHONY: all cpu info caps self-test test bench inspect convert convert-codec tokenizer synthesize oracle \
        metal cuda gpu-selftest leaks ubsan asan clean lib shared install dist

all: $(TARGET)
cpu: all

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(CORE_OBJECTS) $(CLI_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(LIBRARY): $(CORE_OBJECTS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

lib: $(LIBRARY)
shared: $(TARGET)
	@echo "shared-library packaging is not enabled in the v1 CPU slice"

info:
	@printf 'OS=%s\nARCH=%s\nCC=%s\nSIMD=%s\nBLAS=%s\nMETAL=%s\nCUDA=%s\n' "$$(uname -s)" "$$(uname -m)" "$(CC)" "$(SIMD_NAME)" "$(BLAS_NAME)" "$$(command -v metal 2>/dev/null || echo unavailable)" "$$(command -v nvcc 2>/dev/null || echo unavailable)"

caps: $(TARGET)
	@$(TARGET) --version; $(TARGET) --self-test

self-test: $(TARGET)
	@$(TARGET) --self-test

test: self-test
	@python3 tests/test_python_tools.py
	@if test -n "$(MODEL_DIR)"; then $(TARGET) --inspect "$(MODEL_DIR)"; fi

bench: self-test
	@echo "No model benchmark requested; run make test MODEL_DIR=... after conversion."

inspect:
	@test -n "$(MODEL)" || (echo "usage: make inspect MODEL=path/to/model.nemo" >&2; exit 2)
	python3 tools/inspect_nemo.py "$(MODEL)"

convert:
	@test -n "$(MODEL)" || (echo "usage: make convert MODEL=path/to/magpie.nemo" >&2; exit 2)
	python3 tools/convert_magpie.py --tts-archive "$(MODEL)" --codec-archive "$(CODEC)" --output "$(OUTPUT)"

convert-codec:
	@test -n "$(MODEL)" || (echo "usage: make convert-codec MODEL=path/to/codec.nemo" >&2; exit 2)
	python3 tools/convert_magpie.py --codec-only --codec-archive "$(MODEL)" --output "$(OUTPUT)"

tokenizer:
	@test -n "$(MODEL)" || (echo "usage: make tokenizer MODEL=magpie.nemo CODEC=codec.nemo BYT5=tokenizer OUTPUT=pack/tokenizer/english_phoneme.tsv" >&2; exit 2)
	.venv/bin/python tools/export_magpie_tokenizer.py --archive "$(MODEL)" --codec "$(CODEC)" --byt5-tokenizer "$(BYT5)" --output "$(OUTPUT)"

synthesize: $(TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make synthesize MODEL_DIR=pack TEXT='h|ə|...' OUTPUT=out.wav" >&2; exit 2)
	@test -n "$(TEXT)" || (echo "usage: make synthesize MODEL_DIR=pack TEXT='h|ə|...' OUTPUT=out.wav" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "usage: make synthesize MODEL_DIR=pack TEXT='h|ə|...' OUTPUT=out.wav" >&2; exit 2)
	$(TARGET) --synthesize "$(MODEL_DIR)" --normalized "$(TEXT)" --output "$(OUTPUT)" --speaker "$(SPEAKER)" --max-steps "$(MAX_STEPS)"

oracle:
	@test -n "$(MODEL)" || (echo "usage: make oracle MODEL=magpie.nemo CODEC=codec.nemo OUTPUT=oracle.wav" >&2; exit 2)
	@test -n "$(CODEC)" || (echo "usage: make oracle MODEL=magpie.nemo CODEC=codec.nemo OUTPUT=oracle.wav" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "usage: make oracle MODEL=magpie.nemo CODEC=codec.nemo OUTPUT=oracle.wav" >&2; exit 2)
	.venv/bin/python tools/oracle_magpie.py --archive "$(MODEL)" --codec "$(CODEC)" --byt5-tokenizer "$(BYT5)" --output "$(OUTPUT)"

METAL_BUILD_DIR := build/metal
METAL_CPPFLAGS := -Isrc
METAL_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2 -DMYNAH_ENABLE_METAL
METAL_CORE_OBJECTS := $(CORE_SOURCES:%.c=$(METAL_BUILD_DIR)/%.o)
METAL_CLI_OBJECT := $(METAL_BUILD_DIR)/cli/main.o
METAL_HOST_OBJECT := $(METAL_BUILD_DIR)/gpu/metal/backend_metal.o
METAL_TARGET := $(METAL_BUILD_DIR)/mynah-tts

$(METAL_BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(METAL_CPPFLAGS) $(METAL_CFLAGS) -MMD -MP -c $< -o $@

$(METAL_BUILD_DIR)/gpu/metal/backend_metal.o: gpu/metal/backend_metal.m
	@mkdir -p $(@D)
	$(CC) $(METAL_CPPFLAGS) $(METAL_CFLAGS) -fobjc-arc -c $< -o $@

$(METAL_TARGET): $(METAL_CORE_OBJECTS) $(METAL_CLI_OBJECT) $(METAL_HOST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(METAL_CFLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -framework Foundation -framework Metal -o $@

ifeq ($(shell uname -s),Darwin)
metal: $(METAL_TARGET)
	@echo "Metal build ready: $(METAL_TARGET)"
else
metal:
	@echo "Metal is only available on macOS" >&2
	@exit 2
endif

CUDA_BUILD_DIR := build/cuda
CUDA_CPPFLAGS := -Isrc
CUDA_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2 -DMYNAH_ENABLE_CUDA
CUDA_CORE_OBJECTS := $(CORE_SOURCES:%.c=$(CUDA_BUILD_DIR)/%.o)
CUDA_CLI_OBJECT := $(CUDA_BUILD_DIR)/cli/main.o
CUDA_HOST_OBJECT := $(CUDA_BUILD_DIR)/gpu/cuda/backend_cuda.o
CUDA_TARGET := $(CUDA_BUILD_DIR)/mynah-tts

$(CUDA_BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CUDA_CPPFLAGS) $(CUDA_CFLAGS) -MMD -MP -c $< -o $@

$(CUDA_BUILD_DIR)/gpu/cuda/backend_cuda.o: gpu/cuda/backend_cuda.cu
	@mkdir -p $(@D)
	@command -v nvcc >/dev/null 2>&1 || (echo "nvcc is required for CUDA; install the NVIDIA CUDA toolkit" >&2; exit 2)
	nvcc -Isrc -O2 -Xcompiler "-Wall,-Wextra" -c $< -o $@

$(CUDA_TARGET): $(CUDA_CORE_OBJECTS) $(CUDA_CLI_OBJECT) $(CUDA_HOST_OBJECT)
	@mkdir -p $(@D)
	nvcc $(filter %.o,$^) -lm -o $@

cuda: $(CUDA_TARGET)
	@echo "CUDA build ready: $(CUDA_TARGET)"

gpu-selftest:
	@if test "$(DEVICE)" = "cuda"; then $(MAKE) cuda && build/cuda/mynah-tts --gpu-self-test cuda; \
	elif test "$(DEVICE)" = "metal" || test -z "$(DEVICE)"; then $(MAKE) metal && build/metal/mynah-tts --gpu-self-test metal; \
	else echo "usage: make gpu-selftest [DEVICE=metal|cuda]" >&2; exit 2; fi

leaks:
	@$(MAKE) BUILD_DIR=build/leaks CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined' LDFLAGS='-fsanitize=address,undefined' test

ubsan:
	@$(MAKE) BUILD_DIR=build/ubsan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=undefined' LDFLAGS='-fsanitize=undefined' test

asan:
	@$(MAKE) BUILD_DIR=build/asan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address' LDFLAGS='-fsanitize=address' test

install: $(TARGET) $(LIBRARY)
	@test -n "$(PREFIX)" || (echo "usage: make install PREFIX=/path" >&2; exit 2)
	mkdir -p "$(PREFIX)/bin" "$(PREFIX)/lib" "$(PREFIX)/include"
	cp $(TARGET) "$(PREFIX)/bin/mynah-tts"
	cp $(LIBRARY) "$(PREFIX)/lib/libmynah_tts.a"
	cp src/mynah_tts.h "$(PREFIX)/include/mynah_tts.h"

dist:
	@echo "dist is source-only in v1; model weights are never included"

clean:
	rm -rf build

-include $(CORE_OBJECTS:.o=.d) $(CLI_OBJECT:.o=.d)
