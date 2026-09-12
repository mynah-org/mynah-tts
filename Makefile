CC ?= cc
AR ?= ar
BUILD_DIR ?= build/cpu
CPPFLAGS ?= -Isrc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O3 -ffast-math -fno-finite-math-only
LDFLAGS ?=
LDLIBS ?= -lm -lpthread
BLAS ?= auto
SIMD ?= auto
CUDA_ARCH ?= native
SPEAKER ?= 0
MAX_STEPS ?= 0
BENCH_TEXT ?= h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m
BENCH_WARMUP ?= 2
BENCH_RUNS ?= 5
BENCH_OUTPUT ?= build/bench.wav

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Architecture flags: -march=native on macOS and Linux ARM (like mynah/qwen-tts);
# portable -mavx2 -mfma on Linux x86 (override with SIMD=scalar/avx512).
ifeq ($(SIMD),scalar)
CFLAGS += -DMYNAH_DISABLE_SIMD
SIMD_NAME := scalar
else ifeq ($(SIMD),avx2)
CFLAGS += -mavx2 -mfma
SIMD_NAME := avx2/fma
else ifeq ($(SIMD),avx512)
CFLAGS += -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma
SIMD_NAME := avx512
else ifeq ($(SIMD),neon)
SIMD_NAME := neon
else
# auto: -march=native on macOS/ARM, -mavx2 -mfma on x86 Linux
ifeq ($(UNAME_S),Darwin)
CFLAGS += -march=native
SIMD_NAME := native
else ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
CFLAGS += -march=native
SIMD_NAME := native/arm
else
CFLAGS += -mavx2 -mfma
SIMD_NAME := avx2/fma
endif
endif

# BLAS selection. Four values, and only the first one links nothing:
#
#   none      our own f32 GEMM (src/sgemm.c) at every call site. NO external
#             BLAS in the process, which is the whole point: OpenBLAS brings
#             its own thread pool with its own policies and every one of them
#             is a trap to recheck on every host forever (.work/no-blas.md).
#   openblas  Linux vendor BLAS -- a COMPARISON build, kept so the A/B is
#             always available.
#   auto      the current default: Accelerate on macOS, OpenBLAS on Linux when
#             cblas.h is present, scalar otherwise. Also a comparison build.
#   scalar    no GEMM at all: the naive triple loop and the SEANet scalar conv
#             reference. The correctness oracle, never the performance target.
#
# The default is deliberately NOT `none` yet. Landing the kernel and flipping
# the default are two decisions, and the second needs an RTF measurement on
# Linux ARM and x86 that cannot be taken on a development Mac.
ifeq ($(BLAS),none)
CPPFLAGS += -DMYNAH_USE_OWN_SGEMM
BLAS_NAME := none/mynah-sgemm
ifneq ($(UNAME_S),Darwin)
CPPFLAGS += -D_DEFAULT_SOURCE
endif
else
ifeq ($(UNAME_S),Darwin)
ifeq ($(BLAS),scalar)
BLAS_NAME := scalar
else
CPPFLAGS += -DMYNAH_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
LDLIBS += -framework Accelerate
BLAS_NAME := Accelerate
endif
else
# Linux: -D_DEFAULT_SOURCE exposes POSIX/BSD APIs (clock_gettime, strcasecmp, mmap…)
CPPFLAGS += -D_DEFAULT_SOURCE
ifeq ($(BLAS),openblas)
CPPFLAGS += -DMYNAH_USE_OPENBLAS
LDLIBS += -lopenblas
BLAS_NAME := OpenBLAS
else ifeq ($(BLAS),scalar)
BLAS_NAME := scalar
else
# auto: detect system OpenBLAS (fail-early hint like mynah ASR)
ifneq ($(shell printf '\043include <cblas.h>\n' | $(CC) $(CPPFLAGS) -E -xc - >/dev/null 2>&1 && echo ok),)
CPPFLAGS += -DMYNAH_USE_OPENBLAS
LDLIBS += -lopenblas
BLAS_NAME := OpenBLAS
else
BLAS_NAME := scalar
endif
endif
endif
endif

# ingot: the GGUF/safetensors reader, vendored as a subtree. Built by its own
# Makefile so this one never learns how it is compiled.
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
CPPFLAGS += -I$(INGOT_DIR)/include
LDLIBS += $(INGOT_LIB)

# The dispatch report prints the SIMD profile and git revision it was built
# with; without these it honestly says "unset" rather than guessing.
CPPFLAGS += -DMYNAH_SIMD_PROFILE='"$(SIMD)"' -DMYNAH_GIT_REV='"$(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)"'

CORE_SOURCES := src/mynah_tts.c src/weights.c src/mynah_util.c src/conv1d.c src/codec_nanocodec.c src/flow_head.c src/seanet.c src/transformer_ar.c src/voice_clone.c src/engine_magpie.c src/engine_magpie_ctx.c src/engine_pocket.c src/engine_registry.c src/inference.c src/kernels.c src/sgemm.c src/audio.c src/backend.c src/threads.c src/qmat.c src/tokenizer.c src/tokenizer_sentencepiece.c src/dispatch.c src/costmap.c
CLI_SOURCE := cli/main.c
CORE_OBJECTS := $(CORE_SOURCES:%.c=$(BUILD_DIR)/%.o)
CLI_OBJECT := $(CLI_SOURCE:%.c=$(BUILD_DIR)/%.o)
TARGET := $(BUILD_DIR)/mynah-tts
LIBRARY := $(BUILD_DIR)/libmynah_tts.a
STREAM_TEST_OBJECT := $(BUILD_DIR)/tests/test_stream.o
STREAM_TEST_TARGET := $(BUILD_DIR)/tests/test_stream
# The driver's own policies -- decode gang, failure blast radius, quantum ramp
# -- are tested against a synthetic engine, so this one needs no model pack and
# runs inside `make test` (and therefore inside ubsan/asan).
DRIVER_TEST_OBJECT := $(BUILD_DIR)/tests/test_driver.o
DRIVER_TEST_TARGET := $(BUILD_DIR)/tests/test_driver

# The transformer_ar sliding window past `context: 250` (PLAN.md E2-4). Needs no
# model pack and no oracle -- synthetic weights, an independent f64 reference of
# the contract, and two structural invariants -- so it runs inside `make test`
# and therefore inside ubsan/asan. Detail: .work/transformer-ar-sliding-window.md
WINDOW_TEST_OBJECT := $(BUILD_DIR)/tests/test_transformer_ar_window.o
WINDOW_TEST_TARGET := $(BUILD_DIR)/tests/test_transformer_ar_window

.PHONY: all cpu info caps self-test test stream-test driver-test window-test server server-test server-multilang-test \
	server-concurrency-test server-concurrency-test-all bench bench-matrix gen-matrix inspect convert convert-codec tokenizer synthesize oracle \
        oracle-pocket fake-pack goldens goldens-capture tokenizer-parity convert-pocket \
        playback-sim-test serving-profile serving-wave serving-soak serving-quantum-sweep \
        metal cuda gpu-selftest leaks ubsan asan clean lib shared install dist update-ingot

all: $(TARGET)
cpu: all

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib

$(CORE_OBJECTS): | $(INGOT_LIB)

$(TARGET): $(CORE_OBJECTS) $(CLI_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(LIBRARY): $(CORE_OBJECTS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

$(STREAM_TEST_TARGET): $(CORE_OBJECTS) $(STREAM_TEST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

stream-test: $(STREAM_TEST_TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make stream-test MODEL_DIR=pack" >&2; exit 2)
	@$(STREAM_TEST_TARGET) "$(MODEL_DIR)"

$(DRIVER_TEST_TARGET): $(CORE_OBJECTS) $(DRIVER_TEST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

driver-test: $(DRIVER_TEST_TARGET)
	@$(DRIVER_TEST_TARGET)

$(WINDOW_TEST_TARGET): $(CORE_OBJECTS) $(WINDOW_TEST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

window-test: $(WINDOW_TEST_TARGET)
	@$(WINDOW_TEST_TARGET)

SERVER_SOURCES := server/main.c server/http_util.c server/stream_out.c server/prefork.c
SERVER_OBJECTS := $(SERVER_SOURCES:%.c=$(BUILD_DIR)/%.o)
SERVER_TARGET := $(BUILD_DIR)/mynah-tts-server

$(SERVER_OBJECTS): $(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -Iserver $(CFLAGS) -MMD -MP -c $< -o $@

$(SERVER_TARGET): $(CORE_OBJECTS) $(SERVER_OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

server: $(SERVER_TARGET)
	@echo "server ready: $(SERVER_TARGET)"

server-test: $(SERVER_TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make server-test MODEL_DIR=pack" >&2; exit 2)
	@MODEL_DIR="$(MODEL_DIR)" SERVER="$(SERVER_TARGET)" sh tests/test_server.sh

# Multi-language serving (E5-9). MODEL_DIR_B is optional: without it the script
# fabricates a second pack from the first and says which checks that weakens.
# Defaults for the E5-8 concurrency gate (see tests/test_server_concurrency.sh).
# A port of its own so it cannot collide with a server-test left running.
CONC_PORT ?= 8987
PREFORK ?= 4
# Its own step cap: the top-of-file MAX_STEPS defaults to 0 ("no cap") for
# `make synthesize`, and 0 here would turn a correctness gate into a long run.
CONC_MAX_STEPS ?= 48
CONC_LEVELS ?= 2 4 8

# PLAN.md E5-8. Correctness only -- no wall-clock assertion anywhere in it, so
# unlike the `batching` check in tests/test_server.sh it cannot go flaky on a
# busy machine. SERVER_ARGS passes topology through ("--prefork 4"), or point
# SERVER at tests/prefork_server.sh for the same thing.
server-concurrency-test: $(SERVER_TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make server-concurrency-test MODEL_DIR=models/fake-magpie [SERVER_ARGS=--prefork 4]" >&2; exit 2)
	@MODEL_DIR="$(MODEL_DIR)" SERVER="$(SERVER_TARGET)" SERVER_ARGS="$(SERVER_ARGS)" \
	  LEVELS="$(CONC_LEVELS)" MAX_STEPS="$(CONC_MAX_STEPS)" PORT="$(CONC_PORT)" \
	  sh tests/test_server_concurrency.sh

# Both topologies, one command: single process, then a 4-worker prefork pool.
# A request is byte-identical to itself whether one scheduler batched it with
# its neighbours or a router handed it to a private process, or the claim is
# not about the request.
server-concurrency-test-all: $(SERVER_TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make server-concurrency-test-all MODEL_DIR=models/fake-magpie" >&2; exit 2)
	@$(MAKE) --no-print-directory server-concurrency-test MODEL_DIR="$(MODEL_DIR)"
	@$(MAKE) --no-print-directory server-concurrency-test MODEL_DIR="$(MODEL_DIR)" \
	  SERVER_ARGS="--prefork $(PREFORK)"

server-multilang-test: $(SERVER_TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make server-multilang-test MODEL_DIR=pack [MODEL_DIR_B=pack]" >&2; exit 2)
	@MODEL_DIR="$(MODEL_DIR)" MODEL_DIR_B="$(MODEL_DIR_B)" SERVER="$(SERVER_TARGET)" sh tests/test_server_multilang.sh

lib: $(LIBRARY)
shared: $(TARGET)
	@echo "shared-library packaging is not enabled in the v1 CPU slice"

info:
	@printf 'OS=%s\nARCH=%s\nCC=%s\nSIMD=%s\nBLAS=%s\nMETAL=%s\nCUDA=%s\n' "$$(uname -s)" "$$(uname -m)" "$(CC)" "$(SIMD_NAME)" "$(BLAS_NAME)" "$$(command -v metal 2>/dev/null || echo unavailable)" "$$(command -v nvcc 2>/dev/null || echo unavailable)"

caps: $(TARGET)
	@$(TARGET) --version; $(TARGET) --self-test

self-test: $(TARGET)
	@$(TARGET) --self-test

test: self-test driver-test window-test playback-sim-test
	@python3 tests/test_python_tools.py
	@if test -n "$(MODEL_DIR)"; then $(TARGET) --inspect "$(MODEL_DIR)"; fi

bench: self-test
	@test -n "$(MODEL_DIR)" || (echo "usage: make bench MODEL_DIR=pack [QUANT=int8|int4] [MYNAH_THREADS=N]" >&2; exit 2)
	@MYNAH_QUANT="$(QUANT)" $(TARGET) --synthesize "$(MODEL_DIR)" \
		--normalized "$(BENCH_TEXT)" --output "$(BENCH_OUTPUT)" --speaker 4 \
		--max-steps 20 --seed 42 --warmup "$(BENCH_WARMUP)" --runs "$(BENCH_RUNS)"

bench-matrix: self-test
	@test -n "$(MODEL_DIR)" || (echo "usage: make bench-matrix MODEL_DIR=pack [MYNAH_THREADS=N]" >&2; exit 2)
	@for quant in f32 int8 int4; do \
		if test "$$quant" = f32; then qenv=""; else qenv="$$quant"; fi; \
		$(MAKE) --no-print-directory bench MODEL_DIR="$(MODEL_DIR)" QUANT="$$qenv" \
			BENCH_OUTPUT="build/bench-$$quant.wav" BENCH_WARMUP="$(BENCH_WARMUP)" \
			BENCH_RUNS="$(BENCH_RUNS)"; \
	done

gen-matrix: $(TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make gen-matrix MODEL_DIR=pack" >&2; exit 2)
	@mkdir -p build/gen-matrix
	@for entry in \
		"en:hello world" "fr:bonjour le monde" "it:ciao mondo" \
		"es:hola mundo" "de:hallo Welt" "pt:olá mundo" \
		"vi:xin chào thế giới" "ko:안녕하세요 세계" \
		"ja:こんにちは世界" "zh:你好世界" \
		"hi:नमस्ते दुनिया" "ar:مرحبا بالعالم"; do \
		lang="$${entry%%:*}"; text="$${entry#*:}"; \
		echo "  $$lang: $$text"; \
		MYNAH_THREADS=1 MYNAH_QUANT=int8 $(TARGET) --synthesize "$(MODEL_DIR)" \
			--text "$$text" --lang "$$lang" \
			--output "build/gen-matrix/$$lang.wav" \
			--speaker 4 --max-steps 30 --seed 42 --temperature 0 || exit 1; \
	done
	@echo "Done: build/gen-matrix/*.wav"

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

# Serving measurement. playback-sim-test needs no model and no network, so it
# belongs in `make test`; serving-profile starts a real server.
# NOTE on the synthetic pack: its max_decoder_steps is 8, which is 0.37 s of
# audio, and a profile of that has no cadence to measure and correctly reports
# INCONCLUSIVE. --max-steps 64 gives it something to measure.
LEVELS ?= 1,2,4
WAVES ?= 3
PROFILE_ARGS ?= --max-steps 64
PORT ?= 8123
playback-sim-test:
	python3 tests/playback_sim.py

serving-profile: serving-wave

# WAVE is a SCREEN: C requests fired at t=0, repeated. It may disqualify a
# concurrency level; it may never promote one. Cheap, seconds to minutes.
serving-wave:
	@test -n "$(MODEL_DIR)" || (echo "usage: make serving-wave MODEL_DIR=models/pocket-en [LEVELS=1,2,4,8]" >&2; exit 2)
	python3 tools/serving_profile.py --mode wave --model "$(MODEL_DIR)" --port "$(PORT)" \
	  --levels "$(LEVELS)" --waves "$(WAVES)" $(PROFILE_ARGS)

# SOAK is a QUALIFICATION: minutes at fixed concurrency, warm-up discarded, a
# drift gate across windows. The ONLY mode that may promote a configuration.
SOAK_SECONDS ?= 300
SOAK_WARMUP ?= 30
SOAK_WINDOW ?= 60
serving-soak:
	@test -n "$(MODEL_DIR)" || (echo "usage: make serving-soak MODEL_DIR=models/pocket-en [LEVELS=4] [SOAK_SECONDS=300]" >&2; exit 2)
	python3 tools/serving_profile.py --mode soak --model "$(MODEL_DIR)" --port "$(PORT)" \
	  --levels "$(LEVELS)" --soak-seconds "$(SOAK_SECONDS)" \
	  --warmup-seconds "$(SOAK_WARMUP)" --window-seconds "$(SOAK_WINDOW)" $(PROFILE_ARGS)

# The decoder emit quantum is a model.json parameter (audio_emit_frames), so each
# arm is a pack variant and a server restart. Arms run interleaved (A B .. B A)
# because this machine drifts. Chosen on prebuffer and stall, never on RTF.
QUANTA ?= 1,2,4,8
SWEEP_LEVEL ?= 4
SWEEP_REPEATS ?= 2
serving-quantum-sweep:
	@test -n "$(MODEL_DIR)" || (echo "usage: make serving-quantum-sweep MODEL_DIR=models/pocket-en [QUANTA=1,2,4,8] [SWEEP_LEVEL=4]" >&2; exit 2)
	python3 tools/serving_profile.py --model "$(MODEL_DIR)" --port "$(PORT)" \
	  --levels "$(SWEEP_LEVEL)" --quantum-sweep "$(QUANTA)" \
	  --repeats "$(SWEEP_REPEATS)" $(PROFILE_ARGS)

# PocketTTS model pack. Needs the gated Kyutai weights in the HF cache; see
# .work/licensing-and-voice-policy.md before redistributing what this produces.
POCKET_LANG ?= english
POCKET_OUT ?= models/pocket-$(POCKET_LANG)
convert-pocket:
	uv run --with numpy python tools/convert_pocket.py \
	  --language "$(POCKET_LANG)" --output "$(POCKET_OUT)"

# SentencePiece parity against the Python oracle. Generate the cases first with
# `uv run --with sentencepiece python tools/oracle_pocket_tokenizer.py`.
SP_CASES_DIR ?= build/oracle-tokenizer
SP_TEST := $(BUILD_DIR)/tests/test_tokenizer_sp
$(SP_TEST): tests/test_tokenizer_sp.c $(CORE_OBJECTS) | $(INGOT_LIB)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $(BUILD_DIR)/tests/test_tokenizer_sp.o
	$(CC) $(CFLAGS) $(CORE_OBJECTS) $(BUILD_DIR)/tests/test_tokenizer_sp.o $(LDFLAGS) $(LDLIBS) -o $@

tokenizer-parity: $(SP_TEST)
	@test -d "$(SP_CASES_DIR)" || (echo "missing $(SP_CASES_DIR); run: uv run --with sentencepiece python tools/oracle_pocket_tokenizer.py" >&2; exit 2)
	@set -e; for f in $(SP_CASES_DIR)/*.jsonl; do \
	  lang=$$(basename $$f .jsonl); \
	  model=$$(ls -d $$HOME/.cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/languages/$$lang/tokenizer.model 2>/dev/null | head -1); \
	  test -n "$$model" || (echo "no tokenizer.model for $$lang" >&2; exit 2); \
	  $(SP_TEST) "$$model" "$$f"; \
	done

# Synthetic Magpie-shaped pack and the refactor goldens it exists for.
# See .work/engine-seam-refactor.md: models/ is empty and graph.c's self-tests
# are no-ops off Accelerate, so without this nothing guards the E1 split.
FAKE_PACK ?= models/fake-magpie
fake-pack:
	uv run --with numpy --with safetensors python tools/make_fake_pack.py --output "$(FAKE_PACK)"

goldens-capture: $(TARGET)
	tests/refactor_goldens.sh capture "$(FAKE_PACK)"

goldens: $(TARGET)
	tests/refactor_goldens.sh verify "$(FAKE_PACK)"

# PocketTTS oracle. Offline tooling only: `uv` pulls torch into a throwaway
# environment, nothing here is needed to run the binary. The gated Kyutai weights
# require `hf auth login` first.
ORACLE_POCKET_OUT ?= build/oracle-pocket
ORACLE_POCKET_LANG ?= english
ORACLE_POCKET_VOICE ?= alba
oracle-pocket:
	uv run --with pocket-tts --with numpy --with scipy python tools/oracle_pocket.py \
	  --language "$(ORACLE_POCKET_LANG)" --voice "$(ORACLE_POCKET_VOICE)" \
	  --out "$(ORACLE_POCKET_OUT)" --wav "$(ORACLE_POCKET_OUT)/reference.wav"

METAL_BUILD_DIR := build/metal
# The GPU variants define their own flags rather than inheriting CPPFLAGS, so
# the ingot include path has to be repeated here. The library itself arrives
# through LDLIBS, which they do share.
METAL_CPPFLAGS := -Isrc -I$(INGOT_DIR)/include -DMYNAH_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
METAL_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O3 -ffast-math -fno-finite-math-only -DMYNAH_ENABLE_METAL
METAL_CORE_OBJECTS := $(CORE_SOURCES:%.c=$(METAL_BUILD_DIR)/%.o)
$(METAL_CORE_OBJECTS): | $(INGOT_LIB)
METAL_CLI_OBJECT := $(METAL_BUILD_DIR)/cli/main.o
METAL_HOST_OBJECT := $(METAL_BUILD_DIR)/gpu/metal/backend_metal.o
METAL_OPS_OBJECT := $(METAL_BUILD_DIR)/gpu/metal/backend_metal_ops.o
METAL_TARGET := $(METAL_BUILD_DIR)/mynah-tts

$(METAL_BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(METAL_CPPFLAGS) $(METAL_CFLAGS) -MMD -MP -c $< -o $@

$(METAL_BUILD_DIR)/gpu/metal/backend_metal.o: gpu/metal/backend_metal.m
	@mkdir -p $(@D)
	$(CC) $(METAL_CPPFLAGS) $(METAL_CFLAGS) -fobjc-arc -c $< -o $@

$(METAL_BUILD_DIR)/gpu/metal/backend_metal_ops.o: gpu/metal/backend_metal_ops.m
	@mkdir -p $(@D)
	$(CC) $(METAL_CPPFLAGS) $(METAL_CFLAGS) -fobjc-arc -c $< -o $@

$(METAL_TARGET): $(METAL_CORE_OBJECTS) $(METAL_CLI_OBJECT) $(METAL_HOST_OBJECT) $(METAL_OPS_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(METAL_CFLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -framework Foundation -framework Metal -framework MetalPerformanceShaders -o $@

ifeq ($(shell uname -s),Darwin)
metal: $(METAL_TARGET)
	@echo "Metal build ready: $(METAL_TARGET)"
else
metal:
	@echo "Metal is only available on macOS" >&2
	@exit 2
endif

CUDA_BUILD_DIR := build/cuda
CUDA_CPPFLAGS := -Isrc -I$(INGOT_DIR)/include
CUDA_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O2 -DMYNAH_ENABLE_CUDA
ifneq ($(UNAME_S),Darwin)
CUDA_CPPFLAGS += -D_DEFAULT_SOURCE
endif
CUDA_CORE_OBJECTS := $(CORE_SOURCES:%.c=$(CUDA_BUILD_DIR)/%.o)
$(CUDA_CORE_OBJECTS): | $(INGOT_LIB)
CUDA_CLI_OBJECT := $(CUDA_BUILD_DIR)/cli/main.o
CUDA_HOST_OBJECT := $(CUDA_BUILD_DIR)/gpu/cuda/backend_cuda.o
CUDA_TARGET := $(CUDA_BUILD_DIR)/mynah-tts

$(CUDA_BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CUDA_CPPFLAGS) $(CUDA_CFLAGS) -MMD -MP -c $< -o $@

ifeq ($(CUDA_ARCH),native)
CUDA_ARCH_FLAGS := -arch=native
else
CUDA_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

$(CUDA_BUILD_DIR)/gpu/cuda/backend_cuda.o: gpu/cuda/backend_cuda.cu
	@mkdir -p $(@D)
	@command -v nvcc >/dev/null 2>&1 || (echo "nvcc is required for CUDA; install the NVIDIA CUDA toolkit" >&2; exit 2)
	nvcc -Isrc -O2 $(CUDA_ARCH_FLAGS) -Xcompiler "-Wall,-Wextra" -c $< -o $@

$(CUDA_TARGET): $(CUDA_CORE_OBJECTS) $(CUDA_CLI_OBJECT) $(CUDA_HOST_OBJECT)
	@mkdir -p $(@D)
	nvcc $(CUDA_ARCH_FLAGS) $(filter %.o,$^) -lm -lcublas -o $@

cuda: $(CUDA_TARGET)
	@echo "CUDA build ready: $(CUDA_TARGET)"

gpu-selftest:
	@if test "$(DEVICE)" = "cuda"; then $(MAKE) cuda && build/cuda/mynah-tts --gpu-self-test cuda; \
	elif test "$(DEVICE)" = "metal" || test -z "$(DEVICE)"; then $(MAKE) metal && build/metal/mynah-tts --gpu-self-test metal; \
	else echo "usage: make gpu-selftest [DEVICE=metal|cuda]" >&2; exit 2; fi

leaks:
ifeq ($(UNAME_S),Darwin)
	@command -v leaks >/dev/null 2>&1 || (echo "macOS leaks tool is unavailable" >&2; exit 2)
	@$(MAKE) BUILD_DIR=$(SAN_DIR)/leaks-native CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g' $(SAN_DIR)/leaks-native/mynah-tts
	@leaks --atExit -- $(SAN_DIR)/leaks-native/mynah-tts --self-test
else
	@echo "make leaks is macOS-only; use make asan on Linux" >&2
	@exit 2
endif

# The sanitizer build directory carries the BLAS name.  It did not, and that
# was a FALSE GATE: `make ubsan` and `make ubsan BLAS=none` wrote the same
# build/ubsan, make found the objects up to date, nothing recompiled, and the
# second run re-tested the first one's binary -- Accelerate still linked, with
# a green "PASSED" on top. Discovered while qualifying E4-16, on the first run
# that needed the two configurations to differ.
SAN_DIR := build/san-$(subst /,-,$(BLAS_NAME))

ubsan:
	@$(MAKE) BUILD_DIR=$(SAN_DIR)/ubsan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=undefined' LDFLAGS='-fsanitize=undefined' test

asan:
	@$(MAKE) BUILD_DIR=$(SAN_DIR)/asan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address' LDFLAGS='-fsanitize=address' test

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
	@# Without this, libingot.a survives a clean: update the subtree and the
	@# next build silently links the previous library.
	@test -d $(INGOT_DIR) && $(MAKE) -C $(INGOT_DIR) clean || true

# Refresh the vendored ingot subtree from upstream. A plain clone already
# contains ingot (subtree = real files in-tree, nothing to init); this is only
# needed to pick up new upstream commits. Requires a clean working tree.
update-ingot:
	git subtree pull --prefix $(INGOT_DIR) https://github.com/mynah-org/ingot.git main --squash
	@$(MAKE) -C $(INGOT_DIR) clean

# SERVER_OBJECTS was missing here, and the omission is not cosmetic: the server
# objects are compiled with -MMD -MP and write their .d files, but nothing read
# them, so editing server/*.h never rebuilt server/*.o. A widened struct in
# prefork.h therefore produced a binary in which main.o still used the OLD
# layout -- it memset the first five fields and left the rest as stack garbage,
# which reached the admission ladder as nonsense defaults. Same class as the
# mixed-binary trap in .work/linux-production.md: objects reused across a change
# that altered their meaning.
-include $(CORE_OBJECTS:.o=.d) $(SERVER_OBJECTS:.o=.d) $(CLI_OBJECT:.o=.d) $(STREAM_TEST_OBJECT:.o=.d) $(DRIVER_TEST_OBJECT:.o=.d) $(WINDOW_TEST_OBJECT:.o=.d)
