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

# Architecture flags.  Six profiles, and `auto` is the only one that asks the
# host anything:
#
#   auto      resolved by tools/simd-auto.sh: the kernel's /proc/cpuinfo AND a
#             compiler capability probe must BOTH agree before a flag is passed,
#             and the resolution is printed (`make simd-auto`).  Until E4-15 this
#             branch appended a fixed `-mavx2 -mfma` to every Linux x86 build
#             without asking, so a host without AVX2 got a binary that dies on
#             its first vpaddd with no message.  -march=native on macOS/ARM.
#   portable  no ISA flag at all: the compiler's baseline for the target.  On
#             x86-64 that leaves __AVX2__ undefined, so src/kernels.c takes its
#             scalar path while src/qmat.c keeps the target-attributed
#             VNNI/F16C kernels it selects by CPUID.  This is NOT SIMD=scalar,
#             and it is exactly the configuration in which the reference tree
#             shipped unlinkable for days -- hence the E4-13 CI matrix.
#   scalar    -DMYNAH_DISABLE_SIMD: every intrinsic compiled out.  The oracle.
#   neon      no flag, but names the profile; aarch64 always has AdvSIMD.
#   avx2      -mavx2 -mfma, the portable x86 release profile.
#   avx512    explicit opt-in.  `auto` will not choose it: no f32 kernel in src/
#             dispatches on AVX-512 (see the isa.x86.avx512f row in
#             src/dispatch.c), so it only widens autovectorization, which is an
#             unmeasured change that also pins the artifact to the build host.
MYNAH_CPUINFO ?= /proc/cpuinfo
SIMD_AUTO_MK := $(BUILD_DIR)/simd-auto.mk

ifeq ($(SIMD),scalar)
CFLAGS += -DMYNAH_DISABLE_SIMD
SIMD_NAME := scalar
else ifeq ($(SIMD),portable)
SIMD_NAME := portable
else ifeq ($(SIMD),avx2)
CFLAGS += -mavx2 -mfma
SIMD_NAME := avx2/fma
else ifeq ($(SIMD),avx512)
CFLAGS += -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma
SIMD_NAME := avx512
else ifeq ($(SIMD),neon)
SIMD_NAME := neon
else ifeq ($(SIMD),auto)
SIMD_AUTO_RUN := $(shell tools/simd-auto.sh --cc '$(CC)' --cpuinfo '$(MYNAH_CPUINFO)' --out '$(SIMD_AUTO_MK)' && echo ok)
-include $(SIMD_AUTO_MK)
CFLAGS += $(SIMD_AUTO_FLAGS)
SIMD_NAME := $(if $(SIMD_AUTO_NAME),$(SIMD_AUTO_NAME),auto/unresolved)
else
$(error SIMD=$(SIMD) is not a profile. Use auto, portable, scalar, neon, avx2 or avx512)
endif

# Appended after the profile, so `make SIMD=portable EXTRA_CFLAGS=-march=armv8-a`
# is a baseline build.  Setting CFLAGS on the command line instead would replace
# it wholesale -- it is `?=` -- and silently drop -std=c11, -O3 and the warning
# flags, which is how a "baseline build passed" result stops meaning anything.
CFLAGS += $(EXTRA_CFLAGS)

# BLAS selection. Five values, and the one Linux resolves by default links
# nothing:
#
#   none      our own f32 GEMM (src/sgemm.c) at every call site. NO external
#             BLAS in the process. THE LINUX DEFAULT since E4-16, and the
#             reason is ownership, not speed: OpenBLAS brings its own thread
#             pool with its own policies, and a pool we do not own is a
#             permanent source of non-reproducible measurement. Measured on
#             the Axion box (.work/no-blas.md §3c): a worker pinned to eight
#             cpus carried 63 threads with OpenBLAS linked and 32 without it,
#             and our own clamp was responsible for 24 of that difference.
#             It is also 9% FASTER on that box, but that is the tiebreak,
#             not the argument.
#   openblas  Linux vendor BLAS -- a COMPARISON build, kept forever so the
#             A/B never stops being available. Never the default again.
#   accelerate  macOS Accelerate -- the macOS default, and a comparison build
#             on the Linux question. It is NOT dropped here: Accelerate also
#             owns vForce's vvtanhf in the GELU and the BNNS conv filter
#             cache, which are a separate numerical qualification and not a
#             GEMM decision (.work/no-blas.md §3b, tier 3).
#   auto      Accelerate on macOS, `none` everywhere else. No probing: the
#             old auto asked the host whether cblas.h existed and silently
#             linked a vendor BLAS if it did, which made the production build
#             a property of the build machine.
#   scalar    no GEMM at all: the naive triple loop and the SEANet scalar conv
#             reference. The correctness oracle, never the performance target.
ifeq ($(UNAME_S),Darwin)
BLAS_RESOLVED := $(if $(filter auto,$(BLAS)),accelerate,$(BLAS))
else
# Linux: -D_DEFAULT_SOURCE exposes POSIX/BSD APIs (clock_gettime, strcasecmp, mmap…)
CPPFLAGS += -D_DEFAULT_SOURCE
BLAS_RESOLVED := $(if $(filter auto,$(BLAS)),none,$(BLAS))
endif

ifeq ($(BLAS_RESOLVED),none)
CPPFLAGS += -DMYNAH_USE_OWN_SGEMM
BLAS_NAME := none/mynah-sgemm
else ifeq ($(BLAS_RESOLVED),accelerate)
ifneq ($(UNAME_S),Darwin)
$(error BLAS=accelerate is macOS-only; this host is $(UNAME_S))
endif
CPPFLAGS += -DMYNAH_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
LDLIBS += -framework Accelerate
BLAS_NAME := Accelerate
else ifeq ($(BLAS_RESOLVED),openblas)
ifeq ($(UNAME_S),Darwin)
$(error BLAS=openblas is the Linux comparison build; this host is Darwin)
endif
CPPFLAGS += -DMYNAH_USE_OPENBLAS
LDLIBS += -lopenblas
BLAS_NAME := OpenBLAS
else ifeq ($(BLAS_RESOLVED),scalar)
BLAS_NAME := scalar
else
$(error BLAS=$(BLAS) is not a profile. Use auto, none, openblas, accelerate or scalar)
endif

# ingot: the GGUF/safetensors reader, vendored as a subtree. Built by its own
# Makefile so this one never learns how it is compiled.
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
CPPFLAGS += -I$(INGOT_DIR)/include
LDLIBS += $(INGOT_LIB)

# The dispatch report prints the SIMD profile and git revision it was built
# with; without these it honestly says "unset" rather than guessing.  The
# profile carries BOTH halves -- what was asked for and what it resolved to --
# because "SIMD=auto" alone is not an answer to "which ISA is this binary".
CPPFLAGS += -DMYNAH_SIMD_PROFILE='"$(SIMD)->$(SIMD_NAME)"' -DMYNAH_GIT_REV='"$(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)"'

# ---------------------------------------------------------------------------
# E4-14: the flag stamp.
#
# `make` then `make SIMD=portable` without a clean reused every object: the
# -march=native qmat.o from the first run was linked into the second binary,
# which then carried native kernels behind a portable dispatch.  It is silent,
# it survives a passing test run, and the only symptom is a SIGILL on another
# host.  So every object depends on a file holding the effective flags, which is
# rewritten only when that text changes -- with `sleep 1`, because GNU make 3.81
# (which is what /usr/bin/make still is on macOS) compares whole seconds.
#
# The git revision is excluded on purpose: it is compiled into exactly one
# string in the dispatch report, and stamping it would rebuild the whole tree on
# every commit.  Quotes and `$` are stripped before the text reaches the shell.
# Scope is the CPU build dir; build/metal and build/cuda have their own fixed
# flags and their own object directories, so they cannot mix with these.
# ---------------------------------------------------------------------------
MYNAH_SQ := '
MYNAH_DQ := "
BUILD_STAMP := $(BUILD_DIR)/.build-flags
STAMP_TEXT := CC=$(CC) SIMD=$(SIMD)/$(SIMD_NAME) BLAS=$(BLAS)/$(BLAS_NAME) CFLAGS=$(CFLAGS) CPPFLAGS=$(filter-out -DMYNAH_GIT_REV=%,$(CPPFLAGS)) LDFLAGS=$(LDFLAGS) LDLIBS=$(LDLIBS)
STAMP_SAFE := $(subst $(MYNAH_DQ),,$(subst $(MYNAH_SQ),,$(subst $$,,$(STAMP_TEXT))))
STAMP_WRITE := $(shell mkdir -p $(BUILD_DIR) && \
	if [ "x$$(cat $(BUILD_STAMP) 2>/dev/null)" != "x$(STAMP_SAFE)" ]; then \
		sleep 1; printf '%s\n' "$(STAMP_SAFE)" > $(BUILD_STAMP); echo rewritten; \
	fi)


CORE_SOURCES := src/mynah_tts.c src/json.c src/weights.c src/mynah_util.c src/conv1d.c src/codec_nanocodec.c src/flow_head.c src/seanet.c src/transformer_ar.c src/voice_clone.c src/engine_magpie.c src/engine_magpie_ctx.c src/engine_pocket.c src/engine_registry.c src/inference.c src/kernels.c src/sgemm.c src/sgemm_rt.c src/convq8.c src/audio.c src/backend.c src/threads.c src/qmat.c src/tokenizer.c src/tokenizer_sentencepiece.c src/dispatch.c src/costmap.c
CLI_SOURCE := cli/main.c
# E14-4.  On x86 src/sgemm.c is built TWICE and src/sgemm_rt.c picks between
# them at runtime; everywhere else it is built once as before.  The reason it
# cannot be a target attribute on the micro-kernel is at the top of sgemm.c:
# SG_LANES reaches the packed panel geometry and the public
# mynah_sgemm_narrow_max(), so the ISA is in the DATA LAYOUT, and an AVX2 kernel
# would be handed panels packed for the baseline shape.
#
# ASKED OF THE COMPILER, NOT OF `uname -m`.  `make x86-cross` builds x86_64
# objects on an arm64 host, and a decision keyed on the host would silently skip
# the variant that cross build exists to produce -- then link, and run, and look
# fine, having tested nothing.
CC_TARGET_X86 := $(shell echo | $(CC) $(CPPFLAGS) $(CFLAGS) -E -dM - 2>/dev/null | grep -cE '__x86_64__|__i386__')
# SIMD=scalar compiles every intrinsic out, so both variants would be the same
# scalar code and one of them would carry -mavx2 for nothing -- an object that
# invites the question "why does the scalar build contain AVX2 flags" and has no
# good answer. One variant there, and the dispatcher picks it.
ifneq ($(findstring -DMYNAH_DISABLE_SIMD,$(CFLAGS)),)
CC_TARGET_X86 := 0
endif
ifeq ($(CC_TARGET_X86),0)
sgemm_objects = $(1)/src/sgemm.o
else
sgemm_objects = $(1)/src/sgemm_base.o $(1)/src/sgemm_avx2.o
endif
sgemm_object_list = $(filter-out $(1)/src/sgemm.o,$(2)) $(call sgemm_objects,$(1))

CORE_OBJECTS := $(call sgemm_object_list,$(BUILD_DIR),$(CORE_SOURCES:%.c=$(BUILD_DIR)/%.o))
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

.PHONY: all cpu info caps simd-auto simd-auto-test self-test test test-c x86-cross stream-test driver-test window-test kernels-test qmat-test qmat-negative-control perf-profile-test ternary-test server server-test server-multilang-test \
	server-concurrency-test server-concurrency-test-all bench bench-matrix gen-matrix inspect convert convert-codec tokenizer synthesize oracle \
        oracle-pocket fake-pack goldens goldens-capture tokenizer-parity convert-pocket \
        playback-sim-test json-test json-negative-control kernels-negative-control serving-profile serving-wave serving-soak serving-quantum-sweep \
        metal cuda gpu-selftest leaks ubsan asan clean lib shared install dist update-ingot \
        doctor census-test census-parity census-overhead alloc-shim alloc-constant-test observability-test

all: $(TARGET)
cpu: all

# Self-healing only: the $(shell) above already wrote this at parse time.  The
# rule exists so a deleted stamp is not a "No rule to make target" error, and it
# must live AFTER `all` so it can never become the default goal.
$(BUILD_STAMP):
	@mkdir -p $(@D) && printf '%s\n' "$(STAMP_SAFE)" > $@

$(BUILD_DIR)/%.o: %.c $(BUILD_STAMP)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

# The two x86 builds of one source.  `base` takes the build's own flags
# unchanged -- on SIMD=portable that is a genuine baseline binary, which is the
# whole point -- and `avx2` ADDS -mavx2 -mfma rather than replacing anything, so
# a build that already carries them is not silently narrowed.  The pattern
# stem is the build directory, so the sanitizer, Metal and CUDA trees each get
# their own pair without repeating the recipe.
%/src/sgemm_base.o: src/sgemm.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMYNAH_SGEMM_VARIANT=base -MMD -MP -c $< -o $@

%/src/sgemm_avx2.o: src/sgemm.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -mavx2 -mfma -DMYNAH_SGEMM_VARIANT=avx2 -MMD -MP -c $< -o $@

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

# The hot kernels, model-free (PLAN.md E3-6) and the numerical qualification
# for the Accelerate replacements (E4-16d). Needs no model pack and no oracle
# -- an independent f64 reference, a ULP table against libm and, on macOS,
# against Accelerate itself -- so it runs inside `make test` and therefore
# inside ubsan/asan. Detail: .work/accelerate-only-kernels.md
KERNELS_TEST_OBJECT := $(BUILD_DIR)/tests/test_kernels.o
KERNELS_TEST_TARGET := $(BUILD_DIR)/tests/test_kernels
QMAT_TEST_OBJECT := $(BUILD_DIR)/tests/test_qmat.o
QMAT_TEST_TARGET := $(BUILD_DIR)/tests/test_qmat
$(KERNELS_TEST_TARGET): $(CORE_OBJECTS) $(KERNELS_TEST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(QMAT_TEST_TARGET): $(CORE_OBJECTS) $(QMAT_TEST_OBJECT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

kernels-test: $(KERNELS_TEST_TARGET)
	@$(KERNELS_TEST_TARGET)

# The negative control: break the kernels nine ways and require kernels-test to
# catch each break. A suite that has only ever passed is not evidence that it
# can fail. Slow (nine clean rebuilds of the core), so it is NOT in `make test`
# -- run it when the kernels or the suite change.
kernels-negative-control:
	@sh tests/kernels_negative_control.sh

# The int8/int4 determinism suite. Model-free and fast, so it is in `make test`
# and therefore inside ubsan and asan.
#
# It runs the binary FIVE TIMES, and the repeats are the point rather than
# belt-and-braces:
#
#   - MYNAH_QMAT_I8MM=0 / =1 force the ARM SMMLA wiring off and on. The suite
#     also flips it in-process, but a run with the env variable set is what a
#     deployment would actually do, and it is the only way the DEFAULT
#     resolution gets exercised as a default.
#   - MYNAH_QMAT_VNNI=scalar forces the PORTABLE unsigned kernel, which is the
#     only way the x+128 / rowsum correction runs on an ARM machine at all.
#   - MYNAH_QMAT_VNNI=256 / =512 ask for the VEX and EVEX VPDPBUSD kernels.
#     On ARM both are unreachable and the run just re-reports the SDOT path,
#     which costs milliseconds. On x86 they are the only way this project
#     observes its VNNI kernels at all -- there is no x86 host in the fleet, so
#     `ubuntu-latest` in .github/workflows/build.yml is the silicon. §1 of the
#     suite prints whether the level RESOLVED, so a runner without the unit
#     reports "NOT RESOLVED" instead of quietly passing on the scalar path and
#     being read later as a VNNI result.
#
# A level the CPU cannot run is not an error: qmat_u8_level() falls back and §1
# says so. This target fails only on a numeric disagreement.
qmat-test: $(QMAT_TEST_TARGET)
	@$(QMAT_TEST_TARGET)
	@MYNAH_QMAT_I8MM=0 $(QMAT_TEST_TARGET)
	@MYNAH_QMAT_I8MM=1 $(QMAT_TEST_TARGET)
	@MYNAH_QMAT_VNNI=scalar $(QMAT_TEST_TARGET)
	@MYNAH_QMAT_VNNI=256 $(QMAT_TEST_TARGET)
	@MYNAH_QMAT_VNNI=512 $(QMAT_TEST_TARGET)

# The codec conv stack's int8 path against the SAME binary with it off, on real
# weights (E10-5). The C self-test proves the kernel on synthetic data; this is
# the only thing that can say what real weights -- which have outliers, unlike
# uniform random data -- do to real audio, and whether the generated FRAME COUNT
# moved. Needs a pack, so it is not in `make test`:
#     make codec-int8-quality MODEL_DIR=models/pocket-en
# Two modes, because there are two trades. `conv` is the conv1d stack, which
# ships on and costs almost nothing measurable. `convtr` is the transposed
# convolutions, which are OFF in the default spec and are one string away
# (`MYNAH_QUANT_GROUPS=...,codec_convtr:int8`): 1.44x more on the region for
# eight decibels on the waveform. Its bounds are its own.
codec-int8-quality: $(TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make codec-int8-quality MODEL_DIR=models/pocket-en" >&2; exit 2)
	@python3 tests/codec_int8_quality.py --binary $(TARGET) --model "$(MODEL_DIR)" --mode conv
	@python3 tests/codec_int8_quality.py --binary $(TARGET) --model "$(MODEL_DIR)" --mode convtr

# The negative control: break the epilogue the four ways it has actually been
# broken, and require qmat-test to catch each one. Slow (rebuilds of the core
# under more than one SIMD profile), so it is NOT in `make test`.
qmat-negative-control:
	@sh tests/qmat_negative_control.sh

# The JSON parser (src/json.c) and the two readers it replaced. Model-free,
# network-free and millisecond-fast, so it runs inside `make test` and therefore
# inside ubsan/asan -- which is where the byte-substitution sweep and the
# truncate-at-every-offset pass earn their keep.
#
# It links only the three translation units it is about, NOT $(CORE_OBJECTS):
# a parser test that needs the whole runtime to build is a parser test nobody
# runs while changing the parser. server/http_util.c comes in because the
# server's wrappers are part of what is under test.
JSON_TEST_SOURCES := tests/test_json.c src/json.c server/http_util.c
JSON_TEST_TARGET := $(BUILD_DIR)/tests/test_json
$(JSON_TEST_TARGET): $(JSON_TEST_SOURCES) src/json.h server/http_util.h $(BUILD_STAMP)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -Iserver $(CFLAGS) $(LDFLAGS) $(JSON_TEST_SOURCES) -lm -lpthread -o $@

json-test: $(JSON_TEST_TARGET)
	@$(JSON_TEST_TARGET)

# The negative control: break the parser four ways and require json-test to
# catch each break. A suite that has only ever passed is not evidence that it
# can fail. Slow-ish (four rebuilds of three files), so it is NOT in `make test`
# -- run it when the parser or the suite changes.
json-negative-control:
	@sh tests/json_negative_control.sh

SERVER_SOURCES := server/main.c server/http_util.c server/stream_out.c server/prefork.c
SERVER_OBJECTS := $(SERVER_SOURCES:%.c=$(BUILD_DIR)/%.o)
SERVER_TARGET := $(BUILD_DIR)/mynah-tts-server

$(SERVER_OBJECTS): $(BUILD_DIR)/%.o: %.c $(BUILD_STAMP)
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
	@printf 'OS=%s\nARCH=%s\nCC=%s\nSIMD=%s (%s)\nBLAS=%s\nMETAL=%s\nCUDA=%s\n' "$$(uname -s)" "$$(uname -m)" "$(CC)" "$(SIMD)" "$(SIMD_NAME)" "$(BLAS_NAME)" "$$(command -v metal 2>/dev/null || echo unavailable)" "$$(command -v nvcc 2>/dev/null || echo unavailable)"
	@printf 'SIMD_FLAGS=%s\n' "$(SIMD_AUTO_FLAGS)"
	@printf 'STAMP=%s\n' "$(BUILD_STAMP)"

# E4-15.  What SIMD=auto resolved on THIS host, and -- as important -- what it
# saw and deliberately did not use.  Print it before quoting any ISA in a
# benchmark note; `--dispatch-map` is the runtime half of the same question.
simd-auto:
	@tools/simd-auto.sh --cc '$(CC)' --cpuinfo '$(MYNAH_CPUINFO)' --human

# The resolution table itself, against captured /proc/cpuinfo from parts this
# project does not own.  Needs no x86 host, no model pack and no compiler for
# the fixture's architecture -- see the header of tests/test_simd_auto.sh for
# what that does and does not prove.
simd-auto-test:
	@sh tests/test_simd_auto.sh

caps: $(TARGET)
	@$(TARGET) --version; $(TARGET) --self-test

# The second run is not a duplicate. MYNAH_QMAT_VNNI=scalar forces the PORTABLE
# UNSIGNED int8 encoding -- x+128 with the -128*rowsum correction, which is what
# VPDPBUSD needs and what the whole x86 half of production runs. On ARM the
# signed SDOT path is the only one that ever resolves, so without this line the
# unsigned algebra is compiled, shipped and never executed here: two mutations
# of it (a dropped rowsum, a broken lane in the four-row unsigned block) pass
# the default run and are caught only by this one. The level exists for exactly
# this and had no gate using it.
self-test: $(TARGET)
	@$(TARGET) --self-test
	@MYNAH_QMAT_VNNI=scalar $(TARGET) --self-test
	@# The third run is the f32 half of the same argument. Since the x86 f32
	@# kernels became runtime-selected, an x86 host runs AVX2 and NEVER the
	@# scalar forms -- which are the numeric reference the AVX2 ones are
	@# written against. MYNAH_KERNELS_X86=scalar forces them, so both halves
	@# are executed on one machine. On aarch64 this is a no-op second run and
	@# costs a second: NEON is compile-time and has no fallback to select.
	@MYNAH_KERNELS_X86=scalar $(TARGET) --self-test

# THE C GATES, and the only ones a sanitizer build can say anything about.
# `make ubsan` and `make asan` run this target, not `test`: the rest of `test`
# is pure Python that the sanitizer never instruments, so a missing Python
# module used to turn the Memory Safety workflow red while saying nothing about
# memory safety. That happened -- numpy, 2026-09-21, four red sanitizer jobs.
test-c: self-test kernels-test qmat-test driver-test window-test json-test simd-auto-test

test: test-c playback-sim-test perf-profile-test ternary-test
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

# The serving profiles under configs/perf: every committed profile validates, and the
# refusals that keep a qualifying run honest actually refuse.
perf-profile-test:
	python3 tests/test_perf_profile.py

# The ternary feasibility analysis (E11) is closed-form solves transcribed from
# papers. A transcription error would not crash and would not obviously corrupt
# audio -- it would quietly make one method look worse than another, which is
# how a wrong number becomes a finding. So the solves are checked against lstsq
# and the orderings the report claims are asserted. No model needed.
# numpy is OFFLINE TOOLING. The repo contract keeps the runtime Python-free, so
# a machine without numpy must not fail `make test` -- but a silent skip would
# turn a green run into a statement about nothing, so the skip names the exact
# command it did not run and how to enable it. CI installs numpy on the jobs
# that run `make test`, which is what keeps this gate from being vacuous there.
ternary-test:
	@if python3 -c 'import numpy' >/dev/null 2>&1; then \
		echo "python3 tools/ternary_feasibility.py self-test"; \
		python3 tools/ternary_feasibility.py self-test; \
	else \
		echo "SKIP ternary-test: numpy is not installed"; \
		echo "     not run: python3 tools/ternary_feasibility.py self-test"; \
		echo "     enable:  python3 -m pip install numpy"; \
	fi

# x86 WITHOUT RENTING ONE. Cross-compiles for x86_64 on an Apple Silicon Mac
# and EXECUTES the result under Rosetta 2 -- which emulates an x86-64 CPU with
# no AVX2, i.e. exactly the old-x86 tier this project has no hardware for. It
# proves the portable binary runs correctly there and that an AVX2 binary
# refuses to start with the ISA guard's message instead of taking a SIGILL.
# It proves nothing about performance: an emulator is not a measurement.
# macOS-only; skips cleanly (exit 0) anywhere else.
x86-cross:
	@sh tests/x86_cross.sh

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
METAL_CORE_OBJECTS := $(call sgemm_object_list,$(METAL_BUILD_DIR),$(CORE_SOURCES:%.c=$(METAL_BUILD_DIR)/%.o))
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
CUDA_CORE_OBJECTS := $(call sgemm_object_list,$(CUDA_BUILD_DIR),$(CORE_SOURCES:%.c=$(CUDA_BUILD_DIR)/%.o))
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

# $(LDLIBS), not a hand-written `-lm`: the CPU and Metal targets both link
# through LDLIBS, and this one spelled its libraries out instead -- so when
# third_party/ingot became a dependency it was added to LDLIBS and this rule
# never saw it. `make cuda` then failed to link on EVERY machine, with or
# without a GPU:
#
#     undefined reference to `ingot_st_open'
#
# Nothing caught it because nothing built the CUDA target anywhere; the
# compile-only CI job found it on its first run.
$(CUDA_TARGET): $(CUDA_CORE_OBJECTS) $(CUDA_CLI_OBJECT) $(CUDA_HOST_OBJECT) | $(INGOT_LIB)
	@mkdir -p $(@D)
	nvcc $(CUDA_ARCH_FLAGS) $(filter %.o,$^) $(LDLIBS) -lcublas -o $@

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
	@$(MAKE) BUILD_DIR=$(SAN_DIR)/ubsan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=undefined' LDFLAGS='-fsanitize=undefined' test-c

asan:
	@$(MAKE) BUILD_DIR=$(SAN_DIR)/asan CFLAGS='-std=c11 -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address' LDFLAGS='-fsanitize=address' test-c

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
-include $(CORE_OBJECTS:.o=.d) $(SERVER_OBJECTS:.o=.d) $(CLI_OBJECT:.o=.d) $(STREAM_TEST_OBJECT:.o=.d) $(DRIVER_TEST_OBJECT:.o=.d) $(WINDOW_TEST_OBJECT:.o=.d) $(QMAT_TEST_OBJECT:.o=.d)


# ======================================================================
# OBSERVABILITY (E4-2b / E4-13)
#
# Four tools, and each one declares a refusal
# (.work/engineering-method.md §4):
#
#   make doctor              what this machine is, what this binary will choose
#                            on it, and how the server should probably be run
#                            here. Under a minute, no model pack. Every value
#                            is labelled MEASURED / CACHED / TRANSFERRED /
#                            PREDICTED / UNKNOWN, and it prints [UNKNOWN]
#                            rather than a plausible number.
#   make census-parity       the proof that the instrumentation does not change
#                            what it measures: same binary, same workload,
#                            census-only vs census+costmap, censuses diffed and
#                            audio compared byte for byte.
#   make census-overhead     what it costs, in interleaved A/B/B/A arms, never
#                            a clean run followed by an instrumented one.
#   make alloc-constant-test the allocation count is CONSTANT across
#                            --max-steps, which is the evidence that the AR
#                            loop allocates nothing.
#
# The cost map and the census themselves are env-driven, so they need no CLI
# surface and no flag to forget:
#
#   MYNAH_COST_MAP=1|2   region profile (2 adds the per-layer regions)
#   MYNAH_CENSUS=1       shape and kernel census
#   *_STRICT=1           exit non-zero instead of printing a number nobody
#                        should trust
# ======================================================================

doctor:
	@python3 tools/doctor.py --binary $(TARGET)

# The census machinery has a model-free self-test, inside
# mynah_dispatch_self_test.  This comment used to say that this target
# exercised it.  It did not: `--dispatch-map` calls mynah_dispatch_report()
# and nothing else, and mynah_dispatch_self_test() had no caller anywhere in
# the tree, so the census self-test, the region-id collision check and the
# thread-pool litmus had never run in any gate.  It is now called from
# `--self-test` (cli/main.c), which the second line below runs.
census-test: $(TARGET)
	@$(TARGET) --dispatch-map >/dev/null
	@echo "census + dispatch self-test: PASS (via --dispatch-map collect)"
	@MYNAH_CENSUS=1 $(TARGET) --self-test >/dev/null
	@echo "census enabled during --self-test: PASS"

census-parity: $(TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make census-parity MODEL_DIR=models/pocket-en" >&2; exit 2)
	@bash tests/census_parity.sh parity $(TARGET) "$(MODEL_DIR)"

census-overhead: $(TARGET)
	@test -n "$(MODEL_DIR)" || (echo "usage: make census-overhead MODEL_DIR=models/pocket-en [REPS=3]" >&2; exit 2)
	@bash tests/census_parity.sh overhead $(TARGET) "$(MODEL_DIR)" $(or $(REPS),3)

# The shim is PRELOADED, never linked into the runtime: libmynah_tts contains
# no allocator hook at all. See tests/alloc_shim.c.
# dlsym lives in libdl on glibc < 2.34 and in libc after it; linking -ldl when
# it exists is harmless, and this is the only object that needs it. The runtime
# link line is deliberately untouched.
ALLOC_SHIM_LIBS := $(shell uname -s | grep -qi darwin || echo -ldl)
ALLOC_SHIM := $(BUILD_DIR)/alloc_shim.so
alloc-shim: $(ALLOC_SHIM)
$(ALLOC_SHIM): tests/alloc_shim.c
	@mkdir -p $(dir $@)
	$(CC) -std=c11 -O2 -fPIC -shared -Wall -Wextra $< -o $@ $(ALLOC_SHIM_LIBS)

alloc-constant-test: $(TARGET) $(ALLOC_SHIM)
	@test -n "$(MODEL_DIR)" || (echo "usage: make alloc-constant-test MODEL_DIR=models/pocket-en" >&2; exit 2)
	@SHIM=$(ALLOC_SHIM) bash tests/census_parity.sh alloc $(TARGET) "$(MODEL_DIR)"

# Everything above that does not need a model pack.
observability-test: census-test
	@python3 tools/doctor.py --binary $(TARGET) >/dev/null
	@echo "doctor: PASS (ran, produced a labelled report)"
