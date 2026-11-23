# GPU-Accelerated Authentication Threat Detection -- manual build.
# run_pipeline.py builds everything itself; use this to build by hand.
#   make                 # sm_75 (Turing: T4, RTX 20xx)
#   make ARCH=sm_86      # e.g. RTX 30xx; sm_80 A100, sm_89 RTX 40xx, sm_90 H100

ARCH  ?= sm_75
NVCC  ?= nvcc
NVFLAGS = -O2 -arch=$(ARCH) -I.
OMP   = -Xcompiler -fopenmp -lgomp
B     = build

DETECTORS = $(B)/auth_lateral $(B)/auth_fanout $(B)/auth_failburst $(B)/auth_score
LEGACY    = $(B)/auth_scan_memtypes $(B)/auth_streams_libs $(B)/cpu_baseline

all: $(DETECTORS)
legacy: $(LEGACY)

$(B):
	mkdir -p $(B)

$(B)/auth_%: auth_%.cu auth_common.cuh | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ $(OMP)

$(B)/auth_scan_memtypes: auth_scan_memtypes.cu | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

$(B)/auth_streams_libs: auth_streams_libs.cu | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ -lcurand

$(B)/cpu_baseline: cpu_baseline.cpp | $(B)
	g++ -O2 -fopenmp $< -o $@

clean:
	rm -rf $(B)

.PHONY: all legacy clean
