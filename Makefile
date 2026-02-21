# Makefile for FPGA Design Optimization Agent

# Configuration
PYTHON := python3
PIP := $(PYTHON) -m pip

# Vivado executable - can be overridden with: make setup VIVADO_EXEC=/path/to/vivado
VIVADO_EXEC ?= vivado
export VIVADO_EXEC

# Set JAVA_HOME from PATH or Vivado if not already set
# Python RapidWright may need JAVA_HOME to be set, but often users only have `java` on PATH
# See: https://www.rapidwright.io/docs/Install.html#using-java-distributed-with-vivado
ifndef JAVA_HOME
  JAVA_PATH := $(shell command -v java 2>/dev/null)
  ifneq ($(JAVA_PATH),)
    # Found java on PATH - resolve symlinks and derive JAVA_HOME
    # Try readlink -f (Linux), fall back to direct path (macOS/others)
    REAL_JAVA_PATH := $(shell readlink -f "$(JAVA_PATH)" 2>/dev/null || readlink "$(JAVA_PATH)" 2>/dev/null || echo "$(JAVA_PATH)")
    # java is at $JAVA_HOME/bin/java, so go up two directories
    export JAVA_HOME := $(shell dirname $(shell dirname $(REAL_JAVA_PATH)))
  else
    # java not on PATH - try to find Java bundled with Vivado
    # Vivado includes Java at: <VIVADO_ROOT>/tps/lnx64/jre11*/bin/java
    VIVADO_PATH := $(shell command -v $(VIVADO_EXEC) 2>/dev/null)
    ifneq ($(VIVADO_PATH),)
      VIVADO_ROOT := $(shell dirname $(shell dirname $(VIVADO_PATH)))
      VIVADO_JAVA := $(shell ls $(VIVADO_ROOT)/tps/lnx64/jre11*/bin/java 2>/dev/null | head -n 1)
      ifneq ($(VIVADO_JAVA),)
        export JAVA_HOME := $(shell dirname $(shell dirname $(VIVADO_JAVA)))
        export PATH := $(JAVA_HOME)/bin:$(PATH)
      endif
    endif
  endif
endif

# Example DCPs to download
EXAMPLE_DCP_1 := demo_corundum_25g_misses_timing.dcp
EXAMPLE_DCP_2 := logicnets_jscl.dcp
DCP_URL_BASE := http://data.rapidwright.io/example-dcps

# Colors for output
COLOR_GREEN := \033[0;32m
COLOR_YELLOW := \033[0;33m
COLOR_RED := \033[0;31m
COLOR_BLUE := \033[0;34m
COLOR_RESET := \033[0m

.PHONY: setup run_optimizer validate validate_demo run-submission clean veryclean help

# Default target
help:
	@echo "FPGA Design Optimization Agent - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup           - Install dependencies and download example DCPs"
	@echo "  run_optimizer   - Run optimizer on a DCP file (agent or test mode)"
	@echo "  validate        - Validate functional equivalence between two DCPs"
	@echo "  validate_demo   - Run validation demo (self-check)"
	@echo "  clean           - Remove generated files (run directories, logs, Vivado outputs)"
	@echo "  veryclean       - Remove all generated files including example DCPs"
	@echo ""
	@echo "Usage examples:"
	@echo "  make setup"
	@echo "  make setup VIVADO_EXEC=/tools/Xilinx/Vivado/2025.2/bin/vivado"
	@echo "  make run_optimizer DCP=logicnets_jscl.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp VECTORS=50000"
	@echo "  make validate_demo"
	@echo "  make clean"
	@echo ""
	@echo "Environment variables:"
	@echo "  VIVADO_EXEC     - Path to Vivado executable (default: vivado)"
	@echo "  JAVA_HOME       - Java installation directory (auto-detected from PATH if not set)"
	@echo "  DCP             - Input DCP file for run_optimizer target"
	@echo "  GOLDEN          - Golden (reference) DCP for validation"
	@echo "  REVISED         - Revised (optimized) DCP for validation"
	@echo "  VECTORS         - Number of test vectors for validation (default: 10000)"
	@echo ""
	@echo "Output structure:"
	@echo "  - Optimized DCP: <input_name>_optimized-<timestamp>.dcp (next to input)"
	@echo "  - Run directory: dcp_optimizer_run-<timestamp>/ (contains all logs)"
	@echo "  - Validation:    /tmp/dcp_validation_*/ (contains simulation logs)"

# Setup target: Install dependencies, check Vivado, set up Java, download DCPs
setup:
	@printf "$(COLOR_GREEN)===== FPGA Design Optimization Setup =====$(COLOR_RESET)\n"
	@echo ""
	
	@printf "$(COLOR_YELLOW)[1/5] Installing Python dependencies...$(COLOR_RESET)\n"
	$(PIP) install -r requirements.txt
	@printf "$(COLOR_GREEN)✓ Python dependencies installed$(COLOR_RESET)\n"
	@echo ""
	
	@printf "$(COLOR_YELLOW)[2/5] Checking Vivado...$(COLOR_RESET)\n"
	@if command -v $(VIVADO_EXEC) >/dev/null 2>&1; then \
		printf "$(COLOR_GREEN)✓ Vivado found: %s$(COLOR_RESET)\n" "$$(command -v $(VIVADO_EXEC))"; \
		$(VIVADO_EXEC) -version | head -n 1; \
	else \
		printf "$(COLOR_RED)✗ Vivado not found on PATH$(COLOR_RESET)\n"; \
		echo ""; \
		echo "Please either:"; \
		echo "  1. Source Vivado settings: source /path/to/Vivado/*/settings64.sh"; \
		echo "  2. Specify Vivado path: make setup VIVADO_EXEC=/path/to/vivado"; \
		exit 1; \
	fi
	@echo ""
	
	@printf "$(COLOR_YELLOW)[3/5] Checking Java...$(COLOR_RESET)\n"
	@if command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_GREEN)✓ Java found: %s$(COLOR_RESET)\n" "$$(command -v java)"; \
		java -version 2>&1 | head -n 1; \
	else \
		printf "$(COLOR_YELLOW)⚠ Java not found on PATH$(COLOR_RESET)\n"; \
		echo "Attempting to locate Java from Vivado installation..."; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC)); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			VIVADO_JAVA="$$VIVADO_ROOT/tps/lnx64/jre11*/bin/java"; \
			if ls $$VIVADO_JAVA >/dev/null 2>&1; then \
				JAVA_FOUND=$$(ls $$VIVADO_JAVA | head -n 1); \
				printf "$(COLOR_GREEN)✓ Found Java in Vivado: %s$(COLOR_RESET)\n" "$$JAVA_FOUND"; \
				echo ""; \
				printf "$(COLOR_YELLOW)NOTE: Set JAVA_HOME before running optimizer:$(COLOR_RESET)\n"; \
				JAVA_HOME_DIR=$$(dirname $$(dirname $$JAVA_FOUND)); \
				echo "  export JAVA_HOME=$$JAVA_HOME_DIR"; \
				echo "  export PATH=\$$JAVA_HOME/bin:\$$PATH"; \
			else \
				printf "$(COLOR_RED)✗ Could not find Java in Vivado installation$(COLOR_RESET)\n"; \
				echo "Please install Java 11 or later"; \
				exit 1; \
			fi; \
		else \
			printf "$(COLOR_RED)✗ Cannot locate Java$(COLOR_RESET)\n"; \
			echo "Please install Java 11 or later"; \
			exit 1; \
		fi; \
	fi
	@echo ""
	
	@printf "$(COLOR_YELLOW)[4/5] Downloading example DCP: $(EXAMPLE_DCP_1)...$(COLOR_RESET)\n"
	@if [ -f "$(EXAMPLE_DCP_1)" ]; then \
		printf "$(COLOR_GREEN)✓ $(EXAMPLE_DCP_1) already exists$(COLOR_RESET)\n"; \
	else \
		if command -v wget >/dev/null 2>&1; then \
			wget -q --show-progress $(DCP_URL_BASE)/$(EXAMPLE_DCP_1); \
			printf "$(COLOR_GREEN)✓ Downloaded $(EXAMPLE_DCP_1)$(COLOR_RESET)\n"; \
		elif command -v curl >/dev/null 2>&1; then \
			curl -# -O $(DCP_URL_BASE)/$(EXAMPLE_DCP_1); \
			printf "$(COLOR_GREEN)✓ Downloaded $(EXAMPLE_DCP_1)$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_RED)✗ Neither wget nor curl found$(COLOR_RESET)\n"; \
			echo "Please install wget or curl, or manually download:"; \
			echo "  $(DCP_URL_BASE)/$(EXAMPLE_DCP_1)"; \
			exit 1; \
		fi; \
	fi
	@echo ""
	
	@printf "$(COLOR_YELLOW)[5/5] Downloading example DCP: $(EXAMPLE_DCP_2)...$(COLOR_RESET)\n"
	@if [ -f "$(EXAMPLE_DCP_2)" ]; then \
		printf "$(COLOR_GREEN)✓ $(EXAMPLE_DCP_2) already exists$(COLOR_RESET)\n"; \
	else \
		if command -v wget >/dev/null 2>&1; then \
			wget -q --show-progress $(DCP_URL_BASE)/$(EXAMPLE_DCP_2); \
			printf "$(COLOR_GREEN)✓ Downloaded $(EXAMPLE_DCP_2)$(COLOR_RESET)\n"; \
		elif command -v curl >/dev/null 2>&1; then \
			curl -# -O $(DCP_URL_BASE)/$(EXAMPLE_DCP_2); \
			printf "$(COLOR_GREEN)✓ Downloaded $(EXAMPLE_DCP_2)$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_RED)✗ Neither wget nor curl found$(COLOR_RESET)\n"; \
			echo "Please install wget or curl, or manually download:"; \
			echo "  $(DCP_URL_BASE)/$(EXAMPLE_DCP_2)"; \
			exit 1; \
		fi; \
	fi
	@echo ""
	
	@printf "$(COLOR_GREEN)===== Setup Complete! =====$(COLOR_RESET)\n"
	@echo ""
	@echo "Next steps:"
	@echo "  1. If Java was found in Vivado, set JAVA_HOME as shown above"
	@echo "  2. Run the optimizer:"
	@echo "     make run_optimizer DCP=$(EXAMPLE_DCP_1)"
	@echo ""
	@echo "Output will be in:"
	@echo "  - Optimized DCP: <input_name>_optimized-<timestamp>.dcp"
	@echo "  - Run logs: dcp_optimizer_run-<timestamp>/"
	@echo ""

# Run optimizer target: Run dcp_optimizer.py (output DCP name generated automatically)
run_optimizer: 
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_optimizer DCP=input.dcp"; \
		exit 1; \
	fi
	@if [ ! -f "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP file not found: $(DCP)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_GREEN)Running optimizer on $(DCP)...$(COLOR_RESET)\n"
	@# Set up Java from Vivado if Java is not available
	@if ! command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)Java not found on PATH, attempting to use Java from Vivado...$(COLOR_RESET)\n"; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC) 2>/dev/null); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			VIVADO_JAVA="$$VIVADO_ROOT/tps/lnx64/jre11*/bin/java"; \
			if ls $$VIVADO_JAVA >/dev/null 2>&1; then \
				JAVA_FOUND=$$(ls $$VIVADO_JAVA | head -n 1); \
				export JAVA_HOME=$$(dirname $$(dirname $$JAVA_FOUND)); \
				export PATH="$$JAVA_HOME/bin:$$PATH"; \
				printf "$(COLOR_GREEN)Using Java from Vivado: %s$(COLOR_RESET)\n" "$$JAVA_HOME"; \
			fi; \
		fi; \
	fi; \
	echo ""; \
	$(PYTHON) dcp_optimizer.py "$(DCP)" 

# Validation target: Validate functional equivalence between two DCPs
validate:
	@printf "$(COLOR_BLUE)╔══════════════════════════════════════════════════════════════════╗$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)║         DCP Equivalence Validation (2-Phase Approach)           ║$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)╚══════════════════════════════════════════════════════════════════╝$(COLOR_RESET)\n"
	@echo ""
	@# Check if GOLDEN and REVISED are provided
	@if [ -z "$(GOLDEN)" ]; then \
		printf "$(COLOR_RED)✗ Error: GOLDEN DCP not specified$(COLOR_RESET)\n"; \
		echo "Usage: make validate GOLDEN=<golden.dcp> REVISED=<revised.dcp> [VECTORS=10000]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make validate GOLDEN=logicnets_jscl.dcp REVISED=logicnets_jscl_optimized.dcp"; \
		exit 1; \
	fi
	@if [ -z "$(REVISED)" ]; then \
		printf "$(COLOR_RED)✗ Error: REVISED DCP not specified$(COLOR_RESET)\n"; \
		echo "Usage: make validate GOLDEN=<golden.dcp> REVISED=<revised.dcp> [VECTORS=10000]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make validate GOLDEN=logicnets_jscl.dcp REVISED=logicnets_jscl_optimized.dcp"; \
		exit 1; \
	fi
	@# Check if files exist
	@if [ ! -f "$(GOLDEN)" ]; then \
		printf "$(COLOR_RED)✗ Error: Golden DCP not found: $(GOLDEN)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@if [ ! -f "$(REVISED)" ]; then \
		printf "$(COLOR_RED)✗ Error: Revised DCP not found: $(REVISED)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@# Run validation
	@printf "$(COLOR_GREEN)Golden DCP:$(COLOR_RESET)  $(GOLDEN)\n"
	@printf "$(COLOR_GREEN)Revised DCP:$(COLOR_RESET) $(REVISED)\n"
	@printf "$(COLOR_GREEN)Test Vectors:$(COLOR_RESET) $(or $(VECTORS),10000)\n"
	@echo ""
	@if [ -n "$(VECTORS)" ]; then \
		$(PYTHON) validate_dcps.py "$(GOLDEN)" "$(REVISED)" --vectors $(VECTORS); \
	else \
		$(PYTHON) validate_dcps.py "$(GOLDEN)" "$(REVISED)"; \
	fi

# Quick validation example using demo DCPs
validate_demo:
	@printf "$(COLOR_BLUE)╔══════════════════════════════════════════════════════════════════╗$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)║                  Validation Demo (Simulated)                    ║$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)╚══════════════════════════════════════════════════════════════════╝$(COLOR_RESET)\n"
	@echo ""
	@echo "This demo validates a DCP against itself (should always PASS)."
	@echo "For real validation, first optimize a design, then validate:"
	@echo ""
	@echo "  1. python dcp_optimizer.py design.dcp --output design_optimized.dcp"
	@echo "  2. make validate GOLDEN=design.dcp REVISED=design_optimized.dcp"
	@echo ""
	@# Check if example DCP exists
	@if [ ! -f "$(EXAMPLE_DCP_2)" ]; then \
		printf "$(COLOR_YELLOW)Example DCP not found, downloading...$(COLOR_RESET)\n"; \
		$(MAKE) download_dcps; \
	fi
	@# For demo, validate DCP against itself (should always pass)
	@printf "$(COLOR_GREEN)Running demo validation (self-check)...$(COLOR_RESET)\n"
	@echo ""
	$(PYTHON) validate_dcps.py "$(EXAMPLE_DCP_2)" "$(EXAMPLE_DCP_2)" --vectors 1000

run-submission:
	@echo "Running submission...[Will be implemented later]"
	
# Clean target: Remove run directories and Vivado-generated .Xil directories
clean:
	@printf "$(COLOR_YELLOW)Cleaning generated files...$(COLOR_RESET)\n"
	@# Remove run directories (contain all logs, journals, intermediate files)
	@if ls dcp_optimizer_run-* >/dev/null 2>&1; then \
		rm -rf dcp_optimizer_run-*; \
		echo "Removed dcp_optimizer_run-* directories"; \
	fi
	@# Remove .Xil directories (Vivado generates these outside run directories)
	@if [ -d ".Xil" ]; then \
		rm -rf .Xil; \
		echo "Removed .Xil/"; \
	fi
	@if [ -d "VivadoMCP/.Xil" ]; then \
		rm -rf VivadoMCP/.Xil; \
		echo "Removed VivadoMCP/.Xil/"; \
	fi
	@printf "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)\n"
	@echo "Note: Optimized DCP files were preserved"

# Very clean target: Clean + remove __pycache__ and example DCPs
veryclean: clean
	@printf "$(COLOR_YELLOW)Performing deep clean...$(COLOR_RESET)\n"
	@# Remove Python cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Removed __pycache__ directories"
	@# Remove example DCPs
	@rm -f $(EXAMPLE_DCP_1) $(EXAMPLE_DCP_2)
	@echo "Removed example DCPs"
	@printf "$(COLOR_GREEN)✓ Deep clean complete$(COLOR_RESET)\n"
