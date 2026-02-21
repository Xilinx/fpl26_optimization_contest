# Agentic FPGA Backend Optimization Competition  @ [FPL'26](https://2026.fpl.org/)

This is a preliminary example agent for contestants getting started, see details below:

Contest website can be found here: https://xilinx.github.io/fpl26_optimization_contest/


# FPGA Design Optimization Agent

An example LLM-powered autonomous agent that optimizes FPGA designs for timing using RapidWright and Vivado optimizations and tools via MCP (Model Context Protocol) servers.

## Overview

This project provides an intelligent agent that:
1. Analyzes FPGA design checkpoints (.dcp files) for timing issues
2. Identifies high-fanout nets on critical timing paths
3. Applies fanout optimization using RapidWright
4. Routes the optimized design in Vivado
5. Iteratively improves timing until the design meets targets or no further improvement is possible

The agent uses two MCP servers that enable AI assistants to interact with FPGA design tools:
- **VivadoMCP**: Provides access to Xilinx Vivado's design analysis and implementation tools
- **RapidWrightMCP**: Provides access to RapidWright's netlist manipulation and optimization capabilities

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    dcp_optimizer.py                    │
│                  (AI Optimization Agent)               │
│                         │                              │
│         ┌───────────────┴───────────────┐              │
│         │                               │              │
│         ▼                               ▼              │
│  ┌─────────────────┐           ┌─────────────────┐     │
│  │  RapidWrightMCP │           │    VivadoMCP    │     │
│  │   (Python MCP)  │           │   (Python MCP)  │     │
│  └────────┬────────┘           └────────┬────────┘     │
│           │                             │              │
│           ▼                             ▼              │
│  ┌─────────────────┐           ┌─────────────────┐     │
│  │   RapidWright   │           │  Vivado Tcl     │     │
│  │   (Java/JPype)  │           │ (stdin/stdout)  │     │
│  └─────────────────┘           └─────────────────┘     │
└────────────────────────────────────────────────────────┘
```

## Quick Start

The fastest way to get started is with the Makefile:

```bash
# 1. Run setup (installs dependencies, downloads example DCPs)
make setup

# 2. Set OpenRouter API Key
export OPENROUTER_API_KEY="<your_key_here>"

# 3. Run optimizer on an example design
make run_optimizer DCP=demo_corundum_25g_misses_timing.dcp
```

See the [Running the Optimizer](#running-the-optimizer) section for more details.

### Prerequisites

- **Python 3.8+**
- **Java 11+** (for RapidWright - can be auto-detected from Vivado)
- **AMD/Xilinx Vivado** (2025.1)
  - Vivado must be on your PATH, or you can set the `VIVADO_EXEC` environment variable
- **OpenRouter API key** (for LLM access, optional for test mode)

### Installation

The easiest way to set up the project is using the provided Makefile:

```bash
# Clone the repository
git clone https://github.com/Xilinx/fpl26_optimization_contest.git
cd fpl26_optimization_contest

# Run setup (installs dependencies, checks Vivado/Java, downloads example DCPs)
make setup

# If Vivado is not on your PATH, specify it when running make:
make setup VIVADO_EXEC=/path/to/Vivado/2025.1/bin/vivado

# Or set VIVADO_EXEC as an environment variable:
export VIVADO_EXEC=/path/to/Vivado/2025.1/bin/vivado
make setup
```

The `make setup` command will:
- Install Python dependencies from `requirements.txt`
- Check if Vivado is available (and provide instructions if not)
- Check for Java, or locate Java from the Vivado installation
- Download example DCPs: `demo_corundum_25g_misses_timing.dcp` and `logicnets_jscl.dcp`

#### Manual Installation

If you prefer not to use the Makefile:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify Java is available (required for RapidWright)
java -version

# Download example DCPs
wget http://data.rapidwright.io/example-dcps/demo_corundum_25g_misses_timing.dcp
wget http://data.rapidwright.io/example-dcps/logicnets_jscl.dcp
```

**Important: Set JAVA_HOME**

When running the optimizer scripts directly (without the Makefile), you may need to set `JAVA_HOME`. The Makefile automatically detects JAVA_HOME from either:
1. The `java` command on your PATH, or
2. The Java bundled with Vivado (at `<VIVADO_ROOT>/tps/lnx64/jre21*/bin/java`)

For manual invocation, you need to set it explicitly:

```bash
# Option 1: If java is on your PATH
# Linux (with readlink -f support):
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))

# macOS / Linux alternative:
export JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null || dirname $(dirname $(which java)))

# Option 2: Use Java bundled with Vivado (if java is not on PATH)
# See: https://www.rapidwright.io/docs/Install.html#using-java-distributed-with-vivado
VIVADO_ROOT=$(dirname $(dirname $(which vivado)))
export JAVA_HOME=$(ls -d $VIVADO_ROOT/tps/lnx64/jre21* | head -n 1)
export PATH=$JAVA_HOME/bin:$PATH

# Verify it's set correctly
echo $JAVA_HOME
java -version
```

RapidWright (used for netlist manipulation) may need `JAVA_HOME` to locate the Java installation.

## Running the Optimizer

### Using the Makefile (Recommended)

The Makefile provides convenient targets for running the optimizer:

```bash
# Run optimizer on a DCP file (uses test mode by default)
make run_optimizer DCP=demo_corundum_25g_misses_timing.dcp

# Or with the other example
make run_optimizer DCP=logicnets_jscl.dcp
```

The Makefile will:
- Generate timestamped output: `<input>_optimized-YYYYMMDD_HHMMSS.dcp` (in same directory as input)
- Create a run directory: `dcp_optimizer_run-YYYYMMDD_HHMMSS/` (contains all logs and intermediate files)

#### Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show available targets and usage (default) |
| `make setup` | Install dependencies, check tools, download example DCPs |
| `make run_optimizer DCP=<file>` | Run optimizer on specified DCP file (auto-generates output name) |
| `make clean` | Remove run directories and .Xil directories (preserves optimized DCPs) |
| `make veryclean` | Deep clean including example DCPs and Python cache |

### Test Mode (No LLM Required)

Test mode runs a deterministic optimization flow without using an LLM, useful for debugging and verifying the MCP servers work correctly. The test mode automatically detects which example DCP is being used and applies the appropriate optimization strategy:

| DCP File | Optimization Strategy |
|----------|----------------------|
| `demo_corundum_25g_misses_timing.dcp` | High fanout net optimization |
| `logicnets_jscl.dcp` | Pblock-based re-placement |

```bash
# Corundum: High fanout optimization flow
python3 dcp_optimizer.py demo_corundum_25g_misses_timing.dcp --test

# LogicNets: Pblock optimization flow
python3 dcp_optimizer.py logicnets_jscl.dcp --test

# Or specify custom output name
python3 dcp_optimizer.py demo_corundum_25g_misses_timing.dcp --output optimized_output.dcp --test
```

**Note:** Test mode only works with the two example DCPs. For other designs, use the full agent mode (LLM-guided) which can analyze the design and select the appropriate optimization strategy.

#### Corundum Optimization Flow (High Fanout)

For `demo_corundum_25g_misses_timing.dcp`, the test mode:
1. Opens the DCP and reports initial timing
2. Identifies high fanout nets on critical paths
3. Uses RapidWright to replicate drivers and split fanout
4. Routes the optimized design in Vivado
5. Reports timing improvement

#### LogicNets Optimization Flow (Pblock)

For `logicnets_jscl.dcp`, the test mode:
1. Opens the DCP and reports initial timing
2. Extracts critical path cells and analyzes their spread
3. Analyzes FPGA fabric to find optimal pblock region
4. Unplaces the design and applies a pblock constraint
5. Re-places and routes within the constrained region
6. Reports timing improvement

#### Test Mode Options

```bash
# Optimize with up to 3 high-fanout nets (Corundum only, default is 5)
python3 dcp_optimizer.py demo_corundum_25g_misses_timing.dcp --test --max-nets 3

# Enable debug mode (verbose logging, preserve all files)
python3 dcp_optimizer.py demo_corundum_25g_misses_timing.dcp --test --debug
python3 dcp_optimizer.py logicnets_jscl.dcp --test --debug
```

### Full Agent Mode (Requires LLM)

In full agent mode, an LLM guides the optimization process:

```bash
# Using environment variable for API key (output name auto-generated)
export OPENROUTER_API_KEY="your-openrouter-api-key"
python3 dcp_optimizer.py input.dcp

# Or specify API key and custom output name
python3 dcp_optimizer.py input.dcp --output output.dcp --api-key "your-key"

# Use a different model (default: x-ai/grok-4.1-fast)
python3 dcp_optimizer.py input.dcp --model anthropic/claude-sonnet-4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_dcp` | Input design checkpoint (.dcp) file | Required |
| `--output`, `-o` | Output optimized checkpoint (.dcp) file | `<input>_optimized-<timestamp>.dcp` |
| `--api-key` | OpenRouter API key | `OPENROUTER_API_KEY` env var |
| `--model` | LLM model to use | `x-ai/grok-4.1-fast` |
| `--debug` | Enable debug mode (verbose logging, preserve all files) | False |
| `--test` | Test mode: run without LLM. Auto-detects DCP type and applies appropriate optimization (high fanout for Corundum, pblock for LogicNets) | False |
| `--max-nets` | Max high-fanout nets to optimize in test mode (Corundum only) | 5 |

## Validating Optimized Designs

After running `dcp_optimizer.py`, you should validate that the optimized design is functionally equivalent to the original. The `validate_dcps.py` script provides automated equivalence checking using a two-phase approach:

**Phase 1: Structural Sanity Checks** (via RapidWright)
- Verifies top-level module name matches
- Checks I/O ports (names, directions, widths) are identical
- Validates device compatibility
- Ensures cell count is reasonable (can increase but not decrease)

**Phase 2: Functional Simulation** (via Vivado + xsim)
- Exports both designs as Verilog simulation models
- Generates testbench with random stimulus (LFSR-based)
- Runs xsim simulation with configurable test vector count
- Compares outputs cycle-by-cycle for mismatches

### Usage

```bash
# Basic validation with 10,000 test vectors (default)
python3 validate_dcps.py golden.dcp optimized.dcp

# More thorough validation with 100,000 vectors
python3 validate_dcps.py golden.dcp optimized.dcp --vectors 100000

# Enable debug logging
python3 validate_dcps.py golden.dcp optimized.dcp --debug
```

### Example

```bash
# First, optimize a design
python3 dcp_optimizer.py logicnets_jscl.dcp --output logicnets_jscl_optimized.dcp

# Then validate the optimized design
python3 validate_dcps.py logicnets_jscl.dcp logicnets_jscl_optimized.dcp

# Output:
# ======================================================================
# DCP EQUIVALENCE VALIDATION
# ======================================================================
# Golden:  logicnets_jscl.dcp
# Revised: logicnets_jscl_optimized.dcp
# Vectors: 10000
# ======================================================================
#
# ======================================================================
# PHASE 1: STRUCTURAL SANITY CHECKS
# ======================================================================
#
# Comparing design structures...
#
# Structural Checks: 4/4 passed
# Result: PASS
#
# No issues found - designs are structurally compatible
#
# ----------------------------------------------------------------------
# Phase 1: PASSED ✓
# ----------------------------------------------------------------------
#
# ======================================================================
# PHASE 2: FUNCTIONAL SIMULATION
# ======================================================================
#
# Exporting simulation models...
# ✓ Golden model exported: golden_sim.v
# ✓ Revised model exported: revised_sim.v
#
# Parsing design information...
# Golden module: logicnets_jscl
# Revised module: logicnets_jscl
# Inputs: 24
# Outputs: 16
#
# Generating testbench (10000 random vectors)...
# ✓ Testbench generated: testbench.v
#
# Running xsim simulation...
# (This may take a few minutes...)
# ✓ Compilation successful
# ✓ Elaboration successful
#
# Simulating 10000 test vectors...
#
# ----------------------------------------------------------------------
# Simulation Results:
#   Cycles: 10000
#   Mismatches: 0
#   Log: /tmp/dcp_validation_xyz/simulation.log
#
# Phase 2: PASSED ✓
# ----------------------------------------------------------------------
#
# ======================================================================
# VALIDATION SUMMARY
# ======================================================================
# Golden DCP:  logicnets_jscl.dcp
# Revised DCP: logicnets_jscl_optimized.dcp
# Runtime:     3.2 minutes
#
# Phase 1 (Structural): PASSED ✓
#   Checks: 4/4
#
# Phase 2 (Simulation): PASSED ✓
#   Cycles: 10000
#   Mismatches: 0
#
# Overall Result: PASSED ✓
# ======================================================================
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `golden_dcp` | Golden (reference) DCP file | Required |
| `revised_dcp` | Revised (optimized) DCP file to validate | Required |
| `--vectors`, `-n` | Number of random test vectors to simulate | 10000 |
| `--debug` | Enable debug logging | False |

### Notes

- **Simulation time**: Scales with design size and vector count. Typical: 1-5 minutes for 10K vectors.
- **Vector count**: 10K vectors provides good coverage for most designs. For critical designs, use 50K-100K.
- **Working directory**: Preserved after validation in `/tmp/dcp_validation_*` with simulation logs and intermediate files.
- **No testbench required**: The tool automatically generates stimulus based on design I/O structure.
- **Clock/reset detection**: Automatically identifies clock and reset signals by name pattern matching.

### Limitations

- Simulation-based validation is not exhaustive (depends on test vector coverage)
- For formal proof of equivalence, use commercial tools like Synopsys Formality or Cadence Conformal
- Designs with encrypted IP blocks are not supported
- Asynchronous designs may require custom testbenches

## VivadoMCP Server

The Vivado MCP server enables AI assistants to interact with Xilinx Vivado through its Tcl interpreter interface.

### How It Works

```
┌─────────────────┐   stdin/stdout    ┌─────────────────────┐
│   MCP Server    │◄─────────────────►│   Vivado Process    │
│    (Python)     │     (pexpect)     │   ( `-mode tcl` )   │
└─────────────────┘                   └─────────────────────┘
```

The server:
1. Starts Vivado automatically in Tcl mode when first needed
2. Communicates via stdin/stdout using pexpect for direct process control
3. Exposes Vivado commands as MCP tools
4. Handles timeout recovery and graceful cleanup

### Available Vivado Tools

| Tool | Description |
|------|-------------|
| `open_checkpoint` | Open a Vivado Design Checkpoint (.dcp) - starts Vivado automatically |
| `write_checkpoint` | Save current design to .dcp file |
| `report_timing_summary` | Get timing summary (WNS/TNS) |
| `get_critical_high_fanout_nets` | Find high-fanout nets on critical paths |
| `analyze_critical_path_spread` | Measure cell spread across fabric using Manhattan distance |
| `report_utilization_for_pblock` | Get resource utilization for pblock sizing (with 1.5x multiplier) |
| `create_and_apply_pblock` | Create area constraint and apply to design |
| `report_route_status` | Check routing completion status |
| `write_edif` | Export unencrypted EDIF netlist |
| `phys_opt_design` | Run physical optimization (post-place/route) |
| `route_design` | Route the design (with directive option) |
| `place_design` | Run placement (with directive option) |
| `run_tcl` | Execute arbitrary Tcl commands |
| `restart_vivado` | Restart Vivado if hung or stuck |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIVADO_EXEC` | Path to `vivado` executable (used by Makefile, VivadoMCP, and Python scripts) | `vivado` (auto-detect from PATH) |
| `JAVA_HOME` | Java installation directory (required for RapidWright) | Auto-detected from `java` on PATH, or from Vivado's bundled Java |

**Setting VIVADO_EXEC**: You can set this variable in multiple ways:
```bash
# Method 1: Export in your shell (persists for session)
export VIVADO_EXEC=/path/to/Vivado/2025.1/bin/vivado
python3 dcp_optimizer.py input.dcp --test

# Method 2: One-line with command
VIVADO_EXEC=/path/to/vivado python3 dcp_optimizer.py input.dcp --test

# Method 3: Through Makefile
make run_optimizer DCP=input.dcp VIVADO_EXEC=/path/to/vivado
```

**Note**: The server automatically starts Vivado in Tcl mode when first needed. No manual setup required.

## RapidWrightMCP Server

The RapidWright MCP server provides AI access to [RapidWright](https://github.com/Xilinx/RapidWright), an open-source FPGA design tool framework from AMD/Xilinx.

### How It Works

```
┌─────────────────┐
│     MCP Client  │
└────────┬────────┘
         │ MCP Protocol (JSON-RPC over stdio)
┌────────▼────────┐
│    server.py    │  ← Python MCP Server
└────────┬────────┘
         │
┌────────▼────────────┐
│ rapidwright_tools.py│  ← Tool Wrappers
└────────┬────────────┘
         │
┌────────▼────────┐
│   RapidWright   │  ← pip package (JPype + Java libs)
└─────────────────┘
```

The server:
1. Initializes RapidWright's JVM with configurable memory
2. Loads/saves Vivado design checkpoints
3. Provides design analysis and manipulation
4. Implements optimization algorithms (fanout splitting, LUT optimization)

### Available RapidWright Tools

| Tool | Description |
|------|-------------|
| `initialize_rapidwright` | Initialize RapidWright JVM (call first!) |
| `read_checkpoint` | Load a Vivado Design Checkpoint (.dcp) |
| `write_checkpoint` | Save design to .dcp file |
| `get_design_info` | Get design statistics (cells, nets, types) |
| `optimize_fanout` | Split high-fanout nets by replicating drivers |
| `optimize_lut_input_cone` | Combine chained LUTs to reduce logic depth |
| `analyze_fabric_for_pblock` | Find best contiguous fabric region for pblock (avoids delay-heavy columns) |
| `convert_fabric_region_to_pblock` | Convert fabric coordinates to Vivado pblock range strings |
| `get_supported_devices` | List supported FPGA devices |
| `get_device_info` | Get device specifications |
| `search_cells` | Search for cells by name/type |
| `get_tile_info` | Get FPGA tile information |
| `search_sites` | Search for sites by type |

### Fanout Optimization

The `optimize_fanout` tool splits high-fanout nets by replicating the source driver:

```
Before:  Driver -> [1000 loads]
After:   Driver_1 -> [250 loads]
         Driver_2 -> [250 loads]
         Driver_3 -> [250 loads]
         Driver_4 -> [250 loads]
```

**Recommended split factors:**
- k=2: Fanout 500-1000 loads
- k=3-4: Fanout 1000-3000 loads
- k≥5: Fanout >3000 loads

## Optimization Strategies

The agent implements multiple optimization strategies to improve timing:

### 1. High Fanout Net Optimization (Primary for routed designs)

The `optimize_fanout` tool splits high-fanout nets by replicating the source driver:

```
Before:  Driver -> [1000 loads]
After:   Driver_1 -> [250 loads]
         Driver_2 -> [250 loads]
         Driver_3 -> [250 loads]
         Driver_4 -> [250 loads]
```

**When to use:**
- Multiple critical paths share high-fanout signals (fanout > 100)
- Design is already placed and routed
- Routing delays dominate the critical path

**Recommended split factors:**
- k=2-3: Fanout 200-500 loads
- k=3-5: Fanout 500-1500 loads
- k=5-8: Fanout >1500 loads

### 2. Pblock-Based Re-placement (For spread-out designs)

Area constraints (pblocks) restrict placement to a specific contiguous region of the FPGA fabric. This reduces routing distances when cells are spread too far apart.

**When to use:**
- Cells on critical paths are spread >70 tiles apart (Manhattan distance)
- Multiple critical paths (5+) exhibit large cell spread
- Initial placement is poor with excessive routing delays

**Workflow:**
1. **Analyze spread**: `analyze_critical_path_spread` measures Manhattan distance between cells on critical paths
2. **Get utilization**: `report_utilization_for_pblock` provides resource counts
3. **Find region**: `analyze_fabric_for_pblock` (RapidWright) identifies best contiguous fabric region:
   - Avoids delay-heavy columns (URAM, IO)
   - Has 1.5x required resources
   - Maximizes contiguity
4. **Convert coordinates**: `convert_fabric_region_to_pblock` generates Vivado pblock range string
5. **Apply pblock**: `create_and_apply_pblock` creates and applies the constraint (IS_SOFT=0)
6. **Re-implement**: Run `place_design` and `route_design` with the new constraint

**Example:**
```python
# 1. Check if cells are spread out
result = vivado.analyze_critical_path_spread(num_paths=50, distance_threshold=70)
# => "STRONG RECOMMENDATION: Apply pblock-based re-placement"

# 2. Get resource utilization (1.5x multiplier included)
utilization = vivado.report_utilization_for_pblock()
# => LUTs: 45000, FFs: 90000, DSPs: 120, BRAMs: 60

# 3. Find best fabric region (RapidWright)
region = rapidwright.analyze_fabric_for_pblock(
    target_lut_count=45000,
    target_ff_count=90000,
    target_dsp_count=120,
    target_bram_count=60
)
# => Recommended region: cols 10-85, rows 20-150

# 4. Convert to pblock range
pblock_range = rapidwright.convert_fabric_region_to_pblock(
    col_min=10, col_max=85, row_min=20, row_max=150,
    use_clock_regions=True
)
# => "CLOCKREGION_X0Y0:CLOCKREGION_X1Y2"

# 5. Apply pblock and re-implement
vivado.create_and_apply_pblock(
    pblock_name="pblock_tight",
    ranges=pblock_range,
    apply_to="current_design",
    is_soft=False
)
vivado.place_design()
vivado.route_design()
```

**Important notes:**
- Only use pblock if analyze_critical_path_spread strongly recommends it
- For 1-2 paths with spread, try phys_opt_design first
- Pblock must have IS_SOFT=0 (hard constraint) to be effective
- Must re-run place_design and route_design after applying pblock

### 3. Physical Optimization (Post-place/route fine-tuning)

The `phys_opt_design` tool performs timing-driven optimizations on the placed/routed design:

**When to use:**
- 1-2 isolated problematic paths
- After fanout optimization or pblock re-placement
- As a final polish step

**Common directives:**
- `RuntimeOptimized`: Fast optimization (fanout, critical_cell, placement, BRAM enable)
- `Explore`: Multiple passes with replication for high fanout nets
- `AggressiveExplore`: More aggressive algorithms, may temporarily degrade WNS

**Example:**
```python
vivado.phys_opt_design(directive="Explore")
```

## Optimization Workflow

The agent automatically selects the best strategy based on design analysis:

1. **Initialize** RapidWright MCP server (Vivado starts automatically when first used)
2. **Open** the input design in Vivado - this starts Vivado in Tcl mode
3. **Analyze** timing with `report_timing_summary`
4. **Analyze** cell spread with `analyze_critical_path_spread`
5. **Decide strategy:**
   - If timing met (WNS ≥ 0): Save and exit
   - If large cell spread (5+ paths >70 tiles): Use pblock strategy
   - If high fanout nets on critical paths: Use fanout optimization
   - If isolated problematic paths: Use phys_opt_design
6. **Apply optimization:**
   - **Fanout**: Load design in RapidWright → optimize_fanout → save → open in Vivado → route
   - **Pblock**: Analyze fabric → calculate pblock → apply → place → route
   - **Phys_opt**: Run phys_opt_design with appropriate directive
7. **Check** timing - if improved, iterate; otherwise save final result

## Example Session

### Using the Makefile

```
$ make run_optimizer DCP=demo_corundum_25g_misses_timing.dcp

===== FPGA Design Optimization Setup =====

[1/5] Installing Python dependencies...
✓ Python dependencies installed

[2/5] Checking Vivado...
✓ Vivado found: /opt/Xilinx/2025.1/Vivado/bin/vivado
vivado v2025.1 (64-bit)

[3/5] Checking Java...
✓ Java found: /home/user/tools/jdk-17.0.7+7/bin/java
openjdk version "17.0.7" 2023-04-18

[4/5] Downloading example DCP: demo_corundum_25g_misses_timing.dcp...
✓ demo_corundum_25g_misses_timing.dcp already exists

[5/5] Downloading example DCP: logicnets_jscl.dcp...
✓ logicnets_jscl.dcp already exists

===== Setup Complete! =====

Running optimizer on demo_corundum_25g_misses_timing.dcp...

FPGA Design Optimization - TEST MODE
=====================================
Input:       /path/to/demo_corundum_25g_misses_timing.dcp
Output:      /path/to/demo_corundum_25g_misses_timing_optimized-20260119_143022.dcp
Run dir:     /path/to/dcp_optimizer_run-20260119_143022
...
```

### Using Python Directly

```
$ python3 dcp_optimizer.py demo_corundum_25g_misses_timing.dcp --test

FPGA Design Optimization - TEST MODE
=====================================
Input:       /path/to/demo_corundum_25g_misses_timing.dcp
Output:      /path/to/demo_corundum_25g_misses_timing_optimized-20260119_143022.dcp
Run dir:     /path/to/dcp_optimizer_run-20260119_143022
Max nets to optimize: 5

[TEST] Log files in dcp_optimizer_run-20260119_143022/: rapidwright.log, rapidwright-mcp.log, vivado.log, vivado.jou, vivado-mcp.log
[TEST] Starting RapidWright MCP server...
[TEST] RapidWright MCP server started in 3.45s
[TEST] Starting Vivado MCP server...
[TEST] Vivado MCP server started in 45.23s

STEP 1: Open input DCP in Vivado
[TEST] Calling vivado_open_checkpoint...
[TEST] vivado_open_checkpoint completed in 120.34s

STEP 2: Report timing in Vivado
[TEST] vivado_report_timing_summary completed in 23.45s
*** Initial WNS: -0.234 ns ***

STEP 3: Get critical high fanout nets
Found 8 high fanout nets:
  - core_inst/pcie_inst/.../tvalid (fanout=267, paths=6)
  ...

STEP 5: Apply fanout optimizations in RapidWright
[1/5] Optimizing net: core_inst/pcie_inst/.../tvalid
    Fanout: 267, Split factor: 2
    Result: Successfully split net...

...

TEST SUMMARY
======================================================================
Total elapsed time: 892.45s
Initial WNS: -0.234 ns
Final WNS:   -0.089 ns
WNS Change:  +0.145 ns
Nets optimized: 5/5
======================================================================
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Could not find 'vivado' executable" | Set `VIVADO_EXEC` env var or use `make setup VIVADO_EXEC=/path/to/vivado` or source Vivado settings64.sh |
| "RapidWright not initialized" | Ensure Java 11+ is installed |
| "Vivado command timed out" | Command may be still running; use `restart_vivado` to recover |
| Out of memory (RapidWright) | Increase `jvm_max_memory` (e.g., "16G") |
| Vivado hangs | Use `restart_vivado` tool to kill and restart Vivado |

### Debug Mode

Enable debug mode for verbose logging:

```bash
python3 dcp_optimizer.py input.dcp --debug
```

This will:
- Set logging level to DEBUG
- Show MCP server output in console (instead of redirecting to log files)
- Print detailed tool call information

**Note:** All intermediate files are always preserved in the run directory, regardless of debug mode.

### Cleaning Up

The Makefile provides clean targets to remove generated files:

```bash
# Remove run directories and .Xil directories (preserves optimized DCPs)
make clean

# Deep clean: also remove example DCPs and Python cache
make veryclean
```

Files/directories removed by `make clean`:
- `dcp_optimizer_run-*/` directories (contain all logs, journals, intermediate DCPs)
- `.Xil/` directories (Vivado-generated)
- `VivadoMCP/.Xil/` directory

**Note:** Optimized DCP files (e.g., `design_optimized-<timestamp>.dcp`) are preserved.

## Project Structure

```
fpl26_optimization_contest/
├── Makefile                  # Build automation (setup, run, clean)
├── dcp_optimizer.py          # Main optimization agent
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── SYSTEM_PROMPT.TXT         # The system prompt used in dcp_optimizer.py
├── VivadoMCP/               # Vivado MCP server (new stdin/stdout version)
│   ├── vivado_mcp_server.py # MCP server with pexpect control
│   ├── requirements.txt     # Python dependencies
│   └── test_vivado_mcp.py   # Unit tests
└── RapidWrightMCP/          # RapidWright MCP server
    ├── server.py            # MCP server
    ├── rapidwright_tools.py # RapidWright tool wrappers
    └── setup.sh             # Setup script
```

## License

Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See LICENSE-APACHE-2.0.txt for full license details.

## Resources

- [RapidWright Documentation](https://www.rapidwright.io/docs/)
- [RapidWright GitHub](https://github.com/Xilinx/RapidWright)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Vivado Tcl Command Reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands)
