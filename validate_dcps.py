#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
FPGA Design Equivalence Validator

Validates that an optimized DCP is functionally equivalent to the original.
Uses a two-phase approach:
  Phase 1: Structural sanity checks (RapidWright)
  Phase 2: Functional simulation comparison (Vivado + xsim)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class DCPValidator:
    """Validates functional equivalence between two DCPs."""
    
    def __init__(self, golden_dcp: Path, revised_dcp: Path, num_vectors: int = 10000, debug: bool = False):
        self.golden_dcp = golden_dcp
        self.revised_dcp = revised_dcp
        self.num_vectors = num_vectors
        self.debug = debug
        
        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None
        
        # Create temporary directory for intermediate files in workspace
        # (avoids /tmp running out of space for large designs)
        workspace_dir = Path(__file__).parent
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dcp_validation_", dir=workspace_dir))
        logger.info(f"Working directory: {self.temp_dir}")
        
        # Results
        self.phase1_passed = False
        self.phase2_passed = False
        self.phase2_skipped = False
        self.phase2_skip_reason = None
        self.structural_report = None
        self.simulation_report = None
    
    async def start_servers(self):
        """Start both MCP servers."""
        script_dir = Path(__file__).parent.resolve()
        
        # Create log files in temp directory
        rapidwright_log = self.temp_dir / "rapidwright.log"
        rapidwright_mcp_log = self.temp_dir / "rapidwright-mcp.log"
        vivado_log = self.temp_dir / "vivado.log"
        vivado_journal = self.temp_dir / "vivado.jou"
        vivado_mcp_log = self.temp_dir / "vivado-mcp.log"
        
        # RapidWright MCP - with log redirection
        rapidwright_args = [str(script_dir / "RapidWrightMCP" / "server.py")]
        if not self.debug:
            rapidwright_args.extend([
                "--java-log", str(rapidwright_log),
                "--mcp-log", str(rapidwright_mcp_log)
            ])
        
        rapidwright_config = {
            "command": sys.executable,
            "args": rapidwright_args,
            "env": {**os.environ}
        }
        
        logger.info("Starting RapidWright MCP server...")
        rw_params = StdioServerParameters(**rapidwright_config)
        rw_transport = await self.exit_stack.enter_async_context(stdio_client(rw_params))
        rw_read, rw_write = rw_transport
        self.rapidwright_session = await self.exit_stack.enter_async_context(
            ClientSession(rw_read, rw_write)
        )
        await self.rapidwright_session.initialize()
        
        # Vivado MCP - with log redirection
        vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
        if not self.debug:
            vivado_args.extend([
                "--vivado-log", str(vivado_log),
                "--vivado-journal", str(vivado_journal)
            ])
        
        vivado_config = {
            "command": sys.executable,
            "args": vivado_args,
            "env": {**os.environ}
        }
        
        logger.info("Starting Vivado MCP server...")
        v_params = StdioServerParameters(**vivado_config)
        v_transport = await self.exit_stack.enter_async_context(stdio_client(v_params))
        v_read, v_write = v_transport
        self.vivado_session = await self.exit_stack.enter_async_context(
            ClientSession(v_read, v_write)
        )
        await self.vivado_session.initialize()
        
        logger.info("Both MCP servers started")
    
    async def phase1_structural_checks(self) -> bool:
        """Phase 1: Structural sanity checks using RapidWright."""
        print("\n" + "="*70)
        print("PHASE 1: STRUCTURAL SANITY CHECKS")
        print("="*70)
        
        # Initialize RapidWright
        logger.info("Initializing RapidWright...")
        result = await self.rapidwright_session.call_tool("initialize_rapidwright", {})
        
        # Compare designs
        logger.info("Comparing design structures...")
        print("\nComparing design structures...")
        
        result = await self.rapidwright_session.call_tool("compare_design_structure", {
            "golden_dcp": str(self.golden_dcp.resolve()),
            "revised_dcp": str(self.revised_dcp.resolve())
        })
        
        # Parse result
        if result.content:
            text_parts = [c.text for c in result.content if hasattr(c, 'text')]
            result_text = "\n".join(text_parts)
            self.structural_report = json.loads(result_text)
        else:
            self.structural_report = {"error": "No response from tool"}
        
        # Check if passed
        if "error" in self.structural_report:
            print(f"\n✗ ERROR: {self.structural_report['error']}")
            return False
        
        comparison_result = self.structural_report.get("comparison_result", "FAIL")
        checks_passed = self.structural_report.get("checks_passed", 0)
        checks_total = self.structural_report.get("checks_total", 0)
        issues = self.structural_report.get("issues", [])
        
        # Separate INFO issues from real issues
        info_issues = [i for i in issues if i.startswith("INFO:")]
        real_issues = [i for i in issues if not i.startswith("INFO:")]
        
        print(f"\nStructural Checks: {checks_passed}/{checks_total} passed")
        
        if real_issues:
            print("\nIssues found:")
            for issue in real_issues:
                print(f"  - {issue}")
        
        if info_issues:
            print("\nInformational notes:")
            for issue in info_issues:
                print(f"  ℹ {issue[5:].strip()}")  # Remove "INFO:" prefix
        
        if not real_issues and not info_issues:
            print("\nNo issues found - designs are structurally compatible")
        
        self.phase1_passed = (comparison_result == "PASS")
        
        print("\n" + "-"*70)
        if self.phase1_passed:
            print("Phase 1: PASSED ✓")
        else:
            print("Phase 1: FAILED ✗")
        print("-"*70)
        
        return self.phase1_passed
    
    def _check_for_encrypted_ip(self, verilog_path: Path) -> bool:
        """Check if Verilog file contains encrypted or SIP IP blocks."""
        with open(verilog_path, 'r') as f:
            content = f.read(200000)  # Check first 200KB
        
        # Look for SIP modules, encrypted IP, or hard IP blocks that require special libraries
        sip_patterns = [
            r'GTYE4_CHANNEL',       # GTY transceivers
            r'GTYE4_COMMON',        # GTY common blocks
            r'GTHE4_CHANNEL',       # GTH transceivers
            r'GTHE3_CHANNEL',       # GTH transceivers (UltraScale)
            r'GTYE3_CHANNEL',       # GTY transceivers (UltraScale)
            r'PCIE40E4',            # PCIe Gen4 x16
            r'PCIE4CE4',            # PCIe Gen4 CCIX
            r'PCIE_3_1',            # PCIe Gen3
            r'CMAC',                # 100G Ethernet MAC
            r'ILKN',                # Interlaken
            r'SIP_',                # Any SIP module
            r'encrypted',           # Encrypted netlist
            r'ENCRYPTED_VERILOG'    # Encrypted Verilog marker
        ]
        
        for pattern in sip_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.debug(f"Found encrypted/SIP pattern: {pattern}")
                return True
        
        return False
    
    def _is_encrypted_ip_error(self, error_text: str) -> bool:
        """Check if elaboration error is due to encrypted/SIP IP."""
        sip_error_patterns = [
            r'Module\s+<SIP_',           # Module <SIP_xxx> not found
            r'SIP_GTYE4',
            r'SIP_GTHE4',
            r'SIP_PCIE',
            r'instantiating unknown module SIP_',
        ]
        
        for pattern in sip_error_patterns:
            if re.search(pattern, error_text, re.IGNORECASE):
                logger.debug(f"Found SIP error pattern: {pattern}")
                return True
        
        return False
    
    def get_design_info_from_verilog(self, verilog_path: Path) -> dict:
        """Extract design information from Verilog file (module name, ports)."""
        with open(verilog_path, 'r') as f:
            lines = f.readlines()
        
        # Use structural report to find the correct top-level module name
        target_module_name = None
        if self.structural_report:
            if "golden" in str(verilog_path):
                target_module_name = self.structural_report.get("golden_design", {}).get("top_module")
            else:
                target_module_name = self.structural_report.get("revised_design", {}).get("top_module")
        
        # Parse line by line to find target module and its ports
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for module declaration
            if line.startswith('module '):
                module_match = re.search(r'module\s+(\w+)', line)
                if not module_match:
                    i += 1
                    continue
                
                module_name = module_match.group(1)
                
                # Check if this is the module we're looking for
                if target_module_name and module_name != target_module_name:
                    i += 1
                    continue
                
                # Found the target module, now parse its ports
                # Skip past the port list in parentheses to the port declarations
                while i < len(lines) and ');' not in lines[i]:
                    i += 1
                i += 1  # Skip the "); line
                
                # Now parse port declarations (input/output/inout lines)
                # Store as list of dicts with 'name' and 'width' (e.g., [63:0] or None for single bit)
                ports = {"inputs": [], "outputs": [], "inouts": []}
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Stop at wire declarations (ports section is done)
                    if line.startswith('wire ') or line.startswith('reg ') or line == '':
                        break
                    
                    if line.startswith('input '):
                        # Match: input [high:low] name; or input name;
                        port_match = re.search(r'input\s+(?:\[(\d+):(\d+)\]\s*)?(\w+)', line)
                        if port_match:
                            high_bit = port_match.group(1)
                            low_bit = port_match.group(2)
                            name = port_match.group(3)
                            width = f"[{high_bit}:{low_bit}]" if high_bit else None
                            ports["inputs"].append({"name": name, "width": width})
                    elif line.startswith('output '):
                        port_match = re.search(r'output\s+(?:\[(\d+):(\d+)\]\s*)?(\w+)', line)
                        if port_match:
                            high_bit = port_match.group(1)
                            low_bit = port_match.group(2)
                            name = port_match.group(3)
                            width = f"[{high_bit}:{low_bit}]" if high_bit else None
                            ports["outputs"].append({"name": name, "width": width})
                    elif line.startswith('inout '):
                        port_match = re.search(r'inout\s+(?:\[(\d+):(\d+)\]\s*)?(\w+)', line)
                        if port_match:
                            high_bit = port_match.group(1)
                            low_bit = port_match.group(2)
                            name = port_match.group(3)
                            width = f"[{high_bit}:{low_bit}]" if high_bit else None
                            ports["inouts"].append({"name": name, "width": width})
                    
                    i += 1
                
                # Found target module with ports
                return {"module_name": module_name, "ports": ports}
            
            i += 1
        
        # If we get here, we didn't find the target module
        if target_module_name:
            raise ValueError(f"Could not find module '{target_module_name}' in {verilog_path}")
        else:
            raise ValueError(f"Could not find any module in {verilog_path}")
    
    def generate_testbench(self, golden_info: dict, revised_info: dict, tb_path: Path):
        """Generate Verilog testbench for comparing two designs."""
        golden_module = golden_info["module_name"]
        revised_module = revised_info["module_name"] + "_revised"  # Use renamed module
        
        inputs = golden_info["ports"]["inputs"]  # List of {name, width}
        outputs = golden_info["ports"]["outputs"]  # List of {name, width}
        
        # Check for outputs
        if not outputs:
            logger.warning("Design has no outputs - simulation will only verify no crashes occur")
            print("⚠ Warning: Design has no outputs - limited verification possible")
        
        # Filter out clock and reset (we'll drive those separately)
        regular_inputs = [p for p in inputs if 'clk' not in p['name'].lower() and 'rst' not in p['name'].lower() and 'reset' not in p['name'].lower()]
        clocks = [p for p in inputs if 'clk' in p['name'].lower()]
        resets = [p for p in inputs if 'rst' in p['name'].lower() or 'reset' in p['name'].lower()]
        
        if not clocks:
            raise ValueError("No clock signal found in design")
        
        clock = clocks[0]['name']
        reset = resets[0]['name'] if resets else None
        
        # Build port connections carefully to handle edge cases
        def build_port_connections(module_suffix=""):
            """Build port connection string for module instantiation."""
            connections = []
            # Clock
            connections.append(f".{clock}({clock})")
            # Reset if present
            if reset:
                connections.append(f".{reset}({reset})")
            # Regular inputs
            for port in regular_inputs:
                connections.append(f".{port['name']}({port['name']})")
            # Outputs (with module suffix for golden/revised)
            for port in outputs:
                connections.append(f".{port['name']}({module_suffix}{port['name']})")
            return ',\n        '.join(connections)
        
        # Helper to generate stimulus for multi-bit signals
        def generate_stimulus_code():
            """Generate LFSR-based stimulus for all input ports."""
            stim_lines = []
            lfsr_bit_index = 0
            
            for port in regular_inputs:
                name = port['name']
                width = port['width']
                
                if width:  # Multi-bit port like [63:0]
                    # Extract bit width from [high:low] format
                    match = re.match(r'\[(\d+):(\d+)\]', width)
                    if match:
                        high = int(match.group(1))
                        low = int(match.group(2))
                        num_bits = high - low + 1
                        
                        # Use multiple LFSR iterations to generate enough bits
                        if num_bits <= 32:
                            stim_lines.append(f"            {name} = lfsr[{num_bits-1}:0];")
                        else:
                            # For large ports, use multiple LFSR values
                            chunks = (num_bits + 31) // 32
                            assignments = []
                            for chunk in range(chunks):
                                if chunk > 0:
                                    stim_lines.append(f"            lfsr = lfsr_next(lfsr);")
                                start_bit = chunk * 32
                                end_bit = min(start_bit + 32, num_bits)
                                if chunk == chunks - 1:
                                    # Last chunk
                                    assignments.append(f"{name}[{end_bit-1}:{start_bit}] = lfsr[{end_bit-start_bit-1}:0];")
                                else:
                                    assignments.append(f"{name}[{start_bit+31}:{start_bit}] = lfsr;")
                            stim_lines.extend([f"            {a}" for a in assignments])
                else:  # Single-bit port
                    stim_lines.append(f"            {name} = lfsr[{lfsr_bit_index % 32}];")
                    lfsr_bit_index += 1
            
            return '\n'.join(stim_lines) if stim_lines else '            // No inputs to drive'
        
        # Generate testbench
        tb_content = f"""
`timescale 1ns / 1ps

module testbench;

    // Clock and reset
    reg {clock};
    {'reg ' + reset + ';' if reset else ''}
    
    // Inputs (driven by LFSR)
    {chr(10).join(f"    reg {port['width']+' ' if port['width'] else ''}{port['name']};" for port in regular_inputs) if regular_inputs else '    // No regular inputs'}
    
    // Outputs from both designs
    {chr(10).join(f"    wire {port['width']+' ' if port['width'] else ''}golden_{port['name']};" for port in outputs) if outputs else '    // No outputs to compare'}
    {chr(10).join(f"    wire {port['width']+' ' if port['width'] else ''}revised_{port['name']};" for port in outputs) if outputs else ''}
    
    // LFSR for pseudo-random input generation
    reg [31:0] lfsr = 32'hDEADBEEF;
    
    // Instantiate golden design
    {golden_module} golden_dut (
        {build_port_connections("golden_")}
    );
    
    // Instantiate revised design
    {revised_module} revised_dut (
        {build_port_connections("revised_")}
    );
    
    // Clock generation (10ns period = 100MHz)
    initial begin
        {clock} = 0;
        forever #5 {clock} = ~{clock};
    end
    
    // LFSR update function
    function [31:0] lfsr_next;
        input [31:0] lfsr_in;
        begin
            lfsr_next = {{lfsr_in[30:0], lfsr_in[31] ^ lfsr_in[21] ^ lfsr_in[1] ^ lfsr_in[0]}};
        end
    endfunction
    
    // Test stimulus and checking
    integer mismatch_count;
    integer cycle_count;
    
    initial begin
        mismatch_count = 0;
        cycle_count = 0;
        
        // Reset
        {''+reset+' = 1;' if reset else ''}
        {chr(10).join(f"        {port['name']} = 0;" for port in regular_inputs)}
        repeat(10) @(posedge {clock});
        {''+reset+' = 0;' if reset else ''}
        
        // Warm-up period: fill pipeline without checking outputs
        // (for pipelined designs, outputs may be X until pipeline fills)
        repeat(50) begin
            @(posedge {clock});
            lfsr = lfsr_next(lfsr);
{generate_stimulus_code()}
        end
        
        // Run test vectors with output checking
        repeat({self.num_vectors}) begin
            @(posedge {clock});
            cycle_count = cycle_count + 1;
            
            // Generate new inputs from LFSR
            lfsr = lfsr_next(lfsr);
{generate_stimulus_code()}
            
            // Check outputs after settling
            #1; // Small delay for output settling
            
            // Compare all outputs
            {chr(10).join(f'''
            if (golden_{port['name']} !== revised_{port['name']}) begin
                $display("MISMATCH at cycle %0d: {port['name']} golden=%h revised=%h", cycle_count, golden_{port['name']}, revised_{port['name']});
                mismatch_count = mismatch_count + 1;
            end''' for port in outputs) if outputs else '            // No outputs to compare'}
        end
        
        // Report results
        $display("\\n=======================================");
        $display("SIMULATION COMPLETE");
        $display("=======================================");
        $display("Cycles simulated: %0d", cycle_count);
        {'$display("Outputs compared: 0 (design has no outputs)");' if not outputs else '$display("Mismatches found: %0d", mismatch_count);'}
        if (mismatch_count == 0) begin
            {'$display("Result: PASS (no crashes detected)");' if not outputs else '$display("Result: PASS");'}
            $finish(0);
        end else begin
            $display("Result: FAIL");
            $finish(1);
        end
    end
    
    // Timeout watchdog (reset + warmup + test cycles, with 2x safety margin)
    initial begin
        #{(10 + 50 + self.num_vectors) * 20} $display("ERROR: Simulation timeout"); $finish(2);
    end

endmodule
"""
        
        with open(tb_path, 'w') as f:
            f.write(tb_content)
        
        logger.info(f"Generated testbench: {tb_path}")
    
    async def phase2_functional_simulation(self) -> bool:
        """Phase 2: Functional simulation comparison."""
        print("\n" + "="*70)
        print("PHASE 2: FUNCTIONAL SIMULATION")
        print("="*70)
        
        # Export Verilog simulation models
        golden_v = self.temp_dir / "golden_sim.v"
        revised_v = self.temp_dir / "revised_sim.v"
        
        print("\nExporting simulation models...")
        
        # Open golden DCP
        logger.info(f"Opening golden DCP: {self.golden_dcp}")
        result = await self.vivado_session.call_tool("open_checkpoint", {
            "dcp_path": str(self.golden_dcp.resolve())
        })
        
        # Export golden as Verilog
        logger.info("Exporting golden to Verilog...")
        result = await self.vivado_session.call_tool("write_verilog_simulation", {
            "verilog_path": str(golden_v),
            "force": True
        })
        print(f"✓ Golden model exported: {golden_v.name}")
        
        # Open revised DCP
        logger.info(f"Opening revised DCP: {self.revised_dcp}")
        result = await self.vivado_session.call_tool("open_checkpoint", {
            "dcp_path": str(self.revised_dcp.resolve())
        })
        
        # Export revised as Verilog
        logger.info("Exporting revised to Verilog...")
        result = await self.vivado_session.call_tool("write_verilog_simulation", {
            "verilog_path": str(revised_v),
            "force": True
        })
        print(f"✓ Revised model exported: {revised_v.name}")
        
        # Parse design information
        print("\nParsing design information...")
        golden_info = self.get_design_info_from_verilog(golden_v)
        revised_info = self.get_design_info_from_verilog(revised_v)
        
        print(f"Golden module: {golden_info['module_name']}")
        print(f"Revised module: {revised_info['module_name']}")
        
        # Show port details with bit widths
        print(f"\nPort Information:")
        print(f"  Inputs ({len(golden_info['ports']['inputs'])}):")
        for port in golden_info['ports']['inputs']:
            width_str = f" {port['width']}" if port['width'] else ""
            print(f"    - {port['name']}{width_str}")
        print(f"  Outputs ({len(golden_info['ports']['outputs'])}):")
        for port in golden_info['ports']['outputs']:
            width_str = f" {port['width']}" if port['width'] else ""
            print(f"    - {port['name']}{width_str}")
        
        # Generate testbench
        tb_path = self.temp_dir / "testbench.v"
        print(f"\nGenerating testbench ({self.num_vectors} random vectors)...")
        self.generate_testbench(golden_info, revised_info, tb_path)
        print(f"✓ Testbench generated: {tb_path.name}")
        
        # Run xsim simulation
        print("\nRunning xsim simulation...")
        print("(This may take a few minutes...)")
        
        xsim_dir = self.temp_dir / "xsim_work"
        xsim_dir.mkdir(exist_ok=True)
        
        try:
            # Get Vivado installation path for simulation libraries
            # Check VIVADO_EXEC environment variable first, then PATH
            vivado_path = os.environ.get("VIVADO_EXEC")
            if vivado_path:
                # If VIVADO_EXEC is just the name (not a path), search in PATH
                if '/' not in vivado_path:
                    vivado_path = shutil.which(vivado_path)
            else:
                vivado_path = shutil.which("vivado")
            if not vivado_path:
                raise RuntimeError("Vivado not found in PATH. Set VIVADO_EXEC env var or add Vivado to PATH.")
            
            # Vivado sim lib is at: $XILINX_VIVADO/data/verilog/src/
            vivado_bin_dir = Path(vivado_path).parent
            vivado_install = vivado_bin_dir.parent
            unisim_dir = vivado_install / "data" / "verilog" / "src"
            
            if not unisim_dir.exists():
                logger.warning(f"UNISIM library not found at {unisim_dir}, trying glbl.v only")
            
            # To avoid module name conflicts, rename ALL modules in revised file
            logger.info("Renaming all revised modules to avoid conflicts...")
            revised_renamed = xsim_dir / "revised_sim_renamed.v"
            
            with open(revised_v, 'r') as f:
                content = f.read()
            
            # Rename ALL modules (not just top) to avoid sub-module conflicts
            revised_module_name = revised_info["module_name"]
            revised_module_renamed = f"{revised_module_name}_revised"
            
            # Rename module declarations - handle both single-line and multi-line
            # "module name" OR "module name("
            content = re.sub(
                r'\bmodule\s+(\w+)(\s*[\(\n])',
                lambda m: f'module {m.group(1)}_revised{m.group(2)}',
                content
            )
            
            # Rename module instantiations (user modules, not FPGA primitives)
            # Look for patterns like "layer0 inst_name (" or "myreg inst_name ("
            # Primitives typically start with uppercase (LUT6, FDRE, etc.)
            content = re.sub(
                r'\b(layer\d+[_\w]*)\s+(\w+)\s*\(',
                r'\1_revised \2 (',
                content
            )
            content = re.sub(
                r'\b(myreg[_\w]*)\s+(\w+)\s*\(',
                r'\1_revised \2 (',
                content
            )
            
            with open(revised_renamed, 'w') as f:
                f.write(content)
            
            # Compile Verilog files
            logger.info("Compiling with xvlog...")
            compile_cmd = [
                "xvlog",
                "-work", "work",
                str(golden_v),
                str(revised_renamed),
                str(tb_path)
            ]
            
            # Add UNISIM glbl if available
            if unisim_dir.exists():
                glbl_v = unisim_dir / "glbl.v"
                if glbl_v.exists():
                    compile_cmd.insert(3, str(glbl_v))
            
            result = subprocess.run(
                compile_cmd,
                cwd=xsim_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"\n✗ Compilation failed:")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("✓ Compilation successful")
            
            # Elaborate with UNISIM library reference
            logger.info("Elaborating with xelab...")
            elab_cmd = [
                "xelab",
                "-debug", "typical",
                "-L", "unisims_ver",  # Link against UNISIM library
                "-L", "unimacro_ver",
                "work.testbench",  # Specify library.module
                "work.glbl",       # Include glbl for initialization
                "-s", "testbench_sim"
            ]
            
            result = subprocess.run(
                elab_cmd,
                cwd=xsim_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                # Check if failure is due to encrypted/SIP IP
                error_output = result.stdout + result.stderr
                if self._is_encrypted_ip_error(error_output):
                    logger.info("Elaboration failed due to encrypted/SIP IP")
                    print("\n" + "="*70)
                    print("⚠ PHASE 2 SKIPPED")
                    print("="*70)
                    print("\nReason: Design contains encrypted or Secure IP blocks")
                    print("        (e.g., PCIe, GTY transceivers) that cannot be")
                    print("        simulated without vendor-specific libraries.")
                    print("\nStructural checks (Phase 1) are still valid.")
                    print("="*70 + "\n")
                    return {
                        "status": "skipped",
                        "reason": "Design contains encrypted or Secure IP blocks",
                        "details": "xelab elaboration failed due to missing SIP modules"
                    }
                
                print(f"\n✗ Elaboration failed:")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("✓ Elaboration successful")
            
            # Run simulation
            logger.info("Running simulation with xsim...")
            print(f"\nSimulating {self.num_vectors} test vectors...")
            
            sim_cmd = [
                "xsim",
                "testbench_sim",
                "-R"
            ]
            
            result = subprocess.run(
                sim_cmd,
                cwd=xsim_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Parse simulation output
            sim_output = result.stdout + result.stderr
            
            # Save simulation log
            log_file = self.temp_dir / "simulation.log"
            with open(log_file, 'w') as f:
                f.write(sim_output)
            
            # Extract results
            mismatch_count = 0
            cycles_simulated = 0
            
            for line in sim_output.split('\n'):
                if 'MISMATCH' in line:
                    mismatch_count += 1
                    print(f"  {line}")
                elif 'Cycles simulated:' in line:
                    match = re.search(r'Cycles simulated:\s*(\d+)', line)
                    if match:
                        cycles_simulated = int(match.group(1))
                elif 'Mismatches found:' in line:
                    match = re.search(r'Mismatches found:\s*(\d+)', line)
                    if match:
                        mismatch_count = int(match.group(1))
            
            self.simulation_report = {
                "cycles_simulated": cycles_simulated,
                "mismatch_count": mismatch_count,
                "log_file": str(log_file)
            }
            
            # Check if passed
            self.phase2_passed = (mismatch_count == 0 and result.returncode == 0)
            
            print("\n" + "-"*70)
            print(f"Simulation Results:")
            print(f"  Cycles: {cycles_simulated}")
            print(f"  Mismatches: {mismatch_count}")
            print(f"  Log: {log_file}")
            
            if self.phase2_passed:
                print("\nPhase 2: PASSED ✓")
            else:
                print("\nPhase 2: FAILED ✗")
            print("-"*70)
            
            return self.phase2_passed
            
        except subprocess.TimeoutExpired:
            print("\n✗ Simulation timeout")
            return False
        except Exception as e:
            print(f"\n✗ Simulation error: {e}")
            logger.exception("Simulation error")
            return False
    
    async def validate(self) -> bool:
        """Run complete validation (both phases)."""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("DCP EQUIVALENCE VALIDATION")
        print("="*70)
        print(f"Golden:  {self.golden_dcp}")
        print(f"Revised: {self.revised_dcp}")
        print(f"Vectors: {self.num_vectors}")
        print("="*70)
        
        # Phase 1: Structural checks
        phase1_passed = await self.phase1_structural_checks()
        
        if not phase1_passed:
            print("\n⚠ Skipping Phase 2 due to Phase 1 failures")
            elapsed = time.time() - start_time
            self.print_final_report(elapsed)
            return False
        
        # Phase 2: Functional simulation
        phase2_result = await self.phase2_functional_simulation()
        
        # Handle skipped phase2 (e.g., encrypted IP)
        if isinstance(phase2_result, dict) and phase2_result.get("status") == "skipped":
            self.phase2_skipped = True
            self.phase2_skip_reason = phase2_result.get("reason", "Unknown reason")
            phase2_passed = True  # Don't fail validation if phase2 is skipped
        else:
            phase2_passed = phase2_result
        
        elapsed = time.time() - start_time
        self.print_final_report(elapsed)
        
        return phase1_passed and phase2_passed
    
    def print_final_report(self, elapsed_time: float):
        """Print final validation report."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Golden DCP:  {self.golden_dcp.name}")
        print(f"Revised DCP: {self.revised_dcp.name}")
        print(f"Runtime:     {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print()
        
        print("Phase 1 (Structural): " + ("PASSED ✓" if self.phase1_passed else "FAILED ✗"))
        if self.structural_report:
            checks_passed = self.structural_report.get("checks_passed", 0)
            checks_total = self.structural_report.get("checks_total", 0)
            print(f"  Checks: {checks_passed}/{checks_total}")
            issues = self.structural_report.get("issues", [])
            if issues:
                print(f"  Issues: {len(issues)}")
        
        print()
        if self.phase2_skipped:
            print("Phase 2 (Simulation): SKIPPED ⊘")
            print(f"  Reason: {self.phase2_skip_reason}")
        else:
            print("Phase 2 (Simulation): " + ("PASSED ✓" if self.phase2_passed else "FAILED ✗" if self.phase1_passed else "SKIPPED"))
            if self.simulation_report:
                print(f"  Cycles: {self.simulation_report.get('cycles_simulated', 0)}")
                print(f"  Mismatches: {self.simulation_report.get('mismatch_count', 0)}")
        
        print()
        if self.phase2_skipped:
            overall_result = "PASSED ✓ (structural only)" if self.phase1_passed else "FAILED ✗"
        else:
            overall_result = "PASSED ✓" if (self.phase1_passed and self.phase2_passed) else "FAILED ✗"
        print(f"Overall Result: {overall_result}")
        print("="*70)
        
        # Save detailed report
        report_file = self.temp_dir / "validation_report.json"
        report = {
            "golden_dcp": str(self.golden_dcp),
            "revised_dcp": str(self.revised_dcp),
            "num_vectors": self.num_vectors,
            "runtime_seconds": elapsed_time,
            "phase1_passed": self.phase1_passed,
            "phase2_passed": self.phase2_passed,
            "overall_passed": self.phase1_passed and self.phase2_passed,
            "structural_report": self.structural_report,
            "simulation_report": self.simulation_report
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved: {report_file}")
        print(f"Working directory preserved: {self.temp_dir}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    parser = argparse.ArgumentParser(
        description="Validate functional equivalence between two FPGA design checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_dcps.py golden.dcp optimized.dcp
  python validate_dcps.py golden.dcp optimized.dcp --vectors 50000
  python validate_dcps.py golden.dcp optimized.dcp --debug
        """
    )
    parser.add_argument("golden_dcp", type=Path, help="Golden (reference) DCP file")
    parser.add_argument("revised_dcp", type=Path, help="Revised (optimized) DCP file to validate")
    parser.add_argument(
        "--vectors",
        "-n",
        type=int,
        default=10000,
        help="Number of random test vectors to simulate (default: 10000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.golden_dcp.exists():
        print(f"Error: Golden DCP not found: {args.golden_dcp}", file=sys.stderr)
        sys.exit(1)
    
    if not args.revised_dcp.exists():
        print(f"Error: Revised DCP not found: {args.revised_dcp}", file=sys.stderr)
        sys.exit(1)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = DCPValidator(
        golden_dcp=args.golden_dcp,
        revised_dcp=args.revised_dcp,
        num_vectors=args.vectors,
        debug=args.debug
    )
    
    try:
        await validator.start_servers()
        success = await validator.validate()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
