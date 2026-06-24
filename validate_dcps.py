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


def sanitize_identifier(name: str) -> str:
    """Convert an arbitrary interface prefix into a valid Verilog identifier."""
    ident = re.sub(r'[^0-9A-Za-z_]', '_', name)
    ident = re.sub(r'_+', '_', ident).strip('_')
    if not ident:
        ident = "ifc"
    if ident[0].isdigit():
        ident = f"ifc_{ident}"
    return ident


def port_bit_width(port: dict) -> int:
    """Return the bit width for a parsed Verilog port dictionary."""
    width = port.get('width')
    if not width:
        return 1
    match = re.match(r'\[\s*(\d+)\s*:\s*(\d+)\s*\]', width)
    if not match:
        return 1
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def assign_from_seed(port: dict, seed_expr: str, indent: str = "            ") -> str:
    """Generate a Verilog assignment using chunk-varying seed data."""
    bits = port_bit_width(port)
    name = port['name']
    # Rely on Verilog assignment truncation/extension for <=32-bit ports.
    # Indexing a parenthesized expression like "(a ^ b)[7:0]" is rejected by
    # xvlog in the Verilog mode used here.
    if bits <= 32:
        return f"{indent}{name} = {seed_expr};"
    chunks = (bits + 31) // 32
    chunk_exprs = []
    for idx in range(chunks):
        mask = (0x9E3779B9 * (idx + 1)) & 0xFFFFFFFF
        chunk_exprs.append(f"({seed_expr} ^ 32'h{mask:08X})")
    return f"{indent}{name} = {{{', '.join(reversed(chunk_exprs))}}};"


def parse_simulation_output(sim_output: str, returncode: int, expected_cycles: int) -> dict:
    """Parse xsim output and compute the validator pass/fail fields."""
    mismatch_count = 0
    protocol_mismatch_count = 0
    cycles_simulated = 0
    result_pass_seen = False
    result_fail_seen = False
    simulator_failed = False

    fatal_patterns = [
        r'Simulation engine not responding',
        r'Simulator command interrupted',
        r'Command failed:',
        r'terminated in an unexpected manner',
        r'\bFATAL\b',
        r'\bUSF-XSim\b',
        r'\bXSIM\s+\d+-\d+\b',
    ]
    fatal_re = re.compile('|'.join(fatal_patterns), re.IGNORECASE)

    for line in sim_output.split('\n'):
        if 'PROTOCOL MISMATCH' in line:
            protocol_mismatch_count += 1
        elif 'MISMATCH' in line:
            mismatch_count += 1
        elif 'Cycles simulated:' in line:
            match = re.search(r'Cycles simulated:\s*(\d+)', line)
            if match:
                cycles_simulated = int(match.group(1))
        elif 'Mismatches found:' in line:
            match = re.search(r'Mismatches found:\s*(\d+)', line)
            if match:
                mismatch_count = int(match.group(1))
        elif 'Protocol mismatches found:' in line:
            match = re.search(r'Protocol mismatches found:\s*(\d+)', line)
            if match:
                protocol_mismatch_count = int(match.group(1))
        elif 'Result: PASS' in line:
            result_pass_seen = True
        elif 'Result: FAIL' in line:
            result_fail_seen = True

        if fatal_re.search(line):
            simulator_failed = True

    passed = (
        returncode == 0
        and result_pass_seen
        and not result_fail_seen
        and not simulator_failed
        and cycles_simulated == expected_cycles
        and mismatch_count == 0
        and protocol_mismatch_count == 0
    )

    # A definitive logical FAIL (testbench mismatch) must never be promoted to
    # an infrastructure failure. Otherwise a revised netlist that both mismatches
    # and provokes a fatal-pattern xsim message would be tagged INFRASTRUCTURE
    # FAILURE (exit 2 / retry) instead of a clean FAIL (exit 1 / score 0).
    definitive_fail = (
        result_fail_seen
        or mismatch_count > 0
        or protocol_mismatch_count > 0
    )
    infrastructure_failure = not definitive_fail and (
        simulator_failed
        or (not result_pass_seen and not result_fail_seen)
    )
    infrastructure_reason = None
    if infrastructure_failure:
        if simulator_failed:
            infrastructure_reason = "simulator_failed"
        elif not result_pass_seen and not result_fail_seen:
            infrastructure_reason = "testbench_completion_marker_missing"

    return {
        "cycles_simulated": cycles_simulated,
        "mismatch_count": mismatch_count,
        "protocol_mismatch_count": protocol_mismatch_count,
        "result_pass_seen": result_pass_seen,
        "result_fail_seen": result_fail_seen,
        "simulator_failed": simulator_failed,
        "infrastructure_failure": infrastructure_failure,
        "infrastructure_reason": infrastructure_reason,
        "returncode": returncode,
        "passed": passed,
    }


class DCPValidator:
    """Validates functional equivalence between two DCPs."""

    def __init__(
        self,
        golden_dcp: Path,
        revised_dcp: Path,
        num_vectors: int = 200,
        precheck_vectors: int = 100,
        debug: bool = False,
        no_reactive: bool = False,
    ):
        self.golden_dcp = golden_dcp
        self.revised_dcp = revised_dcp
        self.num_vectors = num_vectors
        self.precheck_vectors = precheck_vectors
        self.debug = debug
        self.no_reactive = no_reactive
        
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
        self.preflight_report = None
        self.infrastructure_failure = False
        self.infrastructure_reason = None

    def _copy_dcp_for_rapidwright(self, src_dcp: Path, label: str) -> Path:
        """Copy a DCP into validator-owned space and build a fresh EDIF sidecar.

        RapidWright may consult ``<dcp>.edf`` sidecars when loading DCPs. For
        contest submissions, sidecars beside the submitted DCP should not be
        trusted or required, so create a clean copy and matching sidecar in this
        validation run directory.
        """
        if not src_dcp.exists():
            raise FileNotFoundError(f"{label} DCP not found: {src_dcp}")

        prepared_dcp = self.temp_dir / f"{label}.dcp"
        shutil.copy2(src_dcp, prepared_dcp)

        unzip_result = subprocess.run(
            ["unzip", "-t", str(prepared_dcp)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if unzip_result.returncode != 0:
            detail = (unzip_result.stdout + unzip_result.stderr).strip()
            raise RuntimeError(f"{label} DCP failed zip integrity check: {detail}")

        sidecar_dir = self.temp_dir / f"{label}.dcp.edf"
        if sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
        sidecar_dir.mkdir()
        sidecar_edf = sidecar_dir / f"{label}.edf"
        sidecar_md5 = sidecar_dir / f"{label}.dcp.md5"

        tcl = (
            f"open_checkpoint {{{prepared_dcp}}}\n"
            f"write_edif -force {{{sidecar_edf}}}\n"
            "close_design\n"
        )
        vivado_path = os.environ.get("VIVADO_EXEC")
        if vivado_path:
            if "/" not in vivado_path:
                vivado_path = shutil.which(vivado_path)
        else:
            vivado_path = shutil.which("vivado")
        if not vivado_path:
            raise RuntimeError("Vivado not found in PATH. Set VIVADO_EXEC env var or add Vivado to PATH.")

        tcl_path = self.temp_dir / f"prepare_{label}_readable_edif.tcl"
        tcl_path.write_text(tcl)
        log_path = self.temp_dir / f"prepare_{label}_readable_edif.log"
        jou_path = self.temp_dir / f"prepare_{label}_readable_edif.jou"
        result = subprocess.run(
            [
                vivado_path,
                "-mode", "batch",
                "-source", str(tcl_path),
                "-log", str(log_path),
                "-journal", str(jou_path),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0 or not sidecar_edf.exists() or sidecar_edf.stat().st_size == 0:
            detail = (result.stdout + result.stderr).strip()
            raise RuntimeError(
                f"{label} DCP opened, but readable EDIF generation failed. "
                f"Log: {log_path}. {detail[-1000:]}"
            )

        md5_result = subprocess.run(
            ["md5sum", str(prepared_dcp)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        sidecar_md5.write_text(md5_result.stdout.split()[0] + "\n")

        return prepared_dcp

    def _is_rapidwright_readability_error(self, report: dict) -> bool:
        """Return true for RapidWright DCP/EDIF cache/read failures."""
        error = str(report.get("error", ""))
        patterns = [
            "Unable to find a readable EDIF file",
            "ZipException",
            "zip archive",
            "dcp.xml",
            "Failed to auto-generate an EDIF file",
            "invalid LOC header",
            "bad signature",
        ]
        return any(pattern in error for pattern in patterns)

    async def _compare_design_structures(self, golden_dcp: Path, revised_dcp: Path) -> dict:
        result = await self.rapidwright_session.call_tool("compare_design_structure", {
            "golden_dcp": str(golden_dcp),
            "revised_dcp": str(revised_dcp),
        })

        if result.content:
            text_parts = [c.text for c in result.content if hasattr(c, 'text')]
            result_text = "\n".join(text_parts)
            return json.loads(result_text)

        return {"error": "No response from tool"}

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

        self.structural_report = await self._compare_design_structures(
            self.golden_dcp.resolve(),
            self.revised_dcp.resolve(),
        )

        if (
            "error" in self.structural_report
            and self._is_rapidwright_readability_error(self.structural_report)
        ):
            initial_error = self.structural_report["error"]
            print("\nRapidWright could not read one of the DCPs directly.")
            print("Preparing validator-owned DCP copies and readable EDIF sidecars, then retrying...")
            try:
                prepared_golden = self._copy_dcp_for_rapidwright(self.golden_dcp.resolve(), "golden")
                prepared_revised = self._copy_dcp_for_rapidwright(self.revised_dcp.resolve(), "revised")
                self.preflight_report = {
                    "status": "success",
                    "reason": "rapidwright_readability_retry",
                    "initial_error": initial_error,
                    "golden_prepared_dcp": str(prepared_golden),
                    "revised_prepared_dcp": str(prepared_revised),
                }
                self.structural_report = await self._compare_design_structures(
                    prepared_golden,
                    prepared_revised,
                )
                if "error" in self.structural_report:
                    self.structural_report["preflight_report"] = self.preflight_report
            except Exception as e:
                self.preflight_report = {
                    "status": "error",
                    "stage": "rapidwright_dcp_preparation",
                    "initial_error": initial_error,
                    "error": str(e),
                }
                self.structural_report = {
                    "error": str(e),
                    "preflight_report": self.preflight_report,
                }
                print(f"\n✗ DCP preflight retry failed: {e}")
                return False
        else:
            self.preflight_report = {
                "status": "not_needed",
                "reason": "rapidwright_direct_compare_succeeded_or_non_readability_failure",
            }
        
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
    
    # Tcl helper that emits the top-level clock port names of the currently
    # open design between sentinel markers. Sourced into Vivado on demand.
    #
    # Two strategies, applied in order, with results de-duplicated:
    #   1. ``all_fanin -startpoints_only`` from every pin in the clock
    #      networks back to primary inputs - this handles the common case
    #      where create_clock is bound to an internal pin (e.g. a BUFG
    #      output) rather than a top-level port, which is the case for
    #      Chisel-style designs (e.g. boom_soc whose clock object source
    #      is empty but whose actual port is ``clock_uncore_clock``).
    #   2. Trace primary inputs feeding the I pin of any global clock
    #      buffer (BUFG*, IBUFG*) - this catches designs that have no
    #      ``create_clock`` constraints at all but still have a clearly
    #      identifiable clock input port.
    #
    # The marker strings printed at runtime are assembled with ``format``
    # rather than written as string literals so they don't appear verbatim
    # in this source - if they did, Vivado's echo of the proc body during
    # ``source`` would falsely match the Python-side regex.
    _CLOCK_PORT_DETECT_TCL = r"""
proc __vd_emit_clock_port {sp seenVar} {
    upvar 1 $seenVar seen
    if {[get_property CLASS $sp] ne {port}} { return }
    if {[get_property DIRECTION $sp] ne {IN}} { return }
    set nm [get_property NAME $sp]
    if {[dict exists $seen $nm]} { return }
    dict set seen $nm 1
    set bn [get_property BUS_NAME $sp]
    if {$bn eq {}} { puts $nm } else { puts $bn }
}

proc __vd_find_clock_ports {} {
    set seen [dict create]
    set mB [format {_%s_VDCLKP_%sIN__} {} {BEG}]
    set mE [format {_%s_VDCLKP_%s__}   {} {ENDX}]
    puts $mB
    set clk_pins [get_pins -quiet -of_objects [get_clocks -quiet]]
    if {[llength $clk_pins] > 0} {
        foreach sp [all_fanin -quiet -flat -startpoints_only $clk_pins] {
            __vd_emit_clock_port $sp seen
        }
    }
    foreach c [get_cells -quiet -hier -filter {REF_NAME =~ BUFG* || REF_NAME =~ IBUFG*}] {
        foreach ipin [get_pins -quiet -of_objects $c -filter {DIRECTION == IN}] {
            foreach sp [all_fanin -quiet -flat -startpoints_only $ipin] {
                __vd_emit_clock_port $sp seen
            }
        }
    }
    puts $mE
}
"""
    
    async def _query_clock_ports_from_vivado(self) -> list:
        """Query Vivado for the top-level clock port names of the open design.

        Uses Vivado's clock-network connectivity rather than guessing from
        port names; see ``_CLOCK_PORT_DETECT_TCL`` for the strategies. Returns
        an empty list on any failure (caller falls back to a name heuristic),
        and de-duplicates bus bits like ``clk[0]`` -> ``clk``.
        """
        # Write the helper script to disk once and source it on each call.
        # ``run_tcl`` sends a single line, so a sourced proc keeps the wire
        # protocol simple (vs. encoding multi-line Tcl with semicolons).
        helper_path = self.temp_dir / "find_clock_ports.tcl"
        if not helper_path.exists():
            with open(helper_path, "w") as f:
                f.write(self._CLOCK_PORT_DETECT_TCL)
        
        cmd = f"source {{{helper_path}}}; __vd_find_clock_ports"
        try:
            result = await self.vivado_session.call_tool(
                "run_tcl", {"command": cmd, "timeout": 120}
            )
        except Exception as e:
            logger.warning(f"Failed to query clock ports from Vivado: {e}")
            return []
        
        if not result.content:
            return []
        text = "\n".join(c.text for c in result.content if hasattr(c, 'text'))
        
        # Markers must match the dynamic ``format`` calls in the Tcl helper.
        m = re.search(
            r'__VDCLKP_BEGIN__\s*(.*?)\s*__VDCLKP_ENDX__',
            text, re.DOTALL,
        )
        if not m:
            logger.debug(f"Clock-port markers not found in run_tcl output; got: {text[:500]!r}")
            return []
        
        names: list = []
        seen: set = set()
        # Valid Verilog port identifiers are word characters; reject anything
        # that isn't, which filters out Vivado command echo lines (``# ...``),
        # warning/info banners, and stray punctuation.
        valid = re.compile(r'^\w+(?:\[\d+\])?$')
        for line in m.group(1).splitlines():
            name = line.strip()
            if not name or not valid.match(name):
                continue
            # Strip a single trailing bus index so bit-level port objects
            # like "clk[0]" collapse to the bus name "clk" we parsed from
            # the Verilog port list.
            name = re.sub(r'\[\d+\]$', '', name)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names
    
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
    
    def generate_testbench(self, golden_info: dict, revised_info: dict, tb_path: Path,
                           clock_names: Optional[list] = None):
        """Generate Verilog testbench for comparing two designs.

        ``clock_names`` is the authoritative list of top-level clock port names
        as reported by Vivado's ``get_clocks``/``get_ports`` traversal. When
        provided we use it directly; otherwise we fall back to a name-based
        heuristic (which fails for Chisel/Rocket-Chip-style designs whose
        clocks are named ``clock`` rather than ``clk``).
        """
        golden_module = golden_info["module_name"]
        revised_module = revised_info["module_name"] + "_revised"  # Use renamed module
        
        inputs = golden_info["ports"]["inputs"]  # List of {name, width}
        outputs = golden_info["ports"]["outputs"]  # List of {name, width}
        
        # Check for outputs
        if not outputs:
            logger.warning("Design has no outputs - simulation will only verify no crashes occur")
            print("⚠ Warning: Design has no outputs - limited verification possible")
        
        # Identify clocks. Prefer the constraint-based list from Vivado; fall
        # back to a broadened substring heuristic only if Vivado gave us
        # nothing (e.g. an unconstrained design with no SDC).
        clocks: list = []
        if clock_names:
            # Vivado's all_fanin traversal can return non-clock signals (e.g.
            # data-valid or commit strobe inputs) that happen to feed into
            # the clock network via CE pins or combinatorial paths. Filter the
            # reported list to ports whose names contain 'clk' or 'clock' so
            # those data signals are not stripped from the driven input set.
            clock_like_names = [
                n for n in clock_names
                if 'clk' in n.lower() or 'clock' in n.lower()
            ]
            if not clock_like_names:
                logger.warning(
                    f"Vivado-reported clock ports {sorted(clock_names)} have no "
                    "clock-like names; falling back to name-based heuristic"
                )
                clock_names = []
            else:
                if len(clock_like_names) < len(clock_names):
                    excluded = sorted(set(clock_names) - set(clock_like_names))
                    logger.info(
                        f"Filtered non-clock-like ports from Vivado clock list: "
                        f"{excluded} (likely data/control signals mis-reported by all_fanin)"
                    )
                clock_names = clock_like_names
            wanted = set(clock_names)
            clocks = [p for p in inputs if p['name'] in wanted]
            missing = wanted - {p['name'] for p in clocks}
            if missing:
                logger.warning(
                    f"Vivado-reported clock ports not found among top-level "
                    f"inputs: {sorted(missing)}"
                )
        if not clocks:
            if clock_names:
                logger.warning(
                    "No Vivado-reported clocks matched top-level inputs; "
                    "falling back to name-based heuristic"
                )
            clocks = [
                p for p in inputs
                if 'clk' in p['name'].lower() or 'clock' in p['name'].lower()
            ]
        
        # Reset detection remains heuristic - DCPs do not carry a canonical
        # "this port is the reset" property the way they do for clocks.
        resets = [p for p in inputs if 'rst' in p['name'].lower() or 'reset' in p['name'].lower()]
        
        if not clocks:
            raise ValueError("No clock signal found in design")
        
        # Everything that isn't a clock or reset is driven by the LFSR stimulus.
        special_names = {p['name'] for p in clocks} | {p['name'] for p in resets}
        regular_inputs = [p for p in inputs if p['name'] not in special_names]
        
        clock = clocks[0]['name']
        reset = resets[0]['name'] if resets else None

        # Reset polarity is also heuristic. Active-low resets are conventionally
        # named with an "n" suffix (aresetn, rst_n, resetn, ...). Driving an
        # active-low reset as if it were active-high holds the design IN reset for
        # the entire warm-up + checking window, freezing every output and masking
        # real differences (added latency, etc.). Detect the convention and pick
        # assert/deassert levels accordingly; default to active-high.
        reset_active_low = bool(reset) and bool(
            re.search(r'(rst|reset)_?n$', reset.lower()))
        reset_assert = '0' if reset_active_low else '1'
        reset_deassert = '1' if reset_active_low else '0'

        input_by_name = {p['name']: p for p in regular_inputs}
        output_by_name = {p['name']: p for p in outputs}

        controlled_inputs = set()
        response_ifaces = []
        request_ifaces = []
        ready_bias_inputs = []
        hls_iface = None
        hls_mem_ifaces = []
        used_iface_ids = {}

        def unique_iface_id(prefix: str) -> str:
            base = sanitize_identifier(prefix)
            count = used_iface_ids.get(base, 0)
            used_iface_ids[base] = count + 1
            if count == 0:
                return base
            return f"{base}_{count}"

        # Detect simple master-side request/response interfaces where the DUT
        # emits commands and expects a response on top-level inputs. The
        # validator will emulate a one-deep reactive responder using golden DUT
        # outputs as the reference handshake source.
        for port in regular_inputs:
            name = port['name']
            if not name.endswith('_rsp_valid'):
                continue
            prefix = name[:-len('_rsp_valid')]
            cmd_valid_out = next(
                (
                    candidate for candidate in (
                        f"{prefix}_cmd_valid",
                        f"{prefix}_valid",
                        f"{prefix}_tvalid",
                    )
                    if candidate in output_by_name
                ),
                None,
            )
            if not cmd_valid_out:
                continue

            ready_in = next(
                (
                    candidate for candidate in (
                        f"{prefix}_cmd_ready",
                        f"{prefix}_ready",
                        f"{prefix}_tready",
                    )
                    if candidate in input_by_name
                ),
                None,
            )

            payload_inputs = sorted(
                (
                    p for p in regular_inputs
                    if p['name'].startswith(f"{prefix}_rsp_payload_")
                ),
                key=lambda p: p['name'],
            )

            iface_id = unique_iface_id(prefix)
            response_ifaces.append({
                "id": iface_id,
                "prefix": prefix,
                "cmd_valid_out": cmd_valid_out,
                "ready_in": ready_in,
                "rsp_valid_in": name,
                "payload_inputs": payload_inputs,
            })
            controlled_inputs.add(name)
            if ready_in:
                controlled_inputs.add(ready_in)
            for payload in payload_inputs:
                controlled_inputs.add(payload['name'])

        # Detect simple sink-style request/streaming interfaces where the DUT
        # exposes a ready output and expects the testbench to drive valid/data.
        for port in regular_inputs:
            name = port['name']
            if name in controlled_inputs:
                continue

            ready_out = None
            payload_inputs = []
            valid_in = None

            if name.endswith('_cmd_valid'):
                prefix = name[:-len('_cmd_valid')]
                ready_out = f"{prefix}_cmd_ready"
                if ready_out in output_by_name:
                    valid_in = name
                    payload_inputs = sorted(
                        (
                            p for p in regular_inputs
                            if p['name'].startswith(f"{prefix}_cmd_payload_")
                        ),
                        key=lambda p: p['name'],
                    )
            elif name.endswith('_tvalid'):
                prefix = name[:-len('_tvalid')]
                ready_out = f"{prefix}_tready"
                if ready_out in output_by_name:
                    valid_in = name
                    payload_names = [
                        f"{prefix}_tdata",
                        f"{prefix}_tkeep",
                        f"{prefix}_tstrb",
                        f"{prefix}_tlast",
                        f"{prefix}_tuser",
                    ]
                    payload_inputs = [input_by_name[pn] for pn in payload_names if pn in input_by_name]
            elif name.endswith('_valid') and not name.endswith('_tvalid') and not name.endswith('_cmd_valid'):
                # Generic valid/ready pair (e.g. s_valid/s_ready in custom FIR filters).
                # Only recognised when a matching _ready output is present so we
                # don't misclassify single-bit control strobes.
                prefix = name[:-len('_valid')]
                ready_out = f"{prefix}_ready"
                if ready_out in output_by_name:
                    valid_in = name
                    # Collect companion payload inputs (prefix_data, prefix_*data, etc.)
                    payload_inputs = sorted(
                        (
                            p for p in regular_inputs
                            if p['name'].startswith(f"{prefix}_")
                            and p['name'] != name
                            and not p['name'].endswith('_ready')
                        ),
                        key=lambda p: p['name'],
                    )

            if not valid_in or not ready_out:
                continue

            iface_id = unique_iface_id(prefix)
            request_ifaces.append({
                "id": iface_id,
                "prefix": prefix,
                "ready_out": ready_out,
                "valid_in": valid_in,
                "payload_inputs": payload_inputs,
            })
            controlled_inputs.add(valid_in)
            for payload in payload_inputs:
                controlled_inputs.add(payload['name'])

        # Any remaining ready-style input gets a high-bias driver if the DUT
        # has a matching valid output. This helps generic valid/ready sources.
        for port in regular_inputs:
            name = port['name']
            if name in controlled_inputs:
                continue
            match = re.match(r'^(.*)_(cmd_)?ready$', name)
            if not match:
                continue
            prefix = match.group(1)
            candidate_outputs = [
                f"{prefix}_cmd_valid",
                f"{prefix}_valid",
                f"{prefix}_tvalid",
            ]
            if any(candidate in output_by_name for candidate in candidate_outputs):
                ready_bias_inputs.append(name)
                controlled_inputs.add(name)

        # Detect HLS ap_memory interfaces: {prefix}_q{N} input where
        # {prefix}_address{N} and {prefix}_ce{N} exist as outputs.

        # Ports that need coordinate_byte_clamp: (q_name_lower, q_width, addr_width).
        # Rosetta 3D rendering reads byte-packed coordinates from input_r_q*/q1;
        # clamping keeps triangle areas small enough for the rasterizer to drain
        # within the simulation window.
        _COORDINATE_CLAMP_PORTS: set[tuple[str, int, int]] = {
            ("input_r_q0", 32, 14),
            ("input_r_q1", 32, 14),
        }

        def hls_memory_data_mode(q_name: str, q_width: int, addr_width: int) -> str:
            if ('rendering' in golden_module.lower() and
                    (q_name.lower(), q_width, addr_width) in _COORDINATE_CLAMP_PORTS):
                return "coordinate_byte_clamp"
            return "full_width_hash"

        for port in regular_inputs:
            name = port['name']
            if name in controlled_inputs:
                continue
            m_mem = re.match(r'^(.+)_q(\d+)$', name)
            if not m_mem:
                continue
            prefix, N = m_mem.group(1), m_mem.group(2)
            addr_port_name = f"{prefix}_address{N}"
            ce_port_name = f"{prefix}_ce{N}"
            if addr_port_name not in output_by_name or ce_port_name not in output_by_name:
                continue
            we_port_name = f"{prefix}_we{N}"
            addr_width = port_bit_width(output_by_name[addr_port_name])
            q_width = port_bit_width(port)
            hls_mem_ifaces.append({
                "q_in": name,
                "q_width": q_width,
                "addr_out": addr_port_name,
                "addr_width": addr_width,
                "ce_out": ce_port_name,
                "we_out": we_port_name if we_port_name in output_by_name else None,
                "data_mode": hls_memory_data_mode(name, q_width, addr_width),
            })
            controlled_inputs.add(name)
            logger.info(f"Detected HLS ap_memory: {name} driven by {addr_port_name}/{ce_port_name}")

        # Detect simple HLS-style control interfaces (ap_ctrl_hs) and drive
        # them transactionally. Randomly toggling ap_start and mutating all
        # memory/data inputs every cycle can leave these designs mostly idle
        # or hide latency differences on result-side ports.
        ap_start_in = input_by_name.get("ap_start")
        if ap_start_in:
            ap_done_out = "ap_done" if "ap_done" in output_by_name else None
            ap_idle_out = "ap_idle" if "ap_idle" in output_by_name else None
            ap_ready_out = "ap_ready" if "ap_ready" in output_by_name else None
            if ap_done_out or ap_idle_out or ap_ready_out:
                stable_inputs = [
                    p for p in regular_inputs
                    if p['name'] not in controlled_inputs and p['name'] != ap_start_in['name']
                ]
                hls_iface = {
                    "id": unique_iface_id("hls_ap_ctrl"),
                    "start_in": ap_start_in['name'],
                    "done_out": ap_done_out,
                    "idle_out": ap_idle_out,
                    "ready_out": ap_ready_out,
                    "stable_inputs": stable_inputs,
                }
                controlled_inputs.add(ap_start_in['name'])
                for port in stable_inputs:
                    controlled_inputs.add(port['name'])

        # Skip reactive stimulus if requested — fall back to pure LFSR
        if self.no_reactive:
            response_ifaces.clear()
            request_ifaces.clear()
            ready_bias_inputs.clear()
            hls_iface = None
            hls_mem_ifaces.clear()
            controlled_inputs.clear()

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
        
        def generate_fallback_random_assignments() -> str:
            """Generate plain LFSR stimulus for any input not covered by a reactive driver."""
            stim_lines = []
            lfsr_bit_index = 0
            for port in regular_inputs:
                if port['name'] in controlled_inputs:
                    continue
                name = port['name']
                bits = port_bit_width(port)
                if bits == 1:
                    stim_lines.append(f"            {name} = lfsr[{lfsr_bit_index % 32}];")
                    lfsr_bit_index += 1
                elif bits <= 32:
                    stim_lines.append(f"            {name} = lfsr[{bits-1}:0];")
                else:
                    stim_lines.append(assign_from_seed(port, "lfsr"))
            return '\n'.join(stim_lines) if stim_lines else '            // No fallback-random inputs'

        def generate_env_declarations() -> str:
            decls = []
            for iface in response_ifaces:
                iface_id = iface['id']
                decls.append(f"    reg env_{iface_id}_pending;")
                decls.append(f"    integer env_{iface_id}_delay;")
                decls.append(f"    reg [31:0] env_{iface_id}_seed;")
            if hls_iface:
                iface_id = hls_iface['id']
                decls.append(f"    reg env_{iface_id}_active;")
                decls.append(f"    reg env_{iface_id}_launch;")
                decls.append(f"    integer env_{iface_id}_cycles;")
                decls.append(f"    reg [31:0] env_{iface_id}_seed;")
            return '\n'.join(decls) if decls else '    // No reactive environment state'

        def generate_env_init_code() -> str:
            lines = []
            for iface in response_ifaces:
                iface_id = iface['id']
                lines.append(f"        env_{iface_id}_pending = 0;")
                lines.append(f"        env_{iface_id}_delay = 0;")
                lines.append(f"        env_{iface_id}_seed = 32'h{(0x13579BDF ^ (len(iface_id) * 0x1021)) & 0xFFFFFFFF:08X};")
            if hls_iface:
                iface_id = hls_iface['id']
                lines.append(f"        env_{iface_id}_active = 0;")
                lines.append(f"        env_{iface_id}_launch = 0;")
                lines.append(f"        env_{iface_id}_cycles = 0;")
                lines.append(f"        env_{iface_id}_seed = 32'h2468ACE1;")
            return '\n'.join(lines) if lines else '        // No reactive environment state to initialize'

        def generate_negedge_stimulus_code() -> str:
            lines = []

            for iface in response_ifaces:
                iface_id = iface['id']
                ready_in = iface['ready_in']
                rsp_valid_in = iface['rsp_valid_in']

                lines.append(f"            // Reactive responder for {iface['prefix']}")
                if ready_in:
                    lines.append(f"            if (env_{iface_id}_pending) begin")
                    lines.append(f"                {ready_in} = 0;")
                    lines.append("            end else begin")
                    lines.append(f"                {ready_in} = 1;")
                    lines.append("            end")

                lines.append(f"            if (env_{iface_id}_pending && env_{iface_id}_delay == 0) begin")
                lines.append(f"                {rsp_valid_in} = 1;")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if payload_name.endswith('_error'):
                        lines.append(f"                {payload_name} = 0;")
                    elif payload_name.endswith('_last'):
                        lines.append(f"                {payload_name} = 1;")
                    else:
                        lines.append(assign_from_seed(payload, f"env_{iface_id}_seed", indent="                "))
                lines.append("            end else begin")
                lines.append(f"                {rsp_valid_in} = 0;")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    lines.append(f"                {payload_name} = 0;")
                lines.append("            end")

            for iface in request_ifaces:
                ready_expr = f"golden_{iface['ready_out']}"
                lines.append(f"            // Reactive request driver for {iface['prefix']}")
                lines.append(f"            if ({ready_expr}) begin")
                lines.append(f"                {iface['valid_in']} = 1;")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if payload_name.endswith('_last'):
                        lines.append(f"                {payload_name} = 1;")
                    else:
                        lines.append(assign_from_seed(payload, "lfsr", indent="                "))
                lines.append("            end else begin")
                lines.append(f"                {iface['valid_in']} = lfsr[0];")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if payload_name.endswith('_last'):
                        lines.append(f"                {payload_name} = 0;")
                    else:
                        lines.append(assign_from_seed(payload, "lfsr", indent="                "))
                lines.append("            end")

            for ready_name in ready_bias_inputs:
                lines.append(f"            {ready_name} = 1;")

            if hls_iface:
                iface_id = hls_iface['id']
                start_in = hls_iface['start_in']
                stable_inputs = hls_iface['stable_inputs']
                lines.append("            // Transactional HLS control driver")
                lines.append(f"            if (env_{iface_id}_launch) begin")
                lines.append(f"                {start_in} = 1;")
                for idx, port in enumerate(stable_inputs):
                    seed_expr = f"(env_{iface_id}_seed ^ 32'h{(0x10203040 ^ (idx * 0x1F123BB5)) & 0xFFFFFFFF:08X})"
                    lines.append(assign_from_seed(port, seed_expr, indent="                "))
                lines.append("            end else if (env_{0}_active) begin".format(iface_id))
                lines.append(f"                {start_in} = 1;  // Keep high for streaming/long-running kernels")
                for idx, port in enumerate(stable_inputs):
                    seed_expr = f"(env_{iface_id}_seed ^ 32'h{(0x10203040 ^ (idx * 0x1F123BB5)) & 0xFFFFFFFF:08X})"
                    lines.append(assign_from_seed(port, seed_expr, indent="                "))
                lines.append("            end else begin")
                lines.append(f"                {start_in} = 0;")
                for port in stable_inputs:
                    lines.append(f"                {port['name']} = 0;")
                lines.append("            end")

            lines.append(generate_fallback_random_assignments())
            return '\n'.join(lines)

        def generate_posedge_bookkeeping_code() -> str:
            lines = []
            for idx, iface in enumerate(response_ifaces):
                iface_id = iface['id']
                cmd_valid_out = f"golden_{iface['cmd_valid_out']}"
                ready_gate = iface['ready_in'] if iface['ready_in'] else "1'b1"
                seed_mask = (0x9E3779B9 ^ (idx * 0x45D9F3B)) & 0xFFFFFFFF
                lines.append(f"            if (env_{iface_id}_pending) begin")
                lines.append(f"                if (env_{iface_id}_delay > 0) begin")
                lines.append(f"                    env_{iface_id}_delay = env_{iface_id}_delay - 1;")
                lines.append("                end else begin")
                lines.append(f"                    env_{iface_id}_pending = 0;")
                lines.append("                end")
                lines.append(f"            end else if ({cmd_valid_out} && {ready_gate}) begin")
                lines.append(f"                env_{iface_id}_pending = 1;")
                lines.append(f"                env_{iface_id}_delay = lfsr[0];")
                lines.append(f"                env_{iface_id}_seed = lfsr ^ 32'h{seed_mask:08X};")
                lines.append("            end")
            if hls_iface:
                iface_id = hls_iface['id']
                done_expr = f"golden_{hls_iface['done_out']}" if hls_iface['done_out'] else "1'b0"
                idle_expr = f"golden_{hls_iface['idle_out']}" if hls_iface['idle_out'] else "1'b0"
                ready_expr = f"golden_{hls_iface['ready_out']}" if hls_iface['ready_out'] else "1'b0"
                launch_condition = f"({idle_expr} || {ready_expr})"
                lines.append(f"            if (env_{iface_id}_launch) begin")
                lines.append(f"                env_{iface_id}_launch = 0;")
                lines.append(f"                env_{iface_id}_cycles = 1;")
                lines.append(f"            end else if (env_{iface_id}_active) begin")
                lines.append(f"                env_{iface_id}_cycles = env_{iface_id}_cycles + 1;")
                lines.append(f"                if ({done_expr}) begin")
                lines.append(f"                    // Transaction boundary: refresh seed but stay active for continuous execution.")
                lines.append(f"                    env_{iface_id}_seed = lfsr ^ 32'hA5A55A5A;")
                lines.append(f"                    env_{iface_id}_cycles = 0;")
                lines.append("                end")
                lines.append(f"            end else if ({launch_condition}) begin")
                lines.append(f"                env_{iface_id}_active = 1;")
                lines.append(f"                env_{iface_id}_launch = 1;")
                lines.append(f"                env_{iface_id}_cycles = 0;")
                lines.append(f"                env_{iface_id}_seed = lfsr ^ 32'hA5A55A5A;")
                lines.append("            end")
            return '\n'.join(lines) if lines else '            // No reactive bookkeeping required'

        def generate_protocol_compare_code() -> str:
            protocol_outputs = []
            seen_protocol_outputs = set()

            def add_protocol_output(name: Optional[str]):
                if name and name in output_by_name and name not in seen_protocol_outputs:
                    seen_protocol_outputs.add(name)
                    protocol_outputs.append(name)

            for iface in response_ifaces:
                add_protocol_output(iface['cmd_valid_out'])
            for iface in request_ifaces:
                add_protocol_output(iface['ready_out'])
            if hls_iface:
                add_protocol_output(hls_iface['done_out'])
                add_protocol_output(hls_iface['idle_out'])
                add_protocol_output(hls_iface['ready_out'])

            lines = []
            for name in protocol_outputs:
                lines.append(f'''
            if (golden_{name} !== revised_{name}) begin
                $display("PROTOCOL MISMATCH at cycle %0d: {name} golden=%h revised=%h", cycle_count, golden_{name}, revised_{name});
                protocol_mismatch_count = protocol_mismatch_count + 1;
            end''')
            return '\n'.join(lines) if lines else '            // No protocol outputs to compare'

        def generate_hls_mem_model_code() -> str:
            if not hls_mem_ifaces:
                return ''
            lines = ["    // Behavioral memory model for HLS ap_memory interfaces"]
            if any(iface['data_mode'] == "coordinate_byte_clamp" for iface in hls_mem_ifaces):
                lines.extend([
                    "    function [7:0] hls_mem_byte_hash;",
                    "        input [15:0] addr;",
                    "        input [15:0] prime;",
                    "        input [15:0] offset;",
                    "        begin",
                    "            hls_mem_byte_hash = ((addr * prime + offset) >> 8) & 8'h07;",
                    "        end",
                    "    endfunction",
                    "",
                ])
            lines.append(f"    always @(posedge {clock}) begin")
            for idx, iface in enumerate(hls_mem_ifaces):
                q_in       = iface['q_in']
                q_width    = iface['q_width']
                addr_out   = iface['addr_out']
                addr_width = iface['addr_width']
                ce_out     = iface['ce_out']
                we_out     = iface['we_out']
                data_mode  = iface['data_mode']
                we_check = f" && !golden_{we_out}" if we_out else ""
                lines.append(f"        if (golden_{ce_out}{we_check}) begin")
                if data_mode == "coordinate_byte_clamp":
                    # 32-bit ports carry byte-packed coordinates (e.g., 3D rendering).
                    # Use independent 16-bit hashes per byte so each coordinate field
                    # gets a distinct value, and mask to [0,31] to keep triangle areas
                    # small enough for the HLS rasterizer to drain the input FIFO
                    # within the simulation window.
                    pad16 = max(0, 16 - addr_width)
                    if pad16 > 0:
                        addr16 = f"{{{pad16}'b0, golden_{addr_out}}}"
                    elif addr_width == 16:
                        addr16 = f"golden_{addr_out}"
                    else:
                        addr16 = f"golden_{addr_out}[15:0]"
                    # Four independent primes and offsets; per-interface salt via XOR with idx
                    byte_primes = [0xA15B, 0x6C3D, 0x9E37, 0x4F2B]
                    byte_offsets = [0xA500, 0x5A00, 0x3C00, 0xC300]
                    byte_primes  = [(p ^ (idx * 0x0101) | 1) & 0xFFFF for p in byte_primes]
                    byte_offsets = [(o ^ (idx * 0x1010)) & 0xFFFF for o in byte_offsets]
                    # Build {byte3, byte2, byte1, byte0} — byte3 is MSB of the 32-bit word.
                    # Mask to [0,7] (3 bits) so rasterization stays fast enough to drain the
                    # FIFO within the simulation comparison window.
                    parts_msb_first = []
                    for b in range(3, -1, -1):
                        p = byte_primes[b]
                        o = byte_offsets[b]
                        parts_msb_first.append(
                            f"hls_mem_byte_hash({addr16}, 16'h{p:04X}, 16'h{o:04X})"
                        )
                    lines.append(f"            {q_in} <= {{{', '.join(parts_msb_first)}}};")
                else:
                    # Wide ports (e.g., 64-bit optical-flow frames): full-width multiply-XOR hash.
                    q_mask = (1 << q_width) - 1
                    prime  = 0x9E3779B97F4A7C15 & q_mask
                    salt   = (0xA5A5A5A5A5A5A5A5 ^ (idx * 0x1F1F1F1F1F1F1F1F)) & q_mask
                    ext_zeros = q_width - addr_width
                    if ext_zeros > 0:
                        addr_ext = f"{{{ext_zeros}'b0, golden_{addr_out}}}"
                    elif ext_zeros == 0:
                        addr_ext = f"golden_{addr_out}"
                    else:
                        addr_ext = f"golden_{addr_out}[{q_width-1}:0]"
                    lines.append(f"            {q_in} <= ({addr_ext} * {q_width}'h{prime:X}) ^ {q_width}'h{salt:X};")
                lines.append("        end")
            lines.append("    end")
            return '\n'.join(lines)

        # Generate testbench
        compare_body = chr(10).join(f'''
            if (golden_{port['name']} !== revised_{port['name']}) begin
                $display("MISMATCH at cycle %0d: {port['name']} golden=%h revised=%h", cycle_count, golden_{port['name']}, revised_{port['name']});
                mismatch_count = mismatch_count + 1;
            end''' for port in outputs) if outputs else '            // No outputs to compare'

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
    
    // Reactive environment state
{generate_env_declarations()}
    
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

{generate_hls_mem_model_code()}
    // Test stimulus and checking
    integer mismatch_count;
    integer protocol_mismatch_count;
    integer cycle_count;
    integer num_vectors;
    
    initial begin
        mismatch_count = 0;
        protocol_mismatch_count = 0;
        cycle_count = 0;
        num_vectors = {self.num_vectors};
        if (!$value$plusargs("NUM_VECTORS=%d", num_vectors)) begin
            num_vectors = {self.num_vectors};
        end
        $display("Configured test vectors: %0d", num_vectors);
{generate_env_init_code()}
        
        // Reset ({'active-low' if reset_active_low else 'active-high'})
        {''+reset+' = '+reset_assert+';' if reset else ''}
        {chr(10).join(f"        {port['name']} = 0;" for port in regular_inputs)}
        repeat(10) @(posedge {clock});
        {''+reset+' = '+reset_deassert+';' if reset else ''}
        
        // Warm-up period: fill pipeline without checking outputs.
        // Drive inputs on the inactive edge so sequential logic sees stable
        // values before the active clock edge.
        repeat(50) begin
            @(negedge {clock});
            lfsr = lfsr_next(lfsr);
{generate_negedge_stimulus_code()}
            @(posedge {clock});
{generate_posedge_bookkeeping_code()}
            #1;
{generate_protocol_compare_code()}
        end
        
        // Run test vectors with output checking
        repeat(num_vectors) begin
            // Generate new inputs from LFSR
            @(negedge {clock});
            lfsr = lfsr_next(lfsr);
{generate_negedge_stimulus_code()}

            // Sample outputs on the following active edge
            @(posedge {clock});
            cycle_count = cycle_count + 1;
{generate_posedge_bookkeeping_code()}
            
            // Check outputs after settling
            #1; // Small delay for output settling
            
            // Compare all outputs
{compare_body}

            // Compare protocol outputs that drive the reactive environment
{generate_protocol_compare_code()}
        end
        
        // Report results
        $display("\\n=======================================");
        $display("SIMULATION COMPLETE");
        $display("=======================================");
        $display("Cycles simulated: %0d", cycle_count);
        {'$display("Outputs compared: 0 (design has no outputs)");' if not outputs else '$display("Mismatches found: %0d", mismatch_count);'}
        $display("Protocol mismatches found: %0d", protocol_mismatch_count);
        if (mismatch_count == 0 && protocol_mismatch_count == 0) begin
            {'$display("Result: PASS (no crashes detected)");' if not outputs else '$display("Result: PASS");'}
            $finish(0);
        end else begin
            $display("Result: FAIL");
            $finish(1);
        end
    end
    
    // Timeout watchdog (reset + warmup + test cycles, with 2x safety margin)
    initial begin
        #1;
        #((10 + 50 + num_vectors) * 20) $display("ERROR: Simulation timeout"); $finish(2);
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
        
        # Query clock ports while the golden design is still loaded - more
        # robust than guessing from Verilog port names downstream.
        golden_clocks = await self._query_clock_ports_from_vivado()
        if golden_clocks:
            logger.info(f"Vivado reports golden clock ports: {golden_clocks}")
        else:
            logger.info("Vivado reported no clocks for golden design (will fall back to name heuristic)")
        
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
        
        # Query clock ports for the revised design as well, mainly so we can
        # detect mismatches that would invalidate the testbench (the testbench
        # is built from golden's port list, but revised must agree).
        revised_clocks = await self._query_clock_ports_from_vivado()
        if revised_clocks:
            logger.info(f"Vivado reports revised clock ports: {revised_clocks}")
        if golden_clocks and revised_clocks and set(golden_clocks) != set(revised_clocks):
            logger.warning(
                f"Clock port set differs between golden ({sorted(golden_clocks)}) "
                f"and revised ({sorted(revised_clocks)}); using golden's"
            )
        
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
        self.generate_testbench(
            golden_info, revised_info, tb_path,
            clock_names=golden_clocks,
        )
        print(f"✓ Testbench generated: {tb_path.name}")
        
        # Run xsim simulation
        print("\nRunning xsim simulation...")
        print("(This may take a few minutes...)")
        
        xsim_dir = self.temp_dir / "xsim_work"
        xsim_dir.mkdir(exist_ok=True)
        
        current_step = "setup"
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
            # (both declarations AND instantiations of those modules) so that the
            # renamed-revised top is a self-contained design that doesn't silently
            # link against golden sub-modules at elaboration time.
            logger.info("Renaming all revised modules to avoid conflicts...")
            revised_renamed = xsim_dir / "revised_sim_renamed.v"
            
            with open(revised_v, 'r') as f:
                content = f.read()
            
            revised_module_name = revised_info["module_name"]
            revised_module_renamed = f"{revised_module_name}_revised"
            
            # Pass 1: collect every declared module name in the revised netlist.
            # Vivado's funcsim output is regular: each declaration starts with
            # "module <name>" at the start of a line.
            declared_module_names = set(re.findall(
                r'(?m)^\s*module\s+(\w+)\b',
                content
            ))
            logger.info(
                f"Found {len(declared_module_names)} module declarations in revised netlist"
            )
            
            # Pass 2: per-line scan rewriting both declarations and instantiations
            # of declared modules to use the "_revised" suffix. We use a per-line
            # scan with set-membership lookups (O(file size)) rather than a giant
            # alternation regex, which is intractably slow on big benchmarks
            # (e.g. corescore_500_mod has ~6700 user modules in a 71 MB netlist).
            #
            # Lines we care about in Vivado funcsim output:
            #   "  module NAME"               -> declaration to rename
            #   "  NAME inst_name (..."       -> instantiation to rename
            # Anything else (port lists, assigns, wires, comments, primitives
            # like LUT6/FDRE which are NOT in declared_module_names) is left alone.
            if declared_module_names:
                renamed_lines = []
                suffix = "_revised"
                for line in content.splitlines(keepends=True):
                    s = line.lstrip()
                    if not s:
                        renamed_lines.append(line); continue
                    sp = s.find(' ')
                    tab = s.find('\t')
                    if tab != -1 and (sp == -1 or tab < sp):
                        sp = tab
                    if sp == -1:
                        renamed_lines.append(line); continue
                    first = s[:sp]
                    if first == 'module':
                        rest = s[sp:].lstrip()
                        nm = re.match(r'(\w+)', rest)
                        if nm and nm.group(1) in declared_module_names:
                            name = nm.group(1)
                            indent = line[:len(line) - len(s)]
                            after_name = len(s) - len(rest) + len(name)
                            renamed_lines.append(
                                indent + 'module ' + name + suffix + s[after_name:]
                            )
                            continue
                    elif first in declared_module_names:
                        indent = line[:len(line) - len(s)]
                        renamed_lines.append(
                            indent + first + suffix + s[len(first):]
                        )
                        continue
                    renamed_lines.append(line)
                content = ''.join(renamed_lines)
            
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
            
            # Timeouts are generous because large benchmarks (e.g. corescore_500_mod
            # with thousands of user modules) can take many minutes per step.
            # xvlog/xelab times scale with design size; xsim time scales with both
            # design size and num_vectors.
            xvlog_timeout_s = 1800   # 30 min
            xelab_timeout_s = 3600   # 60 min - elaborates both designs
            # xsim: per-cycle cost is roughly proportional to the number of
            # primitive cells. Two copies of a 100k-LUT design (e.g.
            # corescore_500_mod) measured ~0.3s/cycle. We add a generous baseline
            # for kernel init and cap below by 60 min so small designs still get
            # a comfortable budget.
            def xsim_timeout_for(vector_count: int) -> int:
                return max(3600, 600 + int(vector_count * 1.0))
            
            current_step = "xvlog (compilation)"
            result = subprocess.run(
                compile_cmd,
                cwd=xsim_dir,
                capture_output=True,
                text=True,
                timeout=xvlog_timeout_s
            )
            
            if result.returncode != 0:
                print(f"\n✗ Compilation failed:")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("✓ Compilation successful")
            
            # Elaborate with UNISIM library reference.
            #
            # We pass "--debug off" because the testbench reports results via
            # $display; we never inspect waveforms, so the per-signal debug
            # instrumentation that "-debug typical" enables is pure overhead
            # (relevant for smaller benchmarks where it dominates).
            #
            # "--mt auto" lets xelab parallelise where it can; it is a no-op for
            # the per-module compile loop on huge designs but doesn't hurt.
            logger.info("Elaborating with xelab...")
            elab_cmd = [
                "xelab",
                "--debug", "off",
                "--mt", "auto",
                "-L", "unisims_ver",  # Link against UNISIM library
                "-L", "unimacro_ver",
                "work.testbench",  # Specify library.module
                "work.glbl",       # Include glbl for initialization
                "-s", "testbench_sim"
            ]
            
            current_step = "xelab (elaboration)"
            result = subprocess.run(
                elab_cmd,
                cwd=xsim_dir,
                capture_output=True,
                text=True,
                timeout=xelab_timeout_s
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
            
            def run_xsim(vector_count: int, label: str) -> dict:
                logger.info(f"Running {label} simulation with xsim ({vector_count} vectors)...")
                print(f"\nSimulating {vector_count} test vectors ({label})...")

                sim_cmd = [
                    "xsim",
                    "testbench_sim",
                    "-R",
                    "--testplusarg", f"NUM_VECTORS={vector_count}",
                ]

                result = subprocess.run(
                    sim_cmd,
                    cwd=xsim_dir,
                    capture_output=True,
                    text=True,
                    timeout=xsim_timeout_for(vector_count)
                )

                sim_output = result.stdout + result.stderr

                log_file = self.temp_dir / f"simulation_{label}.log"
                with open(log_file, 'w') as f:
                    f.write(sim_output)

                if label == "full":
                    # Preserve the historical log path for scripts/users that
                    # expect a single final simulation.log.
                    final_log = self.temp_dir / "simulation.log"
                    with open(final_log, 'w') as f:
                        f.write(sim_output)

                parse_result = parse_simulation_output(
                    sim_output,
                    result.returncode,
                    vector_count,
                )

                for line in sim_output.split('\n'):
                    if 'MISMATCH' in line:
                        print(f"  {line}")

                report = {
                    "vectors_requested": vector_count,
                    "cycles_simulated": parse_result["cycles_simulated"],
                    "mismatch_count": parse_result["mismatch_count"],
                    "protocol_mismatch_count": parse_result["protocol_mismatch_count"],
                    "log_file": str(log_file),
                    "result_pass_seen": parse_result["result_pass_seen"],
                    "result_fail_seen": parse_result["result_fail_seen"],
                    "simulator_failed": parse_result["simulator_failed"],
                    "infrastructure_failure": parse_result["infrastructure_failure"],
                    "infrastructure_reason": parse_result["infrastructure_reason"],
                    "returncode": parse_result["returncode"],
                    "passed": parse_result["passed"],
                }

                print("\n" + "-"*70)
                print(f"Simulation Results ({label}):")
                print(f"  Requested vectors: {vector_count}")
                print(f"  Cycles: {parse_result['cycles_simulated']}")
                print(f"  Mismatches: {parse_result['mismatch_count']}")
                print(f"  Protocol mismatches: {parse_result['protocol_mismatch_count']}")
                print(f"  Log: {log_file}")
                if parse_result["simulator_failed"]:
                    print("  Simulator reported an internal failure")
                if not parse_result["result_pass_seen"] and result.returncode == 0:
                    print("  Testbench PASS marker not found")
                print(f"  Result: {'PASSED' if parse_result['passed'] else 'FAILED'}")
                print("-"*70)

                return report

            current_step = "xsim (precheck simulation)"
            precheck_report = None
            run_precheck = (
                self.precheck_vectors > 0
                and self.precheck_vectors < self.num_vectors
            )
            if run_precheck:
                precheck_report = run_xsim(self.precheck_vectors, "precheck")
                if not precheck_report["passed"]:
                    self.simulation_report = {
                        "precheck_vectors": self.precheck_vectors,
                        "precheck_passed": False,
                        "precheck_report": precheck_report,
                        "full_vectors": self.num_vectors,
                        "full_run_skipped": True,
                        "cycles_simulated": precheck_report["cycles_simulated"],
                        "mismatch_count": precheck_report["mismatch_count"],
                        "protocol_mismatch_count": precheck_report["protocol_mismatch_count"],
                        "log_file": precheck_report["log_file"],
                        "result_pass_seen": precheck_report["result_pass_seen"],
                        "result_fail_seen": precheck_report["result_fail_seen"],
                        "simulator_failed": precheck_report["simulator_failed"],
                        "infrastructure_failure": precheck_report["infrastructure_failure"],
                        "infrastructure_reason": precheck_report["infrastructure_reason"],
                        "returncode": precheck_report["returncode"],
                    }
                    if precheck_report["infrastructure_failure"]:
                        self.infrastructure_failure = True
                        self.infrastructure_reason = precheck_report["infrastructure_reason"]
                    self.phase2_passed = False
                    if self.infrastructure_failure:
                        print("\nPhase 2: INFRASTRUCTURE FAILURE ⊘ (precheck crashed or did not complete; full run skipped)")
                    else:
                        print("\nPhase 2: FAILED ✗ (precheck failed; full run skipped)")
                    return False

            current_step = "xsim (full simulation)"
            full_report = run_xsim(self.num_vectors, "full")

            self.simulation_report = {
                "precheck_vectors": self.precheck_vectors if run_precheck else None,
                "precheck_passed": precheck_report["passed"] if precheck_report else None,
                "precheck_report": precheck_report,
                "full_vectors": self.num_vectors,
                "full_report": full_report,
                "cycles_simulated": full_report["cycles_simulated"],
                "mismatch_count": full_report["mismatch_count"],
                "protocol_mismatch_count": full_report["protocol_mismatch_count"],
                "log_file": str(self.temp_dir / "simulation.log"),
                "result_pass_seen": full_report["result_pass_seen"],
                "result_fail_seen": full_report["result_fail_seen"],
                "simulator_failed": full_report["simulator_failed"],
                "infrastructure_failure": full_report["infrastructure_failure"],
                "infrastructure_reason": full_report["infrastructure_reason"],
                "returncode": full_report["returncode"],
            }

            # Check if passed. xsim can report an internal simulator failure
            # while still returning 0, so require the testbench's explicit
            # completion marker and expected cycle count.
            self.phase2_passed = full_report["passed"]
            if full_report["infrastructure_failure"]:
                self.infrastructure_failure = True
                self.infrastructure_reason = full_report["infrastructure_reason"]

            if self.phase2_passed:
                print("\nPhase 2: PASSED ✓")
            elif self.infrastructure_failure:
                print("\nPhase 2: INFRASTRUCTURE FAILURE ⊘")
            else:
                print("\nPhase 2: FAILED ✗")
            
            return self.phase2_passed
            
        except subprocess.TimeoutExpired as e:
            timeout_s = getattr(e, "timeout", "?")
            print(f"\n✗ Timeout in {current_step} after {timeout_s}s")
            logger.error(
                f"subprocess.TimeoutExpired in step '{current_step}' "
                f"(timeout={timeout_s}s)"
            )
            self.infrastructure_failure = True
            self.infrastructure_reason = f"timeout_in_{current_step}"
            self.simulation_report = {
                "infrastructure_failure": True,
                "infrastructure_reason": self.infrastructure_reason,
                "stage": current_step,
                "timeout_seconds": timeout_s,
            }
            return False
        except Exception as e:
            print(f"\n✗ Simulation error in {current_step}: {e}")
            logger.exception(f"Error in step '{current_step}'")
            self.infrastructure_failure = True
            self.infrastructure_reason = f"exception_in_{current_step}"
            self.simulation_report = {
                "infrastructure_failure": True,
                "infrastructure_reason": self.infrastructure_reason,
                "stage": current_step,
                "error": str(e),
            }
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
        if 0 < self.precheck_vectors < self.num_vectors:
            print(f"Precheck vectors: {self.precheck_vectors}")
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
        elif self.infrastructure_failure:
            print("Phase 2 (Simulation): INFRASTRUCTURE FAILURE ⊘")
            print(f"  Reason: {self.infrastructure_reason}")
            if self.simulation_report:
                precheck_report = self.simulation_report.get("precheck_report")
                if precheck_report:
                    print(
                        f"  Precheck: {precheck_report.get('cycles_simulated', 0)} cycles, "
                        f"{precheck_report.get('mismatch_count', 0)} mismatches, "
                        f"{precheck_report.get('protocol_mismatch_count', 0)} protocol mismatches"
                    )
        else:
            print("Phase 2 (Simulation): " + ("PASSED ✓" if self.phase2_passed else "FAILED ✗" if self.phase1_passed else "SKIPPED"))
            if self.simulation_report:
                precheck_report = self.simulation_report.get("precheck_report")
                if precheck_report:
                    print(
                        f"  Precheck: {precheck_report.get('cycles_simulated', 0)} cycles, "
                        f"{precheck_report.get('mismatch_count', 0)} mismatches, "
                        f"{precheck_report.get('protocol_mismatch_count', 0)} protocol mismatches"
                    )
                print(f"  Cycles: {self.simulation_report.get('cycles_simulated', 0)}")
                print(f"  Mismatches: {self.simulation_report.get('mismatch_count', 0)}")
                print(f"  Protocol mismatches: {self.simulation_report.get('protocol_mismatch_count', 0)}")
        
        print()
        if self.phase2_skipped:
            overall_result = "PASSED ✓ (structural only)" if self.phase1_passed else "FAILED ✗"
        elif self.infrastructure_failure:
            overall_result = "INFRASTRUCTURE FAILURE ⊘"
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
            "precheck_vectors": self.precheck_vectors,
            "runtime_seconds": elapsed_time,
            "phase1_passed": self.phase1_passed,
            "phase2_passed": self.phase2_passed,
            "infrastructure_failure": self.infrastructure_failure,
            "infrastructure_reason": self.infrastructure_reason,
            "overall_passed": self.phase1_passed and self.phase2_passed,
            "preflight_report": self.preflight_report,
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
  python validate_dcps.py golden.dcp optimized.dcp --vectors 1000 --precheck-vectors 100
  python validate_dcps.py golden.dcp optimized.dcp --debug
        """
    )
    parser.add_argument("golden_dcp", type=Path, help="Golden (reference) DCP file")
    parser.add_argument("revised_dcp", type=Path, help="Revised (optimized) DCP file to validate")
    parser.add_argument(
        "--vectors",
        "-n",
        type=int,
        default=200,
        help="Number of random test vectors to simulate (default: 200). "
             "Larger benchmarks (e.g. corescore_500_mod) cost ~1s of xsim CPU "
             "per vector, so the default keeps wall-clock reasonable. Bump this "
             "up for higher-confidence runs."
    )
    parser.add_argument(
        "--precheck-vectors",
        type=int,
        default=100,
        help="Run this many vectors before the full simulation and skip the full "
             "run if the precheck fails (default: 100, disabled if 0 or >= --vectors)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-reactive",
        action="store_true",
        help="Disable reactive stimulus generation (use pure LFSR randomness)"
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

    if args.vectors <= 0:
        print("Error: --vectors must be positive", file=sys.stderr)
        sys.exit(1)

    if args.precheck_vectors < 0:
        print("Error: --precheck-vectors must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    # Run validation
    validator = DCPValidator(
        golden_dcp=args.golden_dcp,
        revised_dcp=args.revised_dcp,
        num_vectors=args.vectors,
        precheck_vectors=args.precheck_vectors,
        debug=args.debug,
        no_reactive=args.no_reactive
    )
    
    try:
        await validator.start_servers()
        success = await validator.validate()
        
        if validator.infrastructure_failure:
            sys.exit(2)
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
