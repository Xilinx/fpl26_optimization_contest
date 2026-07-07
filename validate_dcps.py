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
import hashlib
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
        num_vectors: int = 1000,
        precheck_vectors: int = 100,
        debug: bool = False,
        no_reactive: bool = False,
        use_sim_cache: bool = True,
        sim_cache_dir: Optional[Path] = None,
        lfsr_seed: Optional[int] = None,
    ):
        self.golden_dcp = golden_dcp
        self.revised_dcp = revised_dcp
        self.num_vectors = num_vectors
        self.precheck_vectors = precheck_vectors
        self.debug = debug
        self.no_reactive = no_reactive
        self.use_sim_cache = use_sim_cache
        self.lfsr_seed = lfsr_seed if lfsr_seed is not None else int.from_bytes(os.urandom(4), 'big')

        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None

        # Create temporary directory for intermediate files in workspace
        # (avoids /tmp running out of space for large designs)
        workspace_dir = Path(__file__).parent
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dcp_validation_", dir=workspace_dir))
        self.sim_cache_dir = sim_cache_dir or (workspace_dir / ".xsim_validation_cache")
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

    class _LocalToolResult:
        def __init__(self, content):
            self.content = content

    class _LocalToolSession:
        def __init__(self, handler):
            self._handler = handler

        async def call_tool(self, name: str, arguments: dict):
            return DCPValidator._LocalToolResult(
                await self._handler(name, arguments)
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _trace_simulation_cache_key(self, vivado_path: str, role: str) -> Tuple[str, dict]:
        """Return a cache key for split golden-trace/revised-replay snapshots."""
        script_path = Path(__file__).resolve()
        manifest = {
            "cache_schema": 2,
            "role": role,
            "golden_dcp": str(self.golden_dcp.resolve()),
            "golden_sha256": self._sha256_file(self.golden_dcp.resolve()),
            "validator_sha256": self._sha256_file(script_path),
            "vivado_path": str(Path(vivado_path).resolve()),
            "no_reactive": self.no_reactive,
        }
        if role == "revised_replay":
            manifest.update({
                "revised_dcp": str(self.revised_dcp.resolve()),
                "revised_sha256": self._sha256_file(self.revised_dcp.resolve()),
            })
        elif role != "golden_trace":
            raise ValueError(f"Unknown trace simulation cache role: {role}")

        key_material = json.dumps(manifest, sort_keys=True).encode("utf-8")
        # Store seed in manifest after computing the key so it doesn't affect cache lookup.
        # The xelab snapshot is seed-independent (seed passed at xsim runtime via +LFSR_SEED).
        manifest["lfsr_seed"] = f"{self.lfsr_seed:08X}"
        return hashlib.sha256(key_material).hexdigest(), manifest

    @staticmethod
    def _xsim_snapshot_exists(xsim_dir: Path) -> bool:
        xsim_root = xsim_dir / "xsim.dir"
        if not xsim_root.is_dir():
            return False
        return any(path.name == "xsimk" for path in xsim_root.glob("*/xsimk"))

    def _restore_cached_xsim_work(self, cache_entry: Path, xsim_dir: Path) -> bool:
        cache_work = cache_entry / "xsim_work"
        if not self._xsim_snapshot_exists(cache_work):
            return False
        if xsim_dir.exists():
            shutil.rmtree(xsim_dir)
        shutil.copytree(cache_work, xsim_dir)
        return self._xsim_snapshot_exists(xsim_dir)

    def _store_cached_xsim_work(self, cache_entry: Path, xsim_dir: Path, manifest: dict):
        if not self.use_sim_cache or not self._xsim_snapshot_exists(xsim_dir):
            return

        self.sim_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_entry = cache_entry.with_name(f"{cache_entry.name}.tmp-{os.getpid()}")
        if tmp_entry.exists():
            shutil.rmtree(tmp_entry)
        tmp_entry.mkdir(parents=True)
        shutil.copytree(xsim_dir, tmp_entry / "xsim_work")
        manifest = dict(manifest)
        manifest["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(tmp_entry / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        if cache_entry.exists():
            shutil.rmtree(cache_entry)
        os.replace(tmp_entry, cache_entry)

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

        # The installed MCP stdio stack in this environment can initialize
        # subprocess servers but then stall on subsequent tool requests. Use the
        # same tool handlers in-process; the validator-facing call_tool contract
        # stays identical.
        logger.info("Starting in-process RapidWright and Vivado tool sessions...")
        for tool_dir in (script_dir / "RapidWrightMCP", script_dir / "VivadoMCP"):
            tool_dir_str = str(tool_dir)
            if tool_dir_str not in sys.path:
                sys.path.insert(0, tool_dir_str)
        from RapidWrightMCP import server as rapidwright_server
        from VivadoMCP import vivado_mcp_server as vivado_server

        vivado_server._vivado_log_file = str(vivado_log)
        vivado_server._vivado_journal_file = str(vivado_journal)

        self.rapidwright_session = self._LocalToolSession(rapidwright_server.call_tool)
        self.vivado_session = self._LocalToolSession(vivado_server.call_tool)
        logger.info("In-process tool sessions started")
        return

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

    def generate_testbench(
        self,
        golden_info: dict,
        revised_info: dict,
        tb_path: Path,
        clock_names: Optional[list] = None,
        mode: str = "golden_trace",
        trace_filename: str = "golden_trace.hex",
    ):
        """Generate Verilog testbench for comparing two designs.

        ``clock_names`` is the authoritative list of top-level clock port names
        as reported by Vivado's ``get_clocks``/``get_ports`` traversal. When
        provided we use it directly; otherwise we fall back to a name-based
        heuristic (which fails for Chisel/Rocket-Chip-style designs whose
        clocks are named ``clock`` rather than ``clk``).
        """
        if mode not in {"golden_trace", "revised_replay"}:
            raise ValueError(f"Unknown testbench mode: {mode}")

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
        resets = [p for p in inputs
                  if ('rst' in p['name'].lower() or 'reset' in p['name'].lower())
                  and port_bit_width(p) == 1]

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

        # Designs may expose multiple reset-related ports (e.g. VexRiscv has
        # both 'debugReset' and 'reset'). Build assertion/de-assertion lines
        # for ALL of them so the CPU is not left with a floating CLR input.
        def _reset_level(name: str) -> tuple:
            al = bool(re.search(r'(rst|reset)_?n$', name.lower()))
            return ('0' if al else '1', '1' if al else '0')

        all_reset_assert_lines   = '\n'.join(
            f"        {r['name']} = {_reset_level(r['name'])[0]};" for r in resets)
        all_reset_deassert_lines = '\n'.join(
            f"        {r['name']} = {_reset_level(r['name'])[1]};" for r in resets)

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

        # Detect CPU-specific features from the golden Verilog netlist. Both
        # VexRiscv and BoomSoC detection read the text once and share it.
        golden_v_text = None
        icache_bootstrap = None
        tilelink_boom = False
        if golden_info.get('verilog_path') and not self.no_reactive:
            try:
                golden_v_text = Path(golden_info['verilog_path']).read_text(errors='replace')
                icache_bootstrap = self._detect_icache_bootstrap(golden_v_text)
                if icache_bootstrap:
                    logger.info(
                        "VexRiscv InstructionCache detected — enabling 8-word burst "
                        "iBus bootstrap and DSP multiply stimulus"
                    )
                tilelink_boom = self._detect_tilelink_boom(golden_v_text)
                if tilelink_boom:
                    logger.info("BoomSoC TileLink detected — enabling 64-bit DSP multiply stimulus")
            except OSError:
                pass

        dsp_cpu_iface = None
        if not self.no_reactive:
            # iBus-style (VexRiscv)
            if icache_bootstrap:
                _ibus_candidates = []
                for iface in response_ifaces:
                    data_payloads = [p for p in iface['payload_inputs']
                                     if not p['name'].endswith(('_error', '_last'))]
                    if data_payloads and port_bit_width(data_payloads[0]) == 32:
                        _ibus_candidates.append((iface, data_payloads[0]))
                if _ibus_candidates:
                    _best = next(
                        (c for c in _ibus_candidates
                         if 'ibus' in c[0].get('prefix', c[0]['id']).lower()),
                        _ibus_candidates[0]
                    )
                    dsp_cpu_iface = {'type': 'ibus', 'iface': _best[0],
                                     'data_port': _best[1]}
            # TileLink-style (BoomSoC)
            if dsp_cpu_iface is None and tilelink_boom:
                for iface in request_ifaces:
                    data_p = next((p for p in iface['payload_inputs']
                                   if 'd_bits_data' in p['name']
                                   and port_bit_width(p) == 64), None)
                    if data_p:
                        pfx = iface['prefix']
                        a_pfx = (pfx[:-2] + '_a') if pfx.endswith('_d') else pfx
                        a_src_key  = f"{a_pfx}_bits_source"
                        a_size_key = f"{a_pfx}_bits_size"
                        if a_src_key not in output_by_name:
                            logger.warning(
                                f"TileLink A-channel field {a_src_key!r} not found in "
                                f"DUT outputs — source echo will be driven as 0"
                            )
                        if a_size_key not in output_by_name:
                            logger.warning(
                                f"TileLink A-channel field {a_size_key!r} not found in "
                                f"DUT outputs — size echo will be driven as 0"
                            )
                        dsp_cpu_iface = {
                            'type': 'tilelink',
                            'iface': iface,
                            'data_port': data_p,
                            'source_port': next((p for p in iface['payload_inputs']
                                                 if p['name'].endswith('_source')), None),
                            'size_port':   next((p for p in iface['payload_inputs']
                                                 if p['name'].endswith('_size')), None),
                            'opcode_port': next((p for p in iface['payload_inputs']
                                                 if p['name'].endswith('_opcode')), None),
                            'a_source_out': f"golden_{a_src_key}"  if a_src_key  in output_by_name else None,
                            'a_size_out':   f"golden_{a_size_key}" if a_size_key in output_by_name else None,
                        }
                        break
            if dsp_cpu_iface:
                logger.info(f"DSP stimulus mode active ({dsp_cpu_iface['type']})")

        # Build port connections carefully to handle edge cases
        def build_port_connections(module_suffix=""):
            """Build port connection string for module instantiation."""
            connections = []
            # Clock
            connections.append(f".{clock}({clock})")
            # All reset ports
            for r in resets:
                connections.append(f".{r['name']}({r['name']})")
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
                is_ibus_burst = (icache_bootstrap and dsp_cpu_iface
                                 and dsp_cpu_iface['type'] == 'ibus'
                                 and dsp_cpu_iface['iface']['id'] == iface_id)
                decls.append(f"    reg env_{iface_id}_pending;")
                decls.append(f"    integer env_{iface_id}_delay;")
                decls.append(f"    reg [31:0] env_{iface_id}_seed;")
                if is_ibus_burst:
                    decls.append(f"    integer env_{iface_id}_burst_remaining;")
            if hls_iface:
                iface_id = hls_iface['id']
                decls.append(f"    reg env_{iface_id}_active;")
                decls.append(f"    reg env_{iface_id}_launch;")
                decls.append(f"    integer env_{iface_id}_cycles;")
                decls.append(f"    reg [31:0] env_{iface_id}_seed;")
            if dsp_cpu_iface:
                decls += [
                    "    // DSP RV32M injection state machine",
                    "    reg [2:0]  dsp_inj_state;",
                    "    reg [4:0]  dsp_inj_rd1, dsp_inj_rd2, dsp_inj_rd3;",
                    "    reg [11:0] dsp_inj_imm1, dsp_inj_imm2;",
                    "    reg        dsp_inj_fire;",
                ]
            return '\n'.join(decls) if decls else '    // No reactive environment state'

        def generate_env_init_code() -> str:
            lines = []
            for iface in response_ifaces:
                iface_id = iface['id']
                is_ibus_burst = (icache_bootstrap and dsp_cpu_iface
                                 and dsp_cpu_iface['type'] == 'ibus'
                                 and dsp_cpu_iface['iface']['id'] == iface_id)
                lines.append(f"        env_{iface_id}_pending = 0;")
                lines.append(f"        env_{iface_id}_delay = 0;")
                lines.append(f"        env_{iface_id}_seed = 32'h{(0x13579BDF ^ (len(iface_id) * 0x1021)) & 0xFFFFFFFF:08X};")
                if is_ibus_burst:
                    lines.append(f"        env_{iface_id}_burst_remaining = 0;")
            if hls_iface:
                iface_id = hls_iface['id']
                lines.append(f"        env_{iface_id}_active = 0;")
                lines.append(f"        env_{iface_id}_launch = 0;")
                lines.append(f"        env_{iface_id}_cycles = 0;")
                lines.append(f"        env_{iface_id}_seed = 32'h2468ACE1;")
            if dsp_cpu_iface:
                lines += [
                    "        dsp_inj_fire = 1'b0;",
                    "        dsp_inj_state = 3'd0;",
                    "        dsp_inj_rd1 = 5'd1; dsp_inj_rd2 = 5'd2; dsp_inj_rd3 = 5'd3;",
                    "        dsp_inj_imm1 = 12'd0; dsp_inj_imm2 = 12'd0;",
                ]
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

                is_dsp_ibus = (dsp_cpu_iface and dsp_cpu_iface['type'] == 'ibus'
                               and dsp_cpu_iface['iface']['id'] == iface_id)
                is_ibus_burst = is_dsp_ibus and icache_bootstrap
                if is_ibus_burst:
                    active_cond = (f"env_{iface_id}_pending && env_{iface_id}_delay == 0 "
                                   f"&& env_{iface_id}_burst_remaining > 0")
                else:
                    active_cond = f"env_{iface_id}_pending && env_{iface_id}_delay == 0"
                lines.append(f"            if ({active_cond}) begin")
                lines.append(f"                {rsp_valid_in} = 1;")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if payload_name.endswith('_error'):
                        lines.append(f"                {payload_name} = 0;")
                    elif payload_name.endswith('_last'):
                        lines.append(f"                {payload_name} = 1;")
                    elif is_dsp_ibus and payload_name == dsp_cpu_iface['data_port']['name']:
                        lines.append(
                            f"                {payload_name} = "
                            f"dsp_rv32_instr(dsp_inj_state, dsp_inj_rd1, dsp_inj_rd2, "
                            f"dsp_inj_rd3, dsp_inj_imm1, dsp_inj_imm2);"
                        )
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
                is_dsp_tl = (dsp_cpu_iface and dsp_cpu_iface['type'] == 'tilelink'
                             and dsp_cpu_iface['iface']['id'] == iface['id'])
                lines.append(f"            // Reactive request driver for {iface['prefix']}")
                lines.append(f"            if ({ready_expr}) begin")
                lines.append(f"                {iface['valid_in']} = 1;")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if is_dsp_tl:
                        dp = dsp_cpu_iface
                        if payload_name == dp['data_port']['name']:
                            next_st = "(dsp_inj_state == 3'd7) ? 3'd0 : dsp_inj_state + 3'd1"
                            lines.append(
                                f"                {payload_name} = (dsp_inj_state != 3'd0) ? "
                                f"{{dsp_rv32_instr({next_st}, dsp_inj_rd1, dsp_inj_rd2, dsp_inj_rd3, dsp_inj_imm1, dsp_inj_imm2), "
                                f"dsp_rv32_instr(dsp_inj_state, dsp_inj_rd1, dsp_inj_rd2, dsp_inj_rd3, dsp_inj_imm1, dsp_inj_imm2)}} "
                                f": {{lfsr ^ 32'h9E3779B9, lfsr}};"
                            )
                        elif dp.get('source_port') and payload_name == dp['source_port']['name']:
                            src = dp['a_source_out'] if dp['a_source_out'] is not None else "0"
                            lines.append(f"                {payload_name} = {src};")
                        elif dp.get('size_port') and payload_name == dp['size_port']['name']:
                            sz = dp['a_size_out'] if dp['a_size_out'] is not None else "0"
                            lines.append(f"                {payload_name} = {sz};")
                        elif dp.get('opcode_port') and payload_name == dp['opcode_port']['name']:
                            lines.append(f"                {payload_name} = 3'd1;")
                        else:
                            lines.append(f"                {payload_name} = 0;")
                    elif payload_name.endswith('_last'):
                        lines.append(f"                {payload_name} = 1;")
                    else:
                        lines.append(assign_from_seed(payload, "lfsr", indent="                "))
                lines.append("            end else begin")
                # DSP TileLink: hold valid low when DUT is not ready so payload
                # remains stable across the handshake (valid/ready protocol).
                lines.append(f"                {iface['valid_in']} = {'0' if is_dsp_tl else 'lfsr[0]'};")
                for payload in iface['payload_inputs']:
                    payload_name = payload['name']
                    if is_dsp_tl:
                        lines.append(f"                {payload_name} = 0;")
                    elif payload_name.endswith('_last'):
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
            # Capture DSP fire condition BEFORE pending flags are cleared by the
            # management loop below (blocking assignments would otherwise make
            # env_*_pending == 0 by the time the state machine checks it).
            if dsp_cpu_iface and dsp_cpu_iface['type'] == 'ibus':
                iid = dsp_cpu_iface['iface']['id']
                if icache_bootstrap:
                    lines.append(
                        f"            dsp_inj_fire = env_{iid}_pending && env_{iid}_delay == 0 "
                        f"&& env_{iid}_burst_remaining > 0;"
                    )
                else:
                    lines.append(
                        f"            dsp_inj_fire = env_{iid}_pending && env_{iid}_delay == 0;"
                    )
            elif dsp_cpu_iface and dsp_cpu_iface['type'] == 'tilelink':
                d_ready = f"golden_{dsp_cpu_iface['iface']['prefix']}_ready"
                lines.append(
                    f"            dsp_inj_fire = {dsp_cpu_iface['iface']['valid_in']} && {d_ready};"
                )
            for idx, iface in enumerate(response_ifaces):
                iface_id = iface['id']
                cmd_valid_out = f"golden_{iface['cmd_valid_out']}"
                ready_gate = iface['ready_in'] if iface['ready_in'] else "1'b1"
                seed_mask = (0x9E3779B9 ^ (idx * 0x45D9F3B)) & 0xFFFFFFFF
                is_ibus_burst = (icache_bootstrap and dsp_cpu_iface
                                 and dsp_cpu_iface['type'] == 'ibus'
                                 and dsp_cpu_iface['iface']['id'] == iface_id)
                lines.append(f"            if (env_{iface_id}_pending) begin")
                lines.append(f"                if (env_{iface_id}_delay > 0) begin")
                lines.append(f"                    env_{iface_id}_delay = env_{iface_id}_delay - 1;")
                if is_ibus_burst:
                    # Burst mode: deliver one word per cycle; clear pending when all 8 are done
                    lines.append(f"                end else if (env_{iface_id}_burst_remaining > 0) begin")
                    lines.append(f"                    env_{iface_id}_burst_remaining = env_{iface_id}_burst_remaining - 1;")
                    lines.append(f"                    if (env_{iface_id}_burst_remaining == 0) begin")
                    lines.append(f"                        env_{iface_id}_pending = 0;")
                    lines.append(f"                    end")
                else:
                    lines.append("                end else begin")
                    lines.append(f"                    env_{iface_id}_pending = 0;")
                lines.append("                end")
                lines.append(f"            end else if ({cmd_valid_out} && {ready_gate}) begin")
                lines.append(f"                env_{iface_id}_pending = 1;")
                lines.append(f"                env_{iface_id}_delay = lfsr[0];")
                lines.append(f"                env_{iface_id}_seed = lfsr ^ 32'h{seed_mask:08X};")
                if is_ibus_burst:
                    lines.append(f"                env_{iface_id}_burst_remaining = 8;")
                    # Pre-arm DSP injection state so first rsp word carries state=1 (ADDI)
                    lines.append(f"                dsp_inj_state <= 3'd1;")
                    lines.append(f"                dsp_inj_rd1  <= (lfsr[5:1]   == 5'd0) ? 5'd1 : lfsr[5:1];")
                    lines.append(f"                dsp_inj_rd2  <= (lfsr[10:6]  == 5'd0) ? 5'd2 : lfsr[10:6];")
                    lines.append(f"                dsp_inj_rd3  <= (lfsr[15:11] == 5'd0) ? 5'd3 : lfsr[15:11];")
                    lines.append(f"                dsp_inj_imm1 <= 12'd7;")
                    lines.append(f"                dsp_inj_imm2 <= 12'd11;")
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
            if dsp_cpu_iface:
                if icache_bootstrap:
                    # Burst mode: state machine is pre-armed at cmd-fire; just advance per word.
                    lines += [
                        "            // DSP RV32M injection state machine (burst mode: pre-armed at cmd-fire)",
                        "            if (dsp_inj_state != 3'd0 && dsp_inj_fire) begin",
                        "                dsp_inj_state <= (dsp_inj_state == 3'd7) ? 3'd0 : dsp_inj_state + 3'd1;",
                        "            end",
                    ]
                else:
                    lines += [
                        "            // DSP RV32M injection state machine",
                        "            if (dsp_inj_state == 3'd0) begin",
                        "                if (dsp_inj_fire && lfsr[0]) begin",
                        "                    dsp_inj_state <= 3'd1;",
                        "                    dsp_inj_rd1  <= (lfsr[5:1]   == 5'd0) ? 5'd1 : lfsr[5:1];",
                        "                    dsp_inj_rd2  <= (lfsr[10:6]  == 5'd0) ? 5'd2 : lfsr[10:6];",
                        "                    dsp_inj_rd3  <= (lfsr[15:11] == 5'd0) ? 5'd3 : lfsr[15:11];",
                        "                    dsp_inj_imm1 <= 12'd7;",
                        "                    dsp_inj_imm2 <= 12'd11;",
                        "                end",
                        "            end else if (dsp_inj_fire) begin",
                        "                dsp_inj_state <= (dsp_inj_state == 3'd7) ? 3'd0 : dsp_inj_state + 3'd1;",
                        "            end",
                    ]
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

        def protocol_output_names() -> list:
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
            return protocol_outputs

        def trace_scan_format() -> str:
            # phase + regular inputs + expected outputs
            fields = 1 + len(regular_inputs) + len(outputs)
            return ' '.join(['%h'] * fields)

        def trace_signal_args(output_prefix: str) -> str:
            args = ["trace_phase"]
            args.extend(port['name'] for port in regular_inputs)
            args.extend(f"{output_prefix}{port['name']}" for port in outputs)
            return ', '.join(args)

        def trace_expected_declarations() -> str:
            decls = ["    reg trace_phase;"]
            for port in outputs:
                width = f"{port['width']} " if port['width'] else ""
                decls.append(f"    reg {width}expected_{port['name']};")
            return '\n'.join(decls)

        def trace_expected_args() -> str:
            args = ["trace_phase"]
            args.extend(port['name'] for port in regular_inputs)
            args.extend(f"expected_{port['name']}" for port in outputs)
            return ', '.join(args)

        def trace_write_code(phase: int) -> str:
            fmt = trace_scan_format()
            args = trace_signal_args("golden_")
            return (
                f"            trace_phase = 1'b{phase};\n"
                f"            $fwrite(trace_fd, \"{fmt}\\n\", {args});"
            )

        def trace_read_code(expected_phase: int) -> str:
            fmt = trace_scan_format()
            args = trace_expected_args()
            expected_count = 1 + len(regular_inputs) + len(outputs)
            return f"""
            trace_status = $fscanf(trace_fd, "{fmt}\\n", {args});
            if (trace_status != {expected_count}) begin
                $display("ERROR: Trace read failed at cycle %0d: status=%0d expected={expected_count}", cycle_count, trace_status);
                $finish(2);
            end
            if (trace_phase !== 1'b{expected_phase}) begin
                $display("ERROR: Trace phase mismatch at cycle %0d: got=%0d expected={expected_phase}", cycle_count, trace_phase);
                $finish(2);
            end"""

        def generate_replay_protocol_compare_code() -> str:
            lines = []
            for name in protocol_output_names():
                lines.append(f'''
            if (expected_{name} !== revised_{name}) begin
                $display("PROTOCOL MISMATCH at cycle %0d: {name} golden=%h revised=%h", cycle_count, expected_{name}, revised_{name});
                protocol_mismatch_count = protocol_mismatch_count + 1;
            end''')
            return '\n'.join(lines) if lines else '            // No protocol outputs to compare'

        def generate_replay_output_compare_code() -> str:
            lines = []
            for port in outputs:
                name = port['name']
                lines.append(f'''
            if (expected_{name} !== revised_{name}) begin
                $display("MISMATCH at cycle %0d: {name} golden=%h revised=%h", cycle_count, expected_{name}, revised_{name});
                mismatch_count = mismatch_count + 1;
            end''')
            return '\n'.join(lines) if lines else '            // No outputs to compare'

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

        if dsp_cpu_iface:
            dsp_instr_fn = """
    // RV32M multiply-sequence instruction encoder used by DSP injection stimulus.
    function [31:0] dsp_rv32_instr;
        input [2:0]  state;
        input [4:0]  rd1, rd2, rd3;
        input [11:0] imm1, imm2;
        begin
            case (state)
                3'd1: dsp_rv32_instr = {imm1, 5'd0, 3'd0, rd1, 7'b0010011}; // ADDI rd1,x0,imm1
                3'd2: dsp_rv32_instr = 32'h00000013;                          // NOP
                3'd3: dsp_rv32_instr = {imm2, 5'd0, 3'd0, rd2, 7'b0010011}; // ADDI rd2,x0,imm2
                3'd4: dsp_rv32_instr = 32'h00000013;                          // NOP
                3'd5: dsp_rv32_instr = {7'b0000001, rd2, rd1, 3'd0, rd3, 7'b0110011}; // MUL rd3,rd1,rd2
                3'd6: dsp_rv32_instr = 32'h00000013;                          // NOP
                3'd7: dsp_rv32_instr = {7'b0000010, 5'd0, rd3, 3'b001, 5'b00000, 7'b1100011}; // BNE rd3,x0,+64
                default: dsp_rv32_instr = 32'h00000013;                          // NOP (state 0 = idle)
            endcase
        end
    endfunction
"""
        else:
            dsp_instr_fn = ""

        # No force block needed: the VexRiscv naturally boots after the icache
        # flush completes (~256 cycles post-reset), at which point lineLoader_valid
        # fires naturally without any external forcing.  Forcing lineLoader_valid
        # early (before flush completes) corrupts the tag state and blocks the
        # natural boot.
        icache_force_block = ""

        golden_trace_tb_content = f"""
`timescale 1ns / 1ps

module testbench;

    // Clock and reset
    reg {clock};
    {chr(10).join(f"    reg {r['name']};" for r in resets) if resets else '    // no reset port detected'}

    // Inputs driven by the golden reactive environment
    {chr(10).join(f"    reg {port['width']+' ' if port['width'] else ''}{port['name']};" for port in regular_inputs) if regular_inputs else '    // No regular inputs'}

    // Outputs from golden design
    {chr(10).join(f"    wire {port['width']+' ' if port['width'] else ''}golden_{port['name']};" for port in outputs) if outputs else '    // No outputs to record'}

    reg [31:0] lfsr;
    integer cycle_count;
    integer num_vectors;
    integer trace_fd;
    reg [1023:0] trace_file;
    reg trace_phase;

    // Reactive environment state
{generate_env_declarations()}

    // Instantiate golden design
    {golden_module} golden_dut (
        {build_port_connections("golden_")}
    );

    initial begin
        {clock} = 0;
        forever #5 {clock} = ~{clock};
    end

    function [31:0] lfsr_next;
        input [31:0] lfsr_in;
        begin
            lfsr_next = {{lfsr_in[30:0], lfsr_in[31] ^ lfsr_in[21] ^ lfsr_in[1] ^ lfsr_in[0]}};
        end
    endfunction
{dsp_instr_fn}
{generate_hls_mem_model_code()}

    initial begin
        cycle_count = 0;
        num_vectors = {self.num_vectors};
        if (!$value$plusargs("NUM_VECTORS=%d", num_vectors)) begin
            num_vectors = {self.num_vectors};
        end
        if (!$value$plusargs("TRACE_FILE=%s", trace_file)) begin
            trace_file = "{trace_filename}";
        end
        if (!$value$plusargs("LFSR_SEED=%h", lfsr)) begin
            lfsr = 32'hDEADBEEF;
        end
        trace_fd = $fopen(trace_file, "w");
        if (trace_fd == 0) begin
            $display("ERROR: Could not open trace file %0s", trace_file);
            $finish(2);
        end
        $display("Configured test vectors: %0d", num_vectors);
        $display("Writing golden trace: %0s", trace_file);
{generate_env_init_code()}

        {all_reset_assert_lines if resets else '// no reset port detected'}
        {chr(10).join(f"        {port['name']} = 0;" for port in regular_inputs)}
        repeat(10) @(posedge {clock});
        {all_reset_deassert_lines if resets else ''}

        repeat(50) begin
            @(negedge {clock});
            lfsr = lfsr_next(lfsr);
{generate_negedge_stimulus_code()}
            @(posedge {clock});
{generate_posedge_bookkeeping_code()}
            #1;
{trace_write_code(0)}
        end

        repeat(num_vectors) begin
            @(negedge {clock});
            lfsr = lfsr_next(lfsr);
{generate_negedge_stimulus_code()}

            @(posedge {clock});
            cycle_count = cycle_count + 1;
{generate_posedge_bookkeeping_code()}
            #1;
{trace_write_code(1)}
        end

        $fclose(trace_fd);
        $display("\\n=======================================");
        $display("SIMULATION COMPLETE");
        $display("=======================================");
        $display("Cycles simulated: %0d", cycle_count);
        $display("Mismatches found: 0");
        $display("Protocol mismatches found: 0");
        $display("Result: PASS");
        $finish(0);
    end

    initial begin
        #1;
        #((10 + 50 + num_vectors) * 20) $display("ERROR: Simulation timeout"); $finish(2);
    end
{icache_force_block}
endmodule
"""

        revised_replay_tb_content = f"""
`timescale 1ns / 1ps

module testbench;

    // Clock and reset
    reg {clock};
    {chr(10).join(f"    reg {r['name']};" for r in resets) if resets else '    // no reset port detected'}

    // Inputs replayed from golden trace
    {chr(10).join(f"    reg {port['width']+' ' if port['width'] else ''}{port['name']};" for port in regular_inputs) if regular_inputs else '    // No regular inputs'}

    // Outputs from revised design
    {chr(10).join(f"    wire {port['width']+' ' if port['width'] else ''}revised_{port['name']};" for port in outputs) if outputs else '    // No outputs to compare'}

{trace_expected_declarations()}

    integer mismatch_count;
    integer protocol_mismatch_count;
    integer cycle_count;
    integer num_vectors;
    integer trace_fd;
    integer trace_status;
    reg [1023:0] trace_file;

    // Instantiate revised design
    {revised_module} revised_dut (
        {build_port_connections("revised_")}
    );

    initial begin
        {clock} = 0;
        forever #5 {clock} = ~{clock};
    end

    initial begin
        mismatch_count = 0;
        protocol_mismatch_count = 0;
        cycle_count = 0;
        num_vectors = {self.num_vectors};
        if (!$value$plusargs("NUM_VECTORS=%d", num_vectors)) begin
            num_vectors = {self.num_vectors};
        end
        if (!$value$plusargs("TRACE_FILE=%s", trace_file)) begin
            trace_file = "{trace_filename}";
        end
        trace_fd = $fopen(trace_file, "r");
        if (trace_fd == 0) begin
            $display("ERROR: Could not open trace file %0s", trace_file);
            $finish(2);
        end
        $display("Configured test vectors: %0d", num_vectors);
        $display("Replaying golden trace: %0s", trace_file);

        {all_reset_assert_lines if resets else '// no reset port detected'}
        {chr(10).join(f"        {port['name']} = 0;" for port in regular_inputs)}
        repeat(10) @(posedge {clock});
        {all_reset_deassert_lines if resets else ''}

        repeat(50) begin
            @(negedge {clock});
{trace_read_code(0)}
            @(posedge {clock});
            #1;
{generate_replay_protocol_compare_code()}
        end

        repeat(num_vectors) begin
            @(negedge {clock});
{trace_read_code(1)}
            @(posedge {clock});
            cycle_count = cycle_count + 1;
            #1;
{generate_replay_output_compare_code()}
{generate_replay_protocol_compare_code()}
        end

        $fclose(trace_fd);
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

    initial begin
        #1;
        #((10 + 50 + num_vectors) * 20) $display("ERROR: Simulation timeout"); $finish(2);
    end
endmodule
"""

        if mode == "golden_trace":
            tb_content = golden_trace_tb_content
        else:
            tb_content = revised_replay_tb_content

        with open(tb_path, 'w') as f:
            f.write(tb_content)

        logger.info(f"Generated testbench: {tb_path}")

    @staticmethod
    def _detect_icache_bootstrap(golden_verilog_text: str) -> Optional[dict]:
        """Detect VexRiscv InstructionCache — signals that burst iBus bootstrap is needed.

        Returns a dict with instance info if the cache hierarchy is present,
        else None.  When non-None, the testbench must deliver 8 consecutive
        iBus_rsp_valid pulses per cmd (one per cache-line word); the CPU
        boots naturally after the icache flush completes (~256 cycles
        post-reset) without any external forcing of lineLoader_valid.
        """
        if ('IBusCachedPlugin_cache' in golden_verilog_text
                and 'lineLoader_valid' in golden_verilog_text):
            return {'instance': 'IBusCachedPlugin_cache', 'line_size_words': 8}
        return None

    @staticmethod
    def _detect_tilelink_boom(golden_verilog_text: str) -> bool:
        """Detect BoomSoC TileLink instruction bus — signals 64-bit DSP stimulus is needed.

        Requires three markers that co-occur only in BoomSoC post-implementation netlists:
          - DSP48E2: multiply unit is present
          - d_bits_data: TileLink D-channel data port exposed at top level
          - BoomCore: BoomSoC top-level module, absent from all other TileLink+DSP designs
        """
        return (
            'DSP48E2' in golden_verilog_text
            and 'd_bits_data' in golden_verilog_text
            and 'BoomCore' in golden_verilog_text
        )

    def _resolve_vivado_path(self) -> str:
        vivado_path = os.environ.get("VIVADO_EXEC")
        if vivado_path:
            if "/" not in vivado_path:
                vivado_path = shutil.which(vivado_path)
        else:
            vivado_path = shutil.which("vivado")
        if not vivado_path:
            raise RuntimeError("Vivado not found in PATH. Set VIVADO_EXEC env var or add Vivado to PATH.")
        return vivado_path

    @staticmethod
    def _vivado_tool_env(vivado_path: str) -> dict:
        """Return an environment that lets XSim kernels find Vivado runtime libs."""
        env = os.environ.copy()
        vivado_install = Path(vivado_path).parent.parent
        lib_dir = vivado_install / "lib" / "lnx64.o"
        if lib_dir.exists():
            existing = env.get("LD_LIBRARY_PATH", "")
            parts = [str(lib_dir)]
            if existing:
                parts.append(existing)
            env["LD_LIBRARY_PATH"] = ":".join(parts)
        return env

    @staticmethod
    def _rename_revised_verilog(revised_v: Path, revised_info: dict, revised_renamed: Path):
        with open(revised_v, 'r') as f:
            content = f.read()

        declared_module_names = set(re.findall(
            r'(?m)^\s*module\s+(\w+)\b',
            content
        ))
        logger.info(
            f"Found {len(declared_module_names)} module declarations in revised netlist"
        )

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

    def _run_trace_replay_simulation(
        self,
        golden_v: Path,
        revised_v: Path,
        golden_info: dict,
        revised_info: dict,
        golden_clocks: list,
    ):
        """Run phase 2 as cached golden trace generation plus revised replay."""
        print("\nRunning xsim simulation...")
        print("(Golden trace is reusable across revised DCPs for this golden design.)")

        vivado_path = self._resolve_vivado_path()
        vivado_env = self._vivado_tool_env(vivado_path)
        vivado_bin_dir = Path(vivado_path).parent
        vivado_install = vivado_bin_dir.parent
        unisim_dir = vivado_install / "data" / "verilog" / "src"
        glbl_v = unisim_dir / "glbl.v"

        xvlog_timeout_s = 1800
        xelab_timeout_s = 3600

        def xsim_timeout_for(vector_count: int) -> int:
            return max(3600, 600 + int(vector_count * 1.0))

        golden_tb = self.temp_dir / "golden_trace_tb.v"
        replay_tb = self.temp_dir / "revised_replay_tb.v"
        trace_file = self.temp_dir / f"golden_trace_{self.num_vectors}.hex"

        self.generate_testbench(
            golden_info,
            revised_info,
            golden_tb,
            clock_names=golden_clocks,
            mode="golden_trace",
            trace_filename=str(trace_file),
        )
        self.generate_testbench(
            golden_info,
            revised_info,
            replay_tb,
            clock_names=golden_clocks,
            mode="revised_replay",
            trace_filename=str(trace_file),
        )

        def compile_and_elaborate(
            work_dir: Path,
            sources: list,
            snapshot: str,
            top: str = "work.testbench",
        ):
            work_dir.mkdir(exist_ok=True)
            compile_cmd = ["xvlog", "-work", "work"]
            if glbl_v.exists():
                compile_cmd.append(str(glbl_v))
            compile_cmd.extend(sources)

            result = subprocess.run(
                compile_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=xvlog_timeout_s,
            )
            if result.returncode != 0:
                print("\n✗ Compilation failed:")
                print(result.stdout)
                print(result.stderr)
                return False

            elab_cmd = [
                "xelab",
                "--debug", "off",
                "--mt", "auto",
                "-L", "unisims_ver",
                "-L", "unimacro_ver",
                top,
                "work.glbl",
                "-s", snapshot,
            ]
            result = subprocess.run(
                elab_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=xelab_timeout_s,
            )
            if result.returncode != 0:
                error_output = result.stdout + result.stderr
                if self._is_encrypted_ip_error(error_output):
                    logger.info("Elaboration failed due to encrypted/SIP IP")
                    return {
                        "status": "skipped",
                        "reason": "Design contains encrypted or Secure IP blocks",
                        "details": "xelab elaboration failed due to missing SIP modules",
                    }
                print("\n✗ Elaboration failed:")
                print(result.stdout)
                print(result.stderr)
                return False
            return True

        def run_snapshot(work_dir: Path, snapshot: str, vector_count: int, label: str, trace_path: Path, use_lfsr_seed: bool = False) -> dict:
            logger.info(f"Running {label} simulation with xsim ({vector_count} vectors)...")
            sim_cmd = [
                "xsim",
                snapshot,
                "-R",
                "--testplusarg", f"NUM_VECTORS={vector_count}",
                "--testplusarg", f"TRACE_FILE={trace_path}",
            ]
            if use_lfsr_seed:
                sim_cmd += ["--testplusarg", f"LFSR_SEED={self.lfsr_seed:08X}"]
            result = subprocess.run(
                sim_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                env=vivado_env,
                timeout=xsim_timeout_for(vector_count),
            )
            sim_output = result.stdout + result.stderr
            log_file = self.temp_dir / f"simulation_{label}.log"
            with open(log_file, "w") as f:
                f.write(sim_output)
            if label == "full":
                with open(self.temp_dir / "simulation.log", "w") as f:
                    f.write(sim_output)
            parse_result = parse_simulation_output(sim_output, result.returncode, vector_count)
            for line in sim_output.split('\n'):
                if 'MISMATCH' in line:
                    print(f"  {line}")
            return {
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

        try:
            golden_key, golden_manifest = self._trace_simulation_cache_key(vivado_path, "golden_trace")
            golden_cache_entry = self.sim_cache_dir / "golden_trace" / golden_key[:2] / golden_key
            golden_xsim_dir = self.temp_dir / "xsim_golden_trace"
            golden_cache_hit = False
            if self.use_sim_cache:
                try:
                    golden_cache_hit = self._restore_cached_xsim_work(
                        golden_cache_entry,
                        golden_xsim_dir,
                    )
                except Exception as e:
                    logger.warning(f"Unable to restore golden trace xsim cache; rebuilding: {e}")
            if golden_cache_hit:
                logger.info(f"Reusing cached golden trace xelab snapshot: {golden_cache_entry}")
                print(f"✓ Reusing cached golden trace xelab snapshot: {golden_key[:12]}")
            else:
                print("Compiling/elaborating reusable golden trace snapshot...")
                result = compile_and_elaborate(
                    golden_xsim_dir,
                    [str(golden_v), str(golden_tb)],
                    "golden_trace_sim",
                )
                if isinstance(result, dict) or result is False:
                    return result
                if self.use_sim_cache:
                    try:
                        self._store_cached_xsim_work(golden_cache_entry, golden_xsim_dir, golden_manifest)
                    except Exception as e:
                        logger.warning(f"Unable to store golden trace xsim cache; continuing: {e}")

            cached_trace = (
                golden_cache_entry
                / "traces"
                / f"seed_{self.lfsr_seed:08X}_vectors_{self.num_vectors}.hex"
            )
            trace_cache_hit = False
            if self.use_sim_cache and cached_trace.exists():
                try:
                    shutil.copy2(cached_trace, trace_file)
                    trace_cache_hit = True
                    print(
                        f"✓ Reusing cached golden trace: {golden_key[:12]} "
                        f"seed=0x{self.lfsr_seed:08X} vectors={self.num_vectors}"
                    )
                except Exception as e:
                    logger.warning(f"Unable to restore golden trace file cache; regenerating: {e}")
            if not trace_cache_hit:
                print(f"Generating golden trace ({self.num_vectors} vectors)...")
                golden_report = run_snapshot(
                    golden_xsim_dir,
                    "golden_trace_sim",
                    self.num_vectors,
                    "golden_trace",
                    trace_file,
                    use_lfsr_seed=True,
                )
                if not golden_report["passed"] or not trace_file.exists():
                    self.infrastructure_failure = True
                    self.infrastructure_reason = golden_report.get("infrastructure_reason") or "golden_trace_failed"
                    self.simulation_report = {
                        "golden_trace_report": golden_report,
                        "infrastructure_failure": True,
                        "infrastructure_reason": self.infrastructure_reason,
                    }
                    return False
                if self.use_sim_cache:
                    try:
                        cached_trace.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(trace_file, cached_trace)
                    except Exception as e:
                        logger.warning(f"Unable to store golden trace file cache; continuing: {e}")

            revised_key, revised_manifest = self._trace_simulation_cache_key(vivado_path, "revised_replay")
            revised_cache_entry = self.sim_cache_dir / "revised_replay" / revised_key[:2] / revised_key
            revised_xsim_dir = self.temp_dir / "xsim_revised_replay"
            revised_cache_hit = False
            if self.use_sim_cache:
                try:
                    revised_cache_hit = self._restore_cached_xsim_work(
                        revised_cache_entry,
                        revised_xsim_dir,
                    )
                except Exception as e:
                    logger.warning(f"Unable to restore revised replay xsim cache; rebuilding: {e}")
            if revised_cache_hit:
                logger.info(f"Reusing cached revised replay xelab snapshot: {revised_cache_entry}")
                print(f"✓ Reusing cached revised replay xelab snapshot: {revised_key[:12]}")
            else:
                revised_xsim_dir.mkdir(exist_ok=True)
                revised_renamed = revised_xsim_dir / "revised_sim_renamed.v"
                logger.info("Renaming all revised modules to avoid conflicts...")
                self._rename_revised_verilog(revised_v, revised_info, revised_renamed)
                print("Compiling/elaborating revised replay snapshot...")
                result = compile_and_elaborate(
                    revised_xsim_dir,
                    [str(revised_renamed), str(replay_tb)],
                    "revised_replay_sim",
                )
                if isinstance(result, dict) or result is False:
                    return result
                if self.use_sim_cache:
                    try:
                        self._store_cached_xsim_work(revised_cache_entry, revised_xsim_dir, revised_manifest)
                    except Exception as e:
                        logger.warning(f"Unable to store revised replay xsim cache; continuing: {e}")

            def print_report(report: dict, label: str):
                print("\n" + "-"*70)
                print(f"Simulation Results ({label}):")
                print(f"  Requested vectors: {report['vectors_requested']}")
                print(f"  Cycles: {report['cycles_simulated']}")
                print(f"  Mismatches: {report['mismatch_count']}")
                print(f"  Protocol mismatches: {report['protocol_mismatch_count']}")
                print(f"  Log: {report['log_file']}")
                if report["simulator_failed"]:
                    print("  Simulator reported an internal failure")
                if not report["result_pass_seen"] and report["returncode"] == 0:
                    print("  Testbench PASS marker not found")
                print(f"  Result: {'PASSED' if report['passed'] else 'FAILED'}")
                print("-"*70)

            precheck_report = None
            run_precheck = self.precheck_vectors > 0 and self.precheck_vectors < self.num_vectors
            if run_precheck:
                print(f"\nSimulating {self.precheck_vectors} test vectors (precheck)...")
                precheck_report = run_snapshot(
                    revised_xsim_dir,
                    "revised_replay_sim",
                    self.precheck_vectors,
                    "precheck",
                    trace_file,
                )
                print_report(precheck_report, "precheck")
                if not precheck_report["passed"]:
                    self.simulation_report = {
                        "mode": "golden_trace_revised_replay",
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

            print(f"\nSimulating {self.num_vectors} test vectors (full)...")
            full_report = run_snapshot(
                revised_xsim_dir,
                "revised_replay_sim",
                self.num_vectors,
                "full",
                trace_file,
            )
            print_report(full_report, "full")

            self.simulation_report = {
                "mode": "golden_trace_revised_replay",
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
            print(f"\n✗ Timeout in trace/replay simulation after {timeout_s}s")
            self.infrastructure_failure = True
            self.infrastructure_reason = "timeout_in_trace_replay_simulation"
            self.simulation_report = {
                "mode": "golden_trace_revised_replay",
                "infrastructure_failure": True,
                "infrastructure_reason": self.infrastructure_reason,
                "timeout_seconds": timeout_s,
            }
            return False

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
        golden_info['verilog_path'] = str(golden_v)

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

        return self._run_trace_replay_simulation(
            golden_v,
            revised_v,
            golden_info,
            revised_info,
            golden_clocks,
        )

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
            "lfsr_seed": f"0x{self.lfsr_seed:08X}",
            "runtime_seconds": elapsed_time,
            "phase1_passed": self.phase1_passed,
            "phase2_passed": self.phase2_passed,
            "infrastructure_failure": self.infrastructure_failure,
            "infrastructure_reason": self.infrastructure_reason,
            "overall_passed": self.phase1_passed and self.phase2_passed,
            "preflight_report": self.preflight_report,
            "structural_report": self.structural_report,
            "simulation_report": self.simulation_report,
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved: {report_file}")
        print(f"Working directory preserved: {self.temp_dir}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            from VivadoMCP import vivado_mcp_server as vivado_server
            vivado_server.cleanup_vivado()
        except Exception:
            pass
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
        default=1000,
        help="Number of random test vectors to simulate (default: 1000). "
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
    parser.add_argument(
        "--seed",
        type=lambda x: int(x, 0),
        default=None,
        metavar="SEED",
        help="32-bit LFSR seed for reproducible stimulus (hex or decimal, e.g. 0xDEADBEEF). "
             "Defaults to a random value from /dev/urandom."
    )
    parser.add_argument(
        "--no-sim-cache",
        action="store_true",
        help="Disable reuse of successful xelab snapshots"
    )
    parser.add_argument(
        "--sim-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached xsim/xelab snapshots "
             "(default: .xsim_validation_cache)"
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

    if args.seed is not None and not (0 <= args.seed <= 0xFFFFFFFF):
        print("Error: --seed must be a 32-bit unsigned integer (0 to 0xFFFFFFFF)", file=sys.stderr)
        sys.exit(1)

    # Run validation
    validator = DCPValidator(
        golden_dcp=args.golden_dcp,
        revised_dcp=args.revised_dcp,
        num_vectors=args.vectors,
        precheck_vectors=args.precheck_vectors,
        debug=args.debug,
        no_reactive=args.no_reactive,
        use_sim_cache=not args.no_sim_cache,
        sim_cache_dir=args.sim_cache_dir,
        lfsr_seed=args.seed,
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
