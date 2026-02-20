#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
MCP Server for Vivado - manages Vivado via pexpect for stdin/stdout control.

Usage:
    python vivado_mcp_server.py [--vivado-path /path/to/vivado]
"""

import argparse
import atexit
import logging
import os
import re
import signal
import shutil
import sys
from typing import Optional, Dict, Any

import pexpect
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Vivado Tcl prompt pattern
# Pattern requires newline before prompt to avoid matching prompt in command echoes.
# This prevents the issue where pexpect matches stale prompts in the buffer.
VIVADO_PROMPT = r"\r?\nVivado% "

# Global state
_vivado_process: Optional[pexpect.spawn] = None
_vivado_pid: Optional[int] = None
_vivado_path: Optional[str] = None
_vivado_log_file: Optional[str] = None
_vivado_journal_file: Optional[str] = None
_design_open: bool = False
_command_pending: bool = False  # True if a command timed out and may still be running


def get_vivado_path() -> str:
    """Get Vivado executable path from global setting, VIVADO_EXEC env var, or PATH."""
    global _vivado_path
    if _vivado_path:
        return _vivado_path
    # Check VIVADO_EXEC environment variable
    vivado_exec_env = os.environ.get("VIVADO_EXEC")
    if vivado_exec_env:
        return vivado_exec_env
    # Search in PATH
    vivado = shutil.which("vivado")
    if vivado:
        return vivado
    raise RuntimeError("Vivado not found in PATH. Set VIVADO_EXEC env var, provide --vivado-path, or add Vivado to PATH.")


def cleanup_vivado():
    """Kill Vivado process if running. Called on exit."""
    global _vivado_process, _vivado_pid
    if _vivado_pid:
        try:
            os.kill(_vivado_pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        _vivado_pid = None
    if _vivado_process and _vivado_process.isalive():
        try:
            _vivado_process.terminate(force=True)
        except Exception:
            pass
        _vivado_process = None


def signal_handler(signum, frame):
    """Handle termination signals."""
    cleanup_vivado()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_vivado)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def start_vivado(log_file: Optional[str] = None, journal_file: Optional[str] = None) -> pexpect.spawn:
    """Start Vivado in Tcl mode and wait for prompt.
    
    Args:
        log_file: Path to Vivado log file (default: vivado.log in current directory)
        journal_file: Path to Vivado journal file (default: vivado.jou in current directory)
    """
    global _vivado_process, _vivado_pid

    if _vivado_process and _vivado_process.isalive():
        logger.info("Vivado process already running")
        return _vivado_process

    vivado_path = get_vivado_path()
    logger.info(f"Starting Vivado from: {vivado_path}")
    
    # Build Vivado command arguments
    args = ["-mode", "tcl"]
    
    # Set log file if specified
    if log_file:
        args.extend(["-log", log_file])
        logger.info(f"Vivado log file: {log_file}")
    
    # Set journal file if specified
    if journal_file:
        args.extend(["-journal", journal_file])
        logger.info(f"Vivado journal file: {journal_file}")
    
    # Start Vivado in Tcl mode
    # Use large maxread buffer for handling large outputs
    # Set TERM=dumb to prevent terminal line wrapping and ANSI formatting
    # which can corrupt command echo parsing
    env = os.environ.copy()
    env["TERM"] = "dumb"
    
    _vivado_process = pexpect.spawn(
        vivado_path,
        args=args,
        encoding="utf-8",
        timeout=90,  # 15 min default timeout for startup
        maxread=10000000,  # 10MB buffer for large outputs
        searchwindowsize=10000,  # Search window for prompt matching
        env=env,  # Use dumb terminal to prevent line wrapping
        dimensions=(100, 500),  # Set large terminal width to prevent wrapping
    )
    
    # Get the PID for reliable cleanup
    _vivado_pid = _vivado_process.pid
    logger.info(f"Vivado process started with PID: {_vivado_pid}")
    
    # Wait for Vivado prompt
    logger.info("Waiting for Vivado prompt...")
    _vivado_process.expect(VIVADO_PROMPT)
    logger.info("Vivado ready")
    
    return _vivado_process


def ensure_vivado() -> pexpect.spawn:
    """Ensure Vivado is running, start if needed."""
    global _vivado_process, _vivado_log_file, _vivado_journal_file
    if _vivado_process is None or not _vivado_process.isalive():
        return start_vivado(_vivado_log_file, _vivado_journal_file)
    return _vivado_process


def wait_for_prompt(proc: pexpect.spawn, timeout: float) -> str:
    """Wait for Vivado prompt and return captured output."""
    proc.expect(VIVADO_PROMPT, timeout=timeout)
    return proc.before


def sync_after_timeout(proc: pexpect.spawn) -> str:
    """
    After a timeout, wait for the previous command to complete.
    Returns the output from the command that was running.
    """
    global _command_pending
    if not _command_pending:
        return ""
    
    # Wait indefinitely for the prompt (command to complete)
    # Use a very long timeout (1 hour) as a safety
    try:
        output = wait_for_prompt(proc, timeout=3600)
        _command_pending = False
        return f"[Previous command completed]\n{output}"
    except pexpect.TIMEOUT:
        # Still stuck after 1 hour - Vivado is truly hung
        _command_pending = True
        raise RuntimeError("Vivado appears to be hung. Use restart_vivado to recover.")


def run_tcl_command(command: str, timeout: Optional[float] = None) -> str:
    """
    Run a Tcl command in Vivado and return the output.
    
    Args:
        command: Tcl command to execute
        timeout: Timeout in seconds (None for default 300s)
    
    Returns:
        Command output as string
    """
    global _command_pending
    
    proc = ensure_vivado()
    
    # If a previous command timed out, wait for it to complete first
    if _command_pending:
        sync_output = sync_after_timeout(proc)
        if sync_output:
            # Previous command completed, we can continue
            pass
    
    # Use provided timeout or default
    effective_timeout = timeout if timeout is not None else 300
    
    # Log the command (truncate if very long)
    cmd_log = command if len(command) < 200 else command[:200] + "..."
    logger.info(f"Executing Tcl command: {cmd_log}")
    
    # Send command
    proc.sendline(command)
    
    try:
        # Wait for prompt and capture output
        proc.expect(VIVADO_PROMPT, timeout=effective_timeout)
        
        # Get output (everything between command echo and prompt)
        output = proc.before
        
        # Remove the echoed command from output (first line)
        lines = output.split("\n")
        if lines and command in lines[0]:
            output = "\n".join(lines[1:])
        
        logger.info(f"Command completed successfully")
        return output.strip()
    
    except pexpect.TIMEOUT:
        # Mark that we have a pending command
        _command_pending = True
        logger.error(f"Command timed out after {effective_timeout}s: {cmd_log}")
        raise


def restart_vivado_process() -> str:
    """Kill and restart Vivado process."""
    global _design_open, _command_pending, _vivado_log_file, _vivado_journal_file
    cleanup_vivado()
    _design_open = False
    _command_pending = False
    start_vivado(_vivado_log_file, _vivado_journal_file)
    return "Vivado restarted successfully."


def close_current_design() -> str:
    """Close the current design if one is open."""
    global _design_open
    if _design_open:
        output = run_tcl_command("close_design")
        _design_open = False
        return output
    return "No design was open."


def get_critical_high_fanout_nets(
    num_paths: int = 50,
    min_fanout: int = 100,
    exclude_clocks: bool = True,
    timeout: float = 600.0
) -> str:
    """
    Extract high fanout nets from critical timing paths.
    
    Analyzes the worst negative slack (WNS) timing paths to identify non-clock
    nets with high fanout that may be candidates for fanout optimization.
    The output can be used with RapidWright's optimize_fanout tool.
    
    Net names are automatically resolved to their PARENT net names, which is
    required for RapidWright compatibility.
    """
    import re
    from collections import defaultdict
    
    # Flush buffer before generating timing report
    run_tcl_command("puts {fanout_analysis_start}", timeout=5)
    
    # Generate detailed timing report for multiple paths
    cmd = f"report_timing -return_string -max_paths {num_paths} -delay_type max -sort_by slack"
    
    try:
        timing_report = run_tcl_command(cmd, timeout=timeout)
    except Exception as e:
        return f"Error generating timing report: {str(e)}"
    
    # Parse the timing report to extract high fanout nets
    # Dictionary to track nets: net_name -> {fanout, path_count}
    net_info = defaultdict(lambda: {"fanout": 0, "path_count": 0, "paths": set()})
    
    # Split report into individual paths
    lines = timing_report.split('\n')
    current_path_id = 0
    
    # Regex pattern to match net lines with fanout information
    # Example: "net (fo=267, routed)         1.225     4.454    pcie4.../s_axis_cc_tvalid_reg_lower"
    net_pattern = re.compile(r'net\s+\(fo=(\d+),\s*(routed|estimated)\)')
    
    # Clock net patterns to exclude
    clock_patterns = [
        r'CLK[_\[]',       # CLK_ or CLK[ (clock net naming convention)
        r'[_/]CLK$',       # ends with /CLK or _CLK
        r'CLOCK',          # Contains CLOCK
        r'_clk_',          # Contains _clk_
        r'/C$',            # Clock pin (ends with /C)
        r'BUFG',           # BUFG related
        r'MMCM',           # MMCM related
        r'PLL',            # PLL related
        r'TXOUTCLK',       # GT transceiver clock
        r'RXOUTCLK',       # GT transceiver clock
        r'USERCLK',        # User clock
        r'CORECLK',        # Core clock
    ]
    clock_regex = re.compile('|'.join(clock_patterns), re.IGNORECASE)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect new path (usually starts with "Slack" or contains path delimiter)
        if 'Slack' in line and ('ns' in line or 'VIOLATED' in line or 'MET' in line):
            current_path_id += 1
        
        # Look for net with fanout information
        match = net_pattern.search(line)
        if match:
            fanout = int(match.group(1))
            
            # Only process nets meeting the minimum fanout threshold
            if fanout >= min_fanout:
                net_name = None
                
                # First try to find it on the current line after the fanout info
                parts = line.split()
                for part in parts:
                    if '/' in part and not part.startswith('(') and not part.endswith(')'):
                        net_name = part
                        break
                
                # If not found on current line, check next line
                if not net_name and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if '/' in next_line and not next_line.startswith('net') and not 'Delay' in next_line:
                        parts = next_line.split()
                        for part in parts:
                            if '/' in part:
                                net_name = part
                                break
                
                if net_name:
                    # Check if this is a clock net
                    is_clock = False
                    if exclude_clocks and clock_regex.search(net_name):
                        is_clock = True
                    
                    if not is_clock:
                        # Update net info
                        if fanout > net_info[net_name]["fanout"]:
                            net_info[net_name]["fanout"] = fanout
                        net_info[net_name]["paths"].add(current_path_id)
                        net_info[net_name]["path_count"] = len(net_info[net_name]["paths"])
        
        i += 1
    
    if not net_info:
        return f"No high fanout nets (fanout >= {min_fanout}) found in the {num_paths} most critical paths."
    
    # Look up parent net names for all extracted nets
    parent_net_map = {}  # original_name -> parent_name
    
    for net_name in net_info.keys():
        try:
            # First, verify the net exists
            check_cmd = f"get_nets {{{net_name}}}"
            check_result = run_tcl_command(check_cmd, timeout=30.0)
            logger.info(f"[DEBUG] get_nets for '{net_name[-60:]}...': result='{check_result.strip()[:100]}'")
            
            # If get_nets returns empty or an error, use original name
            if not check_result.strip() or "ERROR" in check_result.upper() or "WARNING" in check_result.upper():
                logger.info(f"Net '{net_name}' not found or has errors, using as-is")
                parent_net_map[net_name] = net_name
                continue
            
            # Now get the parent property
            parent_cmd = f"get_property PARENT [get_nets {{{net_name}}}]"
            parent_result = run_tcl_command(parent_cmd, timeout=30.0)
            parent_name = parent_result.strip()
            logger.info(f"[DEBUG] PARENT for '{net_name[-60:]}...': result='{parent_name}'")
            
            # Validate the result - should not be empty, should contain '/' for hierarchical nets,
            # and should not look like a Tcl command or error
            if (parent_name and 
                parent_name != net_name and
                '/' in parent_name and
                not parent_name.startswith('get_') and
                not parent_name.startswith('ERROR') and
                not parent_name.startswith('WARNING')):
                parent_net_map[net_name] = parent_name
                logger.info(f"[DEBUG] Using PARENT name: '{parent_name[-80:]}'")
            else:
                # Use original name if parent lookup returned invalid data
                parent_net_map[net_name] = net_name
                logger.info(f"[DEBUG] PARENT invalid, using original: '{net_name[-80:]}'")
        except Exception as e:
            # If lookup fails, keep original name
            logger.warning(f"Parent lookup failed for net '{net_name}': {e}")
            parent_net_map[net_name] = net_name
    
    # Rebuild net_info with parent net names
    parent_net_info = defaultdict(lambda: {"fanout": 0, "path_count": 0, "paths": set()})
    
    for net_name, info in net_info.items():
        parent_name = parent_net_map[net_name]
        if info["fanout"] > parent_net_info[parent_name]["fanout"]:
            parent_net_info[parent_name]["fanout"] = info["fanout"]
        parent_net_info[parent_name]["paths"].update(info["paths"])
        parent_net_info[parent_name]["path_count"] = len(parent_net_info[parent_name]["paths"])
    
    # Sort nets by path_count, then by fanout
    sorted_nets = sorted(
        parent_net_info.items(),
        key=lambda x: (-x[1]["path_count"], -x[1]["fanout"])
    )
    
    if not sorted_nets:
        return f"No high fanout nets (fanout >= {min_fanout}) found in the {num_paths} most critical paths."
    
    # Format output
    result_lines = [
        f"=== High Fanout Nets in Critical Paths (Parent Net Names) ===",
        f"Analyzed {num_paths} worst timing paths",
        f"Minimum fanout threshold: {min_fanout}",
        f"Clock nets excluded: {exclude_clocks}",
        f"Note: Net names are resolved to parent nets for RapidWright compatibility",
        f"",
        f"Found {len(sorted_nets)} high fanout nets:",
        f"",
        f"{'Paths':>6}  {'Fanout':>8}  Parent Net Name",
        f"{'-'*6}  {'-'*8}  {'-'*50}",
    ]
    
    for net_name, info in sorted_nets:
        result_lines.append(
            f"{info['path_count']:>6}  {info['fanout']:>8}  {net_name}"
        )
    
    result_lines.append("")
    result_lines.append("=== Parent Net Names for RapidWright optimize_fanout ===")
    result_lines.append("(These are parent net names, ready for use with RapidWright's optimize_fanout tool)")
    result_lines.append("")
    
    for net_name, info in sorted_nets:
        result_lines.append(net_name)
    
    return "\n".join(result_lines)


def extract_critical_path_cells(
    num_paths: int = 50,
    output_file: str = None,
    timeout: float = 600.0
) -> str:
    """
    Extract cell names from critical timing paths for spread analysis.
    
    Parses timing report to get ordered list of cells on each critical path.
    Output is JSON format that can be passed to RapidWright's analyze_critical_path_spread.
    
    Args:
        num_paths: Number of critical paths to extract
        output_file: Optional path to write JSON output to file instead of returning it
        timeout: Command timeout in seconds
    
    Returns:
        JSON string with list of paths, or success message if output_file is specified
    """
    import re
    import json
    
    # Generate detailed timing report
    cmd = f"report_timing -return_string -max_paths {num_paths} -delay_type max -sort_by slack -nworst 1"
    
    try:
        timing_report = run_tcl_command(cmd, timeout=timeout)
    except Exception as e:
        return json.dumps({"error": f"Error generating timing report: {str(e)}"})
    
    # Parse paths
    path_sections = re.split(r'Slack \(', timing_report)
    
    all_paths = []
    
    for path_idx, path_section in enumerate(path_sections[1:], 1):  # Skip first (header)
        cell_names = []
        
        for line in path_section.split('\n'):
            # Match cell instances - look for hierarchical paths
            if '/' in line and not line.strip().startswith('net'):
                parts = line.split()
                for part in parts:
                    if '/' in part and not part.startswith('('):
                        # Remove pin suffix (e.g., /C, /D, /O, /Q, /CE, etc.)
                        cell_path = part
                        pin_suffixes = ['/C', '/D', '/Q', '/O', '/CE', '/R', '/S', '/CLR', '/PRE', 
                                       '/I0', '/I1', '/I2', '/I3', '/I4', '/I5', '/I6']
                        for suffix in pin_suffixes:
                            if cell_path.endswith(suffix):
                                cell_path = cell_path[:-len(suffix)]
                                break
                        if cell_path and cell_path not in cell_names:
                            cell_names.append(cell_path)
                        break
        
        if len(cell_names) >= 2:  # Only include paths with at least 2 cells
            all_paths.append(cell_names)
    
    # Write to file if specified, otherwise return JSON
    if output_file:
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_paths, f, indent=2)
            return json.dumps({
                "status": "success",
                "message": f"Extracted {len(all_paths)} critical paths",
                "output_file": output_file,
                "path_count": len(all_paths)
            })
        except Exception as e:
            return json.dumps({"error": f"Error writing to file: {str(e)}"})
    else:
        return json.dumps(all_paths)


def report_utilization_for_pblock(timeout: float = 300.0) -> str:
    """
    Get detailed resource utilization report for pblock sizing.
    
    Returns utilization of key resources:
    - LUTs, FFs, DSPs, BRAMs, URAMs
    - Formatted for easy parsing and pblock size calculation
    """
    cmd = "report_utilization -return_string"
    
    try:
        report = run_tcl_command(cmd, timeout=timeout)
    except Exception as e:
        return f"Error generating utilization report: {str(e)}"
    
    # Parse key resource counts
    resources = {
        "LUT": 0,
        "FF": 0,
        "DSP": 0,
        "BRAM": 0,
        "URAM": 0
    }
    
    lines = report.split('\n')
    for line in lines:
        # Look for slice logic section
        if '| Slice LUTs' in line or '| LUT as Logic' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    resources["LUT"] = int(parts[2].strip().split()[0])
                except:
                    pass
        
        if '| Register as Flip Flop' in line or '| Slice Registers' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    resources["FF"] = int(parts[2].strip().split()[0])
                except:
                    pass
        
        if '| DSPs' in line and '| Block RAM' not in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    resources["DSP"] = int(parts[2].strip().split()[0])
                except:
                    pass
        
        if '| Block RAM Tile' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    resources["BRAM"] = int(parts[2].strip().split()[0])
                except:
                    pass
        
        if '| URAM' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    resources["URAM"] = int(parts[2].strip().split()[0])
                except:
                    pass
    
    # Format output
    result_lines = [
        "=== Design Resource Utilization ===",
        "",
        f"LUTs:  {resources['LUT']:8,}",
        f"FFs:   {resources['FF']:8,}",
        f"DSPs:  {resources['DSP']:8,}",
        f"BRAMs: {resources['BRAM']:8,}",
        f"URAMs: {resources['URAM']:8,}",
        "",
        "=== 1.5x Multiplier (for pblock sizing) ===",
        "",
        f"LUTs:  {int(resources['LUT'] * 1.5):8,}",
        f"FFs:   {int(resources['FF'] * 1.5):8,}",
        f"DSPs:  {int(resources['DSP'] * 1.5):8,}",
        f"BRAMs: {int(resources['BRAM'] * 1.5):8,}",
        f"URAMs: {int(resources['URAM'] * 1.5):8,}",
    ]
    
    return "\n".join(result_lines)


def validate_pblock_resources(pblock_name: str) -> Dict[str, Any]:
    """
    Validate that a pblock has sufficient resources for the design primitives assigned to it.
    
    Returns:
        Dictionary with validation results including:
        - is_valid: True if resources are sufficient
        - resource_checks: Dict of resource type -> {required, available, margin}
        - errors: List of resource insufficiency errors
    """
    import re
    
    # Get pblock properties
    pblock_info = run_tcl_command(f"report_property [get_pblocks {pblock_name}]", timeout=30.0)
    
    # Parse PRIMITIVE_COUNT (total primitives assigned to pblock)
    primitive_count = 0
    cell_count = 0
    for line in pblock_info.split('\n'):
        if 'PRIMITIVE_COUNT' in line:
            parts = line.split()
            for p in parts:
                try:
                    primitive_count = int(p)
                    break
                except ValueError:
                    continue
        if 'CELL_COUNT' in line:
            parts = line.split()
            for p in parts:
                try:
                    cell_count = int(p)
                    break
                except ValueError:
                    continue
    
    # Run DRC to check for resource issues (this is the authoritative check)
    # Use file-based output to avoid buffering issues with -return_string
    import tempfile
    import time as time_module
    
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    drc_file = os.path.join(temp_dir, f"drc_check_{pblock_name}.rpt")
    
    drc_cmd = f"report_drc -checks {{UTLZ-1 UTLZ-2}} -file {{{drc_file}}}"
    run_tcl_command(drc_cmd, timeout=60.0)
    
    # Wait for file to be written using Tcl file size check
    drc_result = ""
    for retry in range(10):
        size_result = run_tcl_command(f"file size {{{drc_file}}}", timeout=10.0)
        try:
            file_size = int(size_result.strip())
            if file_size > 0:
                logger.info(f"DRC file ready: {file_size} bytes")
                break
        except ValueError:
            pass
        time_module.sleep(0.3)
    
    # Read the DRC file
    try:
        with open(drc_file, 'r') as f:
            drc_result = f.read()
        # Clean up temp file
        os.remove(drc_file)
    except Exception as e:
        logger.warning(f"Error reading DRC file: {e}")
    
    # Parse DRC results for resource errors
    errors = []
    resource_issues = {}
    
    logger.info(f"DRC result length: {len(drc_result)} chars")
    
    # Debug: show what we're checking
    utlz1_found = "UTLZ-1" in drc_result
    error_found = "Error" in drc_result
    logger.info(f"DRC content check: 'UTLZ-1' in result={utlz1_found}, 'Error' in result={error_found}")
    if utlz1_found or error_found:
        # Log first 600 chars to understand format
        logger.info(f"DRC result preview: {drc_result[:600]}")
    
    # First, simple check: if UTLZ-1 appears in the output, we have a hard error
    # The DRC summary table shows: "| UTLZ-1 | Error            |"
    # Also check for "UTLZ-1#" which indicates individual errors like "UTLZ-1#1 Error"
    has_utlz1_error = utlz1_found and error_found
    has_utlz2_warning = "UTLZ-2" in drc_result
    
    # Log what we found
    logger.info(f"DRC check: has_utlz1_error={has_utlz1_error}, has_utlz2_warning={has_utlz2_warning}")
    
    # Look for UTLZ-1 errors (hard over-utilization)
    # Format: "LUT6 over-utilized in Pblock ... requires 24377 of such cell types but only 6520 compatible"
    utlz1_pattern = r"(\w+(?:\s+\w+)*?) over-utilized.*?requires (\d+) of such cell types but only (\d+) compatible"
    for match in re.finditer(utlz1_pattern, drc_result, re.IGNORECASE | re.DOTALL):
        resource_type = match.group(1).strip()
        required = int(match.group(2))
        available = int(match.group(3))
        resource_issues[resource_type] = {
            'required': required,
            'available': available,
            'margin': available / required if required > 0 else 999,
            'shortage': required - available
        }
        errors.append(f"{resource_type}: requires {required}, only {available} available (shortage: {required - available})")
        logger.info(f"Found UTLZ-1 error: {resource_type} requires {required}, available {available}")
    
    # Look for UTLZ-2 warnings (over-utilized but placer might handle)
    # Format: "LUT as Logic over-utilized ... has 31370 LUT as Logic(s) assigned ... only 6520 ... available"
    utlz2_pattern = r"(\w+(?:\s+\w+)*?) over-utilized.*?has (\d+).*?only (\d+).*?available"
    for match in re.finditer(utlz2_pattern, drc_result, re.IGNORECASE | re.DOTALL):
        resource_type = match.group(1).strip()
        assigned = int(match.group(2))
        available = int(match.group(3))
        if resource_type not in resource_issues:  # Don't override UTLZ-1 errors
            resource_issues[resource_type] = {
                'required': assigned,
                'available': available,
                'margin': available / assigned if assigned > 0 else 999,
                'shortage': assigned - available,
                'warning_only': True
            }
            errors.append(f"{resource_type}: {assigned} assigned, only {available} available (may cause issues)")
            logger.info(f"Found UTLZ-2 warning: {resource_type} has {assigned}, available {available}")
    
    # Fallback: if we detected UTLZ-1 errors but couldn't parse details, add generic error
    if has_utlz1_error and not resource_issues:
        logger.warning("UTLZ-1 error detected but could not parse details")
        errors.append("UTLZ-1 error detected - pblock resources insufficient")
        resource_issues['unknown'] = {'required': 1, 'available': 0, 'margin': 0, 'shortage': 1}
    
    # is_valid only if there are no UTLZ-1 errors (hard failures)
    hard_errors = [e for e in resource_issues.values() if not e.get('warning_only', False)]
    is_valid = len(hard_errors) == 0 and not has_utlz1_error
    
    logger.info(f"Pblock validation: is_valid={is_valid}, hard_errors={len(hard_errors)}, total_issues={len(resource_issues)}")
    
    return {
        'is_valid': is_valid,
        'primitive_count': primitive_count,
        'cell_count': cell_count,
        'resource_issues': resource_issues,
        'errors': errors,
        'drc_output': drc_result[:1000] if len(drc_result) > 1000 else drc_result
    }


def expand_pblock_range(ranges: str, expansion_factor: float = 1.5) -> str:
    """
    Expand a pblock range by the given factor.
    
    Parses SLICE_X#Y#:SLICE_X#Y# format and expands the range.
    Area scales with the square of the linear factor, so expansion_factor=2.0 gives ~4x area.
    """
    import re
    
    expanded_parts = []
    
    logger.info(f"Expanding pblock range by factor {expansion_factor:.2f}x: {ranges}")
    
    for part in ranges.split():
        # Match pattern like SLICE_X67Y220:SLICE_X80Y272
        match = re.match(r'(\w+)_X(\d+)Y(\d+):(\w+)_X(\d+)Y(\d+)', part)
        if match:
            site_type = match.group(1)
            x_min = int(match.group(2))
            y_min = int(match.group(3))
            x_max = int(match.group(5))
            y_max = int(match.group(6))
            
            # Calculate expansion
            x_span = x_max - x_min
            y_span = y_max - y_min
            
            # Expand around the center
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            new_x_span = int(x_span * expansion_factor)
            new_y_span = int(y_span * expansion_factor)
            
            new_x_min = max(0, int(x_center - new_x_span / 2))
            new_x_max = int(x_center + new_x_span / 2)
            new_y_min = max(0, int(y_center - new_y_span / 2))
            new_y_max = int(y_center + new_y_span / 2)
            
            logger.info(f"  {site_type}: X{x_min}Y{y_min}:X{x_max}Y{y_max} -> X{new_x_min}Y{new_y_min}:X{new_x_max}Y{new_y_max}")
            expanded_parts.append(f"{site_type}_X{new_x_min}Y{new_y_min}:{site_type}_X{new_x_max}Y{new_y_max}")
        else:
            # Keep non-matching parts as-is
            logger.info(f"  Keeping as-is: {part}")
            expanded_parts.append(part)
    
    result = " ".join(expanded_parts)
    logger.info(f"Expanded pblock range: {result}")
    return result


def create_and_apply_pblock(
    pblock_name: str,
    ranges: str,
    apply_to: str = "current_design",
    is_soft: bool = False,
    timeout: float = 300.0,
    validate_resources: bool = True,
    max_expansion_attempts: int = 3
) -> str:
    """
    Create a pblock and apply it to the design with resource validation.
    
    Args:
        pblock_name: Name for the pblock (e.g., "pblock_tight")
        ranges: Pblock range specification (e.g., "SLICE_X0Y0:SLICE_X100Y100" or 
                "CLOCKREGION_X0Y0:CLOCKREGION_X2Y3")
        apply_to: What to apply pblock to - "current_design" applies to all cells in the design,
                 or provide a cell pattern (e.g., "design_1_wrapper_i/*")
        is_soft: If False, sets IS_SOFT property to 0 (hard constraint)
        validate_resources: If True, validate resources and auto-expand if needed
        max_expansion_attempts: Maximum times to try expanding the pblock
    
    Returns:
        Status message
    """
    result_lines = []
    current_ranges = ranges
    
    logger.info(f"Creating pblock '{pblock_name}' with range: {ranges}")
    logger.info(f"validate_resources={validate_resources}, max_expansion_attempts={max_expansion_attempts}")
    
    for attempt in range(max_expansion_attempts + 1):
        try:
            logger.info(f"Pblock creation attempt {attempt+1}/{max_expansion_attempts+1}")
            
            # Delete existing pblock if it exists (for retry attempts)
            if attempt > 0:
                try:
                    run_tcl_command(f"delete_pblocks [get_pblocks {pblock_name}]", timeout=10.0)
                    result_lines.append(f"\n=== Retry attempt {attempt} with expanded pblock ===")
                except Exception:
                    pass  # Pblock might not exist
            
            # Create the pblock
            create_cmd = f"create_pblock {pblock_name}"
            result = run_tcl_command(create_cmd, timeout=30.0)
            result_lines.append(f"Created pblock: {pblock_name}")
            
            # Add the range to the pblock
            resize_cmd = f"resize_pblock {pblock_name} -add {{{current_ranges}}}"
            result = run_tcl_command(resize_cmd, timeout=30.0)
            result_lines.append(f"Set pblock range: {current_ranges}")
            
            # Set IS_SOFT property
            soft_value = "1" if is_soft else "0"
            soft_cmd = f"set_property IS_SOFT {soft_value} [get_pblocks {pblock_name}]"
            result = run_tcl_command(soft_cmd, timeout=30.0)
            result_lines.append(f"Set IS_SOFT = {soft_value}")
            
            # Apply pblock to cells
            if apply_to == "current_design":
                add_cmd = f"add_cells_to_pblock {pblock_name} [get_cells -hierarchical]"
            else:
                add_cmd = f"add_cells_to_pblock {pblock_name} [get_cells {apply_to}]"
            
            result = run_tcl_command(add_cmd, timeout=timeout)
            result_lines.append(f"Applied pblock to: {apply_to}")
            
            # Validate resources if requested
            if validate_resources:
                validation = validate_pblock_resources(pblock_name)
                
                if not validation['is_valid']:
                    result_lines.append(f"\n⚠ Resource validation FAILED:")
                    for error in validation['errors']:
                        result_lines.append(f"  - {error}")
                    
                    if attempt < max_expansion_attempts:
                        # Calculate expansion factor based on worst shortage
                        worst_margin = min(
                            (issue['margin'] for issue in validation['resource_issues'].values()),
                            default=1.0
                        )
                        # Expand by inverse of margin plus some buffer
                        expansion_factor = max(1.5, 1.0 / worst_margin * 1.3)
                        result_lines.append(f"\n  Expanding pblock by factor {expansion_factor:.2f}x...")
                        
                        current_ranges = expand_pblock_range(current_ranges, expansion_factor)
                        continue  # Try again with expanded pblock
                    else:
                        result_lines.append(f"\n  Maximum expansion attempts reached. Consider using a larger region.")
                else:
                    result_lines.append(f"\n✓ Resource validation PASSED")
            
            # Verify the pblock
            verify_cmd = f"report_property [get_pblocks {pblock_name}]"
            verify_result = run_tcl_command(verify_cmd, timeout=30.0)
            
            result_lines.extend([
                "",
                "=== Pblock Created Successfully ===",
                f"Name: {pblock_name}",
                f"Range: {current_ranges}",
                f"IS_SOFT: {soft_value}",
                f"Applied to: {apply_to}",
                "",
                "Next steps:",
                "1. Run place_design to re-place with pblock constraint",
                "2. Run route_design to route the newly placed design",
                "3. Check timing with report_timing_summary"
            ])
            
            return "\n".join(result_lines)
            
        except Exception as e:
            result_lines.append(f"Error in attempt {attempt}: {str(e)}")
            if attempt >= max_expansion_attempts:
                return f"Error creating/applying pblock: {str(e)}\n" + "\n".join(result_lines)
    
    return "\n".join(result_lines)


# Create MCP server
server = Server("vivado-mcp")


@server.list_tools()
async def list_tools():
    """List available Vivado tools."""
    return [
        Tool(
            name="open_checkpoint",
            description="Open a Vivado Design Checkpoint (.dcp) file. Closes any currently open design first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dcp_path": {
                        "type": "string",
                        "description": "Path to the .dcp file to open"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["dcp_path"]
            }
        ),
        Tool(
            name="write_checkpoint",
            description="Write the current design to a Vivado Design Checkpoint (.dcp) file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dcp_path": {
                        "type": "string",
                        "description": "Path where the .dcp file will be saved"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Overwrite existing file if True (default: False)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["dcp_path"]
            }
        ),
        Tool(
            name="report_route_status",
            description="Get the routing status report for the current design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                }
            }
        ),
        Tool(
            name="report_timing_summary",
            description="Get a timing summary report for the current design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                }
            }
        ),
        Tool(
            name="get_wns",
            description="Get the Worst Negative Slack (WNS) value directly. Returns just the numeric slack value in nanoseconds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 60)"
                    }
                }
            }
        ),
        Tool(
            name="place_design",
            description="Run placement on the current design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": "Placement directive (e.g., 'Default', 'Explore', 'Quick')"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 3600 for placement)"
                    }
                }
            }
        ),
        Tool(
            name="route_design",
            description="Run routing on the current design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": "Routing directive (e.g., 'Default', 'Explore', 'Quick')"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 3600 for routing)"
                    }
                }
            }
        ),
        Tool(
            name="run_tcl",
            description="Execute an arbitrary Tcl command in Vivado.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Tcl command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="restart_vivado",
            description="Kill the current Vivado instance and start a fresh one. Use if Vivado is hung or stuck.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_critical_high_fanout_nets",
            description="Extract high fanout nets from critical timing paths for optimization. Returns parent net names for RapidWright compatibility.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_paths": {
                        "type": "number",
                        "description": "Number of critical paths to analyze (default: 50)"
                    },
                    "min_fanout": {
                        "type": "number",
                        "description": "Minimum fanout threshold to report a net (default: 100)"
                    },
                    "exclude_clocks": {
                        "type": "boolean",
                        "description": "If True, exclude clock nets from results (default: True)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 600)"
                    }
                }
            }
        ),
        Tool(
            name="write_edif",
            description="Write an unencrypted EDIF netlist file. This is required when exporting designs for use with RapidWright, as the EDIF netlist inside DCPs is typically encrypted.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edif_path": {
                        "type": "string",
                        "description": "Path where the .edf or .edif file will be saved"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Overwrite existing file if True (default: False)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["edif_path"]
            }
        ),
        Tool(
            name="extract_critical_path_cells",
            description="""Extract cell names from critical timing paths for spread analysis.
            
            Parses timing report to get ordered list of cells on each critical path.
            Output is JSON that can be passed to RapidWright's analyze_critical_path_spread 
            to calculate Manhattan distances.
            
            Can optionally write to a file for efficient data transfer.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_paths": {
                        "type": "number",
                        "description": "Number of critical paths to extract (default: 50)"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional: path to write JSON output to file instead of returning it"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 600)"
                    }
                }
            }
        ),
        Tool(
            name="report_utilization_for_pblock",
            description="""Get design resource utilization for pblock sizing.
            
            Returns counts of LUTs, FFs, DSPs, BRAMs, URAMs with both actual usage and 
            1.5x multiplied values for pblock size calculation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                }
            }
        ),
        Tool(
            name="create_and_apply_pblock",
            description="""Create a pblock (area constraint) and apply it to the design.
            
            A pblock restricts placement to a specific region of the FPGA. This can improve timing
            by reducing routing distances for spread-out designs. After applying a pblock, you must
            run place_design and route_design to implement the constraint.
            
            Range format examples:
            - SLICE_X0Y0:SLICE_X100Y200 (specific slice ranges)
            - CLOCKREGION_X0Y0:CLOCKREGION_X2Y3 (clock region ranges)
            
            Set is_soft=False for hard constraints that must be met.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pblock_name": {
                        "type": "string",
                        "description": "Name for the pblock (e.g., 'pblock_tight')"
                    },
                    "ranges": {
                        "type": "string",
                        "description": "Pblock range (e.g., 'SLICE_X0Y0:SLICE_X100Y100' or 'CLOCKREGION_X0Y0:CLOCKREGION_X2Y3')"
                    },
                    "apply_to": {
                        "type": "string",
                        "description": "What to constrain: 'current_design' (all cells) or a cell pattern (default: 'current_design')"
                    },
                    "is_soft": {
                        "type": "boolean",
                        "description": "If false, creates hard constraint (IS_SOFT=0) (default: false)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["pblock_name", "ranges"]
            }
        ),
        Tool(
            name="write_verilog_simulation",
            description="""Export design as a Verilog functional simulation model.
            
            Generates a Verilog netlist suitable for simulation. This is required for
            functional equivalence checking via simulation. The output netlist can be
            used with xsim or other Verilog simulators.
            
            Use -mode funcsim for functional simulation (no timing).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "verilog_path": {
                        "type": "string",
                        "description": "Path where the .v file will be saved"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Overwrite existing file if True (default: False)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["verilog_path"]
            }
        ),
        Tool(
            name="phys_opt_design",
            description="""Run physical optimization on the current design to improve timing (WNS/TNS). 
            
            Can be run post-place (after place_design) or post-route (after route_design). Performs timing-driven 
            optimization on negative-slack paths. The command operates on the in-memory design and can be run 
            iteratively for additional improvements.
            
            Post-place optimizations (default): fanout optimization, placement optimization, LUT restructure, 
            critical-cell optimization, DSP/BRAM/URAM register optimization.
            
            Post-route optimizations (default): placement optimization, routing optimization, LUT restructure, 
            critical-cell optimization.
            
            NOTE: Using specific optimization options disables default optimizations - only specified ones run.
            The directive option is incompatible with specific optimization options.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": """Physical optimization directive. Only one can be specified at a time, and incompatible with other options:
                        - Default: Run phys_opt_design with default settings
                        - Explore: Multiple passes with replication for very high fanout nets, SLR crossing optimization, and critical path optimization
                        - ExploreWithHoldFix: Multiple passes including hold violation fixing, SLR crossing optimization, and replication for very high fanout nets
                        - ExploreWithAggressiveHoldFix: Multiple passes with aggressive hold violation fixing, SLR crossing optimization, and replication
                        - AggressiveExplore: Similar to Explore but with more aggressive algorithms; includes SLR crossing optimization that may temporarily degrade WNS
                        - AlternateReplication: Use different algorithms for performing critical cell replication
                        - AggressiveFanoutOpt: Use different algorithms for fanout-related optimizations with more aggressive goals
                        - AlternateFlowWithRetiming: Perform more aggressive replication and DSP/BRAM optimization with register retiming enabled
                        - AddRetime: Performs the default phys_opt_design flow and adds register retiming
                        - RuntimeOptimized: Reduced set of optimizations (fanout_opt, critical_cell_opt, placement_opt, bram_enable_opt) for shortest runtime
                        - RQS: Select directive from report_qor_suggestions strategy (requires RQS file)"""
                    },
                    "fanout_opt": {
                        "type": "boolean",
                        "description": "[Note: Cannot be used for post route design, use the optimization from RapidWright instead.] Delay-driven optimization on high-fanout timing critical nets by replicating drivers (not applicable for Versal)"
                    },
                    "placement_opt": {
                        "type": "boolean",
                        "description": "Move cells to reduce delay on timing-critical nets (not applicable for Versal)"
                    },
                    "routing_opt": {
                        "type": "boolean",
                        "description": "Perform routing optimization on timing-critical nets to reduce delay"
                    },
                    "slr_crossing_opt": {
                        "type": "boolean",
                        "description": "Optimize placement of inter-SLR connections (UltraScale/UltraScale+ only)"
                    },
                    "insert_negative_edge_ffs": {
                        "type": "boolean",
                        "description": "Insert negative edge triggered FFs for hold optimization"
                    },
                    "restruct_opt": {
                        "type": "boolean",
                        "description": "Advanced LUT restructure optimization to reduce logic levels and delay on critical signals"
                    },
                    "interconnect_retime": {
                        "type": "boolean",
                        "description": "Perform interconnect retiming by moving/replicating FF or LUT-FF pairs (Versal only)"
                    },
                    "lut_opt": {
                        "type": "boolean",
                        "description": "Perform LUT movement/replication to improve critical path timing (Versal only)"
                    },
                    "casc_opt": {
                        "type": "boolean",
                        "description": "Perform LUT cascade optimization for creating/moving LUT cascades (Versal only)"
                    },
                    "cell_group_opt": {
                        "type": "boolean",
                        "description": "Perform critical cell group optimization"
                    },
                    "equ_drivers_opt": {
                        "type": "boolean",
                        "description": "Rewire load pins to equivalent drivers"
                    },
                    "critical_cell_opt": {
                        "type": "boolean",
                        "description": "Cell-duplication based optimization on timing critical nets (not applicable for Versal)"
                    },
                    "dsp_register_opt": {
                        "type": "boolean",
                        "description": "Move registers between slices and DSP blocks to improve critical path delay"
                    },
                    "bram_register_opt": {
                        "type": "boolean",
                        "description": "Move registers between slices and block RAMs to improve critical path delay"
                    },
                    "uram_register_opt": {
                        "type": "boolean",
                        "description": "Move registers between slices and UltraRAMs to improve critical path delay"
                    },
                    "bram_enable_opt": {
                        "type": "boolean",
                        "description": "Improve timing on critical paths involving power-optimized block RAMs by reversing enable-logic optimization"
                    },
                    "shift_register_opt": {
                        "type": "boolean",
                        "description": "Perform shift register optimization by extracting registers from SRL chains to improve timing"
                    },
                    "hold_fix": {
                        "type": "boolean",
                        "description": "Insert data path delay to fix hold time violations"
                    },
                    "aggressive_hold_fix": {
                        "type": "boolean",
                        "description": "Aggressively insert data path delay to fix hold time violations (considers more violations than standard hold fix)"
                    },
                    "retime": {
                        "type": "boolean",
                        "description": "Re-time registers forward through combinational logic to balance path delays (property-driven)"
                    },
                    "force_replication_on_nets": {
                        "type": "string",
                        "description": "Force replication on specific nets regardless of slack (e.g., net names or Tcl command like '[get_nets -hier *phy_reset*]')"
                    },
                    "critical_pin_opt": {
                        "type": "boolean",
                        "description": "Perform LUT pin-swapping (remap logical to physical pins) to improve critical path timing. Skips cells with LOCK_PINS property."
                    },
                    "clock_opt": {
                        "type": "boolean",
                        "description": "Perform clock skew optimization during post-route optimization by inserting global clock buffers"
                    },
                    "path_groups": {
                        "type": "string",
                        "description": "Perform optimizations on specified path groups only (e.g., 'clk_group1 clk_group2')"
                    },
                    "tns_cleanup": {
                        "type": "boolean",
                        "description": "Total Negative Slack cleanup (use with slr_crossing_opt). Allows some slack degradation if overall WNS doesn't degrade."
                    },
                    "sll_reg_hold_fix": {
                        "type": "boolean",
                        "description": "Perform SLL register hold fix optimization for SLR crossing paths (not applicable for Versal)"
                    },
                    "memory_rewire_opt": {
                        "type": "boolean",
                        "description": "Rewire critical signals to faster pins of BRAM/URAM (Versal only, not for cascaded/ECC memories)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 3600 for physical optimization)"
                    }
                }
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    global _design_open
    
    logger.info(f"Tool called: {name}")
    
    try:
        if name == "open_checkpoint":
            dcp_path = arguments["dcp_path"]
            timeout = arguments.get("timeout", 300)
            
            # Close existing design if open
            if _design_open:
                close_current_design()
            
            # Open the checkpoint
            output = run_tcl_command(f"open_checkpoint {{{dcp_path}}}", timeout=timeout)
            _design_open = True
            return [TextContent(type="text", text=f"Opened checkpoint: {dcp_path}\n\n{output}")]
        
        elif name == "write_checkpoint":
            dcp_path = arguments["dcp_path"]
            force = arguments.get("force", False)
            timeout = arguments.get("timeout", 300)
            
            force_flag = " -force" if force else ""
            output = run_tcl_command(f"write_checkpoint{force_flag} {{{dcp_path}}}", timeout=timeout)
            return [TextContent(type="text", text=f"Wrote checkpoint: {dcp_path}\n\n{output}")]
        
        elif name == "report_route_status":
            timeout = arguments.get("timeout", 300)
            # Run a quick command first to flush any leftover output from previous commands
            run_tcl_command("puts {route_status_start}", timeout=5)
            output = run_tcl_command("report_route_status -return_string", timeout=timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "report_timing_summary":
            timeout = arguments.get("timeout", 300)
            # Run a quick command first to flush any leftover output from previous commands
            run_tcl_command("puts {timing_summary_start}", timeout=5)
            output = run_tcl_command("report_timing_summary -return_string", timeout=timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "get_wns":
            timeout = arguments.get("timeout", 60)
            # Flush buffer first
            run_tcl_command("puts {get_wns_start}", timeout=5)
            # Get WNS directly from timing path slack property (single line to avoid buffer issues)
            tcl_cmd = "set wns_path [get_timing_paths -max_paths 1 -slack_lesser_than 999]; if {[llength $wns_path] > 0} {get_property SLACK $wns_path} else {puts 0.0}"
            output = run_tcl_command(tcl_cmd, timeout=timeout)
            # Clean up the output to just return the number
            wns_value = output.strip().split('\n')[-1].strip()
            return [TextContent(type="text", text=wns_value)]
        
        elif name == "place_design":
            directive = arguments.get("directive")
            timeout = arguments.get("timeout", 3600)  # 1 hour default for placement
            
            cmd = "place_design"
            if directive:
                cmd += f" -directive {directive}"
            
            output = run_tcl_command(cmd, timeout=timeout)
            return [TextContent(type="text", text=f"Placement complete.\n\n{output}")]
        
        elif name == "route_design":
            directive = arguments.get("directive")
            timeout = arguments.get("timeout", 3600)  # 1 hour default for routing
            
            cmd = "route_design"
            if directive:
                cmd += f" -directive {directive}"
            
            output = run_tcl_command(cmd, timeout=timeout)
            return [TextContent(type="text", text=f"Routing complete.\n\n{output}")]
        
        elif name == "run_tcl":
            command = arguments["command"]
            timeout = arguments.get("timeout", 300)
            output = run_tcl_command(command, timeout=timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "restart_vivado":
            output = restart_vivado_process()
            return [TextContent(type="text", text=output)]
        
        elif name == "get_critical_high_fanout_nets":
            num_paths = arguments.get("num_paths", 50)
            min_fanout = arguments.get("min_fanout", 100)
            exclude_clocks = arguments.get("exclude_clocks", True)
            timeout = arguments.get("timeout", 600)
            
            output = get_critical_high_fanout_nets(num_paths, min_fanout, exclude_clocks, timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "write_edif":
            edif_path = arguments["edif_path"]
            force = arguments.get("force", False)
            timeout = arguments.get("timeout", 300)
            
            force_flag = " -force" if force else ""
            output = run_tcl_command(f"write_edif{force_flag} {{{edif_path}}}", timeout=timeout)
            return [TextContent(type="text", text=f"Wrote EDIF netlist: {edif_path}\n\n{output}")]
        
        elif name == "extract_critical_path_cells":
            num_paths = arguments.get("num_paths", 50)
            output_file = arguments.get("output_file")
            timeout = arguments.get("timeout", 600)
            
            output = extract_critical_path_cells(num_paths, output_file, timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "report_utilization_for_pblock":
            timeout = arguments.get("timeout", 300)
            output = report_utilization_for_pblock(timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "create_and_apply_pblock":
            pblock_name = arguments["pblock_name"]
            ranges = arguments["ranges"]
            apply_to = arguments.get("apply_to", "current_design")
            is_soft = arguments.get("is_soft", False)
            timeout = arguments.get("timeout", 300)
            
            output = create_and_apply_pblock(pblock_name, ranges, apply_to, is_soft, timeout)
            return [TextContent(type="text", text=output)]
        
        elif name == "write_verilog_simulation":
            verilog_path = arguments["verilog_path"]
            force = arguments.get("force", False)
            timeout = arguments.get("timeout", 300)
            
            force_flag = " -force" if force else ""
            # Use -mode funcsim for functional simulation
            output = run_tcl_command(f"write_verilog{force_flag} -mode funcsim {{{verilog_path}}}", timeout=timeout)
            return [TextContent(type="text", text=f"Wrote Verilog simulation model: {verilog_path}\n\n{output}")]
        
        elif name == "phys_opt_design":
            timeout = arguments.get("timeout", 3600)  # 1 hour default for physical optimization
            
            cmd = "phys_opt_design"
            
            # Directive option (incompatible with other options)
            directive = arguments.get("directive")
            if directive:
                cmd += f" -directive {directive}"
            else:
                # Build command with specific optimization options
                # Boolean flags
                bool_options = [
                    "fanout_opt", "placement_opt", "routing_opt", "slr_crossing_opt",
                    "insert_negative_edge_ffs", "restruct_opt", "interconnect_retime",
                    "lut_opt", "casc_opt", "cell_group_opt", "equ_drivers_opt",
                    "critical_cell_opt", "dsp_register_opt", "bram_register_opt",
                    "uram_register_opt", "bram_enable_opt", "shift_register_opt",
                    "hold_fix", "aggressive_hold_fix", "retime", "critical_pin_opt",
                    "clock_opt", "tns_cleanup", "sll_reg_hold_fix", "memory_rewire_opt"
                ]
                
                for opt in bool_options:
                    if arguments.get(opt):
                        cmd += f" -{opt}"
                
                # String options
                force_replication = arguments.get("force_replication_on_nets")
                if force_replication:
                    cmd += f" -force_replication_on_nets {force_replication}"
                
                path_groups = arguments.get("path_groups")
                if path_groups:
                    cmd += f" -path_groups {{{path_groups}}}"
            
            output = run_tcl_command(cmd, timeout=timeout)
            return [TextContent(type="text", text=f"Physical optimization complete.\n\n{output}")]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except pexpect.TIMEOUT:
        return [TextContent(
            type="text", 
            text=f"Error: Command timed out. Vivado may be stuck. Use restart_vivado to recover."
        )]
    except pexpect.EOF:
        return [TextContent(
            type="text",
            text="Error: Vivado process terminated unexpectedly. Use restart_vivado to restart."
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point."""
    global _vivado_path, _vivado_log_file, _vivado_journal_file
    
    parser = argparse.ArgumentParser(description="Vivado MCP Server")
    parser.add_argument(
        "--vivado-path",
        type=str,
        help="Path to Vivado executable (default: search in PATH)"
    )
    parser.add_argument(
        "--vivado-log",
        type=str,
        help="Path to Vivado log file (default: vivado.log)"
    )
    parser.add_argument(
        "--vivado-journal",
        type=str,
        help="Path to Vivado journal file (default: vivado.jou)"
    )
    
    args = parser.parse_args()
    
    if args.vivado_path:
        _vivado_path = args.vivado_path
    
    if args.vivado_log:
        _vivado_log_file = args.vivado_log
    
    if args.vivado_journal:
        _vivado_journal_file = args.vivado_journal
    
    logger.info("Starting Vivado MCP Server...")
    
    # Run the MCP server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server running on stdio transport")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
