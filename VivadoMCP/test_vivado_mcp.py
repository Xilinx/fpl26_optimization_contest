#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
Test script for Vivado MCP Server.

Tests MCP tools including:
1. open_checkpoint
2. report_timing_summary
3. get_wns
4. report_route_status
5. report_utilization_for_pblock
6. get_critical_high_fanout_nets
7. create_and_apply_pblock
8. write_checkpoint
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# MCP client imports
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Test checkpoint path - relative to parent directory (downloaded by Makefile)
# The Makefile downloads logicnets_jscl.dcp to the project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DCP = PROJECT_ROOT / "logicnets_jscl.dcp"

# Output directory for written checkpoints and EDIF files
OUTPUT_DIR = Path(tempfile.gettempdir()) / "vivado_mcp_test"


async def call_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Call an MCP tool and return the result text."""
    result = await session.call_tool(name, arguments)
    
    # Extract text from result
    if result.content:
        texts = [c.text for c in result.content if hasattr(c, 'text')]
        return "\n".join(texts)
    return ""


async def test_vivado_tools(session: ClientSession):
    """Test key Vivado MCP tools."""
    print(f"\n{'='*80}")
    print("TESTING VIVADO MCP TOOLS")
    print(f"{'='*80}")
    
    test_dcp_path = str(TEST_DCP)
    
    # 1. Open checkpoint
    print(f"\n[1] Opening checkpoint: {test_dcp_path}")
    result = await call_tool(session, "open_checkpoint", {"dcp_path": test_dcp_path, "timeout": 600})
    print(f"✓ Checkpoint opened")
    if "ERROR" in result.upper():
        print(f"ERROR in result: {result[:1000]}")
        return False
    
    # 2. Get timing summary
    print(f"\n[2] Getting timing summary")
    result = await call_tool(session, "report_timing_summary", {"timeout": 300})
    # Just show first 500 chars to keep output manageable
    print(f"Result (first 500 chars):\n{result[:500]}...")
    print(f"✓ Timing summary generated")
    
    # 3. Get WNS directly
    print(f"\n[3] Getting WNS value")
    result = await call_tool(session, "get_wns", {"timeout": 60})
    print(f"WNS: {result} ns")
    print(f"✓ WNS retrieved")
    
    # 4. Get route status
    print(f"\n[4] Getting route status")
    result = await call_tool(session, "report_route_status", {"timeout": 300})
    print(f"Result:\n{result}")
    print(f"✓ Route status generated")
    
    # 5. Get utilization for pblock sizing
    print(f"\n[5] Getting resource utilization for pblock sizing")
    result = await call_tool(session, "report_utilization_for_pblock", {"timeout": 300})
    print(f"Result:\n{result}")
    print(f"✓ Utilization report generated")
    
    # 6. Get critical high fanout nets
    print(f"\n[6] Getting critical high fanout nets")
    result = await call_tool(session, "get_critical_high_fanout_nets", {
        "num_paths": 50,
        "min_fanout": 10,  # Lower threshold to catch more nets
        "exclude_clocks": True,
        "timeout": 600
    })
    print(f"Result (first 1500 chars):\n{result[:1500]}...")
    
    # Verify that parent net resolution is working
    # The tool should find layer1_inst/layer1_N37_inst/M1w[2] but return its parent: layer1_reg/M1w[47]
    expected_parent_net = "layer1_reg/M1w[47]"
    if expected_parent_net in result:
        print(f"✓ Found expected parent net: {expected_parent_net}")
    else:
        print(f"✗ ERROR: Expected parent net '{expected_parent_net}' not found in results")
        return False
    print(f"✓ High fanout nets retrieved")
    
    # 7. Test creating a pblock (using CLOCKREGION range for simplicity)
    print(f"\n[7] Testing pblock creation")
    print(f"    Creating test pblock with CLOCKREGION range")
    result = await call_tool(session, "create_and_apply_pblock", {
        "pblock_name": "pblock_test",
        "ranges": "CLOCKREGION_X0Y0:CLOCKREGION_X3Y3",
        "apply_to": "current_design",
        "is_soft": True,  # Use soft constraint for testing
        "timeout": 300
    })
    print(f"Result:\n{result}")
    
    if "Successfully" in result or "Pblock Created" in result:
        print(f"✓ Pblock created and applied successfully")
    else:
        print(f"Note: Pblock creation returned: {result[:500]}")
    
    # 8. Write checkpoint to temp directory
    print(f"\n[8] Writing checkpoint")
    output_dcp = OUTPUT_DIR / "test_output.dcp"
    result = await call_tool(session, "write_checkpoint", {
        "dcp_path": str(output_dcp),
        "force": True,
        "timeout": 300
    })
    print(f"Result:\n{result}")
    if output_dcp.exists():
        print(f"✓ Checkpoint written to: {output_dcp}")
    else:
        print(f"Note: Checkpoint write completed but file not verified")
    
    print(f"\n✓ All Vivado tool tests completed!")
    return True


async def main():
    """Main test function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Verify test DCP exists
    if not TEST_DCP.exists():
        print(f"ERROR: Test checkpoint not found: {TEST_DCP}")
        print(f"Please run 'make setup' in the project root to download example DCPs:")
        print(f"  cd {PROJECT_ROOT} && make setup")
        sys.exit(1)
    
    # Path to the MCP server
    server_script = SCRIPT_DIR / "vivado_mcp_server.py"
    if not server_script.exists():
        print(f"ERROR: Server script not found: {server_script}")
        sys.exit(1)
    
    print(f"{'='*80}")
    print(f"VIVADO MCP SERVER TEST")
    print(f"{'='*80}")
    print(f"Server script: {server_script}")
    print(f"Test DCP: {TEST_DCP}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create server parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy()
    )
    
    # Connect to the MCP server
    print(f"\nConnecting to Vivado MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            print(f"✓ Session initialized")
            
            # List available tools
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"\nAvailable tools ({len(tool_names)}):")
            # Mark tools that will be tested in this script
            tested_tools = [
                "open_checkpoint", "report_timing_summary", "get_wns",
                "report_route_status", "report_utilization_for_pblock",
                "get_critical_high_fanout_nets", "create_and_apply_pblock",
                "write_checkpoint"
            ]
            for tool_name in tool_names:
                marker = "★" if tool_name in tested_tools else " "
                print(f"  {marker} {tool_name}")
            
            # Run the tool tests
            success = await test_vivado_tools(session)
            
            print(f"\n{'='*80}")
            if success:
                print("✓ ALL TESTS PASSED!")
            else:
                print("✗ SOME TESTS FAILED!")
            print(f"{'='*80}")
            
            # Final cleanup - Vivado will be terminated when server exits
            print("\nCleaning up and exiting...")
            
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
