#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
Simple test script to verify RapidWright MCP server functionality.
Tests the server's tools directly without going through MCP protocol.
"""
import sys
import json
import os
import tempfile
import rapidwright_tools as rw


def print_result(tool_name, result):
    """Pretty print a tool result."""
    print(f"\n{'='*60}")
    print(f"Tool: {tool_name}")
    print('='*60)
    print(json.dumps(result, indent=2))


def main():
    """Run basic tests."""
    print("RapidWright MCP Server - Test Suite")
    print("=" * 60)
    
    # Test 1: Initialize RapidWright
    print("\n[1/4] Initializing RapidWright...")
    result = rw.initialize_rapidwright()
    print_result("initialize_rapidwright", result)
    
    if result.get("status") != "success":
        print("\n❌ ERROR: Failed to initialize RapidWright.")
        print("Please ensure:")
        print("  - Java 11+ is installed: java -version")
        print("  - RapidWright is installed: pip3 install rapidwright")
        return 1
    
    # Test 2: Get supported devices
    print("\n[2/4] Getting supported devices...")
    result = rw.get_supported_devices()
    print_result("get_supported_devices", result)
    
    if result.get("status") != "success":
        print("\n❌ ERROR: Failed to get supported devices.")
        return 1
    
    # Test 3: Get device info
    device_to_test = None
    if result.get("devices"):
        # Try to find a common device
        devices = result["devices"]
        for common in ["xcvu9p", "xcvu3p", "xcku040"]:
            if common in devices:
                device_to_test = common
                break
        if not device_to_test:
            device_to_test = devices[0]  # Use first available
    
    if device_to_test:
        print(f"\n[3/4] Getting device info for {device_to_test}...")
        result = rw.get_device_info(device_to_test)
        print_result("get_device_info", result)
        
        if result.get("status") != "success":
            print(f"\n⚠️  WARNING: Could not get info for {device_to_test}")
    else:
        print("\n[3/4] Skipping device info test - no devices available")
    
    # Test 4: Search for sites
    if device_to_test:
        print(f"\n[4/4] Searching for SLICE sites on {device_to_test}...")
        result = rw.search_sites(site_type="SLICE", device_name=device_to_test, limit=10)
        print_result("search_sites", result)
    else:
        print("\n[4/4] Skipping site search test - no device selected")
    
    # Test 5: Test write_checkpoint by creating a design from scratch
    print("\n[5/6] Testing write_checkpoint with a new design...")
    result = test_write_checkpoint()
    if result != 0:
        print("\n❌ ERROR: write_checkpoint test failed.")
        return 1
    
    # Test 6: Test overwrite behavior
    print("\n[6/6] Testing write_checkpoint overwrite behavior...")
    result = test_write_checkpoint_overwrite()
    if result != 0:
        print("\n❌ ERROR: write_checkpoint overwrite test failed.")
        return 1
    
    print("\n" + "="*60)
    print("✅ Test suite completed successfully!")
    print("="*60)
    print("\nThe server is working correctly!")
    print("\nNext steps:")
    print("1. Configure Cursor/Claude Desktop (see CURSOR_SETUP.md)")
    print("2. Restart Cursor/Claude Desktop")
    print("3. Ask your AI assistant: 'Initialize RapidWright and show me devices'")
    
    return 0


def test_write_checkpoint():
    """
    Test the write_checkpoint tool by creating a design from scratch,
    writing it to a DCP, and verifying the result.
    """
    try:
        from com.xilinx.rapidwright.design import Design
        
        # Create a simple design from scratch
        # Using a common device that should be available
        design_name = "test_write_checkpoint_design"
        part_name = "xcvu3p-ffvc1517-2-e"  # A common UltraScale+ part
        
        print(f"  Creating new design '{design_name}' for part {part_name}...")
        design = Design(design_name, part_name)
        
        # Store it as the current design for the tools to use
        rw._current_design = design
        
        # Write to a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            dcp_path = os.path.join(tmpdir, "test_output.dcp")
            
            print(f"  Writing checkpoint to {dcp_path}...")
            result = rw.write_checkpoint(dcp_path)
            print_result("write_checkpoint", result)
            
            if result.get("status") != "success":
                print(f"  ❌ write_checkpoint failed: {result.get('error')}")
                return 1
            
            # Verify file was created
            if not os.path.exists(dcp_path):
                print(f"  ❌ DCP file was not created at {dcp_path}")
                return 1
            
            # Verify bytes_written matches actual file size
            actual_size = os.path.getsize(dcp_path)
            reported_size = result.get("bytes_written", 0)
            
            if actual_size != reported_size:
                print(f"  ❌ Size mismatch: reported {reported_size}, actual {actual_size}")
                return 1
            
            print(f"  ✓ DCP file created successfully ({actual_size} bytes)")
            
            # Verify we can read the checkpoint back
            print(f"  Reading checkpoint back to verify...")
            read_result = rw.read_checkpoint(dcp_path)
            
            if read_result.get("status") != "success":
                print(f"  ❌ Could not read back the DCP: {read_result.get('error')}")
                return 1
            
            # Verify the design name matches
            if read_result.get("design_name") != design_name:
                print(f"  ❌ Design name mismatch: expected '{design_name}', got '{read_result.get('design_name')}'")
                return 1
            
            print(f"  ✓ DCP file verified - design name matches")
            print_result("read_checkpoint (verification)", read_result)
        
        print("  ✓ write_checkpoint test passed!")
        return 0
        
    except Exception as e:
        print(f"  ❌ Exception during write_checkpoint test: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_write_checkpoint_overwrite():
    """
    Test the overwrite behavior of write_checkpoint.
    """
    try:
        from com.xilinx.rapidwright.design import Design
        
        # Create a simple design
        design_name = "test_overwrite_design"
        part_name = "xcvu3p-ffvc1517-2-e"
        
        print(f"  Creating design for overwrite test...")
        design = Design(design_name, part_name)
        rw._current_design = design
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dcp_path = os.path.join(tmpdir, "test_overwrite.dcp")
            
            # First write should succeed
            print(f"  First write to {dcp_path}...")
            result = rw.write_checkpoint(dcp_path)
            
            if result.get("status") != "success":
                print(f"  ❌ First write failed: {result.get('error')}")
                return 1
            
            first_size = result.get("bytes_written", 0)
            print(f"  ✓ First write succeeded ({first_size} bytes)")
            
            # Second write without overwrite should fail
            print(f"  Second write without overwrite (should fail)...")
            result = rw.write_checkpoint(dcp_path, overwrite=False)
            
            if result.get("status") == "success":
                print("  ❌ Second write should have failed but succeeded")
                return 1
            
            if "already exists" not in result.get("error", ""):
                print(f"  ❌ Expected 'already exists' error, got: {result.get('error')}")
                return 1
            
            print(f"  ✓ Second write correctly rejected (file exists)")
            
            # Third write with overwrite=True should succeed
            print(f"  Third write with overwrite=True...")
            result = rw.write_checkpoint(dcp_path, overwrite=True)
            
            if result.get("status") != "success":
                print(f"  ❌ Third write with overwrite failed: {result.get('error')}")
                return 1
            
            print(f"  ✓ Third write with overwrite succeeded ({result.get('bytes_written')} bytes)")
        
        print("  ✓ write_checkpoint overwrite test passed!")
        return 0
        
    except Exception as e:
        print(f"  ❌ Exception during overwrite test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
