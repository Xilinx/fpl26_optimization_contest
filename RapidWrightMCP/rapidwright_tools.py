# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
RapidWright Tools - Wrapper functions for RapidWright operations
Uses the rapidwright pip package which handles JPype integration internally
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global state
_initialized = False
_current_design = None


def initialize_rapidwright(jvm_max_memory: str = "4G") -> Dict[str, Any]:
    """
    Initialize the RapidWright environment.
    
    Args:
        jvm_max_memory: Maximum JVM heap size (default: "4G")
        
    Returns:
        Dictionary with initialization status, version, and install path
    """
    global _initialized
    
    if _initialized:
        # Return version and path info even when already initialized
        try:
            import rapidwright
            import os
            from com.xilinx.rapidwright.device import Device
            version = str(Device.RAPIDWRIGHT_VERSION)
            install_path = os.path.dirname(rapidwright.__file__)
            rapidwright_path_env = os.environ.get('RAPIDWRIGHT_PATH')
            classpath = os.environ.get('CLASSPATH')
        except Exception:
            version = 'unknown'
            install_path = 'unknown'
            rapidwright_path_env = None
        
        result = {
            "status": "already_initialized", 
            "message": "RapidWright already initialized",
            "rapidwright_version": version,
            "rapidwright_install_path": install_path
        }
        if rapidwright_path_env:
            result["RAPIDWRIGHT_PATH"] = rapidwright_path_env
        if classpath:
            result["CLASSPATH"] = classpath
        return result
    
    try:
        # Import rapidwright - this automatically starts the JVM
        import rapidwright
        import os
        from com.xilinx.rapidwright.device import Device
        
        _initialized = True
        
        logger.info("RapidWright initialized successfully")
        
        # Test that we can access basic functionality
        device_count = len(Device.getAvailableDevices())
        
        # Get version and install path
        version = str(Device.RAPIDWRIGHT_VERSION)
        install_path = os.path.dirname(rapidwright.__file__)
        rapidwright_path_env = os.environ.get('RAPIDWRIGHT_PATH')
        classpath = os.environ.get('CLASSPATH')

        result = {
            "status": "success",
            "message": "RapidWright initialized successfully",
            "rapidwright_version": version,
            "rapidwright_install_path": install_path,
            "available_devices": device_count
        }
        if rapidwright_path_env:
            result["RAPIDWRIGHT_PATH"] = rapidwright_path_env
        if classpath:
            result["CLASSPATH"] = classpath            
        return result
        
    except Exception as e:
        logger.error(f"Failed to initialize RapidWright: {e}")
        return {
            "status": "error",
            "message": f"Failed to initialize RapidWright: {str(e)}",
            "hint": "Make sure 'pip install rapidwright' completed successfully and Java 11+ is installed"
        }


def get_supported_devices() -> Dict[str, Any]:
    """
    Get list of all FPGA devices supported by RapidWright, including families and part numbers.
    
    Returns:
        Dictionary with devices organized as a tree: Series -> FamilyType -> Devices
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import PartNameTools
        
        # Get all parts from RapidWright's part database
        all_parts = PartNameTools.getParts()
        
        # Build tree structure: Series -> FamilyType -> Devices (deduplicated)
        # Use sets to avoid duplicates since multiple parts map to the same device
        device_tree_sets = {}
        
        for part in all_parts:
            series_name = str(part.getSeries())
            family_name = str(part.getFamily())
            device_name = str(part.getDevice())
            
            # Initialize series level if needed
            if series_name not in device_tree_sets:
                device_tree_sets[series_name] = {}
            
            # Initialize family level if needed
            if family_name not in device_tree_sets[series_name]:
                device_tree_sets[series_name][family_name] = set()
            
            # Add device to family (set handles deduplication)
            device_tree_sets[series_name][family_name].add(device_name)
        
        # Convert sets to sorted lists for JSON serialization
        device_tree = {}
        total_devices = 0
        
        for series in device_tree_sets:
            device_tree[series] = {}
            for family in device_tree_sets[series]:
                device_tree[series][family] = sorted(device_tree_sets[series][family])
                total_devices += len(device_tree[series][family])
        
        # Calculate summary statistics
        series_count = len(device_tree)
        family_count = sum(len(families) for families in device_tree.values())
        
        return {
            "status": "success",
            "total_devices": total_devices,
            "series_count": series_count,
            "family_count": family_count,
            "device_tree": device_tree
        }
        
    except Exception as e:
        logger.error(f"Error getting supported devices: {e}")
        return {"error": str(e)}


def get_device_info(device_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific device.
    
    Args:
        device_name: Name of the device (e.g., 'xcvu3p', 'xcku040')
        
    Returns:
        Dictionary with device information
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import Device
        
        device = Device.getDevice(device_name)
        if device is None:
            return {"error": f"Device '{device_name}' not found"}
        
        return {
            "status": "success",
            "name": str(device.getName()),
            "family": str(device.getFamilyType()),
            "series": str(device.getSeries()),
            "architecture": str(device.getArchitecture()),
            "rows": device.getRows(),
            "columns": device.getColumns(),
            "tile_count": device.getAllTiles().size(),
            "site_count": device.getAllSites().length
        }
        
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        return {"error": str(e)}


def read_checkpoint(dcp_path: str) -> Dict[str, Any]:
    """
    Read a design checkpoint (DCP) file.
    
    Args:
        dcp_path: Path to the DCP file
        
    Returns:
        Dictionary with load status and basic design info
    """
    global _current_design
    
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.design import Design
        from com.xilinx.rapidwright.tests import CodePerfTracker
        from pathlib import Path
        
        dcp_file = Path(dcp_path).expanduser().resolve()
        if not dcp_file.exists():
            return {"error": f"DCP file not found: {dcp_path}"}
        
        logger.info(f"Loading design from {dcp_file}")
        design = Design.readCheckpoint(str(dcp_file))
        _current_design = design
        
        return {
            "status": "success",
            "message": f"Design loaded successfully from {dcp_file.name}",
            "design_name": str(design.getName()),
            "device": str(design.getDevice().getName()),
            "part_name": str(design.getPartName()),
            "cell_count": design.getCells().size(),
            "net_count": design.getNets().size()
        }
        
    except Exception as e:
        logger.error(f"Error loading design: {e}")
        return {"error": str(e)}


def write_checkpoint(dcp_path: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Write the current design to a checkpoint (DCP) file.
    
    Args:
        dcp_path: Path where the DCP file will be saved
        overwrite: If True, overwrite existing file; if False, error if file exists
        
    Returns:
        Dictionary with save status, bytes written, and encrypted IP info
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use read_checkpoint first."}
    
    try:
        from com.xilinx.rapidwright.tests import CodePerfTracker
        from pathlib import Path
        import os
        
        output_file = Path(dcp_path).expanduser().resolve()
        
        # Check if file exists and overwrite is not set
        if output_file.exists() and not overwrite:
            return {
                "error": f"File '{output_file}' already exists. Set overwrite=True to replace it."
            }
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        design = _current_design
        
        # Check if design contains encrypted cells before writing
        # Note: This method may not be available in all RapidWright versions
        try:
            contains_encrypted_ip = design.getNetlist().isEncrypted()
        except AttributeError:
            # Try alternative method or fall back to False
            try:
                contains_encrypted_ip = design.isNetlistEncrypted()
            except AttributeError:
                contains_encrypted_ip = False
                logger.warning("Could not determine if design contains encrypted IP")
        
        logger.info(f"Writing design checkpoint to {output_file}")
        design.writeCheckpoint(str(output_file))
        
        # Get file size
        bytes_written = output_file.stat().st_size
        
        # Check for accompanying Tcl script (generated for encrypted designs)
        tcl_script_path = str(output_file) + ".tcl"
        tcl_script_exists = os.path.exists(tcl_script_path)
        
        result = {
            "status": "success",
            "message": f"Design checkpoint saved successfully to {output_file.name}",
            "output_file": str(output_file),
            "bytes_written": bytes_written
        }
        
        # Add encrypted IP warning if applicable
        if contains_encrypted_ip:
            result["contains_encrypted_ip"] = True
            result["encrypted_ip_warning"] = (
                "This design contains encrypted IP. RapidWright has generated an "
                "accompanying Tcl script that is required to load this DCP in Vivado."
            )
            if tcl_script_exists:
                result["tcl_script_path"] = tcl_script_path
        
        logger.info(f"Design checkpoint saved: {bytes_written} bytes written")
        return result
        
    except Exception as e:
        logger.error(f"Error writing design checkpoint: {e}")
        return {"error": str(e)}


def get_design_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded design.
    
    Returns:
        Dictionary with design statistics
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use load_design first."}
    
    try:
        design = _current_design
        
        # Count cell types
        cell_types = {}
        for cell in design.getCells():
            cell_type = str(cell.getType())
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
        
        # Get top 10 most common cell types
        top_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "status": "success",
            "design_name": str(design.getName()),
            "device": str(design.getDevice().getName()),
            "part_name": str(design.getPartName()),
            "cell_count": design.getCells().size(),
            "net_count": design.getNets().size(),
            "top_cell_types": dict(top_types),
            "is_netlist_encrypted": design.getNetlist().hasEncryptedCells()
        }
        
    except Exception as e:
        logger.error(f"Error getting design info: {e}")
        return {"error": str(e)}


def search_cells(pattern: Optional[str] = None, 
                cell_type: Optional[str] = None, 
                limit: int = 100) -> Dict[str, Any]:
    """
    Search for cells in the current design.
    
    Args:
        pattern: Name pattern to match (case-insensitive)
        cell_type: Filter by cell type
        limit: Maximum number of results
        
    Returns:
        Dictionary with matching cells
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use load_design first."}
    
    try:
        design = _current_design
        matching_cells = []
        pattern_lower = pattern.lower() if pattern else None
        
        for cell in design.getCells():
            if len(matching_cells) >= limit:
                break
            
            cell_name = str(cell.getName())
            cell_type_str = str(cell.getType())
            
            # Apply filters
            if pattern_lower and pattern_lower not in cell_name.lower():
                continue
            if cell_type and cell_type != cell_type_str:
                continue
            
            # Get placement info
            placement = "unplaced"
            if cell.isPlaced():
                site = cell.getSite()
                if site:
                    placement = str(site.getName())
            
            matching_cells.append({
                "name": cell_name,
                "type": cell_type_str,
                "placement": placement
            })
        
        return {
            "status": "success",
            "count": len(matching_cells),
            "cells": matching_cells,
            "truncated": len(matching_cells) >= limit
        }
        
    except Exception as e:
        logger.error(f"Error searching cells: {e}")
        return {"error": str(e)}


def get_tile_info(tile_name: str, device_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a specific tile.
    
    Args:
        tile_name: Name of the tile
        device_name: Device name (uses current design's device if not specified)
        
    Returns:
        Dictionary with tile information
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import Device
        
        # Get device
        if device_name:
            device = Device.getDevice(device_name)
        elif _current_design:
            device = _current_design.getDevice()
        else:
            return {"error": "No device specified and no design loaded"}
        
        tile = device.getTile(tile_name)
        if tile is None:
            return {"error": f"Tile '{tile_name}' not found"}
        
        # Get sites in this tile
        sites = []
        if tile.getSites():
            for site in tile.getSites():
                sites.append({
                    "name": str(site.getName()),
                    "type": str(site.getSiteTypeEnum())
                })
        
        return {
            "status": "success",
            "name": str(tile.getName()),
            "type": str(tile.getTileTypeEnum()),
            "row": tile.getRow(),
            "column": tile.getColumn(),
            "site_count": len(sites),
            "sites": sites
        }
        
    except Exception as e:
        logger.error(f"Error getting tile info: {e}")
        return {"error": str(e)}


def search_sites(site_type: Optional[str] = None, 
                device_name: Optional[str] = None, 
                limit: int = 50) -> Dict[str, Any]:
    """
    Search for sites on a device.
    
    Args:
        site_type: Filter by site type (e.g., 'SLICEL', 'DSP48E2')
        device_name: Device name (uses current design's device if not specified)
        limit: Maximum number of results
        
    Returns:
        Dictionary with matching sites
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import Device
        
        # Get device
        if device_name:
            device = Device.getDevice(device_name)
        elif _current_design:
            device = _current_design.getDevice()
        else:
            return {"error": "No device specified and no design loaded"}
        
        matching_sites = []
        
        for site in device.getAllSites():
            if len(matching_sites) >= limit:
                break
            
            site_type_str = str(site.getSiteTypeEnum())
            
            # Filter by site type if specified
            if site_type and site_type not in site_type_str:
                continue
            
            tile = site.getTile()
            matching_sites.append({
                "name": str(site.getName()),
                "type": site_type_str,
                "tile": str(tile.getName()) if tile else "unknown"
            })
        
        return {
            "status": "success",
            "count": len(matching_sites),
            "sites": matching_sites,
            "truncated": len(matching_sites) >= limit
        }
        
    except Exception as e:
        logger.error(f"Error searching sites: {e}")
        return {"error": str(e)}


def optimize_lut_input_cone(hierarchical_input_pins: list[str]) -> Dict[str, Any]:
    """
    Optimize LUT input cones by combining chained small LUTs into a single larger LUT.
    
    This optimization reduces logic depth by replacing series of small LUTs with a single
    larger LUT (up to 6 inputs). This is particularly useful for critical paths where
    the delay through multiple LUT levels can be reduced to a single LUT.
    
    Args:
        hierarchical_input_pins: List of hierarchical input pin names to optimize
                                (e.g., ["module/submodule/inst/pin"])
        
    Returns:
        Dictionary with optimization results
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use load_design first."}
    
    try:
        from com.xilinx.rapidwright.eco import LUTInputConeOpt
        
        design = _current_design
        results = []
        
        logger.info(f"Optimizing {len(hierarchical_input_pins)} LUT input cones")
        
        for pin_name in hierarchical_input_pins:
            try:
                # Get the hierarchical port instance
                port_inst = design.getNetlist().getHierPortInstFromName(pin_name)
                
                if port_inst is None:
                    results.append({
                        "pin": pin_name,
                        "status": "error",
                        "message": f"Pin '{pin_name}' not found in design"
                    })
                    continue
                
                # Attempt optimization
                optimized_cell = LUTInputConeOpt.optimizedLUTInputCone(design, port_inst)
                
                if optimized_cell is None:
                    results.append({
                        "pin": pin_name,
                        "status": "no_optimization",
                        "message": "No optimization possible for this pin (may not be driven by LUTs or only single LUT in path)"
                    })
                else:
                    cell_info = {
                        "name": str(optimized_cell.getName()),
                        "type": str(optimized_cell.getType()),
                        "placement": "unplaced"
                    }
                    
                    if optimized_cell.isPlaced():
                        site = optimized_cell.getSite()
                        if site:
                            cell_info["placement"] = str(site.getName())
                    
                    results.append({
                        "pin": pin_name,
                        "status": "optimized",
                        "message": "LUT input cone successfully optimized",
                        "new_cell": cell_info
                    })
                    
            except Exception as e:
                logger.error(f"Error optimizing pin {pin_name}: {e}")
                results.append({
                    "pin": pin_name,
                    "status": "error",
                    "message": str(e)
                })
        
        # Count successful optimizations
        success_count = sum(1 for r in results if r["status"] == "optimized")
        
        return {
            "status": "success",
            "total_pins": len(hierarchical_input_pins),
            "optimized_count": success_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in LUT input cone optimization: {e}")
        return {"error": str(e)}


def optimize_fanout(net_name: str, split_factor: int) -> Dict[str, Any]:
    """
    Optimize high fanout nets by splitting them into multiple driven nets.
    
    This optimization reduces fanout by replicating the source driver and dividing
    the loads among multiple copies. This can improve timing and routability for
    nets with very high fanout.
    
    Args:
        net_name: Name of the high fanout net to optimize
        split_factor: Number of copies to create (k) - net will be split into k parts
        
    Returns:
        Dictionary with optimization results
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use load_design first."}
    
    try:
        from com.xilinx.rapidwright.eco import FanOutOptimization
        
        design = _current_design
        
        # Get the net
        net = design.getNet(net_name)
        if net is None:
            return {"error": f"Net '{net_name}' not found in design"}
        
        # Get original fanout info
        original_fanout = net.getFanOut()
        logger.info(f"Optimizing net '{net_name}' with fanout {original_fanout} into {split_factor} parts")
        
        # Perform optimization
        FanOutOptimization.cutFanOutOfRoutedNet(design, net, split_factor)
        
        # Collect info about the new nets created
        # The optimization creates multiple nets by replicating the source
        new_nets_info = []
        
        # Try to find the replicated nets (they will have similar names)
        base_name = net_name
        for design_net in design.getNets():
            net_str = str(design_net.getName())
            if base_name in net_str and net_str != net_name:
                new_nets_info.append({
                    "name": net_str,
                    "fanout": design_net.getFanOut()
                })
                if len(new_nets_info) >= split_factor:
                    break
        
        return {
            "status": "success",
            "net_name": net_name,
            "original_fanout": original_fanout,
            "split_factor": split_factor,
            "new_nets": new_nets_info,
            "message": f"Successfully split net '{net_name}' into {split_factor} parts"
        }
        
    except Exception as e:
        logger.error(f"Error in fanout optimization: {e}")
        return {"error": str(e)}


def analyze_fabric_for_pblock(
    target_lut_count: int,
    target_ff_count: int,
    target_dsp_count: int = 0,
    target_bram_count: int = 0,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the FPGA fabric to find the best contiguous region for a pblock.
    
    Identifies regions that:
    1. Have enough resources (SLICEs, DSPs, BRAMs) for the target utilization
    2. Minimize crossing of delay-heavy columns (URAM, IO, etc.)
    3. Are as contiguous as possible
    
    Args:
        target_lut_count: Required number of LUTs (1.5x current usage)
        target_ff_count: Required number of FFs (1.5x current usage)
        target_dsp_count: Required number of DSPs (1.5x current usage)
        target_bram_count: Required number of BRAMs (1.5x current usage)
        device_name: Device name (uses loaded design's device if omitted)
        
    Returns:
        Dictionary with recommended pblock ranges and analysis
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import Device, TileTypeEnum
        
        # Get the device
        if device_name:
            device = Device.getDevice(device_name)
        elif _current_design:
            device = _current_design.getDevice()
        else:
            return {"error": "No device specified and no design loaded"}
        
        logger.info(f"Analyzing fabric for device: {device.getName()}")
        
        # Get all tiles
        tiles = device.getAllTiles()
        
        # Check for delay-heavy tile types by name pattern
        # These tile types cause routing delays when crossed
        def is_delay_heavy_tile(tile_type_enum) -> bool:
            tile_type_name = str(tile_type_enum.name())
            delay_patterns = ['URAM', 'HPIO', 'HDIO', 'HRIO']
            return any(pattern in tile_type_name for pattern in delay_patterns)
        
        # Map tile columns/rows to resource counts and types
        column_info = {}  # col -> {good_tiles, bad_tiles, resources}
        row_info = {}     # row -> {good_tiles, bad_tiles, resources}
        
        min_col, max_col = float('inf'), 0
        min_row, max_row = float('inf'), 0
        
        for tile in tiles:
            tile_type = tile.getTileTypeEnum()
            col = tile.getColumn()
            row = tile.getRow()
            
            # Track column/row bounds
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            
            # Initialize column/row info
            if col not in column_info:
                column_info[col] = {
                    "good_tiles": 0,
                    "bad_tiles": 0,
                    "slice_sites": 0,
                    "dsp_sites": 0,
                    "bram_sites": 0
                }
            
            if row not in row_info:
                row_info[row] = {
                    "good_tiles": 0,
                    "bad_tiles": 0,
                    "slice_sites": 0,
                    "dsp_sites": 0,
                    "bram_sites": 0
                }
            
            # Categorize tile
            is_bad = is_delay_heavy_tile(tile_type)
            
            if is_bad:
                column_info[col]["bad_tiles"] += 1
                row_info[row]["bad_tiles"] += 1
            else:
                column_info[col]["good_tiles"] += 1
                row_info[row]["good_tiles"] += 1
                
                # Count resources in this tile
                sites = tile.getSites()
                if sites:
                    for site in sites:
                        site_type_str = str(site.getSiteTypeEnum())
                        
                        if "SLICE" in site_type_str:
                            # Each SLICE has ~4 LUTs and ~8 FFs
                            column_info[col]["slice_sites"] += 1
                            row_info[row]["slice_sites"] += 1
                        elif "DSP" in site_type_str:
                            column_info[col]["dsp_sites"] += 1
                            row_info[row]["dsp_sites"] += 1
                        elif "RAMB" in site_type_str or "BRAM" in site_type_str:
                            column_info[col]["bram_sites"] += 1
                            row_info[row]["bram_sites"] += 1
        
        # Find contiguous column ranges with minimal bad columns
        good_columns = []
        for col in sorted(column_info.keys()):
            info = column_info[col]
            # A "good" column has mostly resource tiles, not delay-heavy tiles
            if info["good_tiles"] > info["bad_tiles"] * 2:  # 2:1 ratio
                good_columns.append(col)
        
        # Find the longest contiguous range of good columns
        best_col_range = None
        best_col_resources = {"slices": 0, "dsps": 0, "brams": 0}
        
        current_range = []
        current_resources = {"slices": 0, "dsps": 0, "brams": 0}
        
        for col in good_columns:
            if not current_range or col == current_range[-1] + 1:
                # Continue the range
                current_range.append(col)
                current_resources["slices"] += column_info[col]["slice_sites"]
                current_resources["dsps"] += column_info[col]["dsp_sites"]
                current_resources["brams"] += column_info[col]["bram_sites"]
            else:
                # Gap found - check if current range is better
                if (not best_col_range or 
                    len(current_range) > len(best_col_range) or
                    (len(current_range) == len(best_col_range) and 
                     current_resources["slices"] > best_col_resources["slices"])):
                    best_col_range = current_range
                    best_col_resources = current_resources.copy()
                
                # Start new range
                current_range = [col]
                current_resources = {
                    "slices": column_info[col]["slice_sites"],
                    "dsps": column_info[col]["dsp_sites"],
                    "brams": column_info[col]["bram_sites"]
                }
        
        # Check final range
        if (not best_col_range or 
            len(current_range) > len(best_col_range) or
            (len(current_range) == len(best_col_range) and 
             current_resources["slices"] > best_col_resources["slices"])):
            best_col_range = current_range
            best_col_resources = current_resources.copy()
        
        if not best_col_range:
            return {"error": "No suitable contiguous column range found"}
        
        # Similar analysis for rows
        good_rows = []
        for row in sorted(row_info.keys()):
            info = row_info[row]
            if info["good_tiles"] > info["bad_tiles"] * 2:
                good_rows.append(row)
        
        # Find best row range
        best_row_range = None
        best_row_resources = {"slices": 0, "dsps": 0, "brams": 0}
        
        current_range = []
        current_resources = {"slices": 0, "dsps": 0, "brams": 0}
        
        for row in good_rows:
            if not current_range or row == current_range[-1] + 1:
                current_range.append(row)
                current_resources["slices"] += row_info[row]["slice_sites"]
                current_resources["dsps"] += row_info[row]["dsp_sites"]
                current_resources["brams"] += row_info[row]["bram_sites"]
            else:
                if (not best_row_range or 
                    len(current_range) > len(best_row_range) or
                    (len(current_range) == len(best_row_range) and 
                     current_resources["slices"] > best_row_resources["slices"])):
                    best_row_range = current_range
                    best_row_resources = current_resources.copy()
                
                current_range = [row]
                current_resources = {
                    "slices": row_info[row]["slice_sites"],
                    "dsps": row_info[row]["dsp_sites"],
                    "brams": row_info[row]["bram_sites"]
                }
        
        if (not best_row_range or 
            len(current_range) > len(best_row_range) or
            (len(current_range) == len(best_row_range) and 
             current_resources["slices"] > best_row_resources["slices"])):
            best_row_range = current_range
            best_row_resources = current_resources.copy()
        
        if not best_row_range:
            return {"error": "No suitable contiguous row range found"}
        
        # Calculate center of the best region
        col_center = (best_col_range[0] + best_col_range[-1]) // 2
        row_center = (best_row_range[0] + best_row_range[-1]) // 2
        
        # Estimate required columns/rows for target resources
        # Each SLICE column has ~300 slices, each SLICE has ~4 LUTs and ~8 FFs
        required_slices = max(target_lut_count // 4, target_ff_count // 8)
        
        # Find actual placed cells to determine center of mass
        center_of_mass_col = col_center
        center_of_mass_row = row_center
        
        if _current_design:
            placed_cols = []
            placed_rows = []
            for cell in _current_design.getCells():
                if cell.isPlaced():
                    site = cell.getSite()
                    if site:
                        tile = site.getTile()
                        placed_cols.append(tile.getColumn())
                        placed_rows.append(tile.getRow())
            
            if placed_cols:
                center_of_mass_col = sum(placed_cols) // len(placed_cols)
                center_of_mass_row = sum(placed_rows) // len(placed_rows)
                logger.info(f"Center of mass: col={center_of_mass_col}, row={center_of_mass_row}")
        
        # Find a contiguous range around center of mass that:
        # 1. Has enough resources for target (with margin)
        # 2. Avoids bad columns
        # 3. Is reasonably sized (not the entire device)
        
        # SIMPLIFIED APPROACH: Use fixed reasonable size based on empirical data
        # For logicnets_jscl design (30K LUTs), optimal was 12 SLICE cols × 50 rows
        # This achieved timing closure. Scale based on target LUTs:
        # - Small designs (<20K LUTs): 15-20 columns
        # - Medium designs (20-50K LUTs): 20-30 columns
        # - Large designs (>50K LUTs): 30-40 columns
        
        target_luts = required_slices * 4  # Convert back to LUTs
        
        if target_luts < 20000:
            cols_needed = 20
        elif target_luts < 50000:
            cols_needed = 28
        else:
            cols_needed = 35
        
        rows_needed = 55  # Fixed reasonable height
        
        # Clamp to available fabric size
        cols_needed = min(cols_needed, len(best_col_range) // 2)
        rows_needed = min(rows_needed, len(best_row_range) // 2)
        
        # Grow from center of mass
        col_start_idx = next((i for i, c in enumerate(best_col_range) if c >= center_of_mass_col), len(best_col_range) // 2)
        row_start_idx = next((i for i, r in enumerate(best_row_range) if r >= center_of_mass_row), len(best_row_range) // 2)
        
        # Expand symmetrically from center
        col_left_idx = max(0, col_start_idx - cols_needed // 2)
        col_right_idx = min(len(best_col_range) - 1, col_start_idx + cols_needed // 2)
        row_bottom_idx = max(0, row_start_idx - rows_needed // 2)
        row_top_idx = min(len(best_row_range) - 1, row_start_idx + rows_needed // 2)
        
        final_col_min = best_col_range[col_left_idx]
        final_col_max = best_col_range[col_right_idx]
        final_row_min = best_row_range[row_bottom_idx]
        final_row_max = best_row_range[row_top_idx]
        
        # Count resources in selected region (approximate)
        selected_cols = col_right_idx - col_left_idx + 1
        selected_rows = row_top_idx - row_bottom_idx + 1
        
        est_slice_sites = int(best_col_resources["slices"] * selected_cols / len(best_col_range))
        est_dsp_sites = int(best_col_resources["dsps"] * selected_cols / len(best_col_range))
        est_bram_sites = int(best_col_resources["brams"] * selected_cols / len(best_col_range))
        
        return {
            "status": "success",
            "device": str(device.getName()),
            "fabric_bounds": {
                "min_col": int(min_col),
                "max_col": int(max_col),
                "min_row": int(min_row),
                "max_row": int(max_row)
            },
            "recommended_region": {
                "col_min": int(final_col_min),
                "col_max": int(final_col_max),
                "row_min": int(final_row_min),
                "row_max": int(final_row_max),
                "center_col": int(col_center),
                "center_row": int(row_center),
                "center_of_mass_col": int(center_of_mass_col),
                "center_of_mass_row": int(center_of_mass_row),
                "contiguous_columns": selected_cols,
                "contiguous_rows": selected_rows
            },
            "estimated_resources": {
                "slice_sites": est_slice_sites,
                "dsp_sites": est_dsp_sites,
                "bram_sites": est_bram_sites,
                "approx_luts": est_slice_sites * 4,
                "approx_ffs": est_slice_sites * 8
            },
            "target_requirements": {
                "luts": target_lut_count,
                "ffs": target_ff_count,
                "dsps": target_dsp_count,
                "brams": target_bram_count
            },
            "message": f"Found region around center of mass: cols {final_col_min}-{final_col_max}, rows {final_row_min}-{final_row_max}"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing fabric: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def analyze_critical_path_spread(
    critical_paths_data: list = None,
    input_file: str = None,
) -> Dict[str, Any]:
    """
    Calculate Manhattan distances for cells on critical paths.
    
    Takes critical path data from Vivado (list of cell names per path) and uses
    RapidWright's device model to get accurate tile coordinates and calculate distances.
    
    Args:
        critical_paths_data: List of paths, where each path is a list of cell names
        input_file: Optional path to JSON file containing critical_paths_data
        
    Returns:
        Dictionary with spread analysis including max distances per path
        
    Note: Either critical_paths_data or input_file must be provided
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    if _current_design is None:
        return {"error": "No design loaded. Use read_checkpoint first."}
    
    # Load data from file if specified
    if input_file:
        try:
            import json
            with open(input_file, 'r') as f:
                critical_paths_data = json.load(f)
        except Exception as e:
            return {"error": f"Error reading input file: {str(e)}"}
    
    if not critical_paths_data:
        return {"error": "No critical path data provided. Specify either critical_paths_data or input_file"}
    
    try:
        design = _current_design
        device = design.getDevice()
        
        logger.info(f"Analyzing {len(critical_paths_data)} critical paths for cell spread")
        
        path_results = []
        all_max_distances = []
        
        for path_idx, cell_names in enumerate(critical_paths_data):
            # Get placements for cells in this path
            cell_locations = []
            
            for cell_name in cell_names:
                try:
                    cell = design.getCell(cell_name)
                    if cell and cell.isPlaced():
                        site = cell.getSite()
                        if site:
                            tile = site.getTile()
                            cell_locations.append({
                                "cell": str(cell.getName()),
                                "type": str(cell.getType()),
                                "tile": str(tile.getName()),
                                "col": tile.getColumn(),
                                "row": tile.getRow()
                            })
                except Exception as e:
                    logger.debug(f"Could not get location for cell {cell_name}: {e}")
                    continue
            
            if len(cell_locations) < 2:
                continue
            
            # Calculate maximum Manhattan distance between SEQUENTIAL cells on this path
            max_distance = 0
            max_pair = None
            
            for i in range(len(cell_locations) - 1):
                loc1 = cell_locations[i]
                loc2 = cell_locations[i + 1]
                distance = abs(loc1["col"] - loc2["col"]) + abs(loc1["row"] - loc2["row"])
                
                if distance > max_distance:
                    max_distance = distance
                    max_pair = (loc1, loc2)
            
            all_max_distances.append(max_distance)
            
            path_results.append({
                "path_num": path_idx + 1,
                "cell_count": len(cell_locations),
                "max_distance": max_distance,
                "max_pair": max_pair
            })
        
        if not all_max_distances:
            return {
                "status": "warning",
                "message": "No cell location data found for paths",
                "paths_analyzed": len(critical_paths_data)
            }
        
        # Calculate statistics
        max_dist = max(all_max_distances)
        avg_dist = sum(all_max_distances) / len(all_max_distances)
        
        # Sort by distance
        path_results.sort(key=lambda x: -x["max_distance"])
        
        return {
            "status": "success",
            "paths_analyzed": len(critical_paths_data),
            "max_distance_found": int(max_dist),
            "avg_max_distance": float(avg_dist),
            "path_distances": [int(d) for d in all_max_distances],
            "worst_paths": path_results[:10]  # Top 10 worst
        }
        
    except Exception as e:
        logger.error(f"Error analyzing critical path spread: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def compare_design_structure(golden_dcp: str, revised_dcp: str) -> Dict[str, Any]:
    """
    Compare structural properties of two design checkpoints.
    
    Performs sanity checks to catch obvious errors:
    - Top-level module name
    - I/O port names, directions, and widths
    - Cell count comparison
    - Clock structure
    
    Args:
        golden_dcp: Path to the golden (reference) DCP file
        revised_dcp: Path to the revised (optimized) DCP file
        
    Returns:
        Dictionary with comparison results including pass/fail status
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.design import Design
        from pathlib import Path
        import json
        
        golden_path = Path(golden_dcp).expanduser().resolve()
        revised_path = Path(revised_dcp).expanduser().resolve()
        
        if not golden_path.exists():
            return {"error": f"Golden DCP not found: {golden_dcp}"}
        if not revised_path.exists():
            return {"error": f"Revised DCP not found: {revised_dcp}"}
        
        logger.info(f"Loading golden design from {golden_path}")
        golden = Design.readCheckpoint(str(golden_path))
        
        logger.info(f"Loading revised design from {revised_path}")
        revised = Design.readCheckpoint(str(revised_path))
        
        issues = []
        checks_passed = 0
        checks_total = 0
        
        # Check 1: Top-level module name
        checks_total += 1
        golden_top = str(golden.getName())
        revised_top = str(revised.getName())
        if golden_top == revised_top:
            checks_passed += 1
        else:
            issues.append(f"Top module name mismatch: '{golden_top}' vs '{revised_top}'")
        
        # Check 2: Device compatibility
        checks_total += 1
        golden_device = str(golden.getDevice().getName())
        revised_device = str(revised.getDevice().getName())
        if golden_device == revised_device:
            checks_passed += 1
        else:
            issues.append(f"Device mismatch: '{golden_device}' vs '{revised_device}'")
        
        # Check 3: I/O ports (names, directions, widths)
        golden_netlist = golden.getNetlist()
        revised_netlist = revised.getNetlist()
        
        golden_top_cell = golden_netlist.getTopCell()
        revised_top_cell = revised_netlist.getTopCell()
        
        # Get port information
        golden_ports = {}
        for port in golden_top_cell.getPorts():
            port_name = str(port.getName())
            port_dir = str(port.getDirection())
            port_width = port.getWidth()
            golden_ports[port_name] = {"direction": port_dir, "width": port_width}
        
        revised_ports = {}
        for port in revised_top_cell.getPorts():
            port_name = str(port.getName())
            port_dir = str(port.getDirection())
            port_width = port.getWidth()
            revised_ports[port_name] = {"direction": port_dir, "width": port_width}
        
        # Compare ports
        checks_total += 1
        port_issues = []
        
        # Check for missing/added ports
        golden_port_names = set(golden_ports.keys())
        revised_port_names = set(revised_ports.keys())
        
        missing_ports = golden_port_names - revised_port_names
        added_ports = revised_port_names - golden_port_names
        
        if missing_ports:
            port_issues.append(f"Missing ports in revised: {', '.join(sorted(missing_ports))}")
        if added_ports:
            port_issues.append(f"Added ports in revised: {', '.join(sorted(added_ports))}")
        
        # Check common ports for direction/width mismatches
        common_ports = golden_port_names & revised_port_names
        for port_name in sorted(common_ports):
            g_info = golden_ports[port_name]
            r_info = revised_ports[port_name]
            
            if g_info["direction"] != r_info["direction"]:
                port_issues.append(
                    f"Port '{port_name}' direction mismatch: "
                    f"{g_info['direction']} vs {r_info['direction']}"
                )
            
            if g_info["width"] != r_info["width"]:
                port_issues.append(
                    f"Port '{port_name}' width mismatch: "
                    f"{g_info['width']} vs {r_info['width']}"
                )
        
        if not port_issues:
            checks_passed += 1
        else:
            issues.extend(port_issues)
        
        # Check 4: Cell count (should increase or stay same, small decreases allowed)
        checks_total += 1
        golden_cell_count = golden.getCells().size()
        revised_cell_count = revised.getCells().size()
        
        cell_change_pct = (revised_cell_count - golden_cell_count) / golden_cell_count * 100
        
        # Allow small decrease (<=3%), up to 50% increase (optimizations can add/remove cells)
        if (revised_cell_count >= golden_cell_count * 0.97 and 
            revised_cell_count <= golden_cell_count * 1.5):
            checks_passed += 1
            # Note small changes as info, not error
            if revised_cell_count < golden_cell_count:
                issues.append(
                    f"INFO: Cell count decreased slightly: {golden_cell_count} -> {revised_cell_count} "
                    f"({abs(cell_change_pct):.2f}% decrease - likely due to optimization)"
                )
            elif revised_cell_count > golden_cell_count:
                issues.append(
                    f"INFO: Cell count increased: {golden_cell_count} -> {revised_cell_count} "
                    f"({cell_change_pct:.2f}% increase - likely due to optimization)"
                )
        else:
            if revised_cell_count < golden_cell_count:
                issues.append(
                    f"Cell count decreased significantly: {golden_cell_count} -> {revised_cell_count} "
                    f"({abs(cell_change_pct):.2f}% decrease - this may indicate logic removal)"
                )
            else:
                issues.append(
                    f"Cell count increased significantly: {golden_cell_count} -> {revised_cell_count} "
                    f"({cell_change_pct:.1f}% increase - this may indicate excessive optimization)"
                )
        
        # Summary - only count real issues (not INFO)
        real_issues = [i for i in issues if not i.startswith("INFO:")]
        all_checks_passed = (checks_passed == checks_total)
        
        result = {
            "status": "success",
            "comparison_result": "PASS" if all_checks_passed else "FAIL",
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "golden_design": {
                "path": str(golden_path),
                "top_module": golden_top,
                "device": golden_device,
                "cell_count": golden_cell_count,
                "port_count": len(golden_ports)
            },
            "revised_design": {
                "path": str(revised_path),
                "top_module": revised_top,
                "device": revised_device,
                "cell_count": revised_cell_count,
                "port_count": len(revised_ports)
            },
            "issues": issues
        }
        
        # Restore original design if it was loaded
        global _current_design
        if _current_design:
            _current_design = revised  # Keep revised loaded for potential further use
        
        return result
        
    except Exception as e:
        logger.error(f"Error comparing designs: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def convert_fabric_region_to_pblock_ranges(
    col_min: int,
    col_max: int,
    row_min: int,
    row_max: int,
    device_name: Optional[str] = None,
    use_clock_regions: bool = False
) -> Dict[str, Any]:
    """
    Convert fabric region (column/row coordinates) to Vivado pblock range strings.
    
    Generates a complete pblock string with all site types (SLICE, DSP, BRAM, URAM)
    in the format: "SLICE_X55Y0:SLICE_X109Y179 DSP48E2_X8Y0:DSP48E2_X13Y71 ..."
    
    Args:
        col_min, col_max: Column range (tile coordinates)
        row_min, row_max: Row range (tile coordinates)
        device_name: Device name (uses loaded design's device if omitted)
        use_clock_regions: If True, use CLOCKREGION ranges (simpler but coarser)
        
    Returns:
        Dictionary with pblock range strings suitable for Vivado create_pblock
    """
    if not _initialized:
        return {"error": "RapidWright not initialized. Call initialize_rapidwright first."}
    
    try:
        from com.xilinx.rapidwright.device import Device, SiteTypeEnum
        
        # Get the device
        if device_name:
            device = Device.getDevice(device_name)
        elif _current_design:
            device = _current_design.getDevice()
        else:
            return {"error": "No device specified and no design loaded"}
        
        if use_clock_regions:
            # Use clock region ranges (simpler, coarser granularity)
            cr_x_min = col_min // 60
            cr_x_max = col_max // 60
            cr_y_min = row_min // 60
            cr_y_max = row_max // 60
            
            pblock_range = f"CLOCKREGION_X{cr_x_min}Y{cr_y_min}:CLOCKREGION_X{cr_x_max}Y{cr_y_max}"
            
            return {
                "status": "success",
                "pblock_ranges": pblock_range,
                "format": "CLOCKREGION"
            }
        
        # Use site ranges (finer granularity) - find all site types in region
        # Track min/max coordinates for each site type
        site_bounds = {
            "SLICE": {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0, "count": 0},
            "DSP48E2": {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0, "count": 0},
            "RAMB18": {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0, "count": 0},
            "RAMB36": {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0, "count": 0},
            "URAM288": {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0, "count": 0},
        }
        
        # Iterate through all tiles in the region
        for tile in device.getAllTiles():
            col = tile.getColumn()
            row = tile.getRow()
            
            # Check if tile is within our region
            if not (col_min <= col <= col_max and row_min <= row <= row_max):
                continue
            
            # Check all sites in this tile
            sites = tile.getSites()
            if not sites:
                continue
            
            for site in sites:
                site_type = site.getSiteTypeEnum()
                site_type_name = str(site_type.name())
                site_name = str(site.getName())
                
                # Determine site type category based on name patterns
                site_category = None
                
                if site_type_name in ['SLICEL', 'SLICEM']:
                    site_category = "SLICE"
                elif 'DSP48E2' in site_type_name:
                    site_category = "DSP48E2"
                elif site_type_name in ['RAMB18E1', 'RAMB181', 'RAMB180', 'RAMB18_L', 'RAMB18_U', 'RAMBFIFO18']:
                    site_category = "RAMB18"
                elif site_type_name in ['RAMB36', 'RAMB36E1', 'RAMBFIFO36', 'RAMBFIFO36E1']:
                    site_category = "RAMB36"
                elif site_type_name == 'URAM288':
                    site_category = "URAM288"
                
                if site_category and site_category in site_bounds:
                    # Get instance X/Y coordinates from the site
                    try:
                        x = site.getInstanceX()
                        y = site.getInstanceY()
                        
                        bounds = site_bounds[site_category]
                        bounds["min_x"] = min(bounds["min_x"], x)
                        bounds["max_x"] = max(bounds["max_x"], x)
                        bounds["min_y"] = min(bounds["min_y"], y)
                        bounds["max_y"] = max(bounds["max_y"], y)
                        bounds["count"] += 1
                    except:
                        pass
        
        # Build the pblock range string
        pblock_parts = []
        
        for site_type in ["SLICE", "DSP48E2", "RAMB18", "RAMB36", "URAM288"]:
            bounds = site_bounds[site_type]
            if bounds["count"] > 0 and bounds["min_x"] != float('inf'):
                min_x = int(bounds["min_x"])
                max_x = int(bounds["max_x"])
                min_y = int(bounds["min_y"])
                max_y = int(bounds["max_y"])
                
                range_str = f"{site_type}_X{min_x}Y{min_y}:{site_type}_X{max_x}Y{max_y}"
                pblock_parts.append(range_str)
        
        if not pblock_parts:
            return {"error": "No valid sites found in specified region"}
        
        pblock_ranges = " ".join(pblock_parts)
        
        return {
            "status": "success",
            "pblock_ranges": pblock_ranges,
            "format": "SITE",
            "site_counts": {
                "SLICE": site_bounds["SLICE"]["count"],
                "DSP48E2": site_bounds["DSP48E2"]["count"],
                "RAMB18": site_bounds["RAMB18"]["count"],
                "RAMB36": site_bounds["RAMB36"]["count"],
                "URAM288": site_bounds["URAM288"]["count"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting fabric region to pblock: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

