#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
RapidWright MCP Server
Provides AI assistant access to RapidWright FPGA design tools via the Model Context Protocol
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent, GetPromptResult, PromptMessage
import mcp.server.stdio

import rapidwright_tools as rw

# Global variable for the Java/stdout log file
_java_log_file = None
_original_stderr_fd = None

# Logger will be configured in main() based on command-line arguments
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create MCP server instance
app = Server("rapidwright-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available RapidWright tools."""
    return [
        Tool(
            name="initialize_rapidwright",
            description="Initialize the RapidWright environment. Must be called first before using other tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jvm_max_memory": {
                        "type": "string",
                        "description": "Maximum JVM heap size (default: '4G')",
                        "default": "4G"
                    }
                }
            }
        ),
        Tool(
            name="get_supported_devices",
            description="Get list of all FPGA devices supported by RapidWright, including families and part numbers.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_device_info",
            description="Get detailed information about a specific FPGA device (dimensions, resources, family).",
            inputSchema={
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "FPGA device name (e.g., 'xcvu3p', 'xcvu9p', 'xcku040')"
                    }
                },
                "required": ["device_name"]
            }
        ),
        Tool(
            name="read_checkpoint",
            description="Read a Vivado Design Checkpoint (.dcp) file for inspection and analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dcp_path": {
                        "type": "string",
                        "description": "Path to the .dcp file"
                    }
                },
                "required": ["dcp_path"]
            }
        ),
        Tool(
            name="write_checkpoint",
            description="Write the current design to a Vivado Design Checkpoint (.dcp) file. If the design contains encrypted IP, an accompanying Tcl script will be generated that is required to load the DCP in Vivado.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dcp_path": {
                        "type": "string",
                        "description": "Path where the .dcp file will be saved"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true, overwrite existing file; if false (default), error if file exists",
                        "default": False
                    }
                },
                "required": ["dcp_path"]
            }
        ),
        Tool(
            name="get_design_info",
            description="Get statistics about the currently loaded design (cell/net counts, top cell types).",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="search_cells",
            description="Search for cells in the loaded design by name pattern or cell type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Name pattern to match (case-insensitive, optional)"
                    },
                    "cell_type": {
                        "type": "string",
                        "description": "Cell type to filter by (e.g., 'LUT6', 'FDRE', optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 100)",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="get_tile_info",
            description="Get information about a specific tile on the FPGA (type, location, sites).",
            inputSchema={
                "type": "object",
                "properties": {
                    "tile_name": {
                        "type": "string",
                        "description": "Tile name to query"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Device name (optional, uses loaded design's device if omitted)"
                    }
                },
                "required": ["tile_name"]
            }
        ),
        Tool(
            name="search_sites",
            description="Search for sites on an FPGA device by site type (e.g., SLICEL, DSP48E2, RAMB36).",
            inputSchema={
                "type": "object",
                "properties": {
                    "site_type": {
                        "type": "string",
                        "description": "Site type to search for (e.g., 'SLICEL', 'DSP48E2', 'RAMB36')"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Device name (optional, uses loaded design's device if omitted)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 50)",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="optimize_lut_input_cone",
            description="Optimize LUT input cones by combining chained small LUTs into a single larger LUT to reduce logic depth. This is useful for optimizing critical paths.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hierarchical_input_pins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of hierarchical input pin names to optimize (e.g., ['module/submodule/inst/pin'])"
                    }
                },
                "required": ["hierarchical_input_pins"]
            }
        ),
        Tool(
            name="optimize_fanout",
            description="Optimize high fanout nets by splitting them into multiple driven nets. This reduces fanout by replicating the source driver and can improve timing and routability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "Name of the high fanout net to optimize"
                    },
                    "split_factor": {
                        "type": "integer",
                        "description": "Number of copies to create (k) - net will be split into k parts"
                    }
                },
                "required": ["net_name", "split_factor"]
            }
        ),
        Tool(
            name="analyze_critical_path_spread",
            description="""Calculate Manhattan distances for cells on critical paths.
            
            Takes critical path data from Vivado (cell names from timing report) and uses RapidWright's
            device model to get accurate tile coordinates and calculate Manhattan distances between cells.
            
            Input can be provided either directly as critical_paths_data parameter OR via a JSON file
            specified in input_file parameter (more efficient for large datasets).
            
            Must be called AFTER read_checkpoint.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "critical_paths_data": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "List of paths, each path is a list of cell names from Vivado timing report"
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Optional: path to JSON file containing critical_paths_data (more efficient)"
                    }
                }
            }
        ),
        Tool(
            name="analyze_fabric_for_pblock",
            description="""Analyze FPGA fabric to find the best contiguous region for a pblock (area constraint).
            
            Identifies regions that:
            1. Have enough resources (SLICEs, DSPs, BRAMs) for target utilization
            2. Minimize crossing of delay-heavy columns (URAM, IO, etc.)
            3. Are as contiguous as possible
            
            Use this AFTER getting utilization from Vivado to determine where to place a pblock.
            Requires target resource counts (1.5x current usage from report_utilization_for_pblock).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_lut_count": {
                        "type": "integer",
                        "description": "Required LUTs (1.5x current usage)"
                    },
                    "target_ff_count": {
                        "type": "integer",
                        "description": "Required FFs (1.5x current usage)"
                    },
                    "target_dsp_count": {
                        "type": "integer",
                        "description": "Required DSPs (1.5x current usage, default: 0)"
                    },
                    "target_bram_count": {
                        "type": "integer",
                        "description": "Required BRAMs (1.5x current usage, default: 0)"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Device name (optional, uses loaded design's device if omitted)"
                    }
                },
                "required": ["target_lut_count", "target_ff_count"]
            }
        ),
        Tool(
            name="convert_fabric_region_to_pblock",
            description="""Convert fabric region coordinates to Vivado pblock range strings.
            
            Takes tile column/row coordinates and generates a complete pblock string with all
            site types (SLICE, DSP48E2, RAMB18, RAMB36, URAM288) in proper Vivado format.
            
            Example output: "SLICE_X55Y0:SLICE_X109Y179 DSP48E2_X8Y0:DSP48E2_X13Y71 RAMB18_X4Y0:RAMB18_X7Y71 RAMB36_X4Y0:RAMB36_X7Y35 URAM288_X1Y0:URAM288_X2Y47"
            
            IMPORTANT: Always use detailed site-specific ranges (default) for optimization.
            DO NOT use clock regions (use_clock_regions=True) as they are too coarse.
            
            Must be called AFTER read_checkpoint or with device_name specified.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "col_min": {
                        "type": "integer",
                        "description": "Minimum column coordinate"
                    },
                    "col_max": {
                        "type": "integer",
                        "description": "Maximum column coordinate"
                    },
                    "row_min": {
                        "type": "integer",
                        "description": "Minimum row coordinate"
                    },
                    "row_max": {
                        "type": "integer",
                        "description": "Maximum row coordinate"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Device name (optional, uses loaded design's device if omitted)"
                    },
                    "use_clock_regions": {
                        "type": "boolean",
                        "description": "If true, use coarse CLOCKREGION ranges (NOT RECOMMENDED for optimization); if false (DEFAULT), generate detailed multi-site-type ranges (SLICE_X, DSP48E2_X, etc.) - REQUIRED for pblock optimization"
                    }
                },
                "required": ["col_min", "col_max", "row_min", "row_max"]
            }
        ),
        Tool(
            name="compare_design_structure",
            description="""Compare structural properties of two design checkpoints for equivalence validation.
            
            This is Phase 1 of design equivalence checking. Performs sanity checks to catch obvious errors:
            - Top-level module name must match
            - I/O port names, directions, and widths must match
            - Device must match
            - Cell count can increase (optimizations add cells) but not decrease or increase >50%
            
            Returns PASS/FAIL status with detailed comparison report.
            This should be run BEFORE functional simulation to quickly catch structural errors.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "golden_dcp": {
                        "type": "string",
                        "description": "Path to the golden (reference) DCP file"
                    },
                    "revised_dcp": {
                        "type": "string",
                        "description": "Path to the revised (optimized) DCP file to validate"
                    }
                },
                "required": ["golden_dcp", "revised_dcp"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a tool and return the result."""
    try:
        logger.info(f"Tool called: {name} with arguments: {arguments}")
        
        # Route to appropriate handler
        if name == "initialize_rapidwright":
            result = rw.initialize_rapidwright(
                jvm_max_memory=arguments.get("jvm_max_memory", "4G")
            )
        
        elif name == "get_supported_devices":
            result = rw.get_supported_devices()
        
        elif name == "get_device_info":
            result = rw.get_device_info(arguments["device_name"])
        
        elif name == "read_checkpoint":
            result = rw.read_checkpoint(arguments["dcp_path"])
        
        elif name == "write_checkpoint":
            result = rw.write_checkpoint(
                dcp_path=arguments["dcp_path"],
                overwrite=arguments.get("overwrite", False)
            )
        
        elif name == "get_design_info":
            result = rw.get_design_info()
        
        elif name == "search_cells":
            result = rw.search_cells(
                pattern=arguments.get("pattern"),
                cell_type=arguments.get("cell_type"),
                limit=arguments.get("limit", 100)
            )
        
        elif name == "get_tile_info":
            result = rw.get_tile_info(
                tile_name=arguments["tile_name"],
                device_name=arguments.get("device_name")
            )
        
        elif name == "search_sites":
            result = rw.search_sites(
                site_type=arguments.get("site_type"),
                device_name=arguments.get("device_name"),
                limit=arguments.get("limit", 50)
            )
        
        elif name == "optimize_lut_input_cone":
            result = rw.optimize_lut_input_cone(
                hierarchical_input_pins=arguments["hierarchical_input_pins"]
            )
        
        elif name == "optimize_fanout":
            result = rw.optimize_fanout(
                net_name=arguments["net_name"],
                split_factor=arguments["split_factor"]
            )
        
        elif name == "analyze_critical_path_spread":
            result = rw.analyze_critical_path_spread(
                critical_paths_data=arguments.get("critical_paths_data"),
                input_file=arguments.get("input_file")
            )
        
        elif name == "analyze_fabric_for_pblock":
            result = rw.analyze_fabric_for_pblock(
                target_lut_count=arguments["target_lut_count"],
                target_ff_count=arguments["target_ff_count"],
                target_dsp_count=arguments.get("target_dsp_count", 0),
                target_bram_count=arguments.get("target_bram_count", 0),
                device_name=arguments.get("device_name")
            )
        
        elif name == "convert_fabric_region_to_pblock":
            result = rw.convert_fabric_region_to_pblock_ranges(
                col_min=arguments["col_min"],
                col_max=arguments["col_max"],
                row_min=arguments["row_min"],
                row_max=arguments["row_max"],
                device_name=arguments.get("device_name"),
                use_clock_regions=arguments.get("use_clock_regions", False)  # Default to detailed site ranges
            )
        
        elif name == "compare_design_structure":
            result = rw.compare_design_structure(
                golden_dcp=arguments["golden_dcp"],
                revised_dcp=arguments["revised_dcp"]
            )
        
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        # Return formatted result
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(
            type="text", 
            text=json.dumps({"error": str(e), "tool": name}, indent=2)
        )]


@app.list_prompts()
async def list_prompts() -> list[mcp.types.Prompt]:
    """List available prompt templates."""
    return [
        mcp.types.Prompt(
            name="getting_started",
            description="Get started with RapidWright",
            arguments=[]
        ),
        mcp.types.Prompt(
            name="analyze_design",
            description="Analyze a design checkpoint",
            arguments=[
                mcp.types.PromptArgument(
                    name="dcp_path",
                    description="Path to the .dcp file",
                    required=True
                )
            ]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Get a specific prompt template."""
    if name == "getting_started":
        return GetPromptResult(
            description="Getting started with RapidWright",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""I want to use RapidWright. Please:
1. Initialize RapidWright
2. Show me what devices are supported
3. Explain what I can do with this server"""
                    )
                )
            ]
        )
    
    elif name == "analyze_design":
        dcp_path = arguments.get("dcp_path") if arguments else "/path/to/design.dcp"
        return GetPromptResult(
            description="Analyze a design checkpoint",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Analyze the design at: {dcp_path}

Tell me:
1. What device it targets
2. Cell and net counts
3. Top cell types used
4. Any interesting statistics"""
                    )
                )
            ]
        )
    
    raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Main entry point for the server."""
    global _java_log_file, _original_stderr_fd
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RapidWright MCP Server")
    parser.add_argument(
        "--java-log",
        type=str,
        help="Path to log file for Java/JVM output (stdout/stderr)"
    )
    parser.add_argument(
        "--mcp-log",
        type=str,
        help="Path to log file for MCP server logs"
    )
    args = parser.parse_args()
    
    # Configure logging based on whether mcp-log is specified
    if args.mcp_log:
        # Log MCP server messages to a separate file
        mcp_log_file = open(args.mcp_log, 'w', buffering=1)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(mcp_log_file)
            ]
        )
    else:
        # No mcp-log specified - log to stderr (debug mode or standalone usage)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr)
            ]
        )
    
    # If java-log is specified, redirect stdout and stderr at the file descriptor level
    # This must be done BEFORE importing rapidwright to capture Java output
    # This ensures JPype/JVM output is captured without breaking MCP protocol
    if args.java_log:
        try:
            _java_log_file = open(args.java_log, 'w', buffering=1)  # Line buffered
            
            # Save original stdout and stderr file descriptors
            original_stdout_fd = os.dup(1)  # dup stdout (fd 1)
            _original_stderr_fd = os.dup(2)  # dup stderr (fd 2)
            
            # Redirect both stdout (fd 1) and stderr (fd 2) to the log file
            # This captures all Java output (progress messages, errors, etc.)
            os.dup2(_java_log_file.fileno(), 1)
            os.dup2(_java_log_file.fileno(), 2)
            
            # Restore Python's stdout and stderr to the saved file descriptors
            # This allows Python logging and MCP protocol to work normally
            sys.stdout = os.fdopen(original_stdout_fd, 'w', buffering=1)
            sys.stderr = os.fdopen(_original_stderr_fd, 'w', buffering=1)
            
            logger.info(f"Java/JVM output (stdout/stderr fds) will be redirected to: {args.java_log}")
        except Exception as e:
            logger.error(f"Failed to redirect stdout/stderr to log file: {e}")
    
    logger.info("Starting RapidWright MCP Server...")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running on stdio transport")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
    
    # Close the log file on exit
    if _java_log_file:
        _java_log_file.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
