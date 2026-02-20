# RapidWright MCP Server

An MCP (Model Context Protocol) server that provides AI assistant access to [RapidWright](https://github.com/Xilinx/RapidWright), an open-source FPGA design tool framework from AMD.

This server enables AI assistants like Cursor to interact with FPGA designs, query device information, analyze design checkpoints, and explore Xilinx/AMD device architectures through natural language.

## Features

- **Device Information**: Query supported FPGA devices and families, get detailed device specifications
- **Design Checkpoint Analysis**: Load Vivado .dcp files, inspect design statistics, search for cells
- **Device Architecture Exploration**: Get tile/site information, query device resources
- **Design Optimization**: LUT input cone optimization and high fanout net splitting

## Quick Start

### Prerequisites

- Python 3.8+
- Java 11+

### Installation

```bash
git clone https://gitenterprise.xilinx.com/clavin/RapidWrightMCP
cd RapidWrightMCP
./setup.sh
python3 test_server.py
```

That's it! RapidWright is installed automatically via pip.

## Usage with Cursor

Add to your MCP configuration file:

```json
{
  "mcpServers": {
    "rapidwright": {
      "command": "python3",
      "args": ["/absolute/path/to/RapidWrightMCP/server.py"]
    }
  }
}
```

Restart Cursor after saving.

## Available Tools

| Tool | Description |
|------|-------------|
| `initialize_rapidwright` | Initialize RapidWright (must be called first) |
| `get_supported_devices` | List all supported FPGA devices |
| `get_device_info` | Get detailed information about a specific device |
| `read_checkpoint` | Load a Vivado Design Checkpoint (.dcp) file |
| `write_checkpoint` | Save design to a .dcp file |
| `get_design_info` | Get statistics about the loaded design |
| `search_cells` | Search for cells by name or type |
| `get_tile_info` | Get information about a specific tile |
| `search_sites` | Search for sites by type on a device |
| `optimize_lut_input_cone` | Optimize LUT chains by combining into single LUTs |
| `optimize_fanout` | Split high fanout nets by replicating drivers |

## Example Usage

```
User: "Initialize RapidWright and show me what devices are available"
AI: [calls initialize_rapidwright and get_supported_devices]
    "RapidWright supports 50+ devices including xcvu3p, xcvu9p, xcku040..."

User: "Load my design from ~/my_design.dcp and tell me how many LUTs it uses"
AI: [calls read_checkpoint, then get_design_info and search_cells]
    "Your design uses 15,432 LUT6 cells..."

User: "Optimize the LUT input cone for pin 'top/cpu/alu/result[0]'"
AI: [calls optimize_lut_input_cone]
    "Successfully combined 3 chained LUTs into a single LUT6."

User: "The net 'clk_enable' has very high fanout. Split it into 4 parts."
AI: [calls optimize_fanout]
    "Split 'clk_enable' (original fanout: 2,456) into 4 nets with ~614 loads each."
```

## Tips

- **Initialize once** per session, then use other commands freely
- **Use absolute paths** for .dcp files
- **Be specific** with device names (e.g., "xcvu9p" not "vu9p")
- **Chain requests**: "Load design X and tell me Y"
- Use natural language - no need for exact tool names

## Common Cell & Site Types

| Cell Type | Description |
|-----------|-------------|
| LUT6 | 6-input Lookup Table |
| FDRE | D Flip-Flop with Clock Enable and Sync Reset |
| FDCE | D Flip-Flop with Clock Enable and Async Clear |
| CARRY8 | Carry Logic (UltraScale+) |
| DSP48E2 | DSP Block (UltraScale+) |
| RAMB36E2 | 36Kb Block RAM (UltraScale+) |
| BUFGCE | Global Clock Buffer |

| Site Type | Description |
|-----------|-------------|
| SLICEL | Logic Slice (LUTs, FFs, Carry) |
| SLICEM | Memory Slice (+ Distributed RAM) |
| DSP48E2 | DSP/Math Block |
| RAMB36/RAMB18 | Block RAM |
| URAM288 | 288Kb Ultra RAM |

---

## Design Optimization Guide

### LUT Input Cone Optimization

Combines chained small LUTs into a single larger LUT (up to LUT6) to reduce logic depth.

```
Before:  Input -> LUT2 -> LUT3 -> LUT4 -> Output  (3 logic levels)
After:   Input -> LUT6 -> Output                   (1 logic level)
```

**When to use:**
- Critical path optimization when timing shows multiple LUT levels
- Post-place-and-route ECO fixes without full re-synthesis

**Parameters:**
- `hierarchical_input_pins`: List of pins to optimize (e.g., `["top/cpu/alu/result[0]"]`)
- `output_dcp_path`: Optional path to save the optimized design

**Limitations:**
- Maximum 6 inputs (LUT6 is the largest)
- Only works on paths driven exclusively by LUTs
- Cannot optimize paths with flip-flops, DSPs, or other non-LUT elements

### Fanout Optimization

Splits high-fanout nets by replicating the source driver. Each replica drives a subset of loads.

```
Before:  Driver -> [1000 loads]
After:   Driver_1 -> [250 loads]
         Driver_2 -> [250 loads]
         Driver_3 -> [250 loads]
         Driver_4 -> [250 loads]
```

**When to use:**
- High-fanout enable/control signals (>500 loads)
- Routing congestion from heavily loaded nets
- Timing closure for high-fanout paths

**Choosing split factor:**
- k=2: Moderately high fanout (500-1000 loads)
- k=3-4: Very high fanout (1000-3000 loads)
- k≥5: Extremely high fanout (>3000 loads)

**Trade-offs:**
- Higher k = lower fanout per net = better routing/timing
- Higher k = more driver cells = increased area and power

**Limitations:**
- Only works on routed nets
- Not beneficial for small fanouts (<100 loads)

### Optimization Troubleshooting

| Error | Solution |
|-------|----------|
| "Pin not found" | Check hierarchical path, use full name including top module |
| "No optimization possible" | Pin not driven by LUTs, or already optimal single LUT |
| "6 maximum inputs" | Cone requires >6 inputs; try optimizing a later stage |
| "Net not found" | Use physical net name, not hierarchical logical name |

---

## Architecture

```
┌─────────────────┐
│     Cursor      │
│  AI Assistant   │
└────────┬────────┘
         │ MCP Protocol (JSON-RPC over stdio)
┌────────▼────────┐
│  server.py      │  ← MCP Server
└────────┬────────┘
┌────────▼────────────┐
│ rapidwright_tools.py│  ← Tool Wrappers
└────────┬────────────┘
┌────────▼────────┐
│  RapidWright    │  ← pip package (JPype + Java libs)
└─────────────────┘
```

## Development

### Project Structure

```
RapidWrightMCP/
├── server.py              # Main MCP server
├── rapidwright_tools.py   # RapidWright wrapper functions
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── test_server.py        # Test suite
└── rapidwright_mcp.log   # Log file (created at runtime)
```

### Adding New Tools

To add a new tool:

1. **Add the function** to `rapidwright_tools.py`:
   ```python
   def my_new_tool(param1: str) -> Dict[str, Any]:
       """Your tool implementation."""
       if not _initialized:
           return {"error": "RapidWright not initialized"}
       # ... implementation
       return {"status": "success", "result": ...}
   ```

2. **Register the tool** in `server.py` `list_tools()`:
   ```python
   Tool(
       name="my_new_tool",
       description="What your tool does",
       inputSchema={
           "type": "object",
           "properties": {
               "param1": {"type": "string", "description": "..."}
           },
           "required": ["param1"]
       }
   )
   ```

3. **Add the handler** in `call_tool()`:
   ```python
   elif name == "my_new_tool":
       result = rw.my_new_tool(arguments["param1"])
   ```

### Running Tests

```bash
python3 test_server.py
tail -f rapidwright_mcp.log
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "RapidWright not initialized" | Call `initialize_rapidwright` first; check Java 11+ is installed |
| Server not appearing | Use absolute paths in config; restart Cursor/Claude completely |
| Out of memory | Increase `jvm_max_memory` (e.g., "8G", "16G") |
| Installation issues | `pip3 install --force-reinstall rapidwright` |

**Checklist:**
- [ ] Python 3.8+? (`python3 --version`)
- [ ] Java 11+? (`java -version`)
- [ ] RapidWright installed? (`pip3 show rapidwright`)
- [ ] Config paths absolute?
- [ ] Cursor restarted?
- [ ] Check logs? (`cat rapidwright_mcp.log`)


## Resources

- [RapidWright Documentation](https://www.rapidwright.io/docs/)
- [RapidWright Javadoc](https://www.rapidwright.io/javadoc/)
- [RapidWright GitHub](https://github.com/Xilinx/RapidWright)
- [Model Context Protocol](https://modelcontextprotocol.io/)

