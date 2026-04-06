# Optimization Example

One major way to improve your agent is by creating new backend optimizations or "recipes."  In fact, we hope that you focus most of your development effort in this area ahead of time so that the LLM resources at runtime only have to choose from the available set of optimization options rather than attempting to invent them on-the-fly (which would be both expensive in both time expended and tokens).  Using AI coding agents for development is fully allowed and encouraged to help you in building out these recipes.

To help you get started, here is a simple pattern to follow that can provide success when tackling design optimization.  

---

## The Optimization Recipe Pattern

Physical optimizations can follow this simple four-step cycle:

[![Optimization Recipe Cycle](assets/OptimizationRecipe.svg)](assets/OptimizationRecipe.svg)

| Step | Goal | Typical Tools |
|------|------|---------------|
| **1. Identify** | Pick a specific physical optimization target | Domain knowledge, literature review |
| **2. Analyze** | Build a tool to find where this optimization applies | Vivado timing reports, RapidWright analysis scripts |
| **3. Optimize** | Implement the transformation | RapidWright APIs, Vivado ECO commands |
| **4. Measure** | Quantify the impact on Fmax | Vivado `report_timing_summary` |

The cycle is meant to be iterated: measure results, refine the analysis heuristics, tune the optimization parameters, and repeat.  Different benchmarks will exhibit different optimization opportunites and will be layered like an onion.  When you solve some of the outer ones, there will be different optimization opportunities underneath.

---

## Example Recipe: Critical Path Cell Re-Placement

### The Idea

After place-and-route, some cells on critical paths may be placed far from their ideal location.  (Perhaps the first critical paths encountered in a design would not normally experience this, however, after substantial progress is made in optimizing the first set of critical paths, other paths that become critical may experience this). This causes the router to take long detours, consuming timing margin.  If we can detect these cases and surgically move cells closer to the centroid of their connections, the router can find shorter paths and timing improves.

[![Cell Re-Placement Before/After](assets/cell-replacement-before-after.svg)](assets/cell-replacement-before-after.svg)

---

### Step 1 — Identify the Opportunity

Here is our heuristic: A critical path is **sub-optimal due to placement** when a net segment's *routed path length* (through actual routing PIPs) is significantly longer than the *Manhattan distance* between its source and sink tiles.  We call this ratio the **detour ratio**:

```
detour_ratio = routed_path_length / manhattan_distance
```

A detour ratio of 1.0 means the route is perfectly direct.  Ratios above ~2.0 suggest the cell may benefit from re-placement.  The cell should only be moved if the surrounding path segments have adequate slack to absorb any perturbation.

---

### Step 2 — Build the Analysis Tool

We implement this as an MCP tool (`analyze_net_detour`) in `RapidWrightMCP/rapidwright_tools.py` so the agent can call it directly.

#### 2a. Computing Routed Path Length from PIPs

The core helper walks backward from a sink pin to the net's source, summing tile-to-tile Manhattan distances at each PIP.  Java `Node` objects are used directly as dictionary keys (JPype delegates hashing to the Java `hashCode()`/`equals()` methods):

```python
def _compute_routed_path_length(net, sink_pin):
    pips = net.getPIPs()
    if pips is None or pips.size() == 0:
        return -1

    node_map = {}
    for pip in pips:
        if pip.isReversed():
            end_node, start_node = pip.getStartNode(), pip.getEndNode()
        else:
            end_node, start_node = pip.getEndNode(), pip.getStartNode()
        if end_node is not None and start_node is not None:
            node_map[end_node] = start_node

    src_pin = net.getSource()
    if src_pin is None:
        return -1
    source_node = src_pin.getConnectedNode()
    sink_node = sink_pin.getConnectedNode()
    if source_node is None or sink_node is None:
        return -1

    length = 0
    node = sink_node
    while node is not None and node != source_node:
        prev = node_map.get(node)
        if prev is None:
            return -1
        length += node.getTile().getManhattanDistance(prev.getTile())
        node = prev

    return length if node == source_node else -1
```

A thin wrapper computes the detour ratio by dividing routed path length by Manhattan distance:

```python
def _detour_ratio(net, sink_pin):
    src_pin = net.getSource()
    if src_pin is None or src_pin.getSite() is None:
        return -1
    sink_site = sink_pin.getSite()
    if sink_site is None:
        return -1

    dist = src_pin.getTile().getManhattanDistance(sink_site.getTile())
    if dist == 0:
        return -1
    routed_length = _compute_routed_path_length(net, sink_pin)
    if routed_length <= 0:
        return -1

    return routed_length / dist
```

Key RapidWright APIs used:
* `net.getPIPs()` — returns the list of PIPs that form the net's physical routing
* `pip.getStartNode()` / `pip.getEndNode()` — the routing nodes at each end of a PIP
* `pip.isReversed()` — whether the PIP is driven in reverse direction
* `net.getSource()` — the net's output `SitePinInst`
* `pin.getConnectedNode()` — the routing `Node` at a source or sink pin
* `tile.getManhattanDistance(tile)` — Manhattan distance between two tiles

#### 2b. The Full Analysis Tool

The `analyze_net_detour` tool performs a **cell-centric** analysis.  For each interior cell on the critical path, it examines both the *incoming* net (feeding the cell) and the *outgoing* net (driven by it) to compute the worst-case detour ratio across the source pin and *all* sink pins.  A high detour ratio may be indicative that the cell is poorly placed.

The input is a **pin-path** list as produced by `extract_critical_path_pins`:

```
["src_ff/Q", "lut1/I2", "lut1/O", "lut2/I0", "lut2/O", "dst_ff/D"]
```

Consecutive pins from the same cell (e.g. `lut1/I2`, `lut1/O`) identify the cell's data path.  The analysis uses `EDIFNetlist.getHierPortInstFromName()` for O(1) pin resolution — each `EDIFHierPortInst` gives direct access to the routed physical net and site pin:

```python
for pin_name in pin_list:
    pin = netlist.getHierPortInstFromName(pin_name)
    if pin is None:
        continue
    if prev_pin is not None and pin.getFullHierarchicalInst().equals(
            prev_pin.getFullHierarchicalInst()):
        cells_on_path.append((prev_pin, pin))
    prev_pin = pin

for (in_pin, out_pin) in cells_on_path:
    ratio = -1
    for pin in (in_pin, out_pin):
        if pin is not None:
            net = pin.getRoutedPhysicalNet(design)
            if net is not None and not net.isStaticNet() and not net.isClockNet():
                spi = pin.getRoutedSitePinInst(design)
                if spi is not None:
                    if spi.isOutPin():
                        for sink_spi in net.getSinkPins():
                            cr = _detour_ratio(net, sink_spi)
                            if cr > ratio:
                                ratio = cr
                    else:
                        cr = _detour_ratio(net, spi)
                        if cr > ratio:
                            ratio = cr
```

Note the `isOutPin()` branch: for a cell's output pin the `SitePinInst` is the net's *source*, so `_detour_ratio` would return -1 (zero distance to itself).  Instead, we iterate over the net's sink pins and take the maximum detour ratio.  This is critical for catching cells whose *outgoing* net has a long detour (e.g. a LUT driving a distant register file).

#### 2c. Chaining Vivado and RapidWright

The analysis requires data from both tools:

1. **Vivado** `extract_critical_path_pins` — extracts ordered pin paths from the timing report
2. **Vivado** `report_timing_summary` — provides baseline WNS/TNS and per-path slack
3. **RapidWright** `analyze_net_detour` — compares Manhattan distance to routed path length

---

### Step 3 — Implement the Optimization

The `optimize_cell_placement` MCP tool moves candidate cells to the centroid of their connections:

```python
def optimize_cell_placement(cell_names, max_candidates=10):
    from com.xilinx.rapidwright.design import DesignTools
    from com.xilinx.rapidwright.eco import ECOPlacementHelper
    from com.xilinx.rapidwright.placer.blockplacer import Point
    from com.xilinx.rapidwright.device import SiteTypeEnum
    from java.util import ArrayList, EnumSet

    design = _current_design
    device = design.getDevice()
    target_site_types = EnumSet.of(SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM)

    for cell_name in cell_names[:max_candidates]:
        cell = design.getCell(cell_name)
        old_bel = cell.getBEL()
        is_ff = old_bel.isFF() if old_bel is not None else False
        is_lut = old_bel.isLUT() if old_bel is not None else False

        # 1. Collect tile locations of all connected pins
        connected_nets = _get_cell_physical_nets(design, cell)
        points = ArrayList()
        for net in connected_nets:
            for pin in net.getPins():
                t = pin.getTile()
                if t is not None:
                    points.add(Point(t.getColumn(), t.getRow()))

        # 2. Compute centroid — the ideal placement location
        centroid_site = ECOPlacementHelper.getCentroidOfPoints(
            device, points, target_site_types)

        # 3. Unplace the cell and unroute affected nets
        DesignTools.fullyUnplaceCell(cell, None)
        for net in connected_nets:
            # Note: this removes all routing on the entire net.
            #       For incoming nets of a re-placed cell, this will also unroute
            #       any routing going to other unrelated cells.
            net.unroute()

        # 4. Find an empty site near the centroid and place the cell
        for candidate in ECOPlacementHelper.spiralOutFrom(centroid_site):
            if design.getSiteInstFromSite(candidate) is None:
                bel_name = "AFF" if is_ff else "A6LUT" if is_lut else str(
                    old_bel.getName()) if old_bel else "A6LUT"
                bel = candidate.getBEL(bel_name)
                if bel is not None:
                    design.placeCell(cell, candidate, bel)
                    cell.getSiteInst().routeSite()
                    break
```

Key RapidWright APIs used:

| API | Purpose |
|-----|---------|
| `ECOPlacementHelper.getCentroidOfPoints(device, points, siteTypes)` | Computes the arithmetic mean of tile coordinates and snaps to the nearest site of the requested type |
| `ECOPlacementHelper.spiralOutFrom(site)` | Iterates neighboring sites in an outward spiral to find available placement locations |
| `DesignTools.fullyUnplaceCell(cell, deferRemovals)` | Cleanly removes a cell's physical placement, handling shared site pins |
| `net.unroute()` | Clears all PIPs from a net so Vivado can re-route it |
| `design.placeCell(cell, site, bel)` | Places a cell at a specific site and BEL |
| `siteInst.routeSite()` | Establishes intra-site routing for the newly placed cell |

The helper `_get_cell_physical_nets` traces each of the cell's logical pins through site wires to find the connected physical nets:

```python
def _get_cell_physical_nets(design, cell):
    siteInst = cell.getSiteInst()
    if siteInst is None:
        return []
    net_names = set()
    nets = []
    try:
        edif_inst = cell.getEDIFCellInst()
        if edif_inst is None:
            return []
        for port_inst in edif_inst.getPortInsts():
            logical_pin = str(port_inst.getName())
            try:
                site_wire = cell.getSiteWireNameFromLogicalPin(logical_pin)
                if site_wire is None:
                    continue
                net = siteInst.getNetFromSiteWire(site_wire)
                if net is None or net.isStaticNet() or net.isClockNet():
                    continue
                name = str(net.getName())
                if name not in net_names:
                    net_names.add(name)
                    nets.append(net)
            except Exception:
                continue
    except Exception as e:
        pass
    return nets
```

Key APIs:
* `cell.getEDIFCellInst()` — the logical (EDIF) cell instance
* `cell.getSiteWireNameFromLogicalPin(pin)` — maps a logical pin name to a physical site wire  (used in [FanOutOptimization.java](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/eco/FanOutOptimization.java))
* `siteInst.getNetFromSiteWire(wire)` — returns the physical net connected to a site wire

---

### Step 4 — Measure the Impact

After re-placing cells in RapidWright, write the modified checkpoint and use Vivado to re-route and re-time:

```
rapidwright:  write_checkpoint("optimized.dcp")
vivado:       open_checkpoint("optimized.dcp")
vivado:       route_design                      ← re-routes only the unrouted nets
vivado:       report_timing_summary             ← compare new WNS to baseline
```

Because `optimize_cell_placement` calls `net.unroute()` on every net connected to the moved cells, Vivado's `route_design` will incrementally re-route only those nets—the rest of the design stays intact.

---

## Putting It All Together

Here is the complete sequence of MCP tool calls an agent would make:

```
 ┌─── Baseline ──────────────────────────────────────────────────────────┐
 │ 1. vivado:  open_checkpoint(input.dcp)                                │
 │ 2. vivado:  report_timing_summary  → baseline WNS                     │
 │ 3. vivado:  extract_critical_path_pins(num_paths=50)                  │
 └───────────────────────────────────────────────────────────────────────┘

 ┌─── Analyze ───────────────────────────────────────────────┐
 │ 4. rapidwright:  read_checkpoint(input.dcp)               │
 │ 5. rapidwright:  analyze_net_detour(critical_paths)       │
 │    → returns ranked candidates with detour ratios         │
 └───────────────────────────────────────────────────────────┘

 ┌─── Optimize ──────────────────────────────────────────────┐
 │ 6. rapidwright:  optimize_cell_placement(candidate_cells) │
 │ 7. rapidwright:  write_checkpoint(optimized.dcp)          │
 └───────────────────────────────────────────────────────────┘

 ┌─── Measure ───────────────────────────────────────────────┐
 │  8. vivado:      open_checkpoint(optimized.dcp)           │
 │  9. vivado:      route_design                             │
 │ 10. vivado:      report_timing_summary  → new WNS         │
 │     Compare new WNS to baseline — did Fmax improve?       │
 └───────────────────────────────────────────────────────────┘
```

---

## Try It Yourself

The benchmark `vexriscv_re-place_2025.1.dcp` has a critical path with a deliberately misplaced LUT2 that the recipe can fix.  Here is a complete Python script that runs all four steps.  It assumes the DCP is in the `fpl26_contest_benchmarks/` directory:

```python
#!/usr/bin/env python3
"""End-to-end cell re-placement optimization example."""
import sys, json, os, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VivadoMCP"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RapidWrightMCP"))
import vivado_mcp_server as vivado
import rapidwright_tools as rw

DCP = "fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp"
OPT_DCP = "vexriscv_optimized.dcp"


def get_wns():
    """Return the WNS value in nanoseconds from the current Vivado design."""
    result = vivado.run_tcl_command(
        "set p [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
        "if {[llength $p] > 0} {get_property SLACK $p} else {puts 0.0}",
        timeout=60,
    )
    m = re.search(r"[-]?\d+\.\d+", result)
    return float(m.group()) if m else None


# ── Step 1: Baseline ────────────────────────────────────────────────
print("=" * 60)
print("Step 1  Vivado baseline")
print("=" * 60)
vivado.start_vivado()
vivado.run_tcl_command(f"open_checkpoint {{{DCP}}}", timeout=300)
baseline_wns = get_wns()
print(f"  Baseline WNS: {baseline_wns} ns")

pins_json = vivado.extract_critical_path_pins(num_paths=10)
critical_paths = json.loads(pins_json)
print(f"  Extracted {len(critical_paths)} critical path pin lists")
vivado.cleanup_vivado()

# ── Step 2: Analyze ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2  RapidWright analysis")
print("=" * 60)
rw.initialize_rapidwright()
rw.read_checkpoint(DCP)

analysis = rw.analyze_net_detour(
    critical_paths_data=critical_paths, detour_threshold=2.0
)
candidates = analysis.get("candidates", [])
print(f"  Cells analyzed: {analysis['cells_analyzed']}")
print(f"  Candidates (detour > 2.0): {len(candidates)}")
for c in candidates[:5]:
    print(f"    {str(c['cell']):55s}  ratio={c['max_detour_ratio']}")

# Filter to cells on the worst path(s) — paths 1 and 2
worst_path_cells = [
    str(c["cell"]) for c in candidates if c["path"] <= 2
]
print(f"\n  Targeting {len(worst_path_cells)} cells on paths 1-2:")
for name in worst_path_cells:
    print(f"    {name}")

# ── Step 3: Optimize ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3  RapidWright optimization")
print("=" * 60)
opt_result = rw.optimize_cell_placement(cell_names=worst_path_cells)
for r in opt_result.get("results", []):
    print(f"  {r['cell']}: {r['status']} — {r['message']}")

rw.write_checkpoint(OPT_DCP)
print(f"  Wrote {OPT_DCP}")

# ── Step 4: Measure ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4  Vivado verification")
print("=" * 60)
vivado.start_vivado()
vivado.run_tcl_command(f"open_checkpoint {{{OPT_DCP}}}", timeout=300)
vivado.run_tcl_command("route_design", timeout=600)

route_status = vivado.run_tcl_command(
    "report_route_status -return_string", timeout=60
)
errors = re.search(
    r"# of nets with routing errors.*?:\s+(\d+)", route_status
)
error_count = int(errors.group(1)) if errors else -1

new_wns = get_wns()
print(f"  Routing errors: {error_count}")
print(f"  Baseline WNS:  {baseline_wns} ns")
print(f"  Optimized WNS: {new_wns} ns")
if new_wns is not None and baseline_wns is not None:
    delta = new_wns - baseline_wns
    print(f"  Improvement:   {delta:+.3f} ns")

vivado.cleanup_vivado()
```

On the `vexriscv_re-place_2025.1.dcp` benchmark this moves the misplaced LUT2 from `SLICE_X115Y2` to `SLICE_X111Y17` (18 tiles closer to its connections) and improves WNS from **-1.654 ns** to **-1.285 ns** — a gain of **+0.369 ns**.

---

## Build Your Own Recipes

The cell re-placement example above is just one of many possible optimization recipes.  We encourage contestants to identify new opportunities and build their own.  Here are some ideas to get started:

* **LUT merging** — Combine cascaded small LUTs into a single larger LUT to reduce logic depth.  Already available as the `optimize_lut_input_cone` MCP tool.
* **Register retiming** — Move flip-flops across combinational logic to balance pipeline stage delays.
* **Net swapping** — Swap equivalent nets between BEL pins within a SLICE to reduce routing congestion.
* **DSP/BRAM relocation** — Move hard-block placements closer to their data sources to shorten critical interconnect.
* **Congestion-aware spreading** — Identify regions of high routing congestion and spread cells outward to improve routability.

For each idea, follow the same four-step recipe:

1. **Identify** the scenario and when it hurts timing
2. **Analyze** designs to detect candidates — use Vivado timing reports, RapidWright device queries, or both
3. **Optimize** by implementing the transformation as an MCP tool in RapidWright
4. **Measure** the impact using Vivado's `report_timing_summary`

Remember the [guidelines](details.html#guidelines-for-building-customized-analysis-and-optimizations) for choosing between RapidWright and Vivado for different tasks.  In general: use RapidWright for fast analysis, placement modifications, and netlist ECOs; use Vivado for routing (`route_design`) and authoritative timing (`report_timing_summary`).
