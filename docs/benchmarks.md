# Benchmark Details

This page describes the suite of benchmark designs that are used to assess
contestant performance. The full list of benchmark designs, along with links to
the original sources and some utilization numbers is provided in the following
tables:

## Benchmarks published during contest

|Source |Benchmark Name|LUTs|FFs|DSPs|BRAMs|Fmax (MHz)|
|----------------------|--------------|---:|--:|---:|----:|---------:|
| [AMD](https://github.com/amd/mini-isp)                  |`amd_mini-isp`                                    |3k |4k |40  |12  |307  |
| [BOOM](https://github.com/riscv-boom/riscv-boom)       |`boom_soc`                                    |227k |98k |61  |161  |48.2  |
| [CoreScore](https://github.com/olofk/corescore)              |`corescore_500_mod`                               |100k |120k|0 |250  |344.2 |
| [FINN](https://github.com/Xilinx/finn)              |`finn_radioml`                               |74k |46k|0 |25  |284.9 |
| [ISPD16](https://www.ispd.cc/contests/16/)              |`ispd16_example2`                               |289k |234k|200 |384  |107.6 |
| [LogicNets](https://github.com/Xilinx/logicnets)       |`logicnets_jscl` (Jet Substructure Classification L)|31k  |2k  |0   |0    |403.6 |
| [Rosetta](https://github.com/cornell-zhang/rosetta)     |`rosetta_3d-rendering`                           |14k  |5k  |3   |0    |270.9 |
| [Rosetta](https://github.com/cornell-zhang/rosetta)     |`rosetta_digit-recognition`                      |23k  |23k |0   |161  |367.0 |
| [Rosetta](https://github.com/cornell-zhang/rosetta)     |`rosetta_optical-flow`                           |34k  |37k |42  |61   |324.9 |
| [Rosetta](https://github.com/cornell-zhang/rosetta)     |`rosetta_spam-filter`                            |5k   |13k |224 |3    |437.4 |
| [VexRiscv](https://github.com/SpinalHDL/VexRiscv)      |`vexriscv_re-place`                               |2k   |1k  |4   |6    |310.2 |
| [VTR](https://github.com/verilog-to-routing/vtr-verilog-to-routing) |`vtr_mcml`                         |43k  |15k |105 |142  |62.2  |


## Baseline Agent Results

The following table shows the Fmax improvement achieved by the optimization agent
(`dcp_optimizer.py`) on each benchmark design. All timing is measured on the
`clk_fpl26contest` clock domain. Each design was given a maximum runtime of 1 hour.

| Benchmark | Initial Fmax (MHz) | Best Fmax (MHz) | Improvement (MHz) | Improvement (%) | WNS Change (ns) | Runtime | Status |
|---|---:|---:|---:|---:|---|---:|---|
| `amd_mini-isp` | 307.13 | 375.38 | +68.25 | +22.2% | -1.686 → -1.094 | 619s | Completed |
| `boom_soc` | 48.24 | 50.77 | +2.53 | +5.2% | -19.162 → -18.126 | 3601s | Timed out |
| `corescore_500_mod` | 344.23 | 423.37 | +79.14 | +23.0% | -1.238 → -0.695 | 2131s | Completed |
| `finn_radioml` | 284.90 | 324.46 | +39.56 | +13.9% | -1.910 → -1.482 | 1998s | Completed |
| `ispd16_example2` | 107.64 | 107.64 | +0.00 | +0.0% | -7.752 → -7.752 | 3600s | Timed out |
| `logicnets_jscl` | 403.55 | 434.97 | +31.42 | +7.8% | -0.978 → -0.799 | 825s | Completed |
| `rosetta_3d-rendering` | 270.93 | 279.25 | +8.32 | +3.1% | -2.153 → -2.043 | 1823s | Completed |
| `rosetta_digit-recognition` | 366.97 | 390.78 | +23.81 | +6.5% | -1.025 → -0.859 | 1512s | Completed |
| `rosetta_optical-flow` | 324.89 | 330.14 | +5.26 | +1.6% | -1.078 → -1.029 | 558s | Completed |
| `rosetta_spam-filter` | 437.45 | 494.07 | +56.63 | +12.9% | -0.686 → -0.424 | 1000s | Completed |
| `vexriscv_re-place` | 310.17 | 415.28 | +105.11 | +33.9% | -1.654 → -0.838 | 385s | Completed |
| `vtr_mcml` | 62.25 | 73.05 | +10.80 | +17.4% | -14.527 → -12.152 | 1689s | Completed |

**Summary:** 10 of 12 designs completed within the 1-hour limit. Average Fmax
improvement across completed designs was +44.2 MHz (+14.7%). The best single
improvement was `vexriscv_re-place` at +105.11 MHz (+33.9%). The two timed-out
designs (`boom_soc` and `ispd16_example2`) are the largest benchmarks and spent
most of their allotted time in place and route operations.


## Benchmarks used for final evaluation

To be released after contest concludes.


## Details

Each of the benchmarks targets the `xcvu3p-ffvc1517-2-e` device which has the following resources:

|LUTs|FFs |DSPs|BRAMs|
|----|----|----|-----|
|394k|788k|2280|720  |


## Clock Domain of Interest

Some designs will have multiple clock domains.  To make it obvious which clock domain to optimize, each benchmark has a created XDC clock constraint called `clk_fpl26contest` that should be optimized.  All benchmarks in the contest will have the following attributes:

1. A created clock constraint named `clk_fpl26contest` that should be the target clock domain to optimize.
2. All benchmark designs will meet all hold and pulse width constraints.  If a team's solution does not meet hold or pulse width constraints after evaluation, their score will be 0 for that design.
3. The specific clock constraint given to a benchmark is not relevant for the contest, it is only present to encourage tools to try harder.  The focus is primarily on Fmax improvement.  There is no additional bonus for pushing a design to meet or exceed the constraint (positive WNS) other than the corresponding Fmax improvement.

### Querying the contest clock in Vivado

To retrieve the clock period and WNS for `clk_fpl26contest` in a Vivado Tcl session:

```tcl
# Get the clock period
get_property PERIOD [get_clocks clk_fpl26contest]

# Get setup WNS filtered to the contest clock
set tp [get_timing_paths -max_paths 1 -setup -to [get_clocks clk_fpl26contest]]
get_property SLACK $tp
```

Note that `report_timing_summary` reports the overall WNS across *all* clock
domains, which may differ from the WNS on `clk_fpl26contest` in multi-clock
designs. Tools and scripts should query WNS using `-to [get_clocks
clk_fpl26contest]` to ensure the Fmax calculation reflects the correct clock
domain. Fmax is calculated as `1000 / (period - WNS)` where WNS is negative.

