# Benchmark Details

This page describes the suite of benchmark designs that are used to assess
contestant performance. The full list of benchmark designs, along with links to
the original sources and some utilization numbers is provided in the following
tables:

## Benchmarks published during contest

More benchmarks coming soon...

|Source Benchmark Suite|Benchmark Name|LUTs|FFs|DSPs|BRAMs|OOC [1]|
|----------------------|--------------|----|---|----|-----|-------|
| [LogicNets](https://github.com/Xilinx/logicnets)                                                                        |`jscl` (Jet Substructure Classification L)         |31k |2k  |0   |0  |Y   |
| [Corundum](https://github.com/corundum/corundum)                                                                        |`25g` (ADM_PCIE_9V3 25G)                           |73k |96k |0   |221|N   |


## Benchmarks used for final evaluation

To be released after contest concludes.


## Details

Each of the benchmarks targets the `xcvu3p` device which has the following resources:

|LUTs|FFs |DSPs|BRAMs|
|----|----|----|-----|
|394k|788k|2280|720  |

Throughout the contest framework design files associated with
the benchmarks are named as follows:

```
<source benchmark suite>_<benchmark name>_<file description>.<file extension>
```

