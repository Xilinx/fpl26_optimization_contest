#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache 2.0

import shutil
import tempfile
import unittest
import contextlib
import io
import json
from pathlib import Path

from validate_dcps import (
    DCPValidator,
    assign_from_seed,
    parse_simulation_output,
    port_bit_width,
)


class ValidateDCPHelperTests(unittest.TestCase):
    def test_port_bit_width_accepts_either_range_direction(self):
        self.assertEqual(port_bit_width({"name": "a", "width": "[31:0]"}), 32)
        self.assertEqual(port_bit_width({"name": "a", "width": "[0:31]"}), 32)
        self.assertEqual(port_bit_width({"name": "a", "width": "[ 7 : 0 ]"}), 8)
        self.assertEqual(port_bit_width({"name": "a", "width": None}), 1)

    def test_assign_from_seed_varies_wide_chunks(self):
        assignment = assign_from_seed({"name": "wide", "width": "[95:0]"}, "lfsr")

        self.assertIn("wide = {", assignment)
        self.assertIn("(lfsr ^ 32'h9E3779B9)", assignment)
        self.assertIn("(lfsr ^ 32'h3C6EF372)", assignment)
        self.assertIn("(lfsr ^ 32'hDAA66D2B)", assignment)
        self.assertNotIn("{3{lfsr}}", assignment)


class SimulationParserTests(unittest.TestCase):
    def test_parse_requires_pass_marker_and_expected_cycles(self):
        parsed = parse_simulation_output(
            "Cycles simulated: 200\nMismatches found: 0\n",
            returncode=0,
            expected_cycles=200,
        )

        self.assertFalse(parsed["passed"])
        self.assertFalse(parsed["result_pass_seen"])

    def test_parse_accepts_clean_pass(self):
        parsed = parse_simulation_output(
            "\n".join([
                "Cycles simulated: 200",
                "Mismatches found: 0",
                "Protocol mismatches found: 0",
                "Result: PASS",
            ]),
            returncode=0,
            expected_cycles=200,
        )

        self.assertTrue(parsed["passed"])

    def test_parse_fails_on_protocol_mismatch(self):
        parsed = parse_simulation_output(
            "\n".join([
                "PROTOCOL MISMATCH at cycle 7: s_tready golden=1 revised=0",
                "Cycles simulated: 200",
                "Mismatches found: 0",
                "Protocol mismatches found: 1",
                "Result: FAIL",
            ]),
            returncode=0,
            expected_cycles=200,
        )

        self.assertFalse(parsed["passed"])
        self.assertEqual(parsed["protocol_mismatch_count"], 1)
        self.assertEqual(parsed["mismatch_count"], 0)

    def test_parse_mismatch_with_fatal_marker_is_logical_fail_not_infrastructure(self):
        parsed = parse_simulation_output(
            "\n".join([
                "MISMATCH at cycle 3: out golden=1 revised=0",
                "Cycles simulated: 200",
                "Mismatches found: 1",
                "Protocol mismatches found: 0",
                "FATAL: Simulation engine not responding",
                "Result: FAIL",
            ]),
            returncode=1,
            expected_cycles=200,
        )

        self.assertFalse(parsed["passed"])
        self.assertTrue(parsed["simulator_failed"])
        self.assertFalse(parsed["infrastructure_failure"])
        self.assertIsNone(parsed["infrastructure_reason"])

    def test_parse_fails_on_xsim_fatal_marker_even_with_pass(self):
        parsed = parse_simulation_output(
            "\n".join([
                "Cycles simulated: 200",
                "Mismatches found: 0",
                "Protocol mismatches found: 0",
                "FATAL: Simulation engine not responding",
                "Result: PASS",
            ]),
            returncode=0,
            expected_cycles=200,
        )

        self.assertFalse(parsed["passed"])
        self.assertTrue(parsed["simulator_failed"])


class TestbenchGenerationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="validate_dcps_test_"))

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _generate(
        self,
        inputs,
        outputs,
        no_reactive=False,
        clock_names=None,
        module_name="dut",
        verilog_text=None,
    ):
        validator = DCPValidator(
            self.workspace / "golden.dcp",
            self.workspace / "revised.dcp",
            num_vectors=8,
            no_reactive=no_reactive,
        )
        try:
            tb_path = self.workspace / "testbench.v"
            info = {
                "module_name": module_name,
                "ports": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "inouts": [],
                },
            }
            if verilog_text is not None:
                verilog_path = self.workspace / "golden_sim.v"
                verilog_path.write_text(verilog_text)
                info["verilog_path"] = str(verilog_path)
            validator.generate_testbench(
                info,
                info,
                tb_path,
                clock_names=clock_names if clock_names is not None else [inputs[0]["name"]],
            )
            return tb_path.read_text()
        finally:
            shutil.rmtree(validator.temp_dir, ignore_errors=True)

    def test_sanitized_interface_ids_are_unique(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "foo__bar_rsp_valid", "width": None},
                {"name": "foo_bar_rsp_valid", "width": None},
            ],
            outputs=[
                {"name": "foo__bar_cmd_valid", "width": None},
                {"name": "foo_bar_cmd_valid", "width": None},
            ],
        )

        self.assertIn("reg env_foo_bar_pending;", tb)
        self.assertIn("reg env_foo_bar_1_pending;", tb)

    def test_ready_valid_sink_emits_protocol_check(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "stream_tvalid", "width": None},
                {"name": "stream_tdata", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "stream_tready", "width": None},
            ],
        )

        self.assertIn("if (golden_stream_tready) begin", tb)
        self.assertIn("PROTOCOL MISMATCH at cycle %0d: stream_tready", tb)

    def test_command_response_emits_protocol_check(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "mem_rsp_valid", "width": None},
                {"name": "mem_cmd_ready", "width": None},
                {"name": "mem_rsp_payload_data", "width": "[63:0]"},
            ],
            outputs=[
                {"name": "mem_cmd_valid", "width": None},
            ],
        )

        self.assertIn("Reactive responder for mem", tb)
        self.assertIn("PROTOCOL MISMATCH at cycle %0d: mem_cmd_valid", tb)
        self.assertIn("mem_rsp_payload_data = {", tb)

    def test_hls_control_emits_protocol_checks(self):
        tb = self._generate(
            inputs=[
                {"name": "ap_clk", "width": None},
                {"name": "ap_start", "width": None},
                {"name": "a", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "ap_done", "width": None},
                {"name": "ap_idle", "width": None},
                {"name": "ap_ready", "width": None},
            ],
        )

        self.assertIn("Transactional HLS control driver", tb)
        self.assertIn("PROTOCOL MISMATCH at cycle %0d: ap_done", tb)
        self.assertIn("PROTOCOL MISMATCH at cycle %0d: ap_idle", tb)
        self.assertIn("PROTOCOL MISMATCH at cycle %0d: ap_ready", tb)

    def test_no_reactive_disables_reactive_environment(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "stream_tvalid", "width": None},
            ],
            outputs=[
                {"name": "stream_tready", "width": None},
            ],
            no_reactive=True,
        )

        self.assertIn("No reactive environment state", tb)
        self.assertIn("stream_tvalid = lfsr[0];", tb)
        self.assertNotIn("PROTOCOL MISMATCH", tb)

    def test_non_clock_like_vivado_clock_list_falls_back_to_name_heuristic(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "commit_valid", "width": None},
            ],
            outputs=[
                {"name": "out", "width": None},
            ],
            clock_names=["commit_valid"],
        )

        self.assertIn("forever #5 clk = ~clk;", tb)
        self.assertIn("commit_valid = lfsr[0];", tb)

    def test_generic_hls_ap_memory_32_bit_model_uses_full_width_hash(self):
        tb = self._generate(
            inputs=[
                {"name": "ap_clk", "width": None},
                {"name": "mem_q0", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "mem_address0", "width": "[9:0]"},
                {"name": "mem_ce0", "width": None},
            ],
        )

        self.assertNotIn("function [7:0] hls_mem_byte_hash;", tb)
        self.assertIn("mem_q0 <= ({22'b0, golden_mem_address0} * 32'h", tb)
        self.assertNotIn("mem_q0 = lfsr[31:0];", tb)

    def test_rosetta_rendering_input_memory_uses_coordinate_byte_clamp(self):
        tb = self._generate(
            inputs=[
                {"name": "ap_clk", "width": None},
                {"name": "input_r_q0", "width": "[31:0]"},
                {"name": "input_r_q1", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "input_r_address0", "width": "[13:0]"},
                {"name": "input_r_ce0", "width": None},
                {"name": "input_r_address1", "width": "[13:0]"},
                {"name": "input_r_ce1", "width": None},
            ],
            module_name="xil_internal_svlib_rendering",
        )

        self.assertIn("function [7:0] hls_mem_byte_hash;", tb)
        self.assertIn("input_r_q0 <= {hls_mem_byte_hash(", tb)
        self.assertIn("input_r_q1 <= {hls_mem_byte_hash(", tb)

    def test_rosetta_port_shape_without_rendering_module_uses_full_width_hash(self):
        tb = self._generate(
            inputs=[
                {"name": "ap_clk", "width": None},
                {"name": "input_r_q0", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "input_r_address0", "width": "[13:0]"},
                {"name": "input_r_ce0", "width": None},
            ],
            module_name="dut",
        )

        self.assertNotIn("function [7:0] hls_mem_byte_hash;", tb)
        self.assertIn("input_r_q0 <= ({18'b0, golden_input_r_address0} * 32'h", tb)

    def test_rosetta_spam_filter_label_memory_uses_full_width_hash(self):
        tb = self._generate(
            inputs=[
                {"name": "ap_clk", "width": None},
                {"name": "label_r_q0", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "label_r_address0", "width": "[10:0]"},
                {"name": "label_r_ce0", "width": None},
            ],
            module_name="xil_internal_svlib_SgdLR",
        )

        self.assertNotIn("function [7:0] hls_mem_byte_hash;", tb)
        self.assertIn("label_r_q0 <= ({21'b0, golden_label_r_address0} * 32'h", tb)

    def test_vexriscv_icache_ibus_enables_rv32m_burst_stimulus(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "reset", "width": None},
                {"name": "iBus_rsp_valid", "width": None},
                {"name": "iBus_rsp_payload_data", "width": "[31:0]"},
                {"name": "iBus_rsp_payload_error", "width": None},
            ],
            outputs=[
                {"name": "iBus_cmd_valid", "width": None},
            ],
            verilog_text="""
module dut(input clk);
  // Markers emitted by VexRiscv's cached instruction bus hierarchy.
  wire IBusCachedPlugin_cache;
  wire lineLoader_valid;
endmodule
""",
        )

        self.assertIn("function [31:0] dsp_rv32_instr;", tb)
        self.assertIn("reg [2:0]  dsp_inj_state;", tb)
        self.assertIn("integer env_iBus_burst_remaining;", tb)
        self.assertIn("env_iBus_burst_remaining = 8;", tb)
        self.assertIn("iBus_rsp_payload_data = dsp_rv32_instr(dsp_inj_state,", tb)
        self.assertNotIn("iBus_rsp_payload_data = (dsp_inj_state", tb)
        self.assertIn("default: dsp_rv32_instr = 32'h00000013", tb)
        self.assertNotIn("32'hx", tb)
        self.assertIn("DSP RV32M injection state machine (burst mode", tb)
        # State 7 must be BNE (branch on non-zero multiply result) so mismatch is
        # visible on iBus_cmd_payload_address without needing dBus ports.
        self.assertIn("BNE rd3,x0,+64", tb)
        self.assertNotIn("SW rd3,0(x0)", tb)
        # Operands must be fixed non-zero constants so MUL always produces a
        # non-zero result in the golden design (zero LFSR fields would silence it).
        self.assertIn("dsp_inj_imm1 <= 12'd7;", tb)
        self.assertIn("dsp_inj_imm2 <= 12'd11;", tb)


    def test_wide_reset_named_port_gets_lfsr_stimulus_not_reset_treatment(self):
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "reset", "width": None},
                {"name": "resetVector", "width": "[31:0]"},
            ],
            outputs=[
                {"name": "out", "width": None},
            ],
        )

        self.assertIn("reg [31:0] resetVector;", tb)
        self.assertNotIn("resetVector = 1;", tb)
        self.assertIn("resetVector = lfsr[31:0];", tb)

    def test_detect_tilelink_boom_requires_all_three_markers(self):
        # Missing any one of the three markers must return False.
        self.assertFalse(DCPValidator._detect_tilelink_boom(
            "wire d_bits_data; module BoomCore();"))
        self.assertFalse(DCPValidator._detect_tilelink_boom(
            "DSP48E2 #(.PREG(0)) dsp0 (); module BoomCore();"))
        self.assertFalse(DCPValidator._detect_tilelink_boom(
            "DSP48E2 #(.PREG(0)) dsp0 (); wire d_bits_data;"))
        self.assertTrue(DCPValidator._detect_tilelink_boom(
            "DSP48E2 #(.PREG(0)) dsp0 (); wire d_bits_data; module BoomCore();"
        ))

    def test_boomsoc_tilelink_enables_64bit_dsp_stimulus(self):
        # TileLink D channel: d_valid (input), d_ready (output), d_bits_data (input).
        # The d_ready output is required for the generic valid/ready detector to
        # recognise mem_d as a request interface with mem_d_bits_data as payload.
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "reset", "width": None},
                {"name": "mem_d_valid", "width": None},
                {"name": "mem_d_bits_data", "width": "[63:0]"},
                {"name": "mem_a_ready", "width": None},
            ],
            outputs=[
                {"name": "mem_a_valid", "width": None},
                {"name": "mem_d_ready", "width": None},
            ],
            verilog_text="""
module BoomCore(input clk);
  DSP48E2 #(.PREG(0)) dsp0 (.CLK(clk), .CEP(1'b1));
  wire d_bits_data;
endmodule
""",
        )

        self.assertIn("function [31:0] dsp_rv32_instr;", tb)
        self.assertIn("Reactive request driver for mem_d", tb)

    def test_boom_trace_testbenches_use_separate_elaboration_images(self):
        validator = DCPValidator(
            self.workspace / "golden.dcp",
            self.workspace / "revised.dcp",
            num_vectors=8,
        )
        try:
            inputs = [
                {"name": "clock_uncore_clock", "width": None},
                {"name": "reset_io", "width": None},
                {"name": "tl_slave_0_a_ready", "width": None},
                {"name": "tl_slave_0_d_valid", "width": None},
                {"name": "tl_slave_0_d_bits_data", "width": "[63:0]"},
                {"name": "tl_slave_0_d_bits_source", "width": "[3:0]"},
                {"name": "tl_slave_0_d_bits_size", "width": "[2:0]"},
                {"name": "tl_slave_0_d_bits_opcode", "width": "[2:0]"},
            ]
            outputs = [
                {"name": "tl_slave_0_d_ready", "width": None},
                {"name": "tl_slave_0_a_bits_source", "width": "[3:0]"},
                {"name": "tl_slave_0_a_bits_size", "width": "[2:0]"},
                {"name": "uart_0_txd", "width": None},
            ]
            golden_info = {
                "module_name": "ChipTop",
                "ports": {"inputs": inputs, "outputs": outputs, "inouts": []},
            }
            revised_info = {
                "module_name": "ChipTop",
                # Vivado is free to reorder top-level declarations between
                # golden and revised funcsim exports. Name/width equivalence is
                # what matters because generated connections are named.
                "ports": {
                    "inputs": list(reversed(inputs)),
                    "outputs": list(reversed(outputs)),
                    "inouts": [],
                },
            }
            golden_tb = self.workspace / "golden_trace.v"
            revised_tb = self.workspace / "revised_trace.v"
            trace = self.workspace / "trace.txt"
            validator.generate_boom_trace_testbenches(
                golden_info, revised_info, golden_tb, revised_tb, trace,
                clock_names=["clock_uncore_clock"])
            golden_text = golden_tb.read_text()
            revised_text = revised_tb.read_text()

            self.assertIn("module golden_trace_testbench;", golden_text)
            self.assertIn("ChipTop golden_dut", golden_text)
            self.assertNotIn("revised_dut", golden_text)
            self.assertIn('$fwrite(trace_fd, " %h", tl_slave_0_d_valid);',
                          golden_text)
            self.assertIn('$fwrite(trace_fd, " %h", golden_uart_0_txd);',
                          golden_text)
            self.assertIn('if (!$value$plusargs("NUM_VECTORS=%d", num_vectors))',
                          golden_text)
            self.assertIn("total_cycles = 50 + num_vectors;", golden_text)

            self.assertIn("module revised_trace_testbench;", revised_text)
            self.assertIn("ChipTop revised_dut", revised_text)
            self.assertNotIn("golden_dut", revised_text)
            self.assertIn('$fscanf(trace_fd, " %h", tl_slave_0_d_valid);',
                          revised_text)
            self.assertIn(
                "MISMATCH AT cycle %0d: uart_0_txd golden=%h revised=%h",
                revised_text)
            self.assertIn('if (!$value$plusargs("NUM_VECTORS=%d", num_vectors))',
                          revised_text)
            self.assertIn("total_cycles = 50 + num_vectors;", revised_text)
        finally:
            shutil.rmtree(validator.temp_dir, ignore_errors=True)

    def test_tilelink_a_prefix_derived_from_suffix_not_first_occurrence(self):
        # A prefix like tile_dcache_d must become tile_dcache_a, not tile_acache_d.
        # The old replace('_d', '_a', 1) would corrupt the first _d in the prefix.
        tb = self._generate(
            inputs=[
                {"name": "clk", "width": None},
                {"name": "tile_dcache_d_valid", "width": None},
                {"name": "tile_dcache_d_bits_data", "width": "[63:0]"},
                {"name": "tile_dcache_d_bits_source", "width": "[3:0]"},
                {"name": "tile_dcache_d_bits_size", "width": "[2:0]"},
            ],
            outputs=[
                {"name": "tile_dcache_a_valid", "width": None},
                {"name": "tile_dcache_d_ready", "width": None},
                {"name": "tile_dcache_a_bits_source", "width": "[3:0]"},
                {"name": "tile_dcache_a_bits_size", "width": "[2:0]"},
            ],
            verilog_text="""
module BoomCore(input clk);
  DSP48E2 #(.PREG(0)) dsp0 (.CLK(clk), .CEP(1'b1));
  wire d_bits_data;
endmodule
""",
        )

        # Source and size echoes must reference the correct a-channel outputs.
        self.assertIn("golden_tile_dcache_a_bits_source", tb)
        self.assertIn("golden_tile_dcache_a_bits_size", tb)
        # The corrupted prefix must not appear anywhere.
        self.assertNotIn("tile_acache", tb)


if __name__ == "__main__":
    unittest.main()
