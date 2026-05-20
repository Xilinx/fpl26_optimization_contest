#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache 2.0

import shutil
import tempfile
import unittest
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

    def _generate(self, inputs, outputs, no_reactive=False):
        validator = DCPValidator(
            self.workspace / "golden.dcp",
            self.workspace / "revised.dcp",
            num_vectors=8,
            no_reactive=no_reactive,
        )
        try:
            tb_path = self.workspace / "testbench.v"
            info = {
                "module_name": "dut",
                "ports": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "inouts": [],
                },
            }
            validator.generate_testbench(
                info,
                info,
                tb_path,
                clock_names=[inputs[0]["name"]],
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


if __name__ == "__main__":
    unittest.main()
