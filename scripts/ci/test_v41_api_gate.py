#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("gate",Path(__file__).with_name("v41_api_gate.py"))
gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)

class TestV41Gate(unittest.TestCase):
    def test_png(self):
        self.assertEqual(gate.png((255,0,0))[:8],b"\x89PNG\r\n\x1a\n")
    def test_image_identities(self):
        cases=gate.cases("big")
        self.assertGreaterEqual(len(cases), 28)
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertGreaterEqual(
            sum(case["id"].startswith("v") for case in cases), 8
        )
        self.assertEqual(gate.cases("big", 128)[0]["request"]["max_tokens"], 128)
        a=next(c for c in cases if c["id"]=="v1")
        b=next(c for c in cases if c["id"]=="v2")
        self.assertNotEqual(gate.digest(a["request"]),gate.digest(b["request"]))
    def test_tool_normalization(self):
        x=gate.normalized({"choices":[{"message":{"tool_calls":[{"id":"random","function":{"name":"tool","arguments":"{}"}}]},"finish_reason":"tool_calls"}]})
        self.assertEqual(x["tool_calls"],[{"name":"tool","arguments":{}}])
    def test_comparison(self):
        a={"schema":"dfm-v41-parity/1","model":"big","hashes":{"x":"1"},"results":[{"id":"x","output":{"content":"hello"}}]}
        self.assertEqual(gate.compare(a,a),[])
        b={**a,"results":[{"id":"x","output":{"content":"other"}}]}
        self.assertEqual(gate.compare(a,b),["Output mismatch: x"])
    def test_invalid_output(self):
        with self.assertRaises(RuntimeError): gate.normalized({"choices":[]})

if __name__=="__main__": unittest.main()
