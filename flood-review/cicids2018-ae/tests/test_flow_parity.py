"""The flow builder (app/capture.py) against the real CICFlowMeter flow code and against Wireshark's dissector.

Needs the reference build (python tools/cicflowmeter_ref/build_reference.py: git, a JDK and internet access, once); skipped when it is absent.
The packet-field check needs tshark. Random captures come from tools/make_stress_pcap.py (timeouts, FIN/RST endings, ICMP, port reuse, equal addresses).
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/cicflowmeter_ref"))

import build_reference  # noqa: E402
import make_stress_pcap  # noqa: E402
from scapy.utils import wrpcap  # noqa: E402

needs_reference = pytest.mark.skipif(not (build_reference.built() and shutil.which("java")), reason="CICFlowMeter reference not built (tools/cicflowmeter_ref/build_reference.py)")
TSHARK = shutil.which("tshark") or (r"C:\Program Files\Wireshark\tshark.exe" if Path(r"C:\Program Files\Wireshark\tshark.exe").exists() else None)


@pytest.fixture(scope="module", params=[11, 12, 13])
def stress_pcap(request, tmp_path_factory):
    p = tmp_path_factory.mktemp("stress") / f"s{request.param}.pcap"
    wrpcap(str(p), make_stress_pcap.build(seed=request.param, sessions=250))
    return p


@needs_reference
def test_every_feature_of_every_flow_equals_cicflowmeter(stress_pcap):
    import compare_cicflowmeter as cmp
    res = cmp.compare(stress_pcap, cmp.model_columns())
    assert res["flows_matched"] > 200
    assert res["flows_only_in_reference"] == 0 and res["flows_only_in_this_project"] == 0
    assert res["features_that_differ"] == {}, res["examples"]
    assert res["flows_with_all_features_equal"] == res["flows_matched"]


@pytest.mark.skipif(not TSHARK, reason="tshark not installed")
def test_packet_fields_equal_wireshark(stress_pcap):
    import check_packet_fields as chk
    res = chk.check(str(stress_pcap))
    assert res["ipv4_packets_checked"] > 1000 and res["mismatching_packets"] == 0, res["examples"]


def test_definitions_hold_on_built_flows(stress_pcap):
    """The row-level properties of the original tool's definitions (tools/validate_flow_definitions.py) hold for every flow built."""
    import validate_flow_definitions as val
    cols, n = val.pcap_columns([stress_pcap])
    assert n > 200
    bad = {k: v for k, v in val.run(cols)["properties"].items() if v < 1.0}
    assert not bad, bad
