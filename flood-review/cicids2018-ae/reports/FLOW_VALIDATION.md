# Checking the flow builder against CICFlowMeter

Question asked: the dashboard and the live capture turn packets into flow features with code in `app/capture.py`. The models were trained on records made by CICFlowMeter. Do the two agree, and what happens on ordinary traffic?

**Answer in brief.** The flow builder is a port of CICFlowMeter's flow logic, including its quirks. On 7 captures (102,551 packets, 4,701 flows, 1,149 of them real Wi-Fi traffic) every flow has the same 69 feature values as CICFlowMeter's own flow code, and no flow exists on only one side. The parsed packet fields equal Wireshark's on 98,552 IPv4 packets. 26 row-level properties that follow from the original tool's definitions hold for 1.5 million training rows (lowest share 0.9994) and for every flow the port builds (lowest share 1.0000). On 1,149 flows of ordinary web, DNS and ping traffic the supervised model flagged 2.3%, the one-class model 4.4% and the unsupervised model 69.2%. What this does **not** show: that flags are real attacks. No attack traffic was captured, and the models come from one 2018 lab network.

## 1. What the reference is, and what the training data forced

The reference is the flow code of CICFlowMeter (github.com/ahlashkari/CICFlowMeter), run on the same packets as this project's builder. Only the packet reader of that tool depends on a native library (jnetpcap); the flow classes (`BasicFlow`, `FlowGenerator`) are plain Java. A small driver (`tools/cicflowmeter_ref/RefDriver.java`) reads a pcap and fills the packet fields the way the tool's `PacketReader` does, then feeds the real classes. The source is downloaded when you build the reference (`python tools/cicflowmeter_ref/build_reference.py`) and is not redistributed in this project.

**Which version.** The repository history starts on 2018-02-12, during the CSE-CIC-IDS2018 recording. Commit `eaa853d` (2018-02-13) was chosen because it agrees with the training data on every point tested, and it is used **unmodified**. Later versions do not agree: the 2019-and-later versions count flags over all packets, and the June 2018 commit `6833ccd` counts the first packet as backward (it compares byte arrays by reference after `getSrc()` started returning copies), which the data rules out because no training row has zero forward packets. The exact build that made the CSVs is not published, so this is the closest version, not a proven identity. (An earlier build of the reference carried two patches for the subflow and active-time code; running the source with and without them gave byte-identical output on every capture, so they were removed. The subflow columns equal the flow totals in the unmodified source because its subflow detector runs only for the first packet, and the training data shows the same.)

**What the training data showed** (each was measured on the cleaned training rows, and the unmodified source agrees with each one; `tools/validate_flow_definitions.py`):

| Finding | Evidence in the training data |
|---|---|
| A flow ends at the first FIN (any direction) or when a packet arrives more than 120 s after the flow started; RST does not end a flow | maximum duration 120.0 s; TCP flows that start with a FIN are 6.0% of TCP flows and 90.8% of them have exactly 2 packets |
| A flow with a single packet is never reported | no row has fewer than 2 packets |
| The first packet is counted twice in the all-packet length statistics | mean x (n + 1) - total payload is an integer between 0 and the largest packet in 99.96% of rows (it is the first packet's payload) |
| Flag counts come from the first packet only, and the eight columns hold the flags in the order of a Java HashMap: the column named FIN holds RST, SYN holds PSH, RST holds ECE, PSH holds SYN, ACK holds ACK, URG holds FIN, CWE holds URG, ECE holds CWR | the "SYN" column equals Fwd PSH Flags in 100% of rows; "CWE" equals Fwd URG Flags in 100%; when the "RST" column is 1 the "PSH" column is 1 in 100% of rows (an ECN-setup SYN carries ECE and CWR); "PSH" + "ACK" is 1 in 99.6% of TCP flows (the first packet is a SYN or an ACK) |
| Header length is the transport header (TCP 20 to 60 bytes, UDP 8), not IP + TCP | forward header per packet: TCP quantiles 20 / 21.3 / 34, UDP exactly 8 |
| Payload comes from the IP length and Ethernet padding is not counted | 99.6% of TCP flows have a smallest forward payload of 0 (a padded 60-byte frame would show 6) |
| Init Win Bytes Backward is the window of the last backward packet; both init windows are -1 when there is no such TCP packet (UDP and other protocols: always -1) | most common backward values are small scaled windows (211, 219, 149); the most common forward value is 8192 (a SYN window) |
| Fwd Act Data Pkts does not count the first packet | UDP flows with one forward data packet show 0 in all 274,798 sampled cases; the value never exceeds forward packets - 1 |
| Active and Idle use a 5 s gap; the last active period is never recorded | smallest idle gap 5.000001 s; an active period never appears without an idle gap (0 cases) |
| IP packets that are neither TCP nor UDP form flows of protocol 0 with ports 0, no payload, no header bytes | 1.3% of training rows have protocol 0; all sampled ones have zero payload and header |
| The subflow columns equal the flow totals | 100% of sampled rows (also what the unmodified source gives) |
| Both addresses equal: one flow per direction | follows the source (`generateFlowId`); not observable in the lab data, kept because it is the tool's behaviour |

## 2. Results

### 2.1 Same packets, same flows, same features

`tools/run_flow_validation.py` runs both builders on each capture, matches flows by start time (microseconds), addresses, ports and protocol, and compares the 68 model features plus the protocol (69 values per flow; relative tolerance 1e-9). Zero-duration flows (two packets in one microsecond) give an infinite rate in the original tool; those two rate columns are excluded for such flows (311 flows), and the port reports 0.

| Capture | Kind | Packets | Flows | Flows with all 69 values equal | IPv4 packets checked against Wireshark | Mismatches |
|---|---|---|---|---|---|---|
| ordinary_wifi.pcap | ordinary traffic (real Wi-Fi capture) | 8,990 | 315 | 315 | 7,213 | 0 |
| ordinary_wifi2.pcap | ordinary traffic (real Wi-Fi capture) | 57,164 | 562 | 562 | 54,944 | 0 |
| ordinary_live.pcap | ordinary traffic (real Wi-Fi capture) | 12,377 | 272 | 272 | 12,377 | 0 |
| demo_synthetic.pcap | synthetic demo capture | 383 | 56 | 56 | 381 | 0 |
| stress_seed1.pcap | randomised stress capture | 7,857 | 1,163 | 1,163 | 7,857 | 0 |
| stress_seed7.pcap | randomised stress capture | 7,877 | 1,174 | 1,174 | 7,877 | 0 |
| stress_seed21.pcap | randomised stress capture | 7,903 | 1,159 | 1,159 | 7,903 | 0 |
| **Total** | | **102,551** | **4,701** | **4,701** | **98,552** | **0** |

The stress captures (`tools/make_stress_pcap.py`, random seeds) contain TCP sessions closed by FIN from either side, closed by RST, left open, longer than the 120 s timeout, with idle gaps over 5 s, port reuse, UDP request/response and one-way, ICMP and other IP protocols, equal source and destination addresses, and packets sharing one microsecond. Both tools produce the same flows on every one of them. The ordinary captures were recorded on the Wi-Fi adapter with Wireshark's `dumpcap` (headers only, 96 bytes per packet; 70 s and 90 s) while loading about 35 public web pages, a few downloads, DNS lookups and pings; a third was taken with the dashboard's own Live capture tab on the same adapter (about 70 s, 16 page loads). All three also contain whatever else the computer sent in those minutes. The live capture's 272 flows are exactly the flows that the offline analysis of its saved headers-only file gives. The captures are not part of the report folder (the live one is in `captures/` on this computer and can be deleted).

Limit of this comparison: the reference driver is my reading of what the tool's packet reader does, not jnetpcap itself (which cannot be loaded here), so agreement on packet-level fields rests on the next two checks.

### 2.2 Packet fields against Wireshark

For every IPv4 packet the addresses, protocol, ports, TCP payload (`tcp.len`), TCP header length, flags, window and UDP payload from `packet_fields()` were compared with tshark's own dissection (`tools/check_packet_fields.py`): 98,552 packets, 0 differences.

### 2.3 Row-level properties of the definitions, on the training data and on the port

Each property below follows from one definition of the original tool. The first column of numbers is the share of the 1,500,000 sampled training rows (150,000 random rows per day file) that satisfy it; the second is the share of the 1149 flows built from the ordinary captures. A share below 1.0000 on the training data is real and reported as measured (for example a few rows where float32 storage or a rare packet breaks the identity).

| Property | Training data | Port on ordinary traffic |
|---|---|---|
| flow has at least 2 packets and at least 1 forward packet | 1.0000 | 1.0000 |
| flow duration is at most 120 s | 1.0000 | 1.0000 |
| first packet counted twice: mean*(n+1) - total payload is an integer >= 0 and <= max packet length | 0.9996 | 1.0000 |
| Packet Size Avg = (total payload + first payload) / n | 0.9994 | 1.0000 |
| Packet Length Var = Packet Length Std squared | 1.0000 | 1.0000 |
| Packet Length Min/Max = min/max of the forward and backward extremes | 1.0000 | 1.0000 |
| Down/Up Ratio = integer part of bwd/fwd packets | 1.0000 | 1.0000 |
| subflow columns equal the flow totals | 1.0000 | 1.0000 |
| Fwd/Bwd segment size avg = mean forward/backward payload | 1.0000 | 1.0000 |
| forward mean * forward packets = total forward payload | 1.0000 | 1.0000 |
| Flow IAT mean * (n-1) = flow duration | 1.0000 | 1.0000 |
| Flow Bytes/s = total payload / duration (duration > 0) | 1.0000 | 1.0000 |
| Flow Packets/s = packets / duration (duration > 0) | 1.0000 | 1.0000 |
| the eight flag columns are 0 or 1 | 1.0000 | 1.0000 |
| 'SYN' column = Fwd PSH Flags (HashMap column order, first packet only) | 1.0000 | 1.0000 |
| 'CWE' column = Fwd URG Flags | 1.0000 | 1.0000 |
| 'RST' column = 'ECE' column (ECN-setup SYN carries both) | 1.0000 | 1.0000 |
| UDP and other-IP flows have all flag columns 0 and both init windows -1 | 1.0000 | 1.0000 |
| TCP flows have an init forward window >= 0 | 1.0000 | 1.0000 |
| active data packets <= forward packets - 1 (the first packet is not counted) | 1.0000 | 1.0000 |
| header length is the transport header: UDP forward header = 8 per packet | 1.0000 | 1.0000 |
| TCP header per forward packet is 20 to 60 bytes | 1.0000 | 1.0000 |
| TCP minimum forward header is 20 to 60 and a multiple of 4 | 1.0000 | 1.0000 |
| protocol-0 flows have no payload, no header bytes, windows -1 | 1.0000 | 1.0000 |
| an active period is recorded only together with an idle gap | 1.0000 | 1.0000 |
| every idle gap is longer than 5 s | 1.0000 | 1.0000 |

Statistical regularities (not required to be 1; they depend on the network):

| Regularity | Training data | Port on ordinary traffic |
|---|---|---|
| TCP flows whose first packet is SYN or ACK (the 'PSH' and 'ACK' columns add to 1) | 0.996 | 0.998 |
| TCP flows that start with a FIN (the 'URG' column), all of which have exactly 2 packets | 0.908 | 0.882 |
| share of TCP flows that start with a FIN | 0.060 | 0.116 |
| TCP flows that contain a payload-free packet (Pkt Len Min = 0) | 0.999 | 1.000 |
| share of flows with protocol 0 | 0.013 | 0.003 |

### 2.4 Ordinary traffic: how often does each model flag it?

Ordinary browsing, DNS and ping traffic contains very little that is malicious, so a model that flags much of it would produce false alarms on such a network. 1,149 flows from the ordinary captures were scored with the saved models at their fixed thresholds (`tools/ordinary_traffic_check.py`). The flows are **presumed** benign, not labelled. The last column is the false-alarm rate of the same model on benign flows of the held-out lab test data.

| Model | Flagged | Share [95% Wilson interval] | Lab test (benign flows) |
|---|---|---|---|
| Supervised AE (probability >= 0.5) | 27 of 1149 | 2.3% [1.6, 3.4] | 0.35% |
| Supervised AE + decoder (>= 0.5) | 20 of 1149 | 1.7% [1.1, 2.7] | 2.00% |
| One-class AE (benign-trained) | 50 of 1149 | 4.4% [3.3, 5.7] | 2.94% |
| Unsupervised AE (mixed training data) | 795 of 1149 | 69.2% [66.5, 71.8] | 7.70% |
| Supervised OR unsupervised | 803 of 1149 | 69.9% [67.2, 72.5] | n/a |

Only 5 of the 1,149 flows (0.4%) had any feature outside the training range, against 0.0005% of held-out lab flows: the port's flows look like the training data. Before the port was written, the first version of the flow builder (which used the rules of the CICFlowMeter documentation, not of the tool) gave on the first two ordinary captures 964 flows of which 43.9% had an out-of-range feature, and flagged 21.0% (supervised), 20.0% (supervised + decoder), 68.2% (one-class) and 100% (unsupervised): its features did not look like the training data, which is why this check mattered.

What the flagged flows look like (counted by port and shape, no addresses kept): 25 of the supervised model's 27 flags are one-way flows with no reply (7 to TCP 443, 6 to UDP 443 which is QUIC, 6 to multicast DNS, 4 to other TCP ports, and 4 other UDP flows); the decoder variant's 20 flags are all TCP, 9 of them without a reply; 45 of the one-class model's 50 flags are long multicast DNS flows (UDP 5353). These are plausible leads, not shown to be benign or malicious.

How to read it: the supervised model flags six times more of this traffic than of lab test flows (2.3% against 0.35%) but the share is still small; the decoder variant flags about the same share as in the lab (1.6% against 2.0%); the one-class model somewhat more (4.7% against 2.9%); and the unsupervised model does not transfer at all (70% against 7.7%). This agrees with the report's main result that the unsupervised model trained on mixed data ranks flows at chance level; it is a reason not to use it, or the combined detector that includes it, as a live alarm. With 1,149 flows from one network recorded over a few minutes these are indications, not stable rates.

## 3. What is not shown

* **Detection of real attacks on a live network.** No attack traffic was captured. The ordinary-traffic check measures false alarms only, on one Wi-Fi network over about four minutes; its flows are presumed benign.
* **jnetpcap itself.** The 2018 tool read packets with jnetpcap; this project uses scapy. The byte counting agrees with Wireshark and with the training data (payload and header definitions above), but jnetpcap was not run.
* **The exact build that made the CSVs.** It is not published. The reference is the closest source version, unmodified; every behaviour the training data can reveal agrees with it, and behaviours the data cannot reveal (for example the handling of a flow after its timeout, or of unusual packet orders) follow that source.
* **Zero-duration flows** report 0 for the two rate features; the original tool gives infinity and the training pipeline removed such rows.
* **IPv6** packets are ignored, as in the original tool's default.
* **Live behaviour under load.** The packet handling is pure Python; on a busy link packets can be dropped. Capture with Wireshark and analyse the file for heavy traffic. A flow in the live view is released at a FIN or when it is older than 120 s, which gives the same features as the end-of-file release.

## 4. Reproduce

```bash
python tools/cicflowmeter_ref/build_reference.py          # once: needs git, a JDK and internet
python tools/make_stress_pcap.py stress.pcap --seed 1
python tools/compare_cicflowmeter.py stress.pcap          # port vs CICFlowMeter's flow code, feature by feature
python tools/check_packet_fields.py stress.pcap           # packet fields vs tshark
python tools/validate_flow_definitions.py --training --pcap stress.pcap
python tools/run_flow_validation.py --ordinary ordinary1.pcap ordinary2.pcap
python tools/ordinary_traffic_check.py ordinary1.pcap ordinary2.pcap --json results/ordinary_traffic_check.json
python -m pytest tests/test_flow_parity.py -q
```

Result files: `results/flow_validation_parity.json`, `results/flow_validation_definitions.json`, `results/ordinary_traffic_check.json`. The dashboard reads them (`/api/flow-validation`).
