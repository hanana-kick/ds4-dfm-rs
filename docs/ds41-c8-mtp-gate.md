# DS41 C8 + MTP real-model qualification gate

`tests/ds41_c8_mtp_gate.sh` checks **actual packed GPU execution**, not
just a `--max-seqs 8` configuration. It uses a task-owned server on
`127.0.0.1:18045` (or `DS41_C8_PORT`) and stops only that server.

## Run on a single DGX Spark

```bash
git switch feat/ds41-entrpi-concurrency
git pull --ff-only
bash scripts/build.sh  # only needed if binary is missing or server code changed
bash tests/ds41_c8_mtp_gate.sh
```

The default five modes run **serially** with one model loaded at a time:

| Mode | Purpose |
|---|---|
| C1 MTP OFF | Greedy text and Vision reference |
| C4 MTP OFF | Four-way packed decode and C1 parity |
| C4 MTP ON | Four-way packed target verification and C1 parity |
| C8 MTP OFF | Eight-way packed decode and C1 parity |
| C8 MTP ON | Eight-way packed target verification and C1 parity |

Each width performs long, distinct text prompts (C8 has **8 concurrent
requests**). Mixed Vision uses two different images plus 6 independent
texts in C8 (2 plus 2 in C4), with C1 text/image parity. The DSML tool
step requires one `test_echo(value=ping)` invocation and a tool-result
roundtrip. High Thinking uses explicit **positive seed 424242**,
`temperature=1.0`, a default 8,192 output-token limit, and the committed
yellow-face PNG. In C8 it runs **alongside seven text requests**. High
Thinking must terminate with a nonempty final answer and without detected
100-character repetitions. Stochastic high-thinking text is not asserted
byte-identical across widths.

Greedy text can legitimately end at the fixed 512-token test budget; this
is a batch workload, not a requirement to finish 150 items. C1 greedy
**parity** and observed batch/MTP behavior are required. A batch capacity
failure, transport error, parity mismatch or unfinished/repeating high
Thinking is a **FAIL**, not a silent fallback to a narrower width.

For C8 the log gate requires:

```
[v41-batch] packed_decode rows=8 ... verify_rows=N  # N > 8 with MTP ON
[v41-mtp] request complete rounds=R offered=O ... # R,O > 0
```

A configured width or nonzero MTP tokens **alone** cannot pass. Absence of
`rows=8` means genuine eight-request execution was not observed. All
screenshots, responses and logs are kept; `summary.tsv`, `summary.jsonl`,
`parity.tsv` and `proofs.tsv` identify the observed results. This remains a
bounded test, not long-session, quality or numerical-logit qualification.

## Options

```bash
# Fastest focused qualification, still checks against a C1 baseline
DS41_C8_MODES='c1-off c8-on' bash tests/ds41_c8_mtp_gate.sh

# Reproduce the failing image; use a distinct second image
DS41_C8_IMAGE_A=/absolute/path/to/failing.png \
DS41_C8_IMAGE_B=/absolute/path/to/second.jpg \
bash tests/ds41_c8_mtp_gate.sh

# Isolate batch+MTP without Vision and high Thinking
DS41_C8_MEDIA=off DS41_C8_THINK=off \
DS41_C8_MODES='c1-off c8-on' bash tests/ds41_c8_mtp_gate.sh

# Bigger high-thinking allowance / request timeout
DS41_C8_THINK_TOKENS=16384 DS41_C8_REQUEST_TIMEOUT=3600 \
bash tests/ds41_c8_mtp_gate.sh
```

`DS41_C8_MODES` must start with `c1-off` and include `c8-on`. You can
add `c1-on` to test the serial MTP path. `DS41_C8_OUT` must be a fresh,
nonexistent directory. The script never downloads models, calls Python,
changes clocks or kills an unrelated model owner. Stop other GPU-resident
models before running; C8 admission can fail on insufficient headroom.

## Model-free verification

```bash
bash -n tests/ds41_c8_mtp_gate.sh
node --check tests/fixtures/ds41-c8/mock-server.cjs
bash tests/ds41_c8_mtp_gate.sh --self-test
bash tests/ds41_c8_mtp_mock.sh
```

The mock test checks successful C8 verification and **negative controls**:
8-row packing missing, MTP verification missing, or `seed=0` incorrectly
accepted. The mock synthesizes log messages, so a successful host CI check
**never means an actual C8+MTP GPU run passed**. A real GPU run is required.
