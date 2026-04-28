## GAT-ZKML Strict Full-Dataset Comparison

This checkout contains the final ezkl baseline used for the GAT-ZKML comparison. The only final benchmark route is:

```text
ezkl_gat_native_compressed_csr_attention_full
```

The route proves the full native GAT forward relation over the full datasets `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. Final logits remain public instance cells and are bound to in-circuit derivation. No output-binding replacement, digest, folding, identity, subset, shard, sample, partial, scaled graph, or precomputed-logits route is part of the final benchmark.

### Build

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTC=$PWD/.rustup/toolchains/nightly-2025-12-01-aarch64-unknown-linux-gnu/bin/rustc \
RUSTDOC=$PWD/.rustup/toolchains/nightly-2025-12-01-aarch64-unknown-linux-gnu/bin/rustdoc \
$PWD/.rustup/toolchains/nightly-2025-12-01-aarch64-unknown-linux-gnu/bin/cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j
```

The local `.rustup/` and `.cargo_local/` directories are ignored toolchain/cache material. They are not final experiment artifacts.

### Run

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

Large proof/key/SRS/witness/compiled-circuit payloads under `runs/` are reproducible and intentionally ignored. The repository keeps the small final audit artifacts needed to review strictness, table policy, public-instance binding, memory profile, and metrics.

### Final Results

| Dataset | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

Primary result files:

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`
- `results/control_variable_audit.md`
- `results/large_public_output_policy_review.md`
- `results/native_full_experiment_report.md`

Per-dataset audit artifacts are kept in `runs/<dataset>/`: `run_manifest.json`, `model_info.json`, `graph_attention_spec.json`, `final_logits_binding.json`, `formal_tables.json`, `formal_table_hashes.json`, `native_full_witness_preflight.json`, `keygen_memory_profile.json`, and `prove_memory_profile.json`.

### Necessary Adaptations

- Deterministic inverse table range policy: `degree_bound_exp_bound_sum_scale_v1`, derived from public graph/quant/table metadata with `witness_specific_table=false`.
- Shape-only setup and chunked public instance columns for large public outputs.
- Tagged formal lookup table columns for inverse, LReLU, Exp, and ELU.
- Prove-time payload deduplication and low-memory environment settings for large full datasets.
- Large full-dataset diagnostic MockProver precheck is skipped; strict KZG proof and verify remain required.

<h1 align="center">
	<br>
	 :thought_balloon:
	<br>
	<br>
EZKL
	<br>
	<br>
	<br>
</h1>

> Easy Zero-Knowledge Inference

[![Test](https://github.com/zkonduit/ezkl/workflows/Rust/badge.svg)](https://github.com/zkonduit/ezkl/actions?query=workflow%3ARust)

`ezkl` is a library and command-line tool for doing inference for deep learning models and other computational graphs in a zk-snark (ZKML). It enables the following workflow:

1. Define a computational graph, for instance a neural network (but really any arbitrary set of operations), as you would normally in pytorch or tensorflow.
2. Export the final graph of operations as an [.onnx](https://onnx.ai/) file and some sample inputs to a `.json` file.
3. Point `ezkl` to the `.onnx` and `.json` files to generate a ZK-SNARK circuit with which you can prove statements such as:

> "I ran this publicly available neural network on some private data and it produced this output"

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_network_output.ipynb) 

> "I ran my private neural network on some public data and it produced this output"

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_input_output.ipynb) 

> "I correctly ran this publicly available neural network on some public data and it produced this output"

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_all_public.ipynb) 

In the backend we use the collaboratively-developed [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.

The generated proofs can then be verified with much less computational resources, including on-chain (with the Ethereum Virtual Machine), in a browser, or on a device. 

- If you have any questions, we'd love for you to open up a discussion topic in [Discussions](https://github.com/zkonduit/ezkl/discussions). Alternatively, you can join the ✨[EZKL Community Telegram Group](https://t.me/+QRzaRvTPIthlYWMx)💫.

- For more technical writeups and details check out our [blog](https://blog.ezkl.xyz/).

- To see what you can build with ezkl, check out [cryptoidol.tech](https://cryptoidol.tech/) where ezkl is used to create an AI that judges your singing ... forever.

----------------------

### Getting Started ⚙️

The easiest way to get started is to try out a notebook. 

#### Python
Install the python bindings by calling.

```bash
pip install ezkl
```
Or for the GPU:

```bash
pip install ezkl-gpu
```

Google Colab Example to learn how you can train a neural net and deploy an inference verifier onchain for use in other smart contracts. [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/ezkl_demo.ipynb) 


More notebook tutorials can be found within `examples/notebooks`.

#### CLI
Install the CLI
``` shell
curl https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash
```

https://user-images.githubusercontent.com/45801863/236771676-5bbbbfd1-ba6f-418a-902e-20738ce0e9f0.mp4

For more details visit the [docs](https://docs.ezkl.xyz). The CLI is faster than Python, as it has less overhead. For even more speed and convenience, check out the [remote proving service](https://ei40vx5x6j0.typeform.com/to/sFv1oxvb), which feels like the CLI but is backed by a tuned cluster.

Build the auto-generated rust documentation and open the docs in your browser locally. `cargo doc --open`

### Building the Project 🔨

#### Rust CLI

You can install the library from source

```bash
cargo install --locked --path .
```

`ezkl` now auto-manages solc installation for you.




#### Building Python Bindings
Python bindings exists and can be built using `maturin`. You will need `rust` and `cargo` to be installed.

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop --release --features python-bindings
# dependencies specific to tutorials
pip install torch pandas numpy seaborn jupyter onnx kaggle py-solc-x web3 librosa tensorflow keras tf2onnx
```

### GPU Acceleration

If you have access to NVIDIA GPUs, you can enable acceleration by building with the feature `icicle` and setting the following environment variable:

```sh
export ENABLE_ICICLE_GPU=true
```

GPU acceleration is provided by [Icicle](https://github.com/ingonyama-zk/icicle)

To go back to running with CPU, the previous environment variable must be **unset** instead of being switch to a value of false:

```sh
unset ENABLE_ICICLE_GPU
```

**NOTE:** Even with the above environment variable set, icicle is disabled for circuits where k <= 8. To change the value of `k` where icicle is enabled, you can set the environment variable `ICICLE_SMALL_K`.

### Contributing 🌎

If you're interested in contributing and are unsure where to start, reach out to one of the maintainers:

* dante (alexander-camuto)
* jason (jasonmorton)

More broadly:

- See currently open issues for ideas on how to contribute.

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention.

- To report bugs or request new features [create a new issue within Issues](https://github.com/zkonduit/ezkl/issues) to inform the greater community.


Any contribution intentionally submitted for inclusion in the work by you shall be licensed to Zkonduit Inc. under the terms and conditions specified in the [CLA](https://github.com/zkonduit/ezkl/blob/main/cla.md), which you agree to by intentionally submitting a contribution. In particular, you have the right to submit the contribution and we can distribute it, among other terms and conditions. 


### Audits & Security

[v21.0.0](https://github.com/zkonduit/ezkl/releases/tag/v21.0.0) has been audited by Trail of Bits, the report can be found [here](https://github.com/trailofbits/publications/blob/master/reviews/2025-03-zkonduit-ezkl-securityreview.pdf).

> NOTE: Because operations are quantized when they are converted from an onnx file to a zk-circuit, outputs in python and ezkl may differ slightly.


Check out `docs/advanced_security` for more advanced information on potential threat vectors that are specific to zero-knowledge inference, quantization, and to machine learning models generally.


### No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2026 Zkonduit Inc. 
