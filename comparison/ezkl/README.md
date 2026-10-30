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

### Formal GAT benchmark metrics

The local formal GAT runner writes the four core result fields in milliseconds and bytes. In the current formal SHPLONK path, `commitment_time_ms` measures phase-0 advice KZG commitments plus affine normalization inside native proof creation; `instance_commitment_count` is zero, so no public-instance KZG commitment is included. Public-instance values are absorbed into the proof transcript and their processing is part of the Prove core. `prove_time_ms` measures the native proof-creation residual plus authenticated proving-key materialization. `verify_time_ms` measures proof decoding and the native Halo2 verifier call. These core fields exclude formal command preparation such as input, resource and artifact authentication, public-instance reconstruction, SRS and verification-key loading, and configuration loading. See the linked native metric files in each dataset's `result.json` for the recorded operation and scope.

`prove_command_wall_ms` and `phase_timings_ms.verify` report the outer runner's full Prove command and Verify phase wall times. They include their respective command and preparation costs, so they answer a different question from the three core time fields. Read both sets when comparing end-to-end runtime with the native commitment, proof, or verifier work.

`proof_size_bytes` is the exact file size of that cell's `proof.json`. Formal GAT proofs use a compact JSON envelope containing the native Halo2 transcript as lowercase hexadecimal plus an `instances_ref` digest and count. The raw transcript byte count is `proof.json`'s `proof_transcript.byte_length`; its hex string occupies twice that many bytes. The external public statement and public output are separate files and are not counted in `proof_size_bytes`. The proof artifact does not serialize the full public-instance vector; Verify reconstructs and binds those instances to the external statement and output.

To run only the PPI and OGBN-Arxiv formal cells, start the detached baseline-local session with separate output locations:

```bash
python3 scripts/ezkl_formal_control.py start --datasets ppi ogbn_arxiv --output-root runs/formal_ppi_ogbn --matrix results/formal_ppi_ogbn_matrix.json
```

The selector limits formal Admission and the Setup, Prove, Verify, and result collection stages to those two datasets. Shared static resources are still prepared from the canonical five-dataset resource pack. Each selected proof still undergoes independent Verify and final validation, and reports the same four metrics above. Save the returned session ID to inspect that session; a successful detached launch only means it started, not that the experiment completed. A completed subset is reported as `SUBSET_COMPLETE` / `SUBSET_VALIDATION_OK`; it does not mean the five-dataset matrix is complete. Omitting `--datasets` keeps the default five-dataset run.

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

