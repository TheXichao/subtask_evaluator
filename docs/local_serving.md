# Standing up the evaluator locally (and the general recipe)

How Qwen3-VL-4B got served on this machine (2026-08-12), written as the
reusable playbook: the same steps apply to any open VLM on any box.

## The four separate pieces

```
weights        Hugging Face Hub → ~/.cache/huggingface/hub/   (tool-agnostic)
loader         transformers — resolves a repo id against that cache
harness        LLaMA-Factory / vLLM / llama.cpp — template + preprocessing + HTTP
client         anything speaking OpenAI chat-completions to http://host:port/v1
```

Pick the harness by purpose: vLLM for throughput and faithful model ids,
LLaMA-Factory when the same install will also fine-tune (identical template
and image preprocessing at inference and training), llama.cpp/ollama for
CPU/quantized. Weights and client code never change across that choice.

## 1. Size the hardware

bf16 ≈ 2 bytes/param: a 4B model ≈ 9 GB VRAM plus KV-cache headroom.
`nvidia-smi` for free (not just total) VRAM; disk ≈ 1.2× model size.

## 2. Verify the environment before trusting it

```bash
which <cli>                       # on PATH at all?
<env>/bin/pip list | grep <pkg>   # a PATH next to the version = editable install
<env>/bin/python -c "import <pkg>; import torch; print(torch.cuda.is_available())"
```

An editable install (`pip install -e`) is only a pointer to a source
checkout; delete the checkout and the env silently breaks (exactly what had
happened here). Repair: re-clone, `pip install -e . --no-deps` — never let a
repair re-resolve a working torch/transformers stack.

## 3. Download and VERIFY the weights

```bash
env -u all_proxy -u ALL_PROXY hf download <org>/<model>
```

- `socks://` proxy URLs break httpx-based tools (curl tolerates them); strip
  `all_proxy` per-command with `env -u` rather than editing shell config.
- Exit 0 is not verification. Check every shard listed in
  `model.safetensors.index.json` exists in the snapshot dir, and compare
  `metadata.total_size` with reality.
- The cache is machine-wide and shared by every harness.

## 4. The three load-bearing serving choices

```yaml
# ~/Dev/LLaMA-Factory/qwen3vl_4b_api.yaml (this machine's actual config)
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
template: qwen3_vl_nothink        # 1. wrong template = silent quality loss
infer_backend: huggingface
trust_remote_code: true
image_max_pixels: 1048576         # 2. must exceed your largest input image
```

```bash
API_PORT=9040 API_MODEL_NAME=Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 \
  ~/miniconda3/envs/llama_factory/bin/llamafactory-cli api \
  ~/Dev/LLaMA-Factory/qwen3vl_4b_api.yaml
```

1. **Template**: the harness's recipe for the model family's special-token
   layout. Always use the one the harness's own examples pair with the model.
2. **Pixel budget**: harnesses preprocess images before the model; defaults
   (LF: 768², its Qwen3-VL example: 512²) silently downscale larger inputs.
   Know your input resolution (our grids: 1344×756 = 1,016,064 px,
   `manifest.max_grid_pixels`) and set the knob above it. vLLM instead uses
   the model's own processor config (16 MP for Qwen3-VL) — no knob needed.
3. **Identity**: pick a non-default port deliberately (9040 here; 9020/9030
   mean other things on the team server) and set `API_MODEL_NAME` so
   `/v1/models` reports the truth (LF otherwise says `gpt-3.5-turbo`).
   `HF_HUB_OFFLINE=1` = cache only, no network surprises at load time.

## 5. Smoke-test in layers

```bash
curl -s http://127.0.0.1:9040/v1/models          # liveness (poll bounded, ~6 min)
```

then one real request with `temperature: 0` (reproducible baselines) — and
**read `usage` in the response**: expected visual tokens for Qwen3-VL are
`(W/32)·(H/32)` per image (16 px patches, 2×2 merge). If `prompt_tokens` is
far below that, something downscaled your image. Note the transport
difference from training data: over the API, images are base64
`image_url` content parts and `<image>` placeholder tags must be stripped
from the text.

## 6. Record the result where code will look for it

`model_services.json` → `evaluator.endpoint = "http://127.0.0.1:9040/v1"`,
with the launch command in `notes`. Remote later = same command on the GPU
server + `ssh -N -L 9040:127.0.0.1:9040 <host>`; the endpoint URL — and
therefore all client code — does not change.

Stop the server: `pkill -f llamafactory-cli`.
