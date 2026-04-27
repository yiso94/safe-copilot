# Qwen

> [!WARNING]
> The `convert_checkpoint.py` / `build_trtllm.py` workflow described below is
> **legacy** and will not receive new features. New projects should use
> [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
> or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html)
> instead.

This directory contains a small, review-friendly TensorRT-LLM flow for the
Qwen language model only.

It is designed for two cases:

1. You already have a standalone Hugging Face Qwen causal LM directory.
2. You have a multimodal checkpoint such as ReCogDrive VLM, and you want to
   extract just its `language_model` and build a TensorRT-LLM engine for that
   Qwen submodel.

## Overview

There are two main scripts in this folder:

- [convert_checkpoint.py](./convert_checkpoint.py)
  Converts a Hugging Face Qwen model into a TensorRT-LLM checkpoint. If the
  source model is multimodal, it first extracts the Qwen `language_model` into
  a standalone Hugging Face directory.
- [build_trtllm.py](./build_trtllm.py)
  Builds TensorRT-LLM engine files from that TensorRT-LLM checkpoint. It can
  also build a hidden-state engine that exposes `last_hidden_state_output`.

By default, generated artifacts are stored under:

```text
/workspaces/safe-copilot/models/qwen
```

The default layout is:

```text
models/qwen/
├── hf_model/
├── hidden_state_engine/
├── tllm_checkpoint/
└── engine/
```

You can override the root with `QWEN_MODEL_OUTPUT_ROOT`.

`hf_model/` is mainly used when the source model is multimodal and the script
extracts `language_model` for you. If your source is already a standalone Qwen
Hugging Face directory, you can continue to use that original directory for
hidden-state builds.

## Usage

### 1. Convert To TensorRT-LLM Checkpoint

If your source is already a standalone Qwen Hugging Face model:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir /path/to/qwen_hf_model
```

In that case, `build_trtllm.py --engine_kind hidden_state` should usually use
the same `/path/to/qwen_hf_model` as `--hf_model_dir`.

If your source is a multimodal model that exposes `language_model`, the same
command works and the script will first extract the Qwen submodel into
`models/qwen/hf_model`:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir /path/to/multimodal_model
```

For the ReCogDrive language submodel that was already extracted under this
repository, a typical command is:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir /workspaces/safe-copilot/models/recogdrive/qwen2_submodel
```

Common options:

- `--output_dir`: where to save the TensorRT-LLM checkpoint
- `--hf_output_dir`: where to save the extracted Hugging Face Qwen submodel
- `--dtype`: checkpoint dtype, default `auto`
- `--tp_size`: tensor parallel size
- `--pp_size`: pipeline parallel size

### 2. Build TensorRT-LLM Engine

Build engine files from the TensorRT-LLM checkpoint:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --checkpoint_dir /workspaces/safe-copilot/models/qwen/tllm_checkpoint
```

A slightly more explicit example:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --checkpoint_dir /workspaces/safe-copilot/models/qwen/tllm_checkpoint \
  --output_dir /workspaces/safe-copilot/models/qwen/engine \
  --max_batch_size 1 \
  --max_input_len 1024 \
  --max_output_len 128
```

Common options:

- `--output_dir`: where to save the engine directory
- `--max_batch_size`: maximum batch size
- `--max_input_len`: maximum prompt length
- `--max_output_len`: maximum generated length
- `--max_seq_len`: total sequence length override
- `--workers`: number of parallel build workers

### 3. Build Hidden-State Engine

If you want an engine that exposes the last hidden state, build in
`hidden_state` mode from the Hugging Face Qwen directory:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/models/qwen/hf_model \
  --max_input_len 1024
```

If you want to build the hidden-state engine in `bfloat16`, pass
`--dtype bfloat16` explicitly:

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/models/qwen/hf_model \
  --output_dir /workspaces/safe-copilot/models/qwen \
  --max_input_len 2800 \
  --dtype bfloat16
```

This creates:

```text
/workspaces/safe-copilot/models/qwen/config.json
/workspaces/safe-copilot/models/qwen/rank0.engine
```

The `--dtype` option currently affects the hidden-state build path. The
standard generation-engine path still follows the TensorRT-LLM checkpoint
configuration.

If `--output_dir` is omitted in hidden-state mode, it now defaults to:

```text
/workspaces/safe-copilot/models/qwen/hidden_state_engine
```

This hidden-state engine is built with the same TensorRT-LLM pattern used by
the ReCogDrive hidden-state helper: it marks `last_hidden_state_output` as an
explicit engine output.

In this environment, the hidden-state build defaults also keep
`--remove_input_padding` enabled and `--gpt_attention_plugin auto`. This is the
stable build path for the Qwen hidden-state engine here.

Because that engine works on packed prompt tokens, parity checks should compare
the valid prompt-token region rather than a left-padded `max_length` tensor.

## 日本語メモ: `trt_hidden_state_engine.py` の使い方

[`trt_hidden_state_engine.py`](./trt_hidden_state_engine.py) は CLI スクリプトではなく、
Python から呼ぶための helper module です。役割は大きく 2 つです。

1. Hugging Face 形式の Qwen モデルから、`last_hidden_state_output` を返す
   hidden-state engine を build する
2. その engine を実行して、TensorRT-LLM の `last hidden state` を
   `torch.Tensor` として取得する

通常はこのファイルを直接実行するのではなく、
[`build_trtllm.py`](./build_trtllm.py) の `--engine_kind hidden_state`
から間接的に使います。

### いちばん簡単な使い方

まず hidden-state engine を build します。

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/models/qwen/hf_model \
  --output_dir /workspaces/safe-copilot/models/qwen/hidden_state_engine \
  --max_input_len 1024
```

### ReCogDrive 用に `bfloat16` engine を作る最短手順

`owl10/ReCogDrive-VLM-2B` の Qwen 部分から `bfloat16` の
hidden-state engine を作るときは、通常は次の 2 段階です。

1. `language_model` を Hugging Face 形式で取り出す

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir owl10/ReCogDrive-VLM-2B \
  --hf_output_dir /workspaces/safe-copilot/examples/models/qwen/models/a \
  --output_dir /workspaces/safe-copilot/examples/models/qwen/models/b
```

2. その Hugging Face ディレクトリから `bfloat16` hidden-state engine を作る

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/examples/models/qwen/models/a \
  --output_dir /workspaces/safe-copilot/models/qwen_bf16 \
  --max_input_len 2800 \
  --dtype bfloat16 \
  --max_prompt_embedding_table_size 3328
```

この手順で最終的に使う engine は次です。

```text
/workspaces/safe-copilot/models/qwen/config.json
/workspaces/safe-copilot/models/qwen/rank0.engine
```

`--output_dir` に `/workspaces/safe-copilot/models/qwen` を指定しているのは、
`src/models/internvl_chat.py` の既定値がその場所を読むためです。

このとき内部では `trt_hidden_state_engine.py` の
`build_hidden_state_engine(...)` が呼ばれます。

この helper は既定で packed token 前提の hidden-state engine を作るため、
`last hidden state` の比較テストでは `padding="max_length"` ではなく、
実トークン長のまま比較するのが自然です。

standalone の Qwen Hugging Face ディレクトリを入力にしている場合は、
`--hf_model_dir` にその元ディレクトリをそのまま指定してください。

### Python から直接使う例

すでに engine があるなら、prompt 文字列から `last hidden state` を取れます。

```python
from examples.models.qwen.trt_hidden_state_engine import get_last_hidden_state_tensorrt_llm

hidden_state = get_last_hidden_state_tensorrt_llm(
    model_id="/workspaces/safe-copilot/models/qwen/hf_model",
    prompt="Hello",
    max_input_len=1024,
    engine_dir="/workspaces/safe-copilot/models/qwen/hidden_state_engine",
)
print(hidden_state.shape)
```

`input_ids` を自前で作って渡したい場合は
`get_last_hidden_state_tensorrt_llm_from_input_ids(...)` を使います。

```python
from examples.models.qwen.trt_hidden_state_engine import (
    get_last_hidden_state_tensorrt_llm_from_input_ids,
)

hidden_state = get_last_hidden_state_tensorrt_llm_from_input_ids(
    model_id="/workspaces/safe-copilot/models/qwen/hf_model",
    batch_input_ids=[input_ids],
    max_input_len=1024,
    engine_dir="/workspaces/safe-copilot/models/qwen/hidden_state_engine",
)
```

### 主な関数

- `build_hidden_state_engine(...)`
  Hugging Face モデルから hidden-state engine を build します。
- `get_last_hidden_state_tensorrt_llm(...)`
  prompt 文字列から hidden state を取ります。
- `get_last_hidden_state_tensorrt_llm_from_input_ids(...)`
  tokenized 済みの `input_ids` から hidden state を取ります。

### 主な引数

- `model_id`
  Hugging Face 形式のモデルディレクトリ、または Hugging Face Hub の model id
- `max_input_len`
  engine build と runtime の両方で使う最大入力長
- `engine_dir`
  build 済み engine のディレクトリ
- `max_prompt_embedding_table_size`
  multimodal prompt embedding を使うときの最大サイズ
- `prompt_embedding_table`
  vision 側などで作った埋め込みを直接渡したいときに使います
- `dtype`
  hidden-state engine の build dtype です。`float16` または `bfloat16`
  を指定できます。

### 注意点

- `trtllm-build --checkpoint_dir ...` をそのまま使っても、
  hidden-state 出力は自動では増えません。
  `last_hidden_state_output` を engine に載せる build が必要です。
- `model_id` には pure Qwen の Hugging Face 形式ディレクトリを渡してください。
  `owl10/ReCogDrive-VLM-2B` のような top-level の multimodal checkpoint を直接
  入れるのではなく、先に `language_model` を抽出して使うのが前提です。
- 現在の helper は実用上 batch size 1 を前提にしています。

## Notes

- If the source model is multimodal, `convert_checkpoint.py` looks for
  `config.llm_config` and extracts `language_model`.
- If tokenizer files are not colocated with the source model, set
  `QWEN_TOKENIZER_SOURCE` to a local tokenizer directory before running
  `convert_checkpoint.py`.
- If `rope_scaling.type == "dynamic"` and `alpha` is missing, the conversion
  script patches it automatically in the saved config so TensorRT-LLM can load
  the checkpoint more consistently.

## End-to-End Example

### 既存の standalone Qwen Hugging Face ディレクトリを使う場合

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir /workspaces/safe-copilot/models/recogdrive/qwen2_submodel

/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --checkpoint_dir /workspaces/safe-copilot/models/qwen/tllm_checkpoint \
  --output_dir /workspaces/safe-copilot/models/qwen/engine

/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/models/recogdrive/qwen2_submodel
```

### multimodal モデルから `language_model` を抽出して使う場合

```bash
/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/convert_checkpoint.py \
  --model_dir owl10/ReCogDrive-VLM-2B \
  --hf_output_dir ./examples/models/qwen/models/a \
  --output_dir ./examples/models/qwen/models/b

/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --checkpoint_dir /workspaces/safe-copilot/models/qwen/tllm_checkpoint \
  --output_dir /workspaces/safe-copilot/models/qwen/engine

/workspaces/safe-copilot/.venv/bin/python examples/models/qwen/build_trtllm.py \
  --engine_kind hidden_state \
  --hf_model_dir /workspaces/safe-copilot/examples/models/qwen/models/a \
  --output_dir /workspaces/safe-copilot/models/qwen/ \
  --max_input_len 2800
```
