from typing import Sequence
from pathlib import Path
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
import json
import struct

import numpy as np

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
)


class Qwen2:
    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
    ):
        self._model = None
        self._device = device

        model_path = Path(model_path)

        # ------------------------------------------------------------
        # 1. Read HuggingFace config.json and construct Qwen2 metadata.
        # ------------------------------------------------------------
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        dtype_map = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }

        torch_dtype = config["torch_dtype"]

        if torch_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported Qwen2 dtype: {torch_dtype}"
            )

        dtype = dtype_map[torch_dtype]

        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]

        if hs % nh != 0:
            raise ValueError(
                f"hidden_size ({hs}) must be divisible "
                f"by num_attention_heads ({nh})"
            )

        dh = hs // nh

        eos_token_id = config["eos_token_id"]

        # Some HuggingFace models allow eos_token_id to be a list.
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        meta = LlaisysQwen2Meta(
            dtype,
            config["num_hidden_layers"],
            hs,
            nh,
            nkvh,
            dh,
            config["intermediate_size"],
            config["max_position_embeddings"],
            config["vocab_size"],
            config["rms_norm_eps"],
            config["rope_theta"],
            eos_token_id,
        )

        self._nlayer = config["num_hidden_layers"]
        self._end_token = eos_token_id
        self._maxseq = config["max_position_embeddings"]

        # ------------------------------------------------------------
        # 2. Create the native C++ model.
        # ------------------------------------------------------------
        device_ids = (c_int * 1)(0)

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            device_ids,
            1,
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        try:
            weights = LIB_LLAISYS.llaisysQwen2ModelWeights(
                self._model
            ).contents

            weight_handles = self._build_weight_map(
                weights,
                self._nlayer,
            )

            self._load_safetensors(
                model_path,
                weight_handles,
                dtype,
            )

        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    @staticmethod
    def _build_weight_map(weights, nlayer):
        result = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }

        for i in range(nlayer):
            prefix = f"model.layers.{i}"

            result[
                f"{prefix}.input_layernorm.weight"
            ] = weights.attn_norm_w[i]

            result[
                f"{prefix}.self_attn.q_proj.weight"
            ] = weights.attn_q_w[i]

            result[
                f"{prefix}.self_attn.q_proj.bias"
            ] = weights.attn_q_b[i]

            result[
                f"{prefix}.self_attn.k_proj.weight"
            ] = weights.attn_k_w[i]

            result[
                f"{prefix}.self_attn.k_proj.bias"
            ] = weights.attn_k_b[i]

            result[
                f"{prefix}.self_attn.v_proj.weight"
            ] = weights.attn_v_w[i]

            result[
                f"{prefix}.self_attn.v_proj.bias"
            ] = weights.attn_v_b[i]

            result[
                f"{prefix}.self_attn.o_proj.weight"
            ] = weights.attn_o_w[i]

            result[
                f"{prefix}.post_attention_layernorm.weight"
            ] = weights.mlp_norm_w[i]

            result[
                f"{prefix}.mlp.gate_proj.weight"
            ] = weights.mlp_gate_w[i]

            result[
                f"{prefix}.mlp.up_proj.weight"
            ] = weights.mlp_up_w[i]

            result[
                f"{prefix}.mlp.down_proj.weight"
            ] = weights.mlp_down_w[i]

        return result

    @staticmethod
    def _tensor_shape(tensor):
        ndim = LIB_LLAISYS.tensorGetNdim(tensor)
        shape = (c_size_t * ndim)()

        LIB_LLAISYS.tensorGetShape(
            tensor,
            shape,
        )

        return tuple(shape)

    @classmethod
    def _load_safetensors(
        cls,
        model_path,
        weight_handles,
        model_dtype,
    ):
        safetensor_files = sorted(
            model_path.glob("*.safetensors")
        )

        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        dtype_map = {
            "BF16": (DataType.BF16, 2),
            "F16": (DataType.F16, 2),
            "F32": (DataType.F32, 4),
        }

        loaded = set()

        for file in safetensor_files:
            with open(file, "rb") as f:
                header_size_raw = f.read(8)

                if len(header_size_raw) != 8:
                    raise ValueError(
                        f"Invalid safetensors file: {file}"
                    )

                header_size = struct.unpack(
                    "<Q",
                    header_size_raw,
                )[0]

                header = json.loads(
                    f.read(header_size)
                )

            # Map the whole file as raw bytes.
            raw_file = np.memmap(
                file,
                mode="r",
                dtype=np.uint8,
            )

            data_start = 8 + header_size

            try:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue

                    if name not in weight_handles:
                        raise KeyError(
                            f"Unexpected weight in safetensors: {name}"
                        )

                    if name in loaded:
                        raise ValueError(
                            f"Duplicate weight: {name}"
                        )

                    dtype_name = info["dtype"]

                    if dtype_name not in dtype_map:
                        raise ValueError(
                            f"Unsupported safetensors dtype "
                            f"{dtype_name} for {name}"
                        )

                    tensor_dtype, item_size = dtype_map[
                        dtype_name
                    ]

                    if tensor_dtype != model_dtype:
                        raise ValueError(
                            f"Dtype mismatch for {name}: "
                            f"file={dtype_name}, "
                            f"model={model_dtype}"
                        )

                    expected_shape = tuple(info["shape"])

                    actual_shape = cls._tensor_shape(
                        weight_handles[name]
                    )

                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: "
                            f"file={expected_shape}, "
                            f"model={actual_shape}"
                        )

                    begin, end = info["data_offsets"]
                    nbytes = end - begin

                    expected_nbytes = (
                        int(np.prod(expected_shape))
                        * item_size
                    )

                    if nbytes != expected_nbytes:
                        raise ValueError(
                            f"Byte-size mismatch for {name}: "
                            f"file={nbytes}, "
                            f"expected={expected_nbytes}"
                        )

                    raw_tensor = raw_file[
                        data_start + begin:
                        data_start + end
                    ]

                    LIB_LLAISYS.tensorLoad(
                        weight_handles[name],
                        c_void_p(raw_tensor.ctypes.data),
                    )

                    loaded.add(name)

            finally:
                del raw_file

        missing = set(weight_handles) - loaded

        if missing:
            preview = sorted(missing)[:10]

            raise KeyError(
                f"Missing {len(missing)} Qwen2 weights. "
                f"Examples: {preview}"
            )

        if len(loaded) != len(weight_handles):
            raise RuntimeError(
                f"Weight count mismatch: "
                f"loaded={len(loaded)}, "
                f"expected={len(weight_handles)}"
            )

        print(
            f"Loaded {len(loaded)} Qwen2 weights successfully."
        )

    def __del__(self):
        model = getattr(self, "_model", None)

        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # The native infer function currently returns argmax directly,
        # so this baseline implementation supports greedy decoding only.
        if top_k != 1:
            raise NotImplementedError(
                "Qwen2 generation currently supports greedy decoding only."
            )

        output = list(inputs)

        if not output:
            raise ValueError("Input token sequence must not be empty.")

        if max_new_tokens is None:
            max_new_tokens = self._maxseq - len(output)

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        if len(output) + max_new_tokens > self._maxseq:
            raise ValueError(
                "Requested generation exceeds maximum sequence length."
            )

        for _ in range(max_new_tokens):
            token_array = (c_int64 * len(output))(*output)

            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                len(output),
            )

            next_token = int(next_token)
            output.append(next_token)

            if next_token == self._end_token:
                break

        return output
