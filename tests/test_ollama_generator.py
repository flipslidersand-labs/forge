"""OllamaGenerator のオフラインテスト。ollama サーバー不要。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.ollama_generator import OllamaGenerator, _Proposal


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((512, 4096), torch.float16, True),
            TensorSpec((4096,), torch.float16, True),
        ),
        output_specs=(TensorSpec((512, 4096), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _valid_candidate_json() -> str:
    return _Proposal(
        candidates=[
            {  # type: ignore[arg-type]
                "base_variant": "single_row",
                "block_size": 4096,
                "num_warps": 8,
                "num_stages": 1,
                "acc_dtype": "fp32",
                "rows_per_program": 1,
                "hypothesis": "large block covers full row",
            }
        ]
    ).model_dump_json()


def _mock_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


class TestOllamaGeneratorOffline:
    def _gen(self) -> OllamaGenerator:
        return OllamaGenerator(model="qwen2.5-coder:latest")

    def test_generate_returns_search_params(self) -> None:
        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(
                _valid_candidate_json()
            )
            result = gen.generate(_spec(), "8.6", budget=1)

        assert len(result) == 1
        assert result[0].block_size == 4096
        assert result[0].num_warps == 8
        assert result[0].acc_dtype == "fp32"

    def test_generate_deduplicates(self) -> None:
        # 同一パラメータが2件返ってきても1件にまとめられる
        proposal = _Proposal(
            candidates=[  # type: ignore[arg-type]
                {
                    "base_variant": "single_row",
                    "block_size": 4096,
                    "num_warps": 8,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": "a",
                },
                {
                    "base_variant": "single_row",
                    "block_size": 4096,
                    "num_warps": 8,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": "duplicate",
                },
            ]
        ).model_dump_json()

        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(proposal)
            result = gen.generate(_spec(), "8.6", budget=5)

        assert len(result) == 1

    def test_generate_respects_budget(self) -> None:
        proposal = _Proposal(
            candidates=[  # type: ignore[arg-type]
                {
                    "base_variant": "single_row",
                    "block_size": 4096,
                    "num_warps": w,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": f"warps={w}",
                }
                for w in [4, 8, 16]
            ]
        ).model_dump_json()

        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(proposal)
            result = gen.generate(_spec(), "8.6", budget=2)

        assert len(result) == 2

    def test_generate_returns_empty_on_broken_json(self) -> None:
        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(
                "not valid json at all"
            )
            result = gen.generate(_spec(), "8.6", budget=5)

        assert result == []

    def test_generate_returns_empty_on_ollama_error(self) -> None:
        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = ConnectionError("ollama not running")
            result = gen.generate(_spec(), "8.6", budget=5)

        assert result == []

    def test_generate_drops_invalid_params(self) -> None:
        # block_size が hidden size 未満 → single_row では invalid
        proposal = _Proposal(
            candidates=[  # type: ignore[arg-type]
                {
                    "base_variant": "single_row",
                    "block_size": 128,  # 4096 未満 → 棄却
                    "num_warps": 8,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": "too small",
                }
            ]
        ).model_dump_json()

        gen = self._gen()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(proposal)
            result = gen.generate(_spec(), "8.6", budget=5)

        assert result == []

    def test_default_model_and_host(self) -> None:
        gen = OllamaGenerator()
        assert gen.model == "qwen2.5-coder:latest"
        assert gen.host == "http://localhost:11434"

    def test_custom_model_and_host(self) -> None:
        gen = OllamaGenerator(model="deepseek-coder:latest", host="http://192.168.1.10:11434")
        assert gen.model == "deepseek-coder:latest"
        assert gen.host == "http://192.168.1.10:11434"

    def test_host_passed_to_client(self) -> None:
        gen = OllamaGenerator(host="http://192.168.1.10:11434")
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(
                _valid_candidate_json()
            )
            gen.generate(_spec(), "8.6")
        mock_client.assert_called_once_with(host="http://192.168.1.10:11434")
