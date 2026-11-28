"""OllamaGenerator のオフラインテスト。ollama サーバー不要。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search._proposal_models import Proposal
from forge.search.ollama_generator import OllamaGenerator


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
    return Proposal(
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
        proposal = Proposal(
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
        proposal = Proposal(
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

    def test_propose_logs_error_on_ollama_error(self, caplog: pytest.LogCaptureFixture) -> None:
        gen = self._gen()
        with (
            patch("ollama.Client") as mock_client,
            caplog.at_level("WARNING", logger="forge.search.ollama_generator"),
        ):
            mock_client.return_value.chat.side_effect = ConnectionError("ollama not running")
            gen.generate(_spec(), "8.6", budget=5)

        assert any(
            r.levelname == "ERROR" and "ollama request failed" in r.message for r in caplog.records
        )

    def test_generate_drops_invalid_params(self) -> None:
        # block_size が hidden size 未満 → single_row では invalid
        proposal = Proposal(
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
        mock_client.assert_called_once_with(host="http://192.168.1.10:11434", timeout=60.0)

    def test_default_timeout(self) -> None:
        gen = OllamaGenerator()
        assert gen.timeout == 60.0

    def test_custom_timeout_passed_to_client(self) -> None:
        """#323: timeout が ollama.Client に明示的に渡され、無限待機を防ぐ。"""
        gen = OllamaGenerator(timeout=15.0)
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(
                _valid_candidate_json()
            )
            gen.generate(_spec(), "8.6")
        mock_client.assert_called_once_with(host="http://localhost:11434", timeout=15.0)


class TestOllamaErrorVisibility:
    """#324: 接続障害は error ログ、モデル応答不正は warning で区別する。"""

    def _propose(self, side_effect=None, content=None):
        gen = OllamaGenerator(model="m")
        with patch("ollama.Client") as mc:
            if side_effect is not None:
                mc.return_value.chat.side_effect = side_effect
            else:
                mc.return_value.chat.return_value = _mock_chat_response(content)
            return gen._propose(_spec(), "8.9", 1, [])

    def test_connection_error_logs_error(self, caplog) -> None:
        with caplog.at_level("WARNING", logger="forge.search.ollama_generator"):
            assert self._propose(side_effect=ConnectionError("refused")) == []
        assert [r.levelname for r in caplog.records] == ["ERROR"]

    def test_invalid_response_logs_warning(self, caplog) -> None:
        with caplog.at_level("WARNING", logger="forge.search.ollama_generator"):
            assert self._propose(content="not json") == []
        assert [r.levelname for r in caplog.records] == ["WARNING"]


class TestOllamaCoerceByOpType:
    """非 reduction op でも block_size < N の候補が Ollama 経路で残る。"""

    @pytest.mark.parametrize(
        ("op", "variant", "block"),
        [
            ("gelu", "elementwise", 1024),
            ("linear", "gemm", 64),
            ("scaled_dot_product_attention", "flash", 32),
            ("scaled_dot_product_attention", "flash_causal_opt", 16),
        ],
    )
    def test_non_reduction_candidate_kept(self, op: str, variant: str, block: int) -> None:
        f16 = lambda shape: TensorSpec(shape, torch.float16, True)  # noqa: E731
        specs = {
            "gelu": ((f16((64, 4096)),), (f16((64, 4096)),)),
            "linear": ((f16((128, 1024)), f16((512, 1024)), f16((512,))), (f16((128, 512)),)),
            "scaled_dot_product_attention": ((f16((8, 64, 64)),) * 3, (f16((8, 64, 64)),)),
        }
        ins, outs = specs[op]
        spec = KernelSpec(op, ins, outs, {}, f"{op}_v1", ())
        content = Proposal(
            candidates=[  # type: ignore[arg-type]
                {
                    "base_variant": variant,
                    "block_size": block,
                    "num_warps": 4,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": "tile smaller than hidden",
                }
            ]
        ).model_dump_json()
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.chat.return_value = _mock_chat_response(content)
            result = OllamaGenerator(model="qwen2.5-coder:latest").generate(spec, "8.6", budget=1)

        assert [(p.variant, p.block_size) for p in result] == [(variant, block)]
