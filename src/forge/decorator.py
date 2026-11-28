from __future__ import annotations

import functools
import inspect
import logging
import re
import threading
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from forge.cache.repository import KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.hashing import hash_constants
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.lowering import identify
from forge.ops.registry import OP_REGISTRY, accepts_call
from forge.orchestrator import Orchestrator, ProgressEvent
from forge.runtime.loader import load_kernel_fn
from forge.runtime.worker import warn_python_executable_deprecated
from forge.search.adoption import should_run_search
from forge.search.candidate import CandidateGenerator
from forge.validation.tolerance import Tolerance

_log = logging.getLogger("forge.decorator")

SUPPORTED_BACKENDS: tuple[str, ...] = ("triton",)
SUPPORTED_OBJECTIVES: tuple[str, ...] = ("latency", "economic")


@dataclass(frozen=True)
class _BuildConfig:
    """optimize() のオプションのうち _build / Orchestrator に渡すもの（private）。"""

    budget: int
    repo: KernelRepository | None
    search: CandidateGenerator | None
    min_speedup: float
    python_executable: str | None
    progress: Callable[[ProgressEvent], None]


def optimize(
    budget: int = 50,
    backend: Literal["triton"] = "triton",
    objective: Literal["latency", "economic"] = "latency",
    *,
    repo: KernelRepository | None = None,
    search: CandidateGenerator | None = None,
    min_speedup: float = 1.03,
    min_invocations: int = 0,
    per_candidate_s: float = 2.0,
    python_executable: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """純粋な PyTorch 演算を最速の Triton 実装へ自動置換するデコレータ。

    初回（shape ごと）: torch.fx で op_type を判定 → KernelSpec を組み立て →
    探索・検証・ベンチ・キャッシュ（Orchestrator）→ 最速カーネルを in-process ロード。
    2 回目以降（同 shape）: in-process キャッシュから直接実行。新しい shape は
    再探索するが SQLite キャッシュにヒットすれば即座に返る。

    判定不能の場合、および最良カーネルが eager に min_speedup 倍で勝てない（速くならない）
    場合は、元の eager 関数にフォールバックする（そのカーネルはキャッシュにも書かない）。

    objective="economic" かつ min_invocations > 0 のとき、探索コストが回収できない
    と判断した場合は探索をスキップして eager にフォールバックする。
    per_candidate_s は 1 候補あたりの探索コスト見積もり（秒）。GPU の速度に応じて調整する。

    progress は進捗の人間可読ラベル（str）を 1 件ずつ受け取る。注釈なしの関数・lambda・
    print はラベル文字列を受け取る。第 1 引数に ProgressEvent 注釈を付けた関数だけは
    構造化イベント（ProgressEvent）をそのまま受け取る。
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}. Choose from {SUPPORTED_BACKENDS}.")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective: {objective!r}. Choose from {SUPPORTED_OBJECTIVES}."
        )
    # `not (x >= 1)` / `not (x > 0)` は `<` / `<=` と違い NaN も正しく弾く
    # （`nan < 1` や `nan <= 0` は False のため素通りしてしまう）。
    if not (budget >= 1):
        raise ValueError(f"budget must be >= 1, got {budget!r}.")
    if not (min_speedup > 0):
        raise ValueError(f"min_speedup must be > 0, got {min_speedup!r}.")
    if not (per_candidate_s > 0):
        raise ValueError(f"per_candidate_s must be > 0, got {per_candidate_s!r}.")

    if python_executable is not None:
        warn_python_executable_deprecated(stacklevel=3)

    _emit_progress = _progress_to_event_handler(progress)

    build_config = _BuildConfig(
        budget=budget,
        repo=repo,
        search=search,
        min_speedup=min_speedup,
        python_executable=python_executable,
        progress=_emit_progress,
    )

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        # (shape/dtype シグネチャ, constants ハッシュ) -> in-process カーネル (or None=eager)
        compiled: dict[tuple[Any, ...], Callable[..., Any] | None] = {}
        op_type_box: list[
            str | None
        ] = []  # 一度だけ判定（[] 未判定 / [None] 不可 / [str] 判定済み）
        _op_type_lock = threading.Lock()
        # #329: shape ごとに独立したロックでビルドを保護する。単一ロックだと
        # 無関係な shape（バッチサイズ/seq_len違い等）のビルドまで直列化されて
        # しまうため、key ごとに Lock を割り当てて並行ビルドを許容する。
        # _build_locks_guard は _build_locks 辞書自体への同時書き込みのみを保護
        # する（ビルド処理そのものは保護しない・保持時間は極短）。
        _build_locks_guard = threading.Lock()
        _build_locks: dict[tuple[Any, ...], threading.Lock] = {}

        def _get_build_lock(key: tuple[Any, ...]) -> threading.Lock:
            with _build_locks_guard:
                lock = _build_locks.get(key)
                if lock is None:
                    lock = threading.Lock()
                    _build_locks[key] = lock
                return lock

        def _resolve_op_type() -> str | None:
            # ロック外の高速パス: すでに判定済みならそのまま返す
            if op_type_box:
                return op_type_box[0]
            with _op_type_lock:
                # ダブルチェック: 他スレッドが先に入れた可能性
                if not op_type_box:
                    op_type_box.append(identify(fn) if backend == "triton" else None)
            return op_type_box[0]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            tensors = [v for v in bound.arguments.values() if isinstance(v, torch.Tensor)]
            constants = {
                k: v for k, v in bound.arguments.items() if not isinstance(v, torch.Tensor)
            }

            # GPU テンソルが無い / op 判定不能 → eager
            op_type = _resolve_op_type()
            if not tensors or op_type is None or not tensors[0].is_cuda:
                _log.debug("eager fallback fn=%s op=%s", fn.__qualname__, op_type)
                return fn(*args, **kwargs)

            # #507: 実テンソル数 / constants が op の kernel_fn に対応づけられない形
            # (linear の b=None 等)は、候補生成で全候補が TypeError になる前に eager。
            if not accepts_call(op_type, len(tensors), constants):
                _log.warning(
                    "forge.optimize: 未対応の呼び出し形です fn=%s op=%s tensors=%d constants=%s"
                    " — eager にフォールバックします",
                    fn.__qualname__,
                    op_type,
                    len(tensors),
                    sorted(constants),
                )
                return fn(*args, **kwargs)

            # #328: 引数テンソル間でデバイスが不一致だと Triton/CUDA レベルの
            # 分かりにくいエラーでクラッシュしうる。早期に検出して eager に
            # フォールバックし、原因を特定しやすい warning ログを残す。
            first_device = tensors[0].device
            if any(t.device != first_device for t in tensors[1:]):
                devices = [str(t.device) for t in tensors]
                _log.warning(
                    "forge.optimize: 引数テンソルのデバイスが一致していません "
                    "fn=%s devices=%s — eager にフォールバックします",
                    fn.__qualname__,
                    devices,
                )
                return fn(*args, **kwargs)

            # #372: 生成カーネルは行優先の連続レイアウトを前提にしており、キャッシュキーも
            # shape/dtype のみ。非連続 view(.transpose() 等)は誤った結果を静かに返すため、
            # eager にフォールバックする。
            if any(not t.is_contiguous() for t in tensors):
                _log.warning(
                    "forge.optimize: 非連続テンソルが渡されました fn=%s — "
                    "eager にフォールバックします(.contiguous() で最適化経路に乗ります)",
                    fn.__qualname__,
                )
                return fn(*args, **kwargs)

            # #470: constants(is_causal 等)はレンダリング時にカーネルへ焼き込まれるため、
            # SQLite 側 CacheKey と同様に constants_hash もキーに含める。
            key = (
                tuple((tuple(t.shape), str(t.dtype)) for t in tensors),
                hash_constants(constants),
            )
            # ロック外の高速パス: すでにコンパイル済みならそのまま実行
            if key not in compiled:
                with _get_build_lock(key):
                    # ダブルチェック: 他スレッドが先に _build を完了した可能性
                    if key not in compiled:
                        # economic: eager を 1 回タイムしてから探索判断
                        if objective == "economic" and min_invocations > 0:
                            baseline_us = _time_eager(fn, args, kwargs)
                            search_cost_s = budget * per_candidate_s
                            decision = should_run_search(
                                min_invocations, search_cost_s, baseline_us
                            )
                            _emit_progress(
                                ProgressEvent(kind="info", label=f"adoption: {decision.reason}")
                            )
                            _log.info("adoption fn=%s: %s", fn.__qualname__, decision.reason)
                            if not decision.should_search:
                                compiled[key] = None
                            else:
                                compiled[key] = _build_or_eager(
                                    fn, op_type, tensors, constants, build_config, (args, kwargs)
                                )
                        else:
                            compiled[key] = _build_or_eager(
                                fn, op_type, tensors, constants, build_config, (args, kwargs)
                            )
            kfn = compiled[key]
            if kfn is None:
                return fn(*args, **kwargs)
            return kfn(*tensors, **constants)

        wrapper._forge_compiled = compiled  # type: ignore[attr-defined]  # テスト/内省用
        return wrapper

    return deco


def _fn_graph_hash(fn: Callable[..., Any]) -> str:
    """関数の実装を区別する graph_hash を生成する。

    __qualname__ と inspect.getsource() の SHA-256 先頭 12 桁を組み合わせることで、
    同一 op_type でも実装が異なる関数が別の CacheKey を持つことを保証する。
    getsource() が利用できない（lambda・REPL 等）場合は qualname のみを使う。
    """
    import hashlib

    try:
        src_hash = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        src_hash = "nosrc"
    return f"{fn.__qualname__}:{src_hash}"


def _build_or_eager(*args: Any) -> Callable[..., Any] | None:
    """_build の例外を握って eager フォールバック(None)にする（#319）。

    KeyboardInterrupt / SystemExit は Exception 派生でないため伝播する。
    """
    try:
        return _build(*args)
    except Exception as exc:  # noqa: BLE001 — 探索系の失敗は eager 契約を優先
        _log.warning("build failed — falling back to eager: %s", exc, exc_info=True)
        return None


def _build(
    fn: Callable[..., Any],
    op_type: str,
    tensors: list[Any],
    constants: dict[str, Any],
    config: _BuildConfig,
    user_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
) -> Callable[..., Any] | None:
    """この shape に対して最速カーネルを探索し、in-process 実行関数を返す（無ければ None）。

    user_call=(args, kwargs) を渡すと、採用前にユーザー関数 fn 自体とカーネルの出力を
    その実入力 1 回で照合し、不一致なら None（eager）を返す（#472）。キャッシュから読んだ
    カーネルも同じ経路を通るので、キャッシュ済みの誤カーネルも採用されない。decorator は
    必ず渡す。None は _build を単体で呼ぶ内部テスト用で、照合しない。
    """
    input_specs = tuple(TensorSpec.from_tensor(t) for t in tensors)
    out = TensorSpec.from_tensor(tensors[0])
    spec = KernelSpec(
        op_type=op_type,
        input_specs=input_specs,
        output_specs=(out,),
        constants=constants,
        graph_hash=_fn_graph_hash(fn),
        constraints=(),
    )

    _log.info("build start op=%s", op_type)
    with Orchestrator(
        repo=config.repo,
        min_speedup=config.min_speedup,
        progress=config.progress,
    ) as orch:
        # 非推奨警告は optimize() 側で出済みなので、コンストラクタ経由にせず属性で渡す
        orch.python_executable = config.python_executable
        result = orch.optimize(spec, budget=config.budget, search=config.search)
    if result.best_params is None:
        _log.info("build no best op=%s — falling back to eager", op_type)
        return None
    if not result.adopted:
        # 最良候補が eager に min_speedup 倍で勝たなかった (#475)。キャッシュにも書かれていない
        _log.info(
            "build not adopted op=%s speedup=%s min_speedup=%s — falling back to eager",
            op_type,
            result.speedup,
            config.min_speedup,
        )
        return None
    _log.info("build complete op=%s best=%s", op_type, result.best_params)
    code = generate(spec, result.best_params)
    kfn = load_kernel_fn(code)
    if user_call is not None and not _kernel_matches_user_fn(
        fn, op_type, kfn, tensors, constants, user_call
    ):
        return None
    return kfn


def _kernel_matches_user_fn(
    fn: Callable[..., Any],
    op_type: str,
    kfn: Callable[..., Any],
    tensors: list[Any],
    constants: dict[str, Any],
    user_call: tuple[tuple[Any, ...], dict[str, Any]],
) -> bool:
    """ユーザー関数 fn の出力と、採用候補カーネルの出力が一致するか（実入力 1 回）。

    op の同定は fx ノードの関数名しか見ず、ワーカー検証も registry の reference との比較
    なので、fn と reference が食い違っても通りうる。ここで fn そのものと比べて塞ぐ。
    shape・dtype が違う、テンソルでない、カーネルが例外を投げる場合も不一致。許容誤差は
    ワーカー検証と同じ OP_REGISTRY の値。
    """
    import torch

    args, kwargs = user_call
    defn = OP_REGISTRY.get(op_type)
    tol = defn.tolerance if defn is not None else Tolerance(atol=2e-3, rtol=1e-2)
    from torch.utils._pytree import tree_map

    def _clone(v: Any) -> Any:
        return v.clone() if isinstance(v, torch.Tensor) else v

    with torch.no_grad():
        # fn もカーネルも入力をその場で書き換えうる。ユーザーの実テンソルには触れず、
        # それぞれ別の clone で呼ぶ（実テンソルはこの後の eager / 採用カーネルが使う）。
        c_args, c_kwargs = tree_map(_clone, (args, kwargs))
        expected = fn(*c_args, **c_kwargs)  # fn 自身の例外は _build_or_eager が握って eager に倒す
        try:
            actual = kfn(*[t.clone() for t in tensors], **constants)
            if isinstance(actual, torch.Tensor) and actual.is_cuda:
                torch.cuda.synchronize()  # 非同期のカーネル例外をここで拾う
        except Exception as exc:  # noqa: BLE001 — 動かないカーネルは採用しない
            _log.warning(
                "forge.optimize: カーネルの照合実行に失敗しました fn=%s op=%s: %s %s"
                " — eager にフォールバックします",
                fn.__qualname__,
                op_type,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return False
        if not (isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)):
            reason = "出力がテンソルではありません"
        elif actual.shape != expected.shape or actual.dtype != expected.dtype:
            # allclose はブロードキャストするので shape を先に厳密に比べる。dtype が違うと
            # 採用後の戻り値の型が変わってしまう。
            reason = (
                f"出力の shape/dtype が違います kernel={tuple(actual.shape)}/{actual.dtype}"
                f" fn={tuple(expected.shape)}/{expected.dtype}"
            )
        elif torch.allclose(
            actual.float(),
            expected.float(),
            atol=tol.atol,
            rtol=tol.rtol,
            equal_nan=tol.equal_nan,
        ):
            return True
        else:
            diff = (actual.float() - expected.float()).abs().max().item()
            reason = f"出力が一致しません max_abs_diff={diff:.3g}"
    _log.warning(
        "forge.optimize: 採用候補のカーネルがユーザー関数と一致しません fn=%s op=%s: %s"
        " — eager にフォールバックします",
        fn.__qualname__,
        op_type,
        reason,
    )
    return False


def _mentions_progress_event(ann: Any) -> bool:
    """注釈（文字列 / 型 / Optional・Union の中身）に ProgressEvent が含まれるか。"""
    if isinstance(ann, typing.ForwardRef):
        ann = ann.__forward_arg__
    if isinstance(ann, str):
        # "fo.ProgressEvent" / "ProgressEvent | None" / "Optional[ProgressEvent]" を末尾名で判定
        return any(tok.rsplit(".", 1)[-1] == "ProgressEvent" for tok in re.split(r"[^\w.]+", ann))
    if ann is ProgressEvent:
        return True
    return any(_mentions_progress_event(a) for a in typing.get_args(ann))


def _accepts_progress_event(progress: Callable[..., Any]) -> bool:
    """第 1 引数の注釈が ProgressEvent（修飾名・Optional・別名を含む）なら True。"""
    try:
        params = list(inspect.signature(progress).parameters.values())
    except (ValueError, TypeError):
        return False
    if not params or params[0].annotation is inspect.Parameter.empty:
        return False
    ann = params[0].annotation
    try:
        # 文字列注釈（from __future__ import annotations）や別名 import を解決できれば使う
        ann = typing.get_type_hints(progress).get(params[0].name, ann)
    except Exception:
        pass
    return _mentions_progress_event(ann)


def _progress_to_event_handler(
    progress: Callable[[str], None] | Callable[[ProgressEvent], None] | None,
) -> Callable[[ProgressEvent], None]:
    """利用者の progress を ProgressEvent ハンドラへ揃える（Orchestrator と adoption 通知で共通）。

    宣言型は Callable[[str], None] なので、第 1 引数の注釈が ProgressEvent を含む関数
    （"fo.ProgressEvent"・"ProgressEvent | None"・Optional[ProgressEvent] 等。別名 import は
    typing.get_type_hints で解決できる場合のみ）はイベントをそのまま受ける。それ以外
    （注釈なし・print・lambda・str 注釈・signature 取得不可）や、解決できない別名のように
    判定が曖昧なときは event.label（str）を渡す。
    返す lambda は注釈を持たないため Orchestrator はそのまま素通しする。
    """
    if progress is None:
        return lambda _e: None
    if _accepts_progress_event(progress):
        return progress  # type: ignore[return-value]
    str_fn: Callable[[str], None] = progress  # type: ignore[assignment]
    return lambda e: str_fn(e.label)


def _time_eager(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """eager 関数を warmup + repeat 回 CUDA Event で計測し中央値（µs）を返す。

    単一計測は ±10-20% のノイズがあるため、中央値を使って安定した推定を得る。
    """
    import statistics

    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # ms -> µs

    return statistics.median(samples)
