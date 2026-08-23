## [1.9.2](https://github.com/flipslidersand-labs/forge/compare/v1.9.1...v1.9.2) (2026-08-23)

### 🐛 Bug Fixes

* **#206:** atexit → try/finally で /tmp 一時ファイルを即削除 ([#239](https://github.com/flipslidersand-labs/forge/issues/239)) ([c051b02](https://github.com/flipslidersand-labs/forge/commit/c051b021890d6305e6a871b40467f6b5d22161f2)), closes [#206](https://github.com/flipslidersand-labs/forge/issues/206) [#206](https://github.com/flipslidersand-labs/forge/issues/206)

## [1.9.1](https://github.com/flipslidersand-labs/forge/compare/v1.9.0...v1.9.1) (2026-08-23)

### 🐛 Bug Fixes

* **dx:** pyproject.toml の project.urls を flipslidersand-labs/forge に修正 ([#200](https://github.com/flipslidersand-labs/forge/issues/200)) ([#238](https://github.com/flipslidersand-labs/forge/issues/238)) ([85a9367](https://github.com/flipslidersand-labs/forge/commit/85a9367961152f5fb938e00fdc56eeee910d0f30))

## [1.9.0](https://github.com/flipslidersand-labs/forge/compare/v1.8.0...v1.9.0) (2026-08-22)

### ✨ Features

* **dx:** py.typed マーカー追加 — PEP 561 準拠 ([#199](https://github.com/flipslidersand-labs/forge/issues/199)) ([#237](https://github.com/flipslidersand-labs/forge/issues/237)) ([6b61a42](https://github.com/flipslidersand-labs/forge/commit/6b61a4247ef9984308002dbb2e13791c13ff36f8))

## [1.8.0](https://github.com/flipslidersand-labs/forge/compare/v1.7.2...v1.8.0) (2026-08-22)

### ✨ Features

* **cache:** FORGE_DB_PATH 環境変数でデフォルト DB パスをオーバーライド ([#198](https://github.com/flipslidersand-labs/forge/issues/198)) ([#235](https://github.com/flipslidersand-labs/forge/issues/235)) ([5d6aa3d](https://github.com/flipslidersand-labs/forge/commit/5d6aa3d516f337a754575f5b01a3e95fbbcac6ac))

## [1.7.2](https://github.com/flipslidersand-labs/forge/compare/v1.7.1...v1.7.2) (2026-08-22)

### 🐛 Bug Fixes

* **pareto:** speedup_ratio 常に1.0バグ修正・コストレート定数化 ([#202](https://github.com/flipslidersand-labs/forge/issues/202),[#203](https://github.com/flipslidersand-labs/forge/issues/203)) ([#228](https://github.com/flipslidersand-labs/forge/issues/228)) ([bb173f8](https://github.com/flipslidersand-labs/forge/commit/bb173f8ba017affbfd3662233cdc19b9c06fe643))

## [1.7.1](https://github.com/flipslidersand-labs/forge/compare/v1.7.0...v1.7.1) (2026-08-22)

### 🐛 Bug Fixes

* **runtime,cache:** worker stderr 詳細化・bare Exception 絞り込み・SQLite スレッドセーフ ([#219](https://github.com/flipslidersand-labs/forge/issues/219),[#220](https://github.com/flipslidersand-labs/forge/issues/220),[#221](https://github.com/flipslidersand-labs/forge/issues/221)) ([#227](https://github.com/flipslidersand-labs/forge/issues/227)) ([fb89f91](https://github.com/flipslidersand-labs/forge/commit/fb89f9136b17a0de022ee7baeaf088a013d2044b))

## [1.7.0](https://github.com/flipslidersand-labs/forge/compare/v1.6.4...v1.7.0) (2026-08-22)

### ✨ Features

* **cache:** forge cache prune コマンドを追加 ([#212](https://github.com/flipslidersand-labs/forge/issues/212)) ([#218](https://github.com/flipslidersand-labs/forge/issues/218)) ([0b7cee1](https://github.com/flipslidersand-labs/forge/commit/0b7cee106afd0c031e352115f61794f11df8b87b))

## [1.6.4](https://github.com/flipslidersand-labs/forge/compare/v1.6.3...v1.6.4) (2026-08-22)

### 🐛 Bug Fixes

* **#207:** is_improvement が baseline.p20_us<=0 のとき warning を出力する ([#210](https://github.com/flipslidersand-labs/forge/issues/210)) ([d19b42a](https://github.com/flipslidersand-labs/forge/commit/d19b42a474d8e52bdf5306a071dd759b053d61ac)), closes [#207](https://github.com/flipslidersand-labs/forge/issues/207)
* **#209:** forge cache list/clear が SQLite 破損時に rc=1 とエラーメッセージを返す ([#211](https://github.com/flipslidersand-labs/forge/issues/211)) ([c64e73d](https://github.com/flipslidersand-labs/forge/commit/c64e73dcf8648b3fb10f1c846aa64c03d5495c38)), closes [#209](https://github.com/flipslidersand-labs/forge/issues/209)
* **ci:** coverage-threshold を実測値 76% に基づき 71% に引き上げ ([#195](https://github.com/flipslidersand-labs/forge/issues/195)) ([#226](https://github.com/flipslidersand-labs/forge/issues/226)) ([3e87f24](https://github.com/flipslidersand-labs/forge/commit/3e87f2431ca3350e1e31981693e5e29e269fb11c))
* **ci:** publish.yml を削除し PyPI publish を release.yml に一本化 ([#196](https://github.com/flipslidersand-labs/forge/issues/196)) ([#225](https://github.com/flipslidersand-labs/forge/issues/225)) ([9997cfc](https://github.com/flipslidersand-labs/forge/commit/9997cfc385dbc059e76667c977438c6d01f1c0bf))
* **ci:** search extra を install-extras に追加して collection error を解消 ([#194](https://github.com/flipslidersand-labs/forge/issues/194)) ([#223](https://github.com/flipslidersand-labs/forge/issues/223)) ([4986280](https://github.com/flipslidersand-labs/forge/commit/4986280ee452040868ff7a5fd9e3bc6125c0c9d7))
* **orchestrator:** round_results をループ前に初期化し NameError を排除 ([#193](https://github.com/flipslidersand-labs/forge/issues/193)) ([#222](https://github.com/flipslidersand-labs/forge/issues/222)) ([8f6b711](https://github.com/flipslidersand-labs/forge/commit/8f6b711ceb88c153fa44e6e6100f0230aa87b770))

## [1.6.3](https://github.com/flipslidersand-labs/forge/compare/v1.6.2...v1.6.3) (2026-08-22)

### 🐛 Bug Fixes

* **ci:** push ブランチフィルタに test/** / chore/** 等を追加 ([#201](https://github.com/flipslidersand-labs/forge/issues/201)) ([#224](https://github.com/flipslidersand-labs/forge/issues/224)) ([f2fa5e0](https://github.com/flipslidersand-labs/forge/commit/f2fa5e0c270c706a072e52ca9c6b4dd396504c6b))

## [1.6.2](https://github.com/flipslidersand/forge/compare/v1.6.1...v1.6.2) (2026-08-13)

### 🐛 Bug Fixes

* **ci:** Python 3.11 matrix 削除（3.12 のみに統一） ([#181](https://github.com/flipslidersand/forge/issues/181)) ([c8e511e](https://github.com/flipslidersand/forge/commit/c8e511e8e741b93c9b446d3ff472d1b93fa02b9c))

## [1.6.1](https://github.com/flipslidersand/forge/compare/v1.6.0...v1.6.1) (2026-08-12)

### 🐛 Bug Fixes

* **search:** コードレビュー指摘 6件修正 ([#180](https://github.com/flipslidersand/forge/issues/180)) ([0389b9e](https://github.com/flipslidersand/forge/commit/0389b9e41007ff54c23d96e855e54b7a3af30e00))

## [1.6.0](https://github.com/flipslidersand/forge/compare/v1.5.2...v1.6.0) (2026-08-12)

### ✨ Features

* **publish:** パッケージ名を kernelsmith に変更・Trusted Publishers (OIDC) に切り替え ([b6f70d8](https://github.com/flipslidersand/forge/commit/b6f70d88ec56b3486050dde53fef632910c17f56))

## [1.5.2](https://github.com/flipslidersand/forge/compare/v1.5.1...v1.5.2) (2026-08-12)

### 🐛 Bug Fixes

* **ci:** poetry publish → twine upload に切り替え ([437ebaa](https://github.com/flipslidersand/forge/commit/437ebaa1f335902caca42321d346fbb26246a9df))

## [1.5.1](https://github.com/flipslidersand/forge/compare/v1.5.0...v1.5.1) (2026-08-12)

### 🐛 Bug Fixes

* **ci:** PYPI_PROD_TOKEN 再設定後の再トリガー ([4cc10d4](https://github.com/flipslidersand/forge/commit/4cc10d41263697f185ca361c6f3a67b8d82cf96e))

## [1.5.0](https://github.com/flipslidersand/forge/compare/v1.4.2...v1.5.0) (2026-08-12)

### ✨ Features

* **ci:** 本番 PyPI への publish に切り替え ([caae14d](https://github.com/flipslidersand/forge/commit/caae14dd821d163fadc58e42f0c61edb5195d796))

## [1.4.2](https://github.com/flipslidersand/forge/compare/v1.4.1...v1.4.2) (2026-08-12)

### 🐛 Bug Fixes

* **ci:** Test PyPI トークン再登録後の再トリガー ([23fafa7](https://github.com/flipslidersand/forge/commit/23fafa7f2f1088901dd44fb5c37134fdd7f3ad5f))

## [1.4.1](https://github.com/flipslidersand/forge/compare/v1.4.0...v1.4.1) (2026-08-12)

### 🐛 Bug Fixes

* **ci:** poetry install をConfigureより前に移動 — command not found 修正 ([27b14b3](https://github.com/flipslidersand/forge/commit/27b14b3493a1f0c43a56387debf7489938a63527))

## [1.4.0](https://github.com/flipslidersand/forge/compare/v1.3.1...v1.4.0) (2026-08-12)

### ✨ Features

* **api:** version_info() — 構造化バージョン情報を返すヘルパー追加 ([30b7bbe](https://github.com/flipslidersand/forge/commit/30b7bbef7e7797c6e5cb336c5437489979e5a542))

## [1.3.1](https://github.com/flipslidersand/forge/compare/v1.3.0...v1.3.1) (2026-08-11)

### 🐛 Bug Fixes

* **ci:** Test PyPI 認証設定を修正 ([2bf8610](https://github.com/flipslidersand/forge/commit/2bf8610d2272e21f228c9e116be4fdd0eebb8b46)), closes [#174](https://github.com/flipslidersand/forge/issues/174)

## [1.3.0](https://github.com/flipslidersand/forge/compare/v1.2.0...v1.3.0) (2026-08-11)

### ✨ Features

* Release automation retry (v1.1.0 → v1.2.0) ([75407bb](https://github.com/flipslidersand/forge/commit/75407bb122a30946e40945da9229114c86b7ee86))
* **test:** Test PyPI deployment test ([a3e3abc](https://github.com/flipslidersand/forge/commit/a3e3abc5c82888d5a58d2b0442446951492b8682))

## [1.2.0](https://github.com/flipslidersand/forge/compare/v1.1.0...v1.2.0) (2026-08-11)

### ✨ Features

* Release automation retry (v1.1.0 → v1.2.0) ([7b1430f](https://github.com/flipslidersand/forge/commit/7b1430f0749143dc768b10bb872b029ded5847ce))

## [1.1.0](https://github.com/flipslidersand/forge/compare/v1.0.1...v1.1.0) (2026-08-11)

### ✨ Features

* **test:** Release automation end-to-end test ([d50896e](https://github.com/flipslidersand/forge/commit/d50896e62a4a1068527950394739b1145781a0c6))

## [1.0.1](https://github.com/flipslidersand/forge/compare/v1.0.0...v1.0.1) (2026-08-10)

### 🐛 Bug Fixes

* release.yml の環境変数バグを修正 ([efccfc5](https://github.com/flipslidersand/forge/commit/efccfc530c2bcc95e45b412e1874ed67cf7573ea)), closes [#177](https://github.com/flipslidersand/forge/issues/177)

## 1.0.0 (2026-08-10)

### ✨ Features

* **attention:** Flash Attention 2 対応 op 拡張 ([#30](https://github.com/flipslidersand/forge/issues/30)) ([#41](https://github.com/flipslidersand/forge/issues/41)) ([ec8eaa2](https://github.com/flipslidersand/forge/commit/ec8eaa24ed01bed491c34a1ca7cf5cf6529191ab))
* cache に baseline/speedup を永続化し forge cache list に表示 ([#114](https://github.com/flipslidersand/forge/issues/114)) ([400b49b](https://github.com/flipslidersand/forge/commit/400b49b109cfc0198e8049ef38f4a963692980a8)), closes [#105](https://github.com/flipslidersand/forge/issues/105)
* **ci:** add GPU test job (RTX 4070 / RTX 4060 fallback) ([820c92d](https://github.com/flipslidersand/forge/commit/820c92d6a9f9a3d2024b52a9a11ee5e5234bc60a)), closes [#618](https://github.com/flipslidersand/forge/issues/618)
* **examples:** add optimize_rope.py demo for RoPE forge optimization ([bea0cf0](https://github.com/flipslidersand/forge/commit/bea0cf062ad821efdf0aabe6e89a34e4ea606f17)), closes [#152](https://github.com/flipslidersand/forge/issues/152)
* **examples:** add swiglu/rope/fused_add_rmsnorm to benchmark_all.py ([81c5183](https://github.com/flipslidersand/forge/commit/81c51835942593715ee74c5333e95b8be2ccfa96)), closes [#153](https://github.com/flipslidersand/forge/issues/153)
* **examples:** optimize_swiglu.py — SwiGLU エンドツーエンドサンプル ([#150](https://github.com/flipslidersand/forge/issues/150)) ([db50fa1](https://github.com/flipslidersand/forge/commit/db50fa13163bcf2dd63f011cc303c230159678f4)), closes [#147](https://github.com/flipslidersand/forge/issues/147)
* forge cache list / clear CLI 実装 ([#103](https://github.com/flipslidersand/forge/issues/103)) ([c663187](https://github.com/flipslidersand/forge/commit/c66318769db70b765f8ebcc96ada02fec5abc34e)), closes [#95](https://github.com/flipslidersand/forge/issues/95) [#95](https://github.com/flipslidersand/forge/issues/95)
* forge に Discord 通知を統合 ([#47](https://github.com/flipslidersand/forge/issues/47)) ([9f0f383](https://github.com/flipslidersand/forge/commit/9f0f383be22b0bf05f2f83efb8ff27150810d070)), closes [#completion](https://github.com/flipslidersand/forge/issues/completion) [#completion](https://github.com/flipslidersand/forge/issues/completion) [#errors](https://github.com/flipslidersand/forge/issues/errors) [#completion](https://github.com/flipslidersand/forge/issues/completion) [#errors](https://github.com/flipslidersand/forge/issues/errors)
* KernelRepository にコンテキストマネージャ実装 ([#101](https://github.com/flipslidersand/forge/issues/101)) ([bf8a733](https://github.com/flipslidersand/forge/commit/bf8a733376043f3341a7626771d85aacf2eb1be3)), closes [#72](https://github.com/flipslidersand/forge/issues/72)
* **layernorm:** two_pass/welford variant + SearchSpace 拡張 ([#58](https://github.com/flipslidersand/forge/issues/58)) ([#62](https://github.com/flipslidersand/forge/issues/62)) ([e003468](https://github.com/flipslidersand/forge/commit/e003468f9f0b442e6ce1d1503bdfbecb7101ca46))
* **linear:** linear op (GEMM) 追加 — tl.dot タイル GEMM カーネル ([#64](https://github.com/flipslidersand/forge/issues/64)) ([5ea7870](https://github.com/flipslidersand/forge/commit/5ea7870344b3fac02e56e57f934b2a702630b77b)), closes [#60](https://github.com/flipslidersand/forge/issues/60)
* LLM反復探索ループ + 採用判定 + baseline拡張 + bug fixes ([#9](https://github.com/flipslidersand/forge/issues/9)-[#11](https://github.com/flipslidersand/forge/issues/11) [#14](https://github.com/flipslidersand/forge/issues/14) [#15](https://github.com/flipslidersand/forge/issues/15) [#18](https://github.com/flipslidersand/forge/issues/18)) ([4e9dc9b](https://github.com/flipslidersand/forge/commit/4e9dc9b885ed4a29ee1f2907196c02ada7174702)), closes [#10](https://github.com/flipslidersand/forge/issues/10)
* **ops:** add fused_add_rmsnorm op — 残差加算 + RMSNorm 融合カーネル ([#156](https://github.com/flipslidersand/forge/issues/156)) ([10d03b3](https://github.com/flipslidersand/forge/commit/10d03b3021433f92fbd82d4409a90563de955a44)), closes [#151](https://github.com/flipslidersand/forge/issues/151) [#151](https://github.com/flipslidersand/forge/issues/151)
* **ops:** add LayerNorm and GELU ([#8](https://github.com/flipslidersand/forge/issues/8)) ([95045f7](https://github.com/flipslidersand/forge/commit/95045f72dde9e788700d2fb392ad5120e1bf2c4b))
* **ops:** add rope op — Rotary Position Embedding ([#149](https://github.com/flipslidersand/forge/issues/149)) ([024aa60](https://github.com/flipslidersand/forge/commit/024aa60cc37a22643bd7028c3b4aef62a37ceee6)), closes [#146](https://github.com/flipslidersand/forge/issues/146)
* **ops:** add swiglu op — SiLU-gated linear unit ([#145](https://github.com/flipslidersand/forge/issues/145)) ([683a8fe](https://github.com/flipslidersand/forge/commit/683a8fe6125f9b5dd22a8e03ada2cf086dee42f6)), closes [#144](https://github.com/flipslidersand/forge/issues/144)
* **ops:** OpDefinition dataclass + OP_REGISTRY 設計・実装 ([a8162c4](https://github.com/flipslidersand/forge/commit/a8162c4fc5d06db65ff139c9ebeb6ae446bd3945)), closes [#87](https://github.com/flipslidersand/forge/issues/87)
* **ops:** softmax two_pass variant — online softmax カーネル ([bc9986a](https://github.com/flipslidersand/forge/commit/bc9986a888590d17690c5ec562fe16b1b0ff3d7e)), closes [#155](https://github.com/flipslidersand/forge/issues/155)
* **phase2:** Triton codegen, GPU benchmark, and subprocess worker ([e1cee41](https://github.com/flipslidersand/forge/commit/e1cee4126b5e6aef309ea7c3b613d64076abadbe))
* **phase3:** search space, grid search, correctness suite, orchestrator ([51b4081](https://github.com/flipslidersand/forge/commit/51b4081929f9fee5453712d7abb4bd6ca6fd4fa4)), closes [#4](https://github.com/flipslidersand/forge/issues/4)
* **phase4:** multi_row + two_pass variants and random search ([ad856c9](https://github.com/flipslidersand/forge/commit/ad856c98fc214644bcb8b0122848f75d206bd745))
* **phase5:** CandidateGenerator protocol + LLM candidate generator ([b81303a](https://github.com/flipslidersand/forge/commit/b81303ad4491cf3adb71c3c83c7a2f78424ff8f6))
* **phase6:** [@forge](https://github.com/forge).optimize decorator + torch.fx lowering ([9d3336d](https://github.com/flipslidersand/forge/commit/9d3336d81ad23f498fbfd4ee8fc18fdd5d134e42))
* **sdpa:** flash_causal_opt variant + 探索空間拡張 ([#59](https://github.com/flipslidersand/forge/issues/59)) ([#63](https://github.com/flipslidersand/forge/issues/63)) ([4de29c2](https://github.com/flipslidersand/forge/commit/4de29c2d3c1af934ff987a2dd888bf4f29a2ea59))
* **search:** BayesianGenerator — Optuna TPE ベースの候補生成器 ([#131](https://github.com/flipslidersand/forge/issues/131)) ([4743a92](https://github.com/flipslidersand/forge/commit/4743a923e4d4a5389d8b122a27417ffa46165f94)), closes [#127](https://github.com/flipslidersand/forge/issues/127)
* **search:** OllamaGenerator — ローカル LLM による候補生成 ([#20](https://github.com/flipslidersand/forge/issues/20)) ([8a41e7d](https://github.com/flipslidersand/forge/commit/8a41e7d43c8a940b05a6a5a54632182b0b0f2649))
* **search:** Orchestrator.optimize_sha() — Successive Halving 探索 ([#133](https://github.com/flipslidersand/forge/issues/133)) ([0747678](https://github.com/flipslidersand/forge/commit/0747678b93e3b40d7de22e2bfa41effd342ead15)), closes [#128](https://github.com/flipslidersand/forge/issues/128)
* **search:** コスト考慮型探索 CostModel / BudgetTracker / scalarize 追加 ([#168](https://github.com/flipslidersand/forge/issues/168)) ([061a396](https://github.com/flipslidersand/forge/commit/061a39671f9165cbbdc64bc35854910ad00de81d))
* **search:** ライブ LLM 反復探索 IterativeLLMSearch 追加 ([#167](https://github.com/flipslidersand/forge/issues/167)) ([8b1943d](https://github.com/flipslidersand/forge/commit/8b1943de8aa1de15c400551ed42f11b001f03377))
* **softmax:** add Softmax op end-to-end, proving the pipeline generalizes ([f20cced](https://github.com/flipslidersand/forge/commit/f20cced95e0ab8f20ad1b50f7d6b59575ad60e85))
* torch.compile/autotune baseline 比較 ([#164](https://github.com/flipslidersand/forge/issues/164)) ([a1f4059](https://github.com/flipslidersand/forge/commit/a1f4059bd5cfde105e3221ef500c93c7353e2c0e)), closes [#50](https://github.com/flipslidersand/forge/issues/50) [#51](https://github.com/flipslidersand/forge/issues/51)
* 探索コスト考慮判定（パレート最適化）([#53](https://github.com/flipslidersand/forge/issues/53)) ([51e5814](https://github.com/flipslidersand/forge/commit/51e58142f2cbac2278cd1325fd84ee9408450b74))

### 🐛 Bug Fixes

* **cache:** from_dict に template_hash を追加 — [#116](https://github.com/flipslidersand/forge/issues/116) マージ後の TypeError を修正 ([#119](https://github.com/flipslidersand/forge/issues/119)) ([13ee1cd](https://github.com/flipslidersand/forge/commit/13ee1cd097d9324af27ef115fc766724e5e061e1)), closes [#109](https://github.com/flipslidersand/forge/issues/109)
* **ci:** arc-dev-nodee → ubuntu-latest（lint/test/typecheck） ([#66](https://github.com/flipslidersand/forge/issues/66)) ([3aff784](https://github.com/flipslidersand/forge/commit/3aff78491389bb8b4d5c93c5793f3c6e9dde1d0e))
* cross-round dedup と per-spec token_usage リセット ([#16](https://github.com/flipslidersand/forge/issues/16) [#17](https://github.com/flipslidersand/forge/issues/17)) ([69c7982](https://github.com/flipslidersand/forge/commit/69c79829f2845e6b0d4b72ffd88b7a0bae9420f9))
* gelu correctness_cases の入力生成を _gelu_inputs に分離 ([#97](https://github.com/flipslidersand/forge/issues/97)) ([c14b1a4](https://github.com/flipslidersand/forge/commit/c14b1a4b6f76a2980b36afb76603bb86b9bac705)), closes [#68](https://github.com/flipslidersand/forge/issues/68)
* graph_hash をテンプレート内容ハッシュに変更 ([#104](https://github.com/flipslidersand/forge/issues/104)) ([493334c](https://github.com/flipslidersand/forge/commit/493334cf65f2d86fce5332a4b31b4c7b32bb53b4)), closes [#93](https://github.com/flipslidersand/forge/issues/93) [#93](https://github.com/flipslidersand/forge/issues/93) [#93](https://github.com/flipslidersand/forge/issues/93)
* **kernel:** fp16 exp cast + rope kind=reduction for single_row kernel ([cafe24a](https://github.com/flipslidersand/forge/commit/cafe24a6c9885427061bccb24871c932c91e6ea9)), closes [#160](https://github.com/flipslidersand/forge/issues/160)
* **kernel:** fused_add_rmsnorm fp16 variance overflow ([0a715b8](https://github.com/flipslidersand/forge/commit/0a715b891a8ceab4afe8f3ca9576945fc923d2fe)), closes [#160](https://github.com/flipslidersand/forge/issues/160)
* **kernel:** replace tl.sigmoid with explicit formula; fix rope template key ([1124a58](https://github.com/flipslidersand/forge/commit/1124a58e507220b6ea6ec6f20ee21fa43c85afd7)), closes [#160](https://github.com/flipslidersand/forge/issues/160)
* **lint:** ruff F401/I001/E731/F841 修正 (unused import・lambda→def) ([85f4c3e](https://github.com/flipslidersand/forge/commit/85f4c3eeaadd437c5277f97ce928dd5ed814405e))
* **lint:** ruff UP037/E402 修正 (cost_model/__enter__型注釈・import順序) ([88c7cf5](https://github.com/flipslidersand/forge/commit/88c7cf550e32ceef168e5cb48de67678b9b87b1c))
* loader.py 一時カーネルファイルの atexit クリーンアップ実装 ([#96](https://github.com/flipslidersand/forge/issues/96)) ([18eecb7](https://github.com/flipslidersand/forge/commit/18eecb74cfcfcbeae830c26142550342a6d58daf)), closes [#67](https://github.com/flipslidersand/forge/issues/67)
* **ops:** add swiglu/rope/fused_add_rmsnorm to OP_INFO ([#159](https://github.com/flipslidersand/forge/issues/159)) ([b68c207](https://github.com/flipslidersand/forge/commit/b68c2075b9619445235940236367cbfa40bfe230)), closes [#158](https://github.com/flipslidersand/forge/issues/158)
* **orchestrator:** baseline を最初の成功計測で固定し speedup のブレを解消 ([#141](https://github.com/flipslidersand/forge/issues/141)) ([c87a821](https://github.com/flipslidersand/forge/commit/c87a821b1d026f92bc64f5f39cf998fac019f699)), closes [#140](https://github.com/flipslidersand/forge/issues/140)
* **pyright:** reduce type errors from 124 to 0 ([#36](https://github.com/flipslidersand/forge/issues/36)) ([fa3bc2f](https://github.com/flipslidersand/forge/commit/fa3bc2f3c7496c07bc2451642e24c136b2c13830)), closes [#28](https://github.com/flipslidersand/forge/issues/28)
* **reference:** sdpa_reference エイリアスを復元し ruff format を修正 ([20c05ff](https://github.com/flipslidersand/forge/commit/20c05ffe3621d929b7267ea2aa83f60482010532)), closes [#85](https://github.com/flipslidersand/forge/issues/85)
* SDPA SearchSpace が head_dim 非16倍数でも候補を列挙する問題を修正 ([#49](https://github.com/flipslidersand/forge/issues/49)) ([d6b7639](https://github.com/flipslidersand/forge/commit/d6b763972f18ded0d82bca5b61a23ef3c12a468b)), closes [#48](https://github.com/flipslidersand/forge/issues/48)
* **search:** BayesianGenerator の候補空間から未実装 variant を除外 ([#139](https://github.com/flipslidersand/forge/issues/139)) ([a2cdc3e](https://github.com/flipslidersand/forge/commit/a2cdc3ee1b09158de3464c26734c9d7cf53c7587)), closes [#137](https://github.com/flipslidersand/forge/issues/137)
* **search:** optimize_sha() で initial_budget < 64 のとき UserWarning を発行 ([#143](https://github.com/flipslidersand/forge/issues/143)) ([a8b1b55](https://github.com/flipslidersand/forge/commit/a8b1b55a338d5c5617b49496980a02c93c390a05)), closes [#142](https://github.com/flipslidersand/forge/issues/142)
* **types:** type: ignore コメントに理由を補記し pyright 0 errors を維持 ([7bfdca2](https://github.com/flipslidersand/forge/commit/7bfdca26f0982567a66947b04d535348c06d19fa)), closes [#90](https://github.com/flipslidersand/forge/issues/90)

### ♻️ Refactoring

* **benchmark:** BenchmarkResultDict TypedDict で from_dict の type:ignore を解消 ([#73](https://github.com/flipslidersand/forge/issues/73)) ([#107](https://github.com/flipslidersand/forge/issues/107)) ([a4872ea](https://github.com/flipslidersand/forge/commit/a4872ea4470aa2168b2291aa0d305e9cb429ddf4))
* **cache:** CacheKey.from_dict/from_json 追加と KernelSummary 3重スキーマ解消 ([#115](https://github.com/flipslidersand/forge/issues/115)) ([68598b3](https://github.com/flipslidersand/forge/commit/68598b326d394894f27a470d9924dd5589cef273)), closes [#110](https://github.com/flipslidersand/forge/issues/110)
* CandidateGenerator Protocol に reset_usage() を追加 ([#99](https://github.com/flipslidersand/forge/issues/99)) ([7cf697e](https://github.com/flipslidersand/forge/commit/7cf697ecd3cafcca964b370e01df75b2972683b5)), closes [#70](https://github.com/flipslidersand/forge/issues/70)
* **orchestrator:** _SearchContext + _prepare()/_finalize() 共通化 ([#81](https://github.com/flipslidersand/forge/issues/81), [#79](https://github.com/flipslidersand/forge/issues/79), [#80](https://github.com/flipslidersand/forge/issues/80)) ([#113](https://github.com/flipslidersand/forge/issues/113)) ([921ea8b](https://github.com/flipslidersand/forge/commit/921ea8b7e673ec0fdcde52b6768137b698bbe0ae))
* orchestrator.optimize() を 4 つの private メソッドに分割 ([#176](https://github.com/flipslidersand/forge/issues/176)) ([595d1da](https://github.com/flipslidersand/forge/commit/595d1daa1c87eba2eca59e15f579a12fcd8a7914)), closes [#173](https://github.com/flipslidersand/forge/issues/173)
* **orchestrator:** デフォルト KernelRepository のライフサイクル明示化 ([#73](https://github.com/flipslidersand/forge/issues/73)) ([#102](https://github.com/flipslidersand/forge/issues/102)) ([60e6f52](https://github.com/flipslidersand/forge/commit/60e6f522684dca0c715d85f13da1225698e50112))
* **reference:** OP_REGISTRY へ委譲して重複実装を削除 ([2a816ac](https://github.com/flipslidersand/forge/commit/2a816acc668bf21c0916638b53636370bd266b9d)), closes [#85](https://github.com/flipslidersand/forge/issues/85)
* **search:** _BaseGenerator ABC 導入 — LLM/Ollama Generator の重複集約 ([#78](https://github.com/flipslidersand/forge/issues/78), [#77](https://github.com/flipslidersand/forge/issues/77), [#76](https://github.com/flipslidersand/forge/issues/76)) ([#111](https://github.com/flipslidersand/forge/issues/111)) ([0ed0b3d](https://github.com/flipslidersand/forge/commit/0ed0b3d2dd92688899cf5dad70d585888dd6416a))
* **search:** _Candidate/_Proposal Pydantic モデルを _proposal_models.py に一本化 ([#112](https://github.com/flipslidersand/forge/issues/112)) ([726fbee](https://github.com/flipslidersand/forge/commit/726fbee0de4241260fb4a06d0299411b867df8a2)), closes [#75](https://github.com/flipslidersand/forge/issues/75)
* template_hash を CacheKey の専用フィールドに分離 ([#116](https://github.com/flipslidersand/forge/issues/116)) ([3f8fa75](https://github.com/flipslidersand/forge/commit/3f8fa759d2ee77fb82f60d50373e72dfba0f3017)), closes [#109](https://github.com/flipslidersand/forge/issues/109)
* **validation,codegen,search:** OP_REGISTRY を各所に適用 ([9bc9c54](https://github.com/flipslidersand/forge/commit/9bc9c54bcf83cd70542f297c19b3984021a70627)), closes [#82](https://github.com/flipslidersand/forge/issues/82) [#83](https://github.com/flipslidersand/forge/issues/83) [#84](https://github.com/flipslidersand/forge/issues/84) [#86](https://github.com/flipslidersand/forge/issues/86) [#82](https://github.com/flipslidersand/forge/issues/82) [#83](https://github.com/flipslidersand/forge/issues/83) [#84](https://github.com/flipslidersand/forge/issues/84) [#86](https://github.com/flipslidersand/forge/issues/86)
