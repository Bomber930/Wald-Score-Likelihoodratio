# Wald / Score / LR Comparative Simulation教材

Pythonで `Wald検定 / Score検定 / 尤度比(LR)検定` を体感比較するための教材コードです。  
正則条件での一致だけでなく、有限標本・数値不安定・非正則・ミススペック・p-hackingで差が出る状況を可視化します。

## 要件
- Python 3.11+
- `numpy`, `scipy`, `pandas`, `statsmodels`, `matplotlib`（`seaborn`未使用）

## 実行方法
```bash
python scripts/run_all.py --output-dir outputs
```

主なオプション:
```bash
python scripts/run_all.py --list-scenarios
python scripts/run_all.py --scenarios normal_mean_known_sigma,logistic_separation --reps 200
python scripts/run_all.py --n-list 40,80,160 --effect-list 0,0.2,0.5
```

出力:
- `outputs/results.csv`: 全反復の生データ（失敗記録含む）
- `outputs/<scenario_slug>/*.png`: 必須5図
- `outputs/report.md`: 各シナリオの自動説明付きレポート

## シナリオ一覧（狙い）
1. `normal_mean_known_sigma`  
正則・解析的ベースライン。3検定の漸近一致を確認。
2. `linear_regression_single_constraint`  
線形回帰の単一制約で一致を確認。
3. `linear_regression_multiple_constraints`  
線形回帰の複数制約で一致を確認。
4. `logistic_separation`  
完全/準分離でfull MLEが壊れ、Wald/LRが失敗しやすい状況。
5. `logistic_small_sample`  
小標本でサイズ/パワーがずれる状況。
6. `wald_runtime_batch`  
多数仮説を高速処理するWaldの実務優位（runtime比較）。
7. `zip_boundary`  
境界仮説 `H0: pi=0` で3検定のナイーブ近似崩壊と、LR bootstrap校正。
8. `collinearity`  
強共線性で情報行列がほぼ特異、失敗率と不安定性を観察。
9. `poisson_misspec`  
真がNegBinなのにPoissonを当てるミススペックでサイズ崩壊。
10. `invariance_demo`  
同値仮説 `beta=0` と `OR=1` でWaldのパラメータ化依存を確認。
11. `phacking_stepwise`  
変数選択後の同データ検定で偽陽性増加、サンプルスプリットで回復。

## 実装上のポイント
- 乱数は `numpy.random.Generator(np.random.PCG64(seed))` で管理
- 収束失敗・分散計算失敗・行列特異・NaNは例外で落とさず `failed/error_type` として記録
- 検定結果は統一フォーマットで保存:
  - `df`, `pvalue`, `stat`, `runtime_ms`, `converged_full`, `converged_null`, `error_type`

## Added Scenarios
- `linear_regression_heteroskedastic`: 不均一分散下で通常Wald/Score/LRと`wald_hc3`（robust Wald）を比較。
- `logistic_clustered`: クラスタ依存データで独立仮定の3検定と`wald_cluster`（cluster-robust Wald）を比較。
- `logistic_local_alternatives`: `beta_1=c/sqrt(n)` の局所対立でサイズ・パワーの漸近挙動を比較。
