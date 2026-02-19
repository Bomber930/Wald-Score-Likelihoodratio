# Wald / Score / LR シミュレーション教材レポート

- alpha: 0.05
- total rows: 227920

## このレポートの読み方

- 検定統計量: $W= r(\hat\theta)^\top[R\hat V(\hat\theta)R^\top]^{-1}r(\hat\theta)$, $S=U(\theta_0)^\top I(\theta_0)^{-1}U(\theta_0)$, $LR=2(\ell_{full}-\ell_{null})$。
- `size`: 帰無仮説が真のときの実測第I種過誤率（目標は alpha 付近）。
- `power_at_max_effect`: 最大効果量条件での棄却率。
- `failure_rate`: 計算失敗率（未収束、特異行列、NaN、分散/Hessian 計算失敗など）。
- `mean_runtime_ms`: 検定1件あたりの平均計算時間。


## Regular: Normal mean (sigma known)

- slug: `normal_mean_known_sigma`
- notes: 正則・解析的に扱える設定。nが増えるとWald/Score/LRはほぼ一致する。
- main: H0: mu = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- $X_i \sim \mathcal{N}(\mu,\sigma^2)$（$\sigma^2$ 既知）、帰無仮説は $H_0: \mu=0$。

**なぜこのシナリオを検証するか**
- すべての正則条件が成立する、基準となるクリーンな設定です。

**想定振る舞いが起きる理由**
- 尤度が滑らかで情報行列が安定しているため、Wald・Score・LRは同じ漸近的なカイ二乗近似を共有します。

**結果を見る前の期待パターン**
- 3検定で size/power が近く、failure_rate はほぼゼロになることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- サイズのばらつき（max-min）=0.000、パワーのばらつき（max-min）=0.000、最大 failure_rate=0.000
- 判定: 想定どおり、3検定はほぼ一致しています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.058 | 0.990 | 0.000 | 0.078 |
| score | 0.058 | 0.990 | 0.000 | 0.086 |
| wald | 0.058 | 0.990 | 0.000 | 0.109 |

### Interpretation

n増加で3検定が近づくか、または設定由来の不安定性がどの検定に出るかを確認できます。

### Figures

- ![pvalue_hist.png](normal_mean_known_sigma/pvalue_hist.png)
- ![size_vs_n.png](normal_mean_known_sigma/size_vs_n.png)
- ![power_curves.png](normal_mean_known_sigma/power_curves.png)
- ![failure_rate_vs_n.png](normal_mean_known_sigma/failure_rate_vs_n.png)
- ![runtime_vs_n.png](normal_mean_known_sigma/runtime_vs_n.png)

## Regular: Linear regression (single constraint)

- slug: `linear_regression_single_constraint`
- notes: 正則条件下の基礎例。単一・複数制約ともにn増加で3検定が漸近的一致を示す。
- main: H0: beta_1 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- $y=X\beta+\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma^2I)$、検定は $H_0: \beta_j=0$。

**なぜこのシナリオを検証するか**
- 実務で最も頻出する検定設定であり、正則ケースの基準として有用です。

**想定振る舞いが起きる理由**
- 線形ガウス仮定が正しければ OLS/MLE と共分散推定が安定し、検定統計量は漸近的に揃います。

**結果を見る前の期待パターン**
- 検定間の size と power の差は小さいことが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- サイズのばらつき（max-min）=0.009、パワーのばらつき（max-min）=0.009、最大 failure_rate=0.000
- 判定: 想定どおり、3検定はほぼ一致しています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.059 | 0.964 | 0.000 | 0.261 |
| score | 0.056 | 0.959 | 0.000 | 0.199 |
| wald | 0.064 | 0.969 | 0.000 | 0.206 |

### Interpretation

n増加で3検定が近づくか、または設定由来の不安定性がどの検定に出るかを確認できます。

### Figures

- ![pvalue_hist.png](linear_regression_single_constraint/pvalue_hist.png)
- ![size_vs_n.png](linear_regression_single_constraint/size_vs_n.png)
- ![power_curves.png](linear_regression_single_constraint/power_curves.png)
- ![failure_rate_vs_n.png](linear_regression_single_constraint/failure_rate_vs_n.png)
- ![runtime_vs_n.png](linear_regression_single_constraint/runtime_vs_n.png)

## Regular: Linear regression (multiple constraints)

- slug: `linear_regression_multiple_constraints`
- notes: 正則条件下の基礎例。単一・複数制約ともにn増加で3検定が漸近的一致を示す。
- main: H0: beta_1, beta_2 = 0 (df=2)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 同じ線形モデルだが、帰無仮説は同時制約 $H_0: R\beta=0$（複数制約）。

**なぜこのシナリオを検証するか**
- df > 1 の制約でも同じ漸近的一致が成り立つかを確認します。

**想定振る舞いが起きる理由**
- 正則条件下では、同時 Wald/Score/LR は同じ局所二次近似を対象にします。

**結果を見る前の期待パターン**
- 検定間の振る舞いは似通い、n の増加とともに一致が強まることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- サイズのばらつき（max-min）=0.019、パワーのばらつき（max-min）=0.006、最大 failure_rate=0.000
- 判定: 想定どおり、3検定はほぼ一致しています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.066 | 0.995 | 0.000 | 0.245 |
| score | 0.058 | 0.991 | 0.000 | 0.184 |
| wald | 0.076 | 0.997 | 0.000 | 0.199 |

### Interpretation

n増加で3検定が近づくか、または設定由来の不安定性がどの検定に出るかを確認できます。

### Figures

- ![pvalue_hist.png](linear_regression_multiple_constraints/pvalue_hist.png)
- ![size_vs_n.png](linear_regression_multiple_constraints/size_vs_n.png)
- ![power_curves.png](linear_regression_multiple_constraints/power_curves.png)
- ![failure_rate_vs_n.png](linear_regression_multiple_constraints/failure_rate_vs_n.png)
- ![runtime_vs_n.png](linear_regression_multiple_constraints/runtime_vs_n.png)

## Separation: Logistic (full MLE unstable)

- slug: `logistic_separation`
- notes: 完全/準分離でfull MLEが壊れるとWald/LRは失敗しやすい。帰無モデル依存のScoreは残りやすい。
- main: H0: beta_1 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 二値ロジスティックモデル $\mathrm{logit}(p_i)=\beta_0+\beta_1x_i$ で、（準）分離が起こるようにデータを設計した設定です。
  - 分離...線形予測子の符号でyをほぼ完全に判別できてしまう場面

**なぜこのシナリオを検証するか**
- full モデルの MLE が不安定化・発散したときに何が起きるかを示します。

**想定振る舞いが起きる理由**
- 分離では $|\hat\beta_1|$ が発散し得て、Hessian や共分散推定が不安定になります。Wald/LR は full 推定に依存しますが、Score は帰無モデル推定のみで計算できます。

**結果を見る前の期待パターン**
- Wald/LR の failure が高く、Score の failure は低くなることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- failure_rate: Wald=0.334, Score=0.000, LR=0.334
- 判定: 想定どおり、Score は生き残り、Wald/LR は失敗しやすくなっています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.053 | NA | 0.334 | 2.186 |
| score | 0.050 | 1.000 | 0.000 | 0.843 |
| wald | 0.050 | NA | 0.334 | 1.492 |

### Interpretation

分離設定では失敗率の高い検定が目立ち、特に `lr` が不安定です。 一方で `score` は比較的計算が成立しやすく、帰無点ベース検定の利点が出ます。

### Figures

- ![pvalue_hist.png](logistic_separation/pvalue_hist.png)
- ![size_vs_n.png](logistic_separation/size_vs_n.png)
- ![power_curves.png](logistic_separation/power_curves.png)
- ![failure_rate_vs_n.png](logistic_separation/failure_rate_vs_n.png)
- ![runtime_vs_n.png](logistic_separation/runtime_vs_n.png)

## Finite-sample: Logistic small n

- slug: `logistic_small_sample`
- notes: 小標本では漸近近似誤差でサイズ/パワーに差が出る。n増加で差は縮まる。
- main: H0: beta_1 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 小標本ロジスティック回帰: $y_i\sim\mathrm{Bernoulli}(p_i)$, $\mathrm{logit}(p_i)=\beta_0+\beta_1x_i$。

**なぜこのシナリオを検証するか**
- 漸近近似が十分効く前の有限標本ギャップを可視化します。

**想定振る舞いが起きる理由**
- n が小さいと正規近似・カイ二乗近似が粗く、検定ごとにキャリブレーションや power がずれます。

**結果を見る前の期待パターン**
- 小さい n では、size/power に中程度の差が現れることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- サイズのばらつき（max-min）=0.019、パワーのばらつき（max-min）=0.074
- 判定: 想定どおり、有限標本ギャップが観測されています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.052 | 0.726 | 0.000 | 1.712 |
| score | 0.049 | 0.711 | 0.000 | 0.853 |
| wald | 0.033 | 0.652 | 0.000 | 1.025 |

### Interpretation

n増加で3検定が近づくか、または設定由来の不安定性がどの検定に出るかを確認できます。

### Figures

- ![pvalue_hist.png](logistic_small_sample/pvalue_hist.png)
- ![size_vs_n.png](logistic_small_sample/size_vs_n.png)
- ![power_curves.png](logistic_small_sample/power_curves.png)
- ![failure_rate_vs_n.png](logistic_small_sample/failure_rate_vs_n.png)
- ![runtime_vs_n.png](logistic_small_sample/runtime_vs_n.png)

## Practical: Wald batch speed for many hypotheses

- slug: `wald_runtime_batch`
- notes: 多数係数を順次検定する実務想定。1回のfull推定を再利用できるWaldはLR/Scoreより高速になりやすい。
- beta_1: H0: beta_1 = 0 (df=1)
- beta_2: H0: beta_2 = 0 (df=1)
- beta_3: H0: beta_3 = 0 (df=1)
- beta_4: H0: beta_4 = 0 (df=1)
- beta_5: H0: beta_5 = 0 (df=1)
- beta_6: H0: beta_6 = 0 (df=1)
- beta_7: H0: beta_7 = 0 (df=1)
- beta_8: H0: beta_8 = 0 (df=1)
- beta_9: H0: beta_9 = 0 (df=1)
- beta_10: H0: beta_10 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 1つの full 線形モデルに対して、多数の係数仮説 $H_0:\beta_j=0$ を順次検定します。

**なぜこのシナリオを検証するか**
- 実務では多数係数を検定することが多く、計算時間が重要になります。

**想定振る舞いが起きる理由**
- Wald は1回の full 推定を使い回せますが、LR/Score は制約モデル・帰無モデルの反復計算が必要になりがちです。

**結果を見る前の期待パターン**
- 平均 runtime で Wald が明確に速いことが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- 平均 runtime (ms): Wald=0.142, Score=0.467, LR=0.417
- 判定: 想定どおり、速度優位を確認（Score/Wald=3.30, LR/Wald=2.95）。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.074 | 0.523 | 0.000 | 0.417 |
| score | 0.072 | 0.521 | 0.000 | 0.467 |
| wald | 0.077 | 0.524 | 0.000 | 0.142 |

### Interpretation

多数仮説の同時処理では、Waldがfull推定再利用できるため平均runtimeで有利になりやすい設定です。

補足（size が 0.05 より大きい理由と対策）:
- 今回は `size` が `Wald=0.077, Score=0.072, LR=0.074` と、目標 `alpha=0.05` より高めです。主因は有限標本での漸近近似誤差です。`p_features=20` に対して `n=80,150,300` は「十分大標本」とは限らず、$\chi^2$ 近似がやや楽観的になります。
- 実装上、線形回帰の分散推定は $\hat{\sigma}^2=\mathrm{SSE}/n$（MLE型）を使っており、有限標本では分散を小さめに見積もりやすいです。すると標準誤差が小さくなり、検定統計量が大きくなって棄却率が上振れします。
- 対策1: 分散推定を不偏型 $\hat{\sigma}^2=\mathrm{SSE}/(n-p)$ に変更する。これは自由度補正で分散の過小評価を緩和し、Wald統計量の過大化を抑えます。
- 対策2: 有限標本向けの参照分布（線形回帰なら t/F）を使う。漸近 $\chi^2$ よりも小標本の分布形に合うため、size の校正が改善します。
- 対策3: `n` を増やす（または `p/n` を下げる）。漸近理論が効きやすくなり、Wald/Score/LR の経験的 size は 0.05 に近づきます。

### Figures

- ![pvalue_hist.png](wald_runtime_batch/pvalue_hist.png)
- ![size_vs_n.png](wald_runtime_batch/size_vs_n.png)
- ![power_curves.png](wald_runtime_batch/power_curves.png)
- ![failure_rate_vs_n.png](wald_runtime_batch/failure_rate_vs_n.png)
- ![runtime_vs_n.png](wald_runtime_batch/runtime_vs_n.png)

## Non-regular: ZIP boundary H0: pi=0

- slug: `zip_boundary`
- notes: 境界仮説でχ²近似が壊れ、Wald/Score/LRすべてサイズが歪み得る。LRのparametric bootstrapで校正可能。
- main: H0: pi = 0 (boundary) (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- ZIP モデル $P(Y=0)=\pi+(1-\pi)e^{-\lambda}$、境界帰無仮説は $H_0:\pi=0$。

**なぜこのシナリオを検証するか**
- 境界帰無は非正則であり、ナイーブなカイ二乗近似は破綻し得ます。

**想定振る舞いが起きる理由**
- 境界では通常の漸近展開が壊れます。Parametric bootstrap は帰無分布を経験的に近似します。

**結果を見る前の期待パターン**
- ナイーブ検定は size がずれる可能性があり、bootstrap は計算コスト増と引き換えに校正改善が期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- size: Wald=0.019, Score=0.031, LR=0.019, LR_bootstrap=0.013（目標 alpha=0.05）
- 判定: この実行では bootstrap 改善は明確ではありません（B や n を増やして再評価）。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.019 | 0.531 | 0.000 | 3.294 |
| lr_bootstrap | 0.013 | 0.469 | 0.000 | 159.782 |
| score | 0.031 | 0.531 | 0.000 | 0.176 |
| wald | 0.019 | 0.600 | 0.000 | 3.368 |

### Interpretation

境界仮説では通常のχ²近似が当てはまりにくく、サイズがalphaからずれます。 `lr_bootstrap` のサイズが他よりalphaに近ければ、校正の有効性が確認できます。

### Figures

- ![pvalue_hist.png](zip_boundary/pvalue_hist.png)
- ![size_vs_n.png](zip_boundary/size_vs_n.png)
- ![power_curves.png](zip_boundary/power_curves.png)
- ![failure_rate_vs_n.png](zip_boundary/failure_rate_vs_n.png)
- ![runtime_vs_n.png](zip_boundary/runtime_vs_n.png)

## Weak identification: severe collinearity

- slug: `collinearity`
- notes: 強共線性で情報行列がほぼ特異となり、分散推定不安定や失敗率上昇が起こる。
- main: H0: beta_1 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 強い共線性（例: $x_2\approx x_1$）を持つ線形モデルで、$X^TX$ がほぼ特異になります。

**なぜこのシナリオを検証するか**
- 弱識別と、分散/Hessian の逆行列計算不安定を示す設定です。

**想定振る舞いが起きる理由**
- 情報行列がほぼ特異だと数値誤差が増幅し、検定そのものが失敗し得ます。

**結果を見る前の期待パターン**
- failure_rate が上昇し、推論が不安定になることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- failure_rate: Wald=1.000, Score=0.338, LR=1.000
- 判定: 想定どおり、弱識別に伴う強い不安定性が観測されています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | NA | NA | 1.000 | 0.176 |
| score | 0.056 | 0.061 | 0.338 | 0.187 |
| wald | NA | NA | 1.000 | 0.074 |

### Interpretation

共線性で情報行列が不安定になり、失敗率上昇や推定分散の増大が確認できます。

### Figures

- ![pvalue_hist.png](collinearity/pvalue_hist.png)
- ![size_vs_n.png](collinearity/size_vs_n.png)
- ![power_curves.png](collinearity/power_curves.png)
- ![failure_rate_vs_n.png](collinearity/failure_rate_vs_n.png)
- ![runtime_vs_n.png](collinearity/runtime_vs_n.png)

## Misspecification: True NegBin, fitted Poisson

- slug: `poisson_misspec`
- notes: 真の分布が過分散なのにPoisson仮定で推定。3検定が同時にサイズ崩壊する例。
- main: H0: beta_1 = 0 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- 真のデータは過分散（NegBin 的）だが、Poisson GLM を当てはめる設定です。

**なぜこのシナリオを検証するか**
- モデルミススペック下では、3検定が揃っていても誤ることを示します。

**想定振る舞いが起きる理由**
- 尤度・分散仮定が誤ると、漸近近似が3検定で同時にバイアスし得ます。

**結果を見る前の期待パターン**
- Wald/Score/LR すべてで系統的な size 歪みが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- size: Wald=0.189, Score=0.189, LR=0.189, 平均 |size-alpha|=0.139
- 判定: 想定どおり、モデルミススペック由来の size 歪みが3検定で同時に観測されています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr | 0.189 | 0.989 | 0.000 | 1.811 |
| score | 0.189 | 0.989 | 0.000 | 0.909 |
| wald | 0.189 | 0.989 | 0.000 | 1.068 |

### Interpretation

モデルミススペック下ではサイズがalpha=0.05から体系的に外れ、3検定が同時に崩れることがあります。

### Figures

- ![pvalue_hist.png](poisson_misspec/pvalue_hist.png)
- ![size_vs_n.png](poisson_misspec/size_vs_n.png)
- ![power_curves.png](poisson_misspec/power_curves.png)
- ![failure_rate_vs_n.png](poisson_misspec/failure_rate_vs_n.png)
- ![runtime_vs_n.png](poisson_misspec/runtime_vs_n.png)

## Invariance demo: beta=0 vs OR=1

- slug: `invariance_demo`
- notes: 同値仮説でもWaldはパラメータ化依存でズレ得る。LRは同じfull/null比較なので相対的に不変。
- beta: H0: beta_1 = 0 (df=1)
- or: H0: OR = 1 (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- ロジスティックモデルで $\mathrm{OR}=\exp(\beta_1)$。同値な帰無仮説 $H_0:\beta_1=0$ と $H_0:\mathrm{OR}=1$ を比較します。

**なぜこのシナリオを検証するか**
- 有限標本での Wald のパラメータ化依存性を示します。

**想定振る舞いが起きる理由**
- Wald は推定点近傍の局所線形化に依存するため、再パラメータ化で統計量が変わり得ます。

**結果を見る前の期待パターン**
- Wald(beta) と Wald(OR) はずれ得る一方、LR はこの変換に比較的不変です。

### 今回の結果で実際に起きたか（実測ベース）

- Wald(beta) と Wald(OR): size=0.048 vs 0.067, power=0.688 vs 0.422
- 判定: 想定どおり、Wald のパラメータ化依存が確認できます。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr:beta | 0.068 | 0.736 | 0.000 | 2.303 |
| score:beta | 0.062 | 0.719 | 0.000 | 1.162 |
| wald:beta | 0.048 | 0.688 | 0.000 | 1.361 |
| wald:or | 0.067 | 0.422 | 0.000 | 1.367 |

### Interpretation

同値仮説 `beta=0` と `OR=1` でWaldの値がずれ得る一方、LRは同じfull/null比較のため相対的に不変です。

### Figures

- ![pvalue_hist.png](invariance_demo/pvalue_hist.png)
- ![size_vs_n.png](invariance_demo/size_vs_n.png)
- ![power_curves.png](invariance_demo/power_curves.png)
- ![failure_rate_vs_n.png](invariance_demo/failure_rate_vs_n.png)
- ![runtime_vs_n.png](invariance_demo/runtime_vs_n.png)

## Overfitting/p-hacking: stepwise then test

- slug: `phacking_stepwise`
- notes: 変数選択後に同データで検定すると偽陽性率が増える。探索/検定分離でサイズが回復する。
- same_data: H0: selected beta = 0 (selection and testing on same data) (df=1)
- split_data: H0: selected beta = 0 (selection on train, testing on test split) (df=1)

### 背景（モデル・狙い・想定振る舞い）

**モデル仮定**
- データ駆動で変数選択した後に同じデータで検定する場合と、sample split で検定する場合を比較します。

**なぜこのシナリオを検証するか**
- 選択後バイアスと偽陽性率の膨張を示します。

**想定振る舞いが起きる理由**
- 変数選択は名目 p 値の校正を壊します。データ分割は探索と確認を分離します。

**結果を見る前の期待パターン**
- 帰無下で same_data の棄却率 > split_data の棄却率となることが期待されます。

### 今回の結果で実際に起きたか（実測ベース）

- wald: same_data=0.677, split_data=0.065, 差=0.611
- score: same_data=0.653, split_data=0.058, 差=0.595
- lr: same_data=0.663, split_data=0.062, 差=0.601
- 判定: 想定どおり、p-hacking による膨張が強く観測され、sample split で緩和されています。

### Summary

| test_label | size | power_at_max_effect | failure_rate | mean_runtime_ms |
| --- | --- | --- | --- | --- |
| lr:same_data | 0.663 | 0.663 | 0.000 | 0.289 |
| lr:split_data | 0.062 | 0.062 | 0.000 | 0.292 |
| score:same_data | 0.653 | 0.653 | 0.000 | 0.217 |
| score:split_data | 0.058 | 0.058 | 0.000 | 0.219 |
| wald:same_data | 0.677 | 0.677 | 0.000 | 0.249 |
| wald:split_data | 0.065 | 0.065 | 0.000 | 0.242 |

### Interpretation

同一データ選択の系列（`*:same_data`）は、split系列（`*:split_data`）よりサイズが大きくなりやすく、選択バイアスによる偽陽性増加が見えます。

### Figures

- ![pvalue_hist.png](phacking_stepwise/pvalue_hist.png)
- ![size_vs_n.png](phacking_stepwise/size_vs_n.png)
- ![power_curves.png](phacking_stepwise/power_curves.png)
- ![failure_rate_vs_n.png](phacking_stepwise/failure_rate_vs_n.png)
- ![runtime_vs_n.png](phacking_stepwise/runtime_vs_n.png)
