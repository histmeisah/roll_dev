# Section 4.1 理论推导引用链

## Priority Staleness and Exponential Age Decay

推导链引用了 4 篇文献：

| 引用 | 论文 | 在推导中的作用 |
|------|------|--------------|
| `schaul2016per` | Schaul et al., *Prioritized Experience Replay*, ICLR 2016 | 标准 PER 框架基础，base priority 的来源 |
| `schulman2015trpo` | Schulman et al., *Trust Region Policy Optimization*, ICML 2015 | 提供 "KL divergence 随梯度步数线性增长" 的理论依据（Eq. 4） |
| `metelli2020importance` | Metelli et al., *Importance Sampling in RL*, 2020 | 提供 χ²-divergence 与 Rényi divergence 的关系：χ² = exp(D₂) - 1，建立 importance ratio 方差下界（Eq. 5） |
| `kong1992note` | Kong, *A Note on Importance Sampling*, 1992 | 提供 ESS 经典定义：ESS = n / (1 + Var[ρ])（Eq. 6） |

## 推导逻辑链

1. **Schulman (TRPO)** → KL 随 Δ 线性增长
2. **Metelli** → Var[ρ] 随 KL 指数增长
3. **Kong** → ESS 随 Var[ρ] 衰减 → ESS 随 Δ **指数衰减**
4. 结论 → 用 exp(-Δ/τ) 做 age decay 是理论上合理的
