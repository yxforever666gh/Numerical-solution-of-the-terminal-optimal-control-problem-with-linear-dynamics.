# 具有线性动力学的终端型最优控制数值实验

本目录给出论文第4章和第5章对应的可复现实验代码。

## 运行方式

依赖环境：

```powershell
pip install -r .\代码\requirements.txt
```

在论文目录下运行：

```powershell
python .\代码\terminal_control_experiments.py
```

脚本会重新生成 `代码/results` 下的实验文件；如果该目录中已有旧结果，运行后会被覆盖。代码会生成：

- `代码/results/summary.csv`：所有算例的汇总表。
- `代码/results/*_history.csv`：每个算例的迭代历史。
- `代码/results/*_state.csv`：每个算例的离散状态序列。
- `代码/results/*_control.csv`：每个算例的离散控制序列。
- `代码/results/figures/*.png`：残差下降、状态曲线、控制曲线和相轨线图。

其中 `case2_M_*` 用于二维终端跟踪算例的网格敏感性分析，当前包含
`M=50,100,200,400,800`；`case2_direct_M_200` 为直接离散优化对比实验，
其优化变量是每个时间网格上的控制值。

## 方法对应关系

脚本实现了论文中的以下步骤：

- 显式 Euler 状态递推：

```tex
y_{m+1}=y_m+\tau(Ay_m+Bu_m)
```

- 伴随变量采样：

```tex
\psi_m(q)=e^{A^\top(T-t_m)}q
```

- 盒约束控制生成及光滑化：

```tex
u_{m,i}^{\mu}=
\frac{u_{\max,i}\eta_{m,i}}{\sqrt{\eta_{m,i}^2+\mu^2}}
```

- 二次终端误差残差：

```tex
F(q)=q+y_M(q)-x_T
```

- 仿射终端约束残差：

```tex
G(\lambda)=Cy_M(\lambda)-d
```

- 有限差分 Jacobian、Armijo 线搜索和高斯--牛顿方向。

此外，脚本还实现了一个用于对比的直接离散优化方法。该方法仍采用同一显式 Euler
状态递推，但不使用伴随终端参数，而是直接以
`u_0,\ldots,u_{M-1}` 为优化变量，并通过 `scipy.optimize.minimize`
的 `L-BFGS-B` 方法处理盒约束 `|u_m|\leq u_{\max}`。该直接法仅作为数值对比基准，
不是论文提出的主要算法。
