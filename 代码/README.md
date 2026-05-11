# Numerical Experiments for Terminal Optimal Control Problems

This directory contains the reproducible numerical experiments used in Chapters 4 and 5 of the thesis
*Numerical Solution of Terminal Optimal Control Problems with Linear Dynamics*.

The experiments implement low-dimensional shooting-type formulations for terminal optimal control
problems with linear dynamics, compare them with a direct discretization baseline, and regenerate the
tables and figures used in the numerical section of the thesis.

## Requirements

Install the Python dependencies from the thesis repository root:

```powershell
pip install -r .\代码\requirements.txt
```

The main dependencies are NumPy, SciPy, pandas, and Matplotlib.

## Running the Experiments

Run the script from the thesis repository root:

```powershell
python .\代码\terminal_control_experiments.py
```

The script regenerates the files under `代码/results`. Existing result files in that directory are
overwritten.

Generated outputs include:

- `代码/results/summary.csv`: summary table for all test cases.
- `代码/results/*_history.csv`: iteration history for each test case.
- `代码/results/*_state.csv`: discrete state trajectory for each test case.
- `代码/results/*_control.csv`: discrete control sequence for each test case.
- `代码/results/figures/*.png`: residual histories, state and control plots, and phase portraits.

The `case2_M_*` outputs are used for the grid-sensitivity study of the two-dimensional terminal
tracking example and currently include `M=50,100,200,400,800`. The `case2_direct_M_200` output is a
direct discretization comparison in which the optimization variables are the control values on the time
grid.

## Method Correspondence

The script implements the following steps from the thesis:

- Explicit Euler propagation of the state:

```tex
y_{m+1}=y_m+\tau(Ay_m+Bu_m)
```

- Sampling of the adjoint variable:

```tex
\psi_m(q)=e^{A^\top(T-t_m)}q
```

- Box-constrained control reconstruction and smoothing:

```tex
u_{m,i}^{\mu}=
\frac{u_{\max,i}\eta_{m,i}}{\sqrt{\eta_{m,i}^2+\mu^2}}
```

- Residual for the quadratic terminal-error problem:

```tex
F(q)=q+y_M(q)-x_T
```

- Residual for the affine terminal-constraint problem:

```tex
G(\lambda)=Cy_M(\lambda)-d
```

- Finite-difference Jacobians, Armijo line search, and Gauss-Newton directions.

The script also includes a direct discretization baseline. It uses the same explicit Euler state
propagation, but optimizes directly over `u_0,\ldots,u_{M-1}` instead of a terminal adjoint parameter.
The box constraints `|u_m|\leq u_{\max}` are handled with `scipy.optimize.minimize` and the `L-BFGS-B`
method. This direct method is included only as a numerical comparison baseline; it is not the main
algorithm proposed in the thesis.
