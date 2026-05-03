from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


Array = np.ndarray


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


@dataclass
class SolverOptions:
    max_iter: int = 120
    residual_tol: float = 1e-6
    grad_tol: float = 1e-8
    fd_step: float = 1e-6
    armijo_beta: float = 0.5
    armijo_sigma: float = 1e-4
    damping: float = 1e-8
    method: str = "gn"


@dataclass
class SolveResult:
    z: Array
    residual: Array
    theta: float
    grad_norm: float
    iterations: int
    converged: bool
    history: pd.DataFrame


@dataclass
class DirectSolveResult:
    y: Array
    u: Array
    theta: float
    terminal_error: float
    grad_norm: float
    iterations: int
    success: bool
    message: str
    history: pd.DataFrame


def safe_label(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return text


def as_col(x: Array | list[float] | tuple[float, ...]) -> Array:
    return np.asarray(x, dtype=float).reshape(-1)


def smooth_box_control(eta: Array, umax: Array, mu: float) -> Array:
    eta = as_col(eta)
    umax = as_col(umax)
    if mu <= 0:
        return umax * np.sign(eta)
    return umax * eta / np.sqrt(eta * eta + mu * mu)


def psi_samples(A: Array, q: Array, T: float, M: int) -> Array:
    q = as_col(q)
    times = np.linspace(0.0, T, M + 1)
    return np.vstack([expm(A.T * (T - t)) @ q for t in times])


def transition_samples(A: Array, T: float, M: int) -> Array:
    times = np.linspace(0.0, T, M + 1)
    return np.stack([expm(A.T * (T - t)) for t in times], axis=0)


def simulate_linear(
    A: Array,
    B: Array,
    x0: Array,
    T: float,
    M: int,
    q: Array,
    umax: Array,
    mu: float,
    transitions: Array | None = None,
) -> tuple[Array, Array, Array]:
    x0 = as_col(x0)
    tau = T / M
    if transitions is None:
        transitions = transition_samples(A, T, M)
    psi = np.einsum("mij,j->mi", transitions, q)
    y = np.zeros((M + 1, x0.size), dtype=float)
    u = np.zeros((M, B.shape[1]), dtype=float)
    y[0] = x0
    for m in range(M):
        eta = B.T @ psi[m]
        u[m] = smooth_box_control(eta, umax, mu)
        y[m + 1] = y[m] + tau * (A @ y[m] + B @ u[m])
    return y, u, psi


def simulate_linear_with_controls(A: Array, B: Array, x0: Array, T: float, u: Array) -> Array:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = as_col(x0)
    u = np.asarray(u, dtype=float)
    M = u.shape[0]
    tau = T / M
    y = np.zeros((M + 1, x0.size), dtype=float)
    y[0] = x0
    for m in range(M):
        y[m + 1] = y[m] + tau * (A @ y[m] + B @ u[m])
    return y


def finite_difference_jacobian(residual_fn: Callable[[Array], Array], z: Array, h: float) -> Array:
    z = as_col(z)
    r0 = residual_fn(z)
    jac = np.zeros((r0.size, z.size), dtype=float)
    for j in range(z.size):
        dz = np.zeros_like(z)
        dz[j] = h
        jac[:, j] = (residual_fn(z + dz) - r0) / h
    return jac


def solve_least_squares(
    residual_fn: Callable[[Array], Array],
    z0: Array,
    options: SolverOptions,
) -> SolveResult:
    z = as_col(z0).copy()
    rows: list[dict[str, float | int | bool]] = []
    converged = False
    grad_norm = np.inf

    for k in range(options.max_iter + 1):
        r = residual_fn(z)
        theta = 0.5 * float(r @ r)
        jac = finite_difference_jacobian(residual_fn, z, options.fd_step)
        grad = jac.T @ r
        grad_norm = float(np.linalg.norm(grad))
        res_norm = float(np.linalg.norm(r))
        rows.append(
            {
                "iteration": k,
                "theta": theta,
                "residual_norm": res_norm,
                "grad_norm": grad_norm,
                "step_norm": 0.0,
                "accepted_step": 0.0,
                "converged": False,
            }
        )
        if res_norm <= options.residual_tol:
            converged = True
            rows[-1]["converged"] = True
            break
        if grad_norm <= options.grad_tol:
            break
        if k == options.max_iter:
            break

        if options.method == "gradient":
            direction = -grad
        else:
            lhs = jac.T @ jac + options.damping * np.eye(z.size)
            rhs = -grad
            try:
                direction = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                direction = -grad
            if float(grad @ direction) >= 0:
                direction = -grad

        step = 1.0
        directional = float(grad @ direction)
        if directional >= 0:
            direction = -grad
            directional = float(grad @ direction)

        accepted = False
        for _ in range(40):
            candidate = z + step * direction
            rc = residual_fn(candidate)
            theta_c = 0.5 * float(rc @ rc)
            if theta_c <= theta + options.armijo_sigma * step * directional:
                accepted = True
                break
            step *= options.armijo_beta
        if not accepted:
            step = min(step, 1e-6)
            candidate = z + step * direction

        rows[-1]["step_norm"] = float(np.linalg.norm(step * direction))
        rows[-1]["accepted_step"] = float(step)
        z = candidate

    final_r = residual_fn(z)
    final_theta = 0.5 * float(final_r @ final_r)
    final_jac = finite_difference_jacobian(residual_fn, z, options.fd_step)
    final_grad_norm = float(np.linalg.norm(final_jac.T @ final_r))
    return SolveResult(
        z=z,
        residual=final_r,
        theta=final_theta,
        grad_norm=final_grad_norm,
        iterations=int(rows[-1]["iteration"]),
        converged=converged,
        history=pd.DataFrame(rows),
    )


def solve_direct_problem1(
    A: Array,
    B: Array,
    x0: Array,
    x_target: Array,
    T: float,
    M: int,
    umax: Array,
    max_iter: int = 300,
) -> DirectSolveResult:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = as_col(x0)
    x_target = as_col(x_target)
    umax = as_col(umax)
    n = x0.size
    m = B.shape[1]
    tau = T / M
    ad = np.eye(n) + tau * A
    bd = tau * B

    def objective_and_grad(flat_u: Array) -> tuple[float, Array]:
        u = np.asarray(flat_u, dtype=float).reshape(M, m)
        y_terminal = x0.copy()
        for k in range(M):
            y_terminal = ad @ y_terminal + bd @ u[k]
        err = y_terminal - x_target
        theta = 0.5 * float(err @ err)
        grad = np.zeros((M, m), dtype=float)
        adj = err.copy()
        for k in range(M - 1, -1, -1):
            grad[k] = bd.T @ adj
            adj = ad.T @ adj
        return theta, grad.reshape(-1)

    rows: list[dict[str, float | int | bool]] = []

    def append_history(iteration: int, flat_u: Array) -> None:
        theta, grad = objective_and_grad(flat_u)
        rows.append(
            {
                "iteration": iteration,
                "theta": theta,
                "residual_norm": float(np.sqrt(2.0 * theta)),
                "grad_norm": float(np.linalg.norm(grad)),
                "step_norm": 0.0,
                "accepted_step": 0.0,
                "converged": False,
            }
        )

    u0 = np.zeros(M * m, dtype=float)
    append_history(0, u0)
    callback_state = {"iteration": 0, "previous": u0.copy()}

    def callback(flat_u: Array) -> None:
        callback_state["iteration"] += 1
        step_norm = float(np.linalg.norm(flat_u - callback_state["previous"]))
        append_history(callback_state["iteration"], flat_u)
        rows[-1]["step_norm"] = step_norm
        rows[-1]["accepted_step"] = 1.0
        callback_state["previous"] = flat_u.copy()

    bounds = [(-float(umax[j % m]), float(umax[j % m])) for j in range(M * m)]
    result = minimize(
        objective_and_grad,
        u0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50},
    )
    final_u = np.asarray(result.x, dtype=float).reshape(M, m)
    final_y = simulate_linear_with_controls(A, B, x0, T, final_u)
    final_theta, final_grad = objective_and_grad(result.x)
    terminal_error = float(np.linalg.norm(final_y[-1] - x_target))
    if rows:
        rows[-1]["converged"] = bool(result.success)
    return DirectSolveResult(
        y=final_y,
        u=final_u,
        theta=final_theta,
        terminal_error=terminal_error,
        grad_norm=float(np.linalg.norm(final_grad)),
        iterations=int(result.nit),
        success=bool(result.success),
        message=str(result.message),
        history=pd.DataFrame(rows),
    )


def problem1_residual(
    A: Array,
    B: Array,
    x0: Array,
    x_target: Array,
    T: float,
    M: int,
    umax: Array,
    mu: float,
) -> Callable[[Array], Array]:
    x_target = as_col(x_target)
    transitions = transition_samples(A, T, M)

    def residual(q: Array) -> Array:
        y, _, _ = simulate_linear(A, B, x0, T, M, q, umax, mu, transitions)
        return as_col(q) + y[-1] - x_target

    return residual


def problem2_residual(
    A: Array,
    B: Array,
    x0: Array,
    T: float,
    M: int,
    umax: Array,
    mu: float,
    a: Array,
    C: Array,
    d: Array,
) -> Callable[[Array], Array]:
    a = as_col(a)
    C = np.asarray(C, dtype=float)
    d = as_col(d)
    transitions = transition_samples(A, T, M)

    def residual(lam: Array) -> Array:
        q = -a - C.T @ as_col(lam)
        y, _, _ = simulate_linear(A, B, x0, T, M, q, umax, mu, transitions)
        return C @ y[-1] - d

    return residual


def save_history_plot(history: pd.DataFrame, title: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.semilogy(history["iteration"], history["residual_norm"], label="residual")
    plt.semilogy(history["iteration"], history["theta"], label="theta")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def save_state_control_plot(y: Array, u: Array, T: float, title: str, filename: str) -> None:
    t_state = np.linspace(0.0, T, y.shape[0])
    t_control = np.linspace(0.0, T, u.shape[0], endpoint=False)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=False)
    for i in range(y.shape[1]):
        axes[0].plot(t_state, y[:, i], label=f"x{i + 1}")
    for j in range(u.shape[1]):
        axes[1].step(t_control, u[:, j], where="post", label=f"u{j + 1}")
    axes[0].set_title(title)
    axes[0].set_ylabel("state")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("control")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def save_phase_plot(y: Array, target: Array, title: str, filename: str) -> None:
    if y.shape[1] < 2:
        return
    target = as_col(target)
    plt.figure(figsize=(5, 5))
    plt.plot(y[:, 0], y[:, 1], label="trajectory")
    plt.scatter([y[0, 0]], [y[0, 1]], marker="o", label="start")
    plt.scatter([target[0]], [target[1]], marker="x", label="target")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def run_problem1_case(
    name: str,
    A: Array,
    B: Array,
    x0: Array,
    x_target: Array,
    T: float,
    M: int,
    umax: Array,
    mu: float,
    q0: Array,
    method: str = "gn",
) -> dict[str, float | str | int | bool]:
    options = SolverOptions(method=method)
    residual = problem1_residual(A, B, x0, x_target, T, M, umax, mu)
    started = perf_counter()
    result = solve_least_squares(residual, q0, options)
    runtime_seconds = perf_counter() - started
    y, u, _ = simulate_linear(A, B, x0, T, M, result.z, umax, mu)
    terminal_error = float(np.linalg.norm(y[-1] - as_col(x_target)))
    control_dim = M * np.asarray(B, dtype=float).shape[1]
    result.history.to_csv(RESULTS / f"{name}_history.csv", index=False, encoding="utf-8-sig")
    np.savetxt(RESULTS / f"{name}_state.csv", y, delimiter=",")
    np.savetxt(RESULTS / f"{name}_control.csv", u, delimiter=",")
    save_history_plot(result.history, name, f"{name}_history.png")
    save_state_control_plot(y, u, T, name, f"{name}_state_control.png")
    save_phase_plot(y, x_target, name, f"{name}_phase.png")
    return {
        "case": name,
        "method": method,
        "M": M,
        "mu": mu,
        "iterations": result.iterations,
        "converged": result.converged,
        "residual_norm": float(np.linalg.norm(result.residual)),
        "theta": result.theta,
        "grad_norm": result.grad_norm,
        "terminal_error": terminal_error,
        "z": np.array2string(result.z, precision=6, separator=";"),
        "decision_dim": as_col(q0).size,
        "direct_decision_dim": control_dim,
        "runtime_seconds": runtime_seconds,
        "optimizer_success": result.converged,
        "objective_value": 0.5 * terminal_error * terminal_error,
    }


def run_problem2_case(
    name: str,
    A: Array,
    B: Array,
    x0: Array,
    T: float,
    M: int,
    umax: Array,
    mu: float,
    a: Array,
    C: Array,
    d: Array,
    lambda0: Array,
    method: str = "gn",
) -> dict[str, float | str | int | bool]:
    options = SolverOptions(method=method)
    residual = problem2_residual(A, B, x0, T, M, umax, mu, a, C, d)
    started = perf_counter()
    result = solve_least_squares(residual, lambda0, options)
    runtime_seconds = perf_counter() - started
    q = -as_col(a) - np.asarray(C, dtype=float).T @ result.z
    y, u, _ = simulate_linear(A, B, x0, T, M, q, umax, mu)
    terminal_residual = float(np.linalg.norm(np.asarray(C, dtype=float) @ y[-1] - as_col(d)))
    terminal_objective = float(as_col(a) @ y[-1])
    control_dim = M * np.asarray(B, dtype=float).shape[1]
    result.history.to_csv(RESULTS / f"{name}_history.csv", index=False, encoding="utf-8-sig")
    np.savetxt(RESULTS / f"{name}_state.csv", y, delimiter=",")
    np.savetxt(RESULTS / f"{name}_control.csv", u, delimiter=",")
    save_history_plot(result.history, name, f"{name}_history.png")
    save_state_control_plot(y, u, T, name, f"{name}_state_control.png")
    if y.shape[1] >= 2:
        target = y[-1].copy()
        save_phase_plot(y, target, name, f"{name}_phase.png")
    return {
        "case": name,
        "method": method,
        "M": M,
        "mu": mu,
        "iterations": result.iterations,
        "converged": result.converged,
        "residual_norm": float(np.linalg.norm(result.residual)),
        "theta": result.theta,
        "grad_norm": result.grad_norm,
        "terminal_constraint_residual": terminal_residual,
        "terminal_objective": terminal_objective,
        "z": np.array2string(result.z, precision=6, separator=";"),
        "decision_dim": as_col(lambda0).size,
        "direct_decision_dim": control_dim,
        "runtime_seconds": runtime_seconds,
        "optimizer_success": result.converged,
        "objective_value": terminal_objective,
    }


def run_direct_problem1_case(
    name: str,
    A: Array,
    B: Array,
    x0: Array,
    x_target: Array,
    T: float,
    M: int,
    umax: Array,
) -> dict[str, float | str | int | bool]:
    started = perf_counter()
    result = solve_direct_problem1(A, B, x0, x_target, T, M, umax)
    runtime_seconds = perf_counter() - started
    control_dim = M * np.asarray(B, dtype=float).shape[1]
    result.history.to_csv(RESULTS / f"{name}_history.csv", index=False, encoding="utf-8-sig")
    np.savetxt(RESULTS / f"{name}_state.csv", result.y, delimiter=",")
    np.savetxt(RESULTS / f"{name}_control.csv", result.u, delimiter=",")
    save_history_plot(result.history, name, f"{name}_history.png")
    save_state_control_plot(result.y, result.u, T, name, f"{name}_state_control.png")
    save_phase_plot(result.y, x_target, name, f"{name}_phase.png")
    return {
        "case": name,
        "method": "direct",
        "M": M,
        "mu": np.nan,
        "iterations": result.iterations,
        "converged": result.success,
        "residual_norm": np.nan,
        "theta": result.theta,
        "grad_norm": result.grad_norm,
        "terminal_error": result.terminal_error,
        "z": "",
        "decision_dim": control_dim,
        "direct_decision_dim": control_dim,
        "runtime_seconds": runtime_seconds,
        "optimizer_success": result.success,
        "objective_value": result.theta,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    for pattern in ("*.csv", "*.png"):
        for path in RESULTS.glob(pattern):
            path.unlink()
        for path in FIGURES.glob(pattern):
            path.unlink()
    summaries: list[dict[str, float | str | int | bool]] = []

    # Case 0: analytically checkable integrator examples.
    A0 = np.array([[0.0]], dtype=float)
    B0 = np.array([[1.0]], dtype=float)
    for target in [0.5, 2.0]:
        summaries.append(
            run_problem1_case(
                name=f"case0_integrator_xt_{safe_label(target)}",
                A=A0,
                B=B0,
                x0=np.array([0.0]),
                x_target=np.array([target]),
                T=1.0,
                M=200,
                umax=np.array([1.0]),
                mu=1e-3,
                q0=np.array([target]),
                method="gn",
            )
        )
        summaries.append(
            run_direct_problem1_case(
                name=f"case0_integrator_direct_xt_{safe_label(target)}",
                A=A0,
                B=B0,
                x0=np.array([0.0]),
                x_target=np.array([target]),
                T=1.0,
                M=200,
                umax=np.array([1.0]),
            )
        )

    # Case 1: one-dimensional terminal tracking.
    A1 = np.array([[0.5]], dtype=float)
    B1 = np.array([[1.0]], dtype=float)
    for q0 in [-2.0, 0.0, 2.0]:
        summaries.append(
            run_problem1_case(
                name=f"case1_q0_{safe_label(q0)}",
                A=A1,
                B=B1,
                x0=np.array([0.0]),
                x_target=np.array([1.0]),
                T=1.0,
                M=200,
                umax=np.array([1.0]),
                mu=1e-3,
                q0=np.array([q0]),
                method="gn",
            )
        )

    # Case 2: two-dimensional terminal tracking.
    A2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    B2 = np.array([[0.0], [1.0]], dtype=float)
    x0_2 = np.array([1.0, 0.0])
    x_target_2 = np.array([0.0, 0.0])
    for method in ["gn", "gradient"]:
        summaries.append(
            run_problem1_case(
                name=f"case2_{method}",
                A=A2,
                B=B2,
                x0=x0_2,
                x_target=x_target_2,
                T=2.0,
                M=200,
                umax=np.array([1.0]),
                mu=1e-3,
                q0=np.array([0.0, 0.0]),
                method=method,
            )
        )
    for M in [50, 100, 200, 400, 800]:
        summaries.append(
            run_problem1_case(
                name=f"case2_M_{M}",
                A=A2,
                B=B2,
                x0=x0_2,
                x_target=x_target_2,
                T=2.0,
                M=M,
                umax=np.array([1.0]),
                mu=1e-3,
                q0=np.array([1.0, -1.0]),
                method="gn",
            )
        )
    summaries.append(
        run_direct_problem1_case(
            name="case2_direct_M_200",
            A=A2,
            B=B2,
            x0=x0_2,
            x_target=x_target_2,
            T=2.0,
            M=200,
            umax=np.array([1.0]),
        )
    )

    # Case 3: affine terminal target set.
    C = np.array([[0.0, 1.0]], dtype=float)
    d = np.array([0.0])
    a = np.array([1.0, 0.0])
    for lambda0 in [-3.0, -2.5, -2.0]:
        summaries.append(
            run_problem2_case(
                name=f"case3_lambda0_{safe_label(lambda0)}",
                A=A2,
                B=B2,
                x0=x0_2,
                T=2.0,
                M=200,
                umax=np.array([1.0]),
                mu=1e-3,
                a=a,
                C=C,
                d=d,
                lambda0=np.array([lambda0]),
                method="gn",
            )
        )

    # Case 4: smoothing parameter comparison.
    for mu in [1e-1, 1e-2, 1e-3, 0.0]:
        label = "bangbang" if mu == 0.0 else f"mu_{mu:.0e}".replace("-", "m")
        summaries.append(
            run_problem1_case(
                name=f"case4_{label}",
                A=A2,
                B=B2,
                x0=x0_2,
                x_target=x_target_2,
                T=2.0,
                M=200,
                umax=np.array([1.0]),
                mu=mu,
                q0=np.array([0.0, 0.0]),
                method="gn",
            )
        )

    # Case 5: medium-dimensional linear system.
    A5 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, -0.2, 0.3, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, -1.0, -0.1],
        ],
        dtype=float,
    )
    B5 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    x0_5 = np.array([1.0, 0.0, -0.5, 0.0])
    x_target_5 = np.array([0.0, 0.0, 0.0, 0.0])
    summaries.append(
        run_problem1_case(
            name="case5_medium4d_gn",
            A=A5,
            B=B5,
            x0=x0_5,
            x_target=x_target_5,
            T=3.0,
            M=200,
            umax=np.array([1.0, 1.0]),
            mu=1e-3,
            q0=np.array([0.0, 0.0, 0.0, 0.0]),
            method="gn",
        )
    )
    summaries.append(
        run_direct_problem1_case(
            name="case5_medium4d_direct",
            A=A5,
            B=B5,
            x0=x0_5,
            x_target=x_target_5,
            T=3.0,
            M=200,
            umax=np.array([1.0, 1.0]),
        )
    )

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "summary.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary)


if __name__ == "__main__":
    main()
