# -*- coding: utf-8 -*-
"""
Compare Nesterov + quadratic-gradient preconditioning against standard Nesterov,
using Bidirectional Adaptive Line Search (BALS) for monotone logistic-regression
maximum-likelihood optimization.

Input format: MNIST_train.txt, CSV-like text file with a header row.
Column 0 is the binary label {0, 1}; remaining columns are features.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_ITER = 60
EPSILON = 1e-10

# BALS parameters
INITIAL_ALPHA = 0.01
INCREASE_FACTOR = 1.5
DECREASE_FACTOR = 0.5
ARMIJO_C1 = 1e-4
MAX_LINE_SEARCH_ITER = 30
MIN_ALPHA = 1e-12
MAX_ALPHA = 1e6


def stable_sigmoid(z):
    """Numerically stable sigmoid for a scalar."""
    z = float(z)
    if z >= 0.0:
        e = math.exp(-min(z, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def ROCAUC(score, y, show=False):
    """ROC AUC for labels in {-1, +1}."""
    pairs = sorted(zip(score, y))
    scores = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    positives = labels.count(+1)
    negatives = labels.count(-1)
    if positives == 0 or negatives == 0:
        return 0.5

    roc_x = [1.0]
    roc_y = [1.0]
    fn = tn = 0
    for i in range(len(scores)):
        if labels[i] == +1:
            fn += 1
        else:
            tn += 1
        roc_x.append(1.0 - tn / float(negatives))
        roc_y.append(1.0 - fn / float(positives))

    points = sorted(zip(roc_x, roc_y))
    roc_x = [p[0] for p in points]
    roc_y = [p[1] for p in points]
    auc = 0.0
    previous_x = roc_x[0]
    for i, x in enumerate(roc_x):
        auc += (x - previous_x) * roc_y[i]
        previous_x = x

    if show:
        plt.figure()
        plt.plot(roc_x, roc_y)
        plt.plot([0, 1], [0, 1])
        plt.axis("equal")
        plt.title("AUC = {}".format(auc))
        plt.grid()
        plt.show()
    return auc


def calculate_loglikelihood(weights, MX, Y):
    """Stable binary logistic-regression log-likelihood for y in {-1,+1}."""
    margins = np.asarray(MX * weights).reshape(-1)
    labels = np.asarray(Y, dtype=float)
    # log(sigmoid(margin)) = -logaddexp(0, -margin)
    return float(np.sum(-np.logaddexp(0.0, -labels * margins)))


def calculate_gradient(weights, MX, MXT, Y):
    """Gradient of the log-likelihood (an ascent direction)."""
    margins = np.asarray(MX * weights).reshape(-1)
    labels = np.asarray(Y, dtype=float)
    probs = np.array([stable_sigmoid(v) for v in labels * margins])
    residual = labels * (1.0 - probs)
    gradient = MXT * np.matrix(residual).T
    return gradient


def construct_MB_inverse(MX):
    """
    Preserve the diagonal fixed-Hessian approximation from the supplied code:
        MB[i,i] = 0.25 * (X^T * row_sum(X))[i] + epsilon
    The absolute value and floor prevent a non-positive/near-zero diagonal
    from producing an invalid preconditioner.
    """
    mx = MX.sum(axis=1)
    mxtmx = MX.T.dot(mx)
    diagonal = np.asarray(mxtmx).reshape(-1)
    diagonal = np.maximum(np.abs(0.25 * diagonal), EPSILON)
    return np.matrix(np.diag(1.0 / diagonal))


def evaluate_candidate(base_point, direction, alpha, MX, Y):
    candidate = base_point + alpha * direction
    objective = calculate_loglikelihood(candidate, MX, Y)
    return candidate, objective


def bidirectional_adaptive_line_search(
        base_point, direction, gradient, current_point,
        current_loglikelihood, MX, Y, alpha_init):
    """
    BALS for maximizing log-likelihood.

    First tests alpha_init. If accepted, expands alpha geometrically; otherwise
    backtracks. Acceptance requires both the Armijo ascent condition and
    non-decrease relative to current_loglikelihood.

    Returns (accepted_point, alpha, success).
    """
    directional_derivative = float((gradient.T * direction)[0, 0])
    if (not math.isfinite(directional_derivative)
            or directional_derivative <= 0.0):
        return None, alpha_init, False

    alpha = min(max(float(alpha_init), MIN_ALPHA), MAX_ALPHA)

    def acceptable(point, value, step):
        rhs = (current_loglikelihood
               + ARMIJO_C1 * step * directional_derivative)
        return (math.isfinite(value)
                and value >= current_loglikelihood
                and value >= rhs)

    point, value = evaluate_candidate(
        base_point, direction, alpha, MX, Y
    )
    if acceptable(point, value, alpha):
        accepted_point, accepted_value, accepted_alpha = point, value, alpha

        for _ in range(MAX_LINE_SEARCH_ITER):
            trial_alpha = min(accepted_alpha * INCREASE_FACTOR, MAX_ALPHA)
            if trial_alpha <= accepted_alpha:
                break
            trial_point, trial_value = evaluate_candidate(
                base_point, direction, trial_alpha, MX, Y
            )
            if acceptable(trial_point, trial_value, trial_alpha):
                accepted_point = trial_point
                accepted_value = trial_value
                accepted_alpha = trial_alpha
            else:
                break
        return accepted_point, accepted_alpha, True

    for _ in range(MAX_LINE_SEARCH_ITER):
        alpha *= DECREASE_FACTOR
        if alpha < MIN_ALPHA:
            break
        point, value = evaluate_candidate(base_point, direction, alpha, MX, Y)
        if acceptable(point, value, alpha):
            return point, alpha, True

    return None, max(alpha, MIN_ALPHA), False


def monotone_nesterov_step(
        current_point, previous_point, beta, direction, gradient,
        MX, Y, current_loglikelihood, alpha_init,
        direction_builder):
    """
    Try the Nesterov look-ahead step first. If it cannot produce a monotone
    accepted point, restart momentum and search from current_point.
    direction_builder(point, gradient) constructs either g or MB_inv @ g.
    """
    lookahead = current_point + beta * (current_point - previous_point)
    point, alpha, success = bidirectional_adaptive_line_search(
        base_point=lookahead,
        direction=direction,
        gradient=gradient,
        current_point=current_point,
        current_loglikelihood=current_loglikelihood,
        MX=MX, Y=Y, alpha_init=alpha_init
    )
    if success and calculate_loglikelihood(point, MX, Y) >= current_loglikelihood:
        return point, alpha, False

    # Monotone restart: remove momentum and recompute the gradient at x_k.
    MXT = MX.T
    restart_gradient = calculate_gradient(current_point, MX, MXT, Y)
    restart_direction = direction_builder(current_point, restart_gradient)
    point, alpha, success = bidirectional_adaptive_line_search(
        base_point=current_point,
        direction=restart_direction,
        gradient=restart_gradient,
        current_point=current_point,
        current_loglikelihood=current_loglikelihood,
        MX=MX, Y=Y, alpha_init=max(alpha_init * DECREASE_FACTOR, MIN_ALPHA)
    )
    if success:
        return point, alpha, True

    # If no step passes the line search, preserve the current iterate.
    return current_point.copy(), max(alpha, MIN_ALPHA), True


def load_dataset(path):
    rows = []
    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header, as in the supplied code
        for row in reader:
            if row:
                rows.append([float(value) for value in row])
    if not rows:
        raise ValueError("No data rows found in {!r}".format(path))

    data = np.asarray(rows, dtype=float)
    raw_y = data[:, 0].astype(int)
    X = np.column_stack((np.ones(data.shape[0]), data[:, 1:]))

    # Preserve the supplied code's min-max normalization per feature.
    for j in range(1, X.shape[1]):
        col_min = np.min(X[:, j])
        col_max = np.max(X[:, j])
        if col_max - col_min >= EPSILON:
            X[:, j] = (X[:, j] - col_min) / (col_max - col_min + EPSILON)

    Y = 2 * raw_y - 1
    return np.matrix(X), Y.tolist()


def run_experiment(MX, Y, use_qg, label):
    MXT = MX.T
    MB_inv = construct_MB_inverse(MX)

    weights = np.matrix(np.zeros((MX.shape[1], 1), dtype=float))
    previous = weights.copy()
    t = 1.0
    alpha = INITIAL_ALPHA

    mle_history = []
    auc_history = []
    alpha_history = []
    restart_history = []

    current_L = calculate_loglikelihood(weights, MX, Y)

    def direction_builder(point, grad):
        return MB_inv * grad if use_qg else grad

    print("\n{}".format("=" * 68))
    print(label)
    print("=" * 68)

    for iteration in range(MAX_ITER):
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        beta = (t - 1.0) / t_next

        lookahead = weights + beta * (weights - previous)
        grad = calculate_gradient(lookahead, MX, MXT, Y)
        direction = direction_builder(lookahead, grad)

        new_weights, alpha, restarted = monotone_nesterov_step(
            current_point=weights,
            previous_point=previous,
            beta=beta,
            direction=direction,
            gradient=grad,
            MX=MX, Y=Y,
            current_loglikelihood=current_L,
            alpha_init=alpha,
            direction_builder=direction_builder
        )

        old_weights = weights.copy()
        weights = new_weights.copy()
        previous = weights.copy() if restarted else old_weights
        t = 1.0 if restarted else t_next
        current_L = calculate_loglikelihood(weights, MX, Y)

        margins = np.asarray(MX * weights).reshape(-1)
        predictions = [stable_sigmoid(v) for v in margins]
        auc = ROCAUC(predictions, Y)

        mle_history.append(current_L)
        auc_history.append(auc)
        alpha_history.append(alpha)
        restart_history.append(restarted)

        print("Iteration {:02d} | MLE={:.8f} | AUC={:.6f} | alpha={:.3e} | beta={:.5f} | restart={}".format(
            iteration, current_L, auc, alpha, beta, restarted
        ))

    return {
        "mle": mle_history,
        "auc": auc_history,
        "alpha": alpha_history,
        "restart": restart_history,
    }



def run_fixed_lr_qg(MX, Y, label="Experiment 3: Nesterov + QG (fixed alpha=1)"):
    """
    Baseline: Nesterov + quadratic-gradient (QG) preconditioning with a fixed
    learning rate alpha=1. No BALS, backtracking, expansion, or monotone restart
    is used; the QG-preconditioned gradient is applied directly at each
    Nesterov look-ahead point.
    """
    MXT = MX.T
    MB_inv = construct_MB_inverse(MX)
    weights = np.matrix(np.zeros((MX.shape[1], 1), dtype=float))
    previous = weights.copy()
    t = 1.0
    alpha = 1.0

    mle_history = []
    auc_history = []
    alpha_history = []

    print("\\n{}".format("=" * 68))
    print(label)
    print("=" * 68)

    for iteration in range(MAX_ITER):
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        beta = (t - 1.0) / t_next
        lookahead = weights + beta * (weights - previous)
        grad = calculate_gradient(lookahead, MX, MXT, Y)
        direction = MB_inv * grad

        new_weights = lookahead + alpha * direction
        old_weights = weights.copy()
        weights = new_weights.copy()
        previous = old_weights
        t = t_next

        current_L = calculate_loglikelihood(weights, MX, Y)
        margins = np.asarray(MX * weights).reshape(-1)
        predictions = [stable_sigmoid(v) for v in margins]
        auc = ROCAUC(predictions, Y)

        mle_history.append(current_L)
        auc_history.append(auc)
        alpha_history.append(alpha)

        print("Iteration {:02d} | MLE={:.8f} | AUC={:.6f} | alpha={:.3e} | beta={:.5f}".format(
            iteration, current_L, auc, alpha, beta
        ))

    return {
        "mle": mle_history,
        "auc": auc_history,
        "alpha": alpha_history,
    }

def main():
    data_path = os.environ.get("MNIST_TRAIN_PATH", "MNIST_train.txt")
    MX, Y = load_dataset(data_path)

    results_qg = run_experiment(
        MX, Y, use_qg=True, label="Experiment 1: Nesterov + QG + BALS"
    )
    results_nag = run_experiment(
        MX, Y, use_qg=False, label="Experiment 2: Nesterov + BALS"
    )
    results_qg_fixed = run_fixed_lr_qg(
        MX, Y, label="Experiment 3: Nesterov + QG (fixed alpha=1, no BALS)"
    )

    # MLE plot
    plt.figure()
    plt.plot(results_qg["mle"], "v-b", label="Nesterov + QG + BALS")
    plt.plot(results_nag["mle"], "o--r", label="Nesterov + BALS")
    plt.plot(results_qg_fixed["mle"], "s-.g", label="Nesterov + QG (fixed alpha=1)")
    plt.title("MLE")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("MLE_BALS.png")
    plt.close()

    # AUC plot
    plt.figure()
    plt.plot(results_qg["auc"], "v-b", label="Nesterov + QG + BALS")
    plt.plot(results_nag["auc"], "o--r", label="Nesterov + BALS")
    plt.plot(results_qg_fixed["auc"], "s-.g", label="Nesterov + QG (fixed alpha=1)")
    plt.title("AUC")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("AUC_BALS.png")
    plt.close()

    # CSV summary
    with open("PythonExperiment_NAGvs.NAGG_MNIST_BALS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Iteration", "Nesterov_BALS_MLE", "Nesterov_QG_BALS_MLE",
            "Nesterov_QG_fixed_alpha1_MLE",
            "Nesterov_BALS_AUC", "Nesterov_QG_BALS_AUC",
            "Nesterov_QG_fixed_alpha1_AUC",
            "Nesterov_BALS_alpha", "Nesterov_QG_BALS_alpha",
            "Nesterov_QG_fixed_alpha1_alpha"
        ])
        for i in range(MAX_ITER):
            writer.writerow([
                i,
                results_nag["mle"][i], results_qg["mle"][i],
                results_qg_fixed["mle"][i],
                results_nag["auc"][i], results_qg["auc"][i],
                results_qg_fixed["auc"][i],
                results_nag["alpha"][i], results_qg["alpha"][i],
                results_qg_fixed["alpha"][i]
            ])


if __name__ == "__main__":
    main()
