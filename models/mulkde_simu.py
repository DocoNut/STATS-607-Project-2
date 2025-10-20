from models.mulkde import multi_kde_n0 as mkde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.other_kde import plugin_kde,adaptive_kde

def loss(func, data, eps=1e-12):
    """Compute negative log-likelihood (NLL) loss.

    Args:
        func (callable): A function that takes `data` as input and 
            returns probabilities (values in (0, 1]).
        data (array-like): Input data for evaluating the loss.
        eps (float, optional): Small value to prevent log(0). Defaults to 1e-12.

    Returns:
        float: The mean negative log-likelihood.
    """
    probs = np.clip(func(data), eps, 1.0)  # keep values in [eps,1]
    return -np.log(probs).mean()

def simulation():
    """main function"""
    # load data
    data=pd.read_csv('data/processed/faithful.csv').to_numpy().flatten()

    # Parameters
    N = len(data)                # sample size
    h = N ** (-0.2)                # base bandwidth
    d = 4                        # dimension / number of scalers
    xi = np.sqrt(np.linspace(1, 3.5, d))  # scalers

    # --- Train/test split ---
    np.random.seed(0)  # for reproducibility
    shuffled_data = np.random.permutation(data)

    train_size = int(0.8 * len(shuffled_data))
    train_set, test_set = shuffled_data[:train_size], shuffled_data[train_size:]

    # --- Fit estimators ---
    estimators = {
        "4KDE": mkde(h, train_set, xi, d),
        "AKDE": adaptive_kde(h, train_set),
        "DDE":  plugin_kde(h, train_set),
    }

    # --- Evaluate losses ---
    results = {name: loss(est, test_set) for name, est in estimators.items()}

    # --- Print results ---
    for name, val in results.items():
        print(f"{name} loss: {val:.4f}")

    # Evaluation grid
    x = np.linspace(0, 6, 10000)

    # Density estimators
    estimators = {
        "4KDE": (mkde(h, data, xi, d), "blue"),
        "AKDE": (adaptive_kde(h, data), "red"),
        "DDE":  (plugin_kde(h, data), "green"),
    }

    # Plot
    plt.figure(figsize=(8, 5))
    for label, (estimator, color) in estimators.items():
        plt.plot(x, estimator(x), label=label, color=color)

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimates")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulation()