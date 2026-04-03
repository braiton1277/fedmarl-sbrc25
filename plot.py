"""
Gera gráfico de acurácia por rodada a partir do arquivo JSON de resultados.

Uso:
    python plot.py <arquivo.json>

Exemplo:
    python plot.py experiments/results/exp1_seed2049_abc123.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_results(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"Arquivo não encontrado: {json_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)

    exp_name     = log["resumo"]["experimento"]
    fedavg_acc   = np.array(log["tracks"]["fedavg"]["test_acc"], dtype=float)
    marl_acc     = np.array(log["tracks"]["marl"]["test_acc"],   dtype=float)

    n = min(len(fedavg_acc), len(marl_acc))
    fedavg_acc = fedavg_acc[:n]
    marl_acc   = marl_acc[:n]
    x = np.arange(1, n + 1)

    plt.figure(figsize=(3.9, 2.8), dpi=140)
    ax = plt.gca()

    ax.plot(x, fedavg_acc, linewidth=1, linestyle="-", marker=None, label="FedAvg")
    ax.plot(x, marl_acc,   linewidth=1, linestyle="-", marker="o",
            markersize=2.0, markevery=15, label="MARL")

    ax.set_xlim(1, n)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Rodadas")
    ax.set_ylabel("Acurácia")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.8)
    ticks = [t for t in [1, 250, 500] if 1 <= t <= n]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "250", "500"][:len(ticks)])

    leg = ax.legend(loc="lower right", frameon=True, fontsize=8)
    leg.get_frame().set_linewidth(0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_pdf = path.with_suffix(".pdf")
    out_png = path.with_suffix(".png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em: {out_png} e {out_pdf}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot.py <arquivo.json>")
        sys.exit(1)
    plot_results(sys.argv[1])
