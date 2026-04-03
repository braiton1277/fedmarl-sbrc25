"""
Teste mínimo de execução para verificação da instalação.
Roda 10 rodadas com 10 clientes..
"""
from experiment import run_experiment

run_experiment(
    rounds=10,
    n_clients=10,
    k_select=3,
    dir_alpha=0.3,
    initial_flip_fraction=0.4,
    flip_add_fraction=0.0,
    attack_rounds=[600],
    flip_rate_initial=1.0,
    local_steps=5,
    warmup_transitions=10,
    start_train_round=5,
    updates_per_round=5,
    print_every=1,
    out_dir=".",
)
