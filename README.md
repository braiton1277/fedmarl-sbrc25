# FEDMARL: Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente

Implementação do artefato referente ao artigo **"Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente"**, submetido ao SBRC 2026.

Este repositório contém a implementação de um mecanismo de seleção de clientes para Aprendizado Federado (FL) baseado em Aprendizado por Reforço Multiagente (MARL), com foco em robustez contra ataques de inversão de rótulos (label flipping).

---

## Estrutura do Repositório

```
fedmarl-byzantine/
├── config.py        # Semente global, dispositivo e utilitários de reprodutibilidade
├── model.py         # Definição da SmallCNN para CIFAR-10
├── data.py          # Particionamento Dirichlet e dataset com ataque de label flipping
├── metrics.py       # Funções de avaliação, recompensa e utilitários de parâmetros
├── server.py        # Treinamento local, agregação FedAvg e rastreamento de estado
├── agent.py         # Agente VDN: Q-network, replay buffer PER e seleção Top-K
├── experiment.py    # Loop principal do experimento (RANDOM vs VDN)
├── main.py          # Ponto de entrada com hiperparâmetros configuráveis
└── requirements.txt # Dependências do projeto
```

### Descrição dos Módulos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Define `SEED=2049`, `DEVICE` (cuda/cpu), `seed_worker` para reprodutibilidade nos DataLoaders |
| `model.py` | `SmallCNN`: 3 camadas Conv+Pool → FC(2048→256) → FC(256→10) |
| `data.py` | `SwitchableTargetedLabelFlipSubset` (ataque determinístico e ativável em runtime); `make_clients_dirichlet_indices` (particionamento não-IID via Dirichlet) |
| `metrics.py` | `eval_acc`, `eval_loss`, `probing_loss_random_offset`, `windowed_reward`, `dynamic_batch_size` |
| `server.py` | `compute_deltas_proj_mom_probe_now` (fase de métricas: proj e gener); `apply_fedavg`; `update_staleness_streak` |
| `agent.py` | `PrioritizedReplayJoint` (PER para N agentes); `AgentMLP` (Q-network MLP); `VDNSelector` (Double DQN + VDN); `build_context_matrix_vdn` (vetor de observação 5D) |
| `experiment.py` | `run_experiment`: executa as duas trilhas (RANDOM e VDN) em paralelo, salva resultados em JSON |
| `main.py` | Ponto de entrada; todos os hiperparâmetros do experimento são configurados aqui |

---

## Reivindicações Principais do Artigo e Localização no Código

| Reivindicação | Arquivo(s) | Função/Classe |
|---|---|---|
| Seleção de clientes via MARL com VDN | `agent.py` | `VDNSelector`, `AgentMLP` |
| Vetor de observação 5D (bias, proj, gener, staleness, streak) | `agent.py` | `build_context_matrix_vdn` |
| Métrica de projeção no gradiente do servidor (proj) | `server.py` | `compute_deltas_proj_mom_probe_now` |
| Métrica de generalização local (gener) | `server.py` | `compute_deltas_proj_mom_probe_now` |
| Ataque de label flipping direcionado (40% dos clientes) | `data.py` | `SwitchableTargetedLabelFlipSubset` |
| Particionamento não-IID via Dirichlet (α=0.3) | `data.py` | `make_clients_dirichlet_indices` |
| Comparação RANDOM (FedAvg) vs VDN | `experiment.py` | `run_experiment` |
| Resultados salvos por rodada (test_acc, selection counts) | `experiment.py` | `save_json` |

---

## Dependências

| Biblioteca | Versão mínima |
|---|---|
| Python | 3.9+ |
| torch | 2.0 |
| torchvision | 0.15 |
| numpy | 1.24 |

O CIFAR-10 é baixado automaticamente pelo `torchvision` na primeira execução (~170 MB).

---

## Ambiente de Execução

Os experimentos do artigo foram executados em:

- **GPU**: NVIDIA GeForce RTX 5090 (21.760 núcleos CUDA, 32 GB GDDR7, 1,79 TB/s)
- **Python**: 3.9 ou superior

GPU é recomendado; CPU é suportado mas significativamente mais lento. O código detecta automaticamente se CUDA está disponível via `config.py`.

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/braiton1277/fedmarl-byzantine.git
cd fedmarl-byzantine

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Execução

### Experimento completo (configuração do artigo)

```bash
python main.py
```

Isso executa 500 rodadas com 50 clientes (40% atacantes, label flipping), selecionando 15 clientes por rodada. Os resultados são salvos automaticamente em um arquivo JSON no diretório atual:

```
results_random_vs_vdn_targeted_PROJMOM_FO_seed2049_<run_id>.json
```



Saída esperada: mensagens de progresso por rodada, acurácia de teste para as trilhas RANDOM e VDN, e arquivo JSON salvo ao final.

---

## Reprodução dos Experimentos do Artigo

### Configuração utilizada no artigo

| Hiperparâmetro | Valor |
|---|---|
| `rounds` | 500 |
| `n_clients` | 50 |
| `k_select` | 15 |
| `dir_alpha` | 0.3 |
| `initial_flip_fraction` | 0.4 |
| `flip_rate_initial` | 1.0 |
| `local_steps` | 10 |
| `local_lr` | 0.01 |
| `SEED` | 2049 |

### Para reproduzir os gráficos/tabelas do artigo

```bash
python main.py
```

O arquivo JSON gerado contém:
- `tracks.random.test_acc`: acurácia por rodada da trilha RANDOM (FedAvg)
- `tracks.vdn.test_acc`: acurácia por rodada da trilha VDN (FEDMARL)
- `tracks.*.selection_count_total_per_client`: contagem de seleção por cliente



> **Nota sobre tempo de execução:** 500 rodadas com 50 clientes e `local_steps=10` levam aproximadamente 4–8 horas em GPU e 20–40 horas em CPU, dependendo do hardware. Um resultado pré-computado de referência (`results_random_vs_vdn_targeted_PROJMOM_FO_seed2049_92768206f1.json`) está disponível no repositório para consulta imediata.

---

## Reprodutibilidade

A semente global `SEED=2049` é fixada em `config.py` para `random`, `numpy`, `torch` e CUDA. O particionamento Dirichlet, a atribuição de atacantes e os DataLoaders usam seeds derivadas de `SEED`, garantindo resultados idênticos entre execuções no mesmo hardware com as mesmas versões de dependências.
