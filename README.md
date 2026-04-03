# Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente

Este artefato contém a implementação do mecanismo de seleção de clientes para Aprendizado Federado (FL) proposto no artigo **"Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente"**, submetido ao SBRC 2026. 

No aprendizado federado, o servidor central decide quais clientes devem participar de cada rodada de treinamento. Entretanto, estratégias tradicionais que não estimam a contribuição de cada cliente podem ser vulneráveis a dados de baixa qualidade e a clientes maliciosos. Este trabalho investiga uma abordagem de seleção de clientes baseada no aprendizado por reforço multiagente (MARL). A arquitetura proposta modela cada cliente como um agente, no qual as decisões são tomadas de forma descentralizada e cooperativa. Cada agente avalia características como diversidade de dados, capacidade de processamento e histórico de participação, aprendendo a contribuir para uma seleção mais estratégica dos participantes do treinamento em cenários dinâmicos e não-IID. Os experimentos simulam diferentes graus de heterogeneidade entre os clientes, refletindo distribuições não-IID, além de considerar cenários com inversão de rótulos como estratégia de ataque ao aprendizado. O desempenho, em comparação ao FedAvg e ao single-agent RL (SARL), demonstra melhoria na acurácia final do modelo e no equilíbrio na participação dos clientes.

---

# Estrutura do README

Este README está organizado nas seguintes seções:

- **Selos Considerados**: selos solicitados para avaliação
- **Informações Básicas**: ambiente de hardware e software necessário
- **Dependências**: bibliotecas e versões utilizadas
- **Preocupações com Segurança**: riscos para os avaliadores
- **Instalação**: passo a passo para configurar o ambiente
- **Teste Mínimo**: execução rápida para verificar a instalação
- **Experimentos**: reprodução das reivindicações do artigo
- **LICENSE**: licença do projeto

O repositório está organizado da seguinte forma:

```
fedmarl-sbrc25/
├── experiments/
│   ├── exp1.py          # Experimento 1: 40% atacantes, 100% inversão
│   ├── exp2.py          # Experimento 2: 60% atacantes, 100% inversão
│   ├── exp3.py          # Experimento 3: 40% atacantes, 60% inversão
│   ├── exp4.py          # Experimento 4: 40% atacantes, 40% inversão
│   └── results/         # JSONs e PNGs gerados pelos experimentos
├── config.py            # Semente global, dispositivo e utilitários de reprodutibilidade
├── model.py             # Definição da SmallCNN para CIFAR-10
├── data.py              # Particionamento Dirichlet e dataset com ataque de label flipping
├── metrics.py           # Funções de avaliação, recompensa e utilitários de parâmetros
├── server.py            # Treinamento local, agregação FedAvg e rastreamento de estado
├── agent.py             # Agente MARL: Q-network, replay buffer PER e seleção Top-K
├── experiment.py        # Loop principal do experimento (FedAvg vs MARL)
├── main.py              # Ponto de entrada com hiperparâmetros configuráveis
├── test_min.py          # Script de teste mínimo para verificação da instalação
├── plot.py              # Script de plotagem de resultados a partir do JSON
└── requirements.txt     # Dependências do projeto
```

---

# Correspondência com o Artigo

| Componente | Arquivo | Classe/Função |
|---|---|---|
| Seleção de clientes via MARL | `agent.py` | `VDNSelector` |
| Vetor de observação 5D | `agent.py` | `build_context_matrix_vdn` |
| Métrica proj (alinhamento ao gradiente do servidor) | `server.py` | `compute_deltas_proj_mom_probe_now` |
| Métrica gener (generalização local) | `server.py` | `compute_deltas_proj_mom_probe_now` |
| Métrica staleness (rounds sem seleção) | `server.py` | `update_staleness_streak` |
| Métrica streak (seleções consecutivas) | `server.py` | `update_staleness_streak` |
| Ataque de label flipping | `data.py` | `SwitchableTargetedLabelFlipSubset` |
| Comparação FedAvg vs MARL | `experiment.py` | `run_experiment` |

---

# Selos Considerados

Os selos considerados são: **Disponíveis (SeloD)**, **Funcionais (SeloF)**, **Sustentáveis (SeloS)** e **Reprodutíveis (SeloR)**.

---

# Informações Básicas

**Hardware utilizado nos experimentos do artigo:**

- GPU: NVIDIA GeForce RTX 5090 (21.760 núcleos CUDA, 32 GB GDDR7, 1,79 TB/s)

> GPU é fortemente recomendado para reprodução dos experimentos. O código também executa em CPU, porém com tempo significativamente maior.

**Software:**

- Sistema Operacional: Linux (recomendado)
- Python: 3.9 ou superior
- CUDA: compatível com a versão do PyTorch instalada

---

# Dependências

| Biblioteca | Versão mínima |
|---|---|
| Python | 3.9 |
| torch | 2.0 |
| torchvision | 0.15 |
| numpy | 1.24 |
| matplotlib | 3.7 |

O dataset CIFAR-10 é baixado automaticamente pelo `torchvision` na primeira execução (aproximadamente 170 MB). Não é necessário acesso a recursos externos além do PyPI e dos servidores do CIFAR-10.

---

# Preocupações com Segurança

Não há preocupações de segurança relevantes para os avaliadores.

---

# Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/braiton1277/fedmarl-sbrc26.git
cd fedmarl-sbrc26

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

Ao final deste processo, o CIFAR-10 será baixado automaticamente na primeira execução.

---

# Teste Mínimo

Execute o script de teste mínimo — 10 rodadas com 10 clientes, selecionando 3 por rodada, apenas para verificar que o ambiente está funcionando corretamente:

```bash
python test_min.py
```

**Resultado esperado:** 10 rodadas de progresso impressas no terminal com as acurácias das trilhas FedAvg e MARL, e um arquivo JSON salvo no diretório atual ao final da execução, contendo a acurácia de teste por rodada de cada trilha e a contagem de seleções por cliente.

---

# Experimentos

Os experimentos comparam a seleção de clientes por MARL com a seleção aleatória (FedAvg) sob diferentes cenários de ataque de label flipping e dados não-IID, ao longo de 500 rodadas de treinamento com 50 clientes, selecionando 15 por rodada.

**Recursos esperados:** ~32 GB VRAM (GPU utilizada no artigo); em GPUs com menos memória, reduzir `n_clients` ou `max_per_client`.

**Tempo esperado por experimento:** aproximadamente 1h30 em GPU NVIDIA RTX 5090.

Para executar cada experimento individualmente:

## Reivindicação #1 — Impacto da proporção de clientes atacantes

Avalia o desempenho da abordagem MARL à medida que a fração de clientes maliciosos aumenta, mantendo a taxa de inversão de rótulos em 100%.

**Experimento 1** — 40% de atacantes, 100% de inversão:

```bash
python experiments/exp1.py
```

**Experimento 2** — 60% de atacantes, 100% de inversão:

```bash
python experiments/exp2.py
```

**Resultado esperado:** em ambos os cenários, a trilha MARL apresenta acurácia de teste superior à trilha FedAvg, com a vantagem se tornando mais evidente conforme a proporção de atacantes aumenta.

## Reivindicação #2 — Impacto da porcentagem de inversão de rótulos

Avalia o desempenho da abordagem MARL à medida que a intensidade do ataque diminui, mantendo a fração de clientes maliciosos em 40%.

**Experimento 3** — 40% de atacantes, 60% de inversão:

```bash
python experiments/exp3.py
```

**Experimento 4** — 40% de atacantes, 40% de inversão:

```bash
python experiments/exp4.py
```

**Resultado esperado:** mesmo com ataques de menor intensidade, a trilha MARL mantém acurácia superior à FedAvg, evidenciando a robustez da abordagem em diferentes níveis de ataque.

---

Para executar todos os experimentos automaticamente em sequência:

```bash
bash run_all.sh
```

Cada experimento gera automaticamente em `experiments/results/` um arquivo JSON e um gráfico (PDF e PNG).

O arquivo JSON contém:
- `tracks.fedavg.test_acc`: acurácia por rodada da trilha FedAvg
- `tracks.marl.test_acc`: acurácia por rodada da trilha MARL
- `tracks.*.selection_count_total_per_client`: frequência de seleção por cliente

O gráfico exibe as curvas de acurácia por rodada de cada trilha.

Resultados pré-computados de todos os experimentos (JSON e gráficos em PDF e PNG) estão disponíveis em `experiments/results/` para consulta imediata, sem necessidade de re-executar os experimentos. Caso prefira reproduzir, ao final de cada execução o JSON e o gráfico são gerados automaticamente no mesmo diretório.

---

# LICENSE

MIT License — Copyright (c) 2026
