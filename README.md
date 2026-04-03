# Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente

Este artefato contém a implementação do mecanismo de seleção de clientes para Aprendizado Federado (FL) proposto no artigo **"Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente"**, submetido ao SBRC 2026. A abordagem aplica Aprendizado por Reforço Multiagente (MARL) para a seleção cooperativa de clientes de forma robusta contra ataques de inversão de rótulos (label flipping) em cenários de dados não-IID.

---

# Estrutura do README.md

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
├── config.py        # Semente global, dispositivo e utilitários de reprodutibilidade
├── model.py         # Definição da SmallCNN para CIFAR-10
├── data.py          # Particionamento Dirichlet e dataset com ataque de label flipping
├── metrics.py       # Funções de avaliação, recompensa e utilitários de parâmetros
├── server.py        # Treinamento local, agregação FedAvg e rastreamento de estado
├── agent.py         # Agente MARL: Q-network, replay buffer PER e seleção Top-K
├── experiment.py    # Loop principal do experimento (FedAvg vs MARL)
├── main.py          # Ponto de entrada com hiperparâmetros configuráveis
├── test_min.py      # Script de teste mínimo para verificação da instalação
├── exp1.py          # Experimento 1: 40% atacantes, 100% inversão
├── exp2.py          # Experimento 2: 60% atacantes, 100% inversão
├── exp3.py          # Experimento 3: 40% atacantes, 60% inversão
├── exp4.py          # Experimento 4: 40% atacantes, 40% inversão
└── requirements.txt # Dependências do projeto
```

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

O dataset CIFAR-10 é baixado automaticamente pelo `torchvision` na primeira execução (aproximadamente 170 MB). Não é necessário acesso a recursos externos além do PyPI e dos servidores do CIFAR-10.

---

# Preocupações com Segurança

Não há preocupações de segurança relevantes para os avaliadores.

---

# Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/braiton1277/fedmarl-sbrc25.git
cd fedmarl-sbrc25

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

**Resultado esperado:** 10 rodadas de progresso impressas no terminal com as acurácias das trilhas RANDOM e VDN, e um arquivo JSON salvo no diretório atual ao final da execução, contendo a acurácia de teste por rodada de cada trilha e a contagem de seleções por cliente.

---

# Experimentos

Os experimentos comparam a seleção de clientes por MARL com a seleção aleatória (FedAvg) sob diferentes cenários de ataque de label flipping e dados não-IID, ao longo de 500 rodadas de treinamento com 50 clientes, selecionando 15 por rodada.

Cada experimento gera um arquivo JSON contendo:
- `tracks.random.test_acc`: acurácia por rodada da trilha RANDOM (FedAvg)
- `tracks.vdn.test_acc`: acurácia por rodada da trilha VDN (MARL)
- `tracks.*.selection_count_total_per_client`: frequência de seleção por cliente

Um resultado pré-computado de referência está disponível no repositório (`results_random_vs_vdn_targeted_PROJMOM_FO_seed2049_92768206f1.json`) para consulta imediata, sem necessidade de re-executar os experimentos.

**Recursos esperados:** ~32 GB VRAM (GPU utilizada no artigo); em GPUs com menos memória, reduzir `n_clients` ou `max_per_client`.

**Tempo esperado por experimento:** aproximadamente 1 hora em GPU NVIDIA RTX 5090.

## Reivindicação #1 — Impacto da proporção de clientes atacantes

Avalia o desempenho da abordagem MARL à medida que a fração de clientes maliciosos aumenta, mantendo a taxa de inversão de rótulos em 100%.

**Experimento 1** — 40% de atacantes, 100% de inversão:

```bash
python exp1.py
```

**Experimento 2** — 60% de atacantes, 100% de inversão:

```bash
python exp2.py
```

**Resultado esperado:** em ambos os cenários, a trilha VDN apresenta acurácia de teste superior à trilha RANDOM, com a vantagem se tornando mais evidente conforme a proporção de atacantes aumenta.

## Reivindicação #2 — Impacto da porcentagem de inversão de rótulos

Avalia o desempenho da abordagem MARL à medida que a intensidade do ataque diminui, mantendo a fração de clientes maliciosos em 40%.

**Experimento 3** — 40% de atacantes, 60% de inversão:

```bash
python exp3.py
```

**Experimento 4** — 40% de atacantes, 40% de inversão:

```bash
python exp4.py
```

**Resultado esperado:** mesmo com ataques de menor intensidade, a trilha VDN mantém acurácia superior à seleção aleatória, evidenciando a robustez da abordagem em diferentes níveis de ataque.

---

# LICENSE

MIT License

Copyright (c) 2026 GTA-UFRJ

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
