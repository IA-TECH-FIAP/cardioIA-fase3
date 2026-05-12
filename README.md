# Ir Além 2 — IA em Séries Temporais de Saúde

**CardioIA — Fase 3 | FIAP — Inteligência Artificial**

Comparação entre **Regressão Logística** e **Rede Neuromórfica FitzHugh–Nagumo (FHN)** na classificação de batidas cardíacas em séries temporais de ECG.

---

## Integrantes

| Nome | RM |
|------|-----|
| Giulia Bugatti Fonseca | 562675 |
| Mahmod Ahmad Issa | 561426 |
| Matheus Cardoso Oliveira Lima | 565844 |
| Silas Fernandes de Souza Fonseca | 564246 |

---

## Demonstração em Vídeo

▶️ **Link:** `<INSERIR LINK DO YOUTUBE AQUI>`

> Vídeo "não listado" no YouTube com até 4 minutos apresentando objetivo, metodologia, resultados e análise crítica.

---

## Objetivo

Aplicar e comparar duas abordagens fundamentalmente diferentes de Inteligência Artificial para o mesmo problema — classificação binária de batidas cardíacas em **Normal** vs **Anormal** a partir de séries temporais de ECG:

1. **Regressão Logística** — modelo estatístico clássico, linear, atemporal (trata os 140 pontos como features independentes).
2. **FitzHugh–Nagumo + Classificador Linear** — modelo bio-inspirado simulado com Brian2; injeta o ECG como corrente num neurônio cardíaco simulado e classifica a partir do padrão de disparos resultantes.

---

## Dataset

- **Nome:** ECG5000
- **Origem:** BIDMC Congestive Heart Failure Database (PhysioNet) → UCR Time Series Classification Archive
- **Licença:** Open Data Commons Attribution (ODC-By) — uso acadêmico livre
- **Tamanho:** 5.000 batidas cardíacas isoladas × 140 timesteps cada (univariado)
- **Classes originais:** 5 (Normal, R-on-T PVC, PVC, SP/EB, Unclassified)
- **Binarização aplicada:** classe 1 → Normal (58,4%); classes 2–5 → Anormal (41,6%)
- **Download:** automático via biblioteca `aeon` (1 linha de código no notebook)

---

## Modelos

### Modelo 1 — Regressão Logística

| Característica | Valor |
|----------------|-------|
| Implementação | `sklearn.linear_model.LogisticRegression` |
| Solver | `lbfgs`, `max_iter=1000` |
| Regularização | L2 (C=1.0) |
| Balanceamento | `class_weight='balanced'` |
| Features | 140 timesteps brutos (z-score por amostra) |
| Parâmetros aprendíveis | 140 |

### Modelo 2 — FitzHugh–Nagumo + Classificador Linear

| Característica | Valor |
|----------------|-------|
| Framework de simulação | **Brian2 2.6** |
| Equações | FHN canônico (FitzHugh 1961): `dv/dt = (v − v³/3 − w + I)/τ`, `dw/dt = ε(v + a − bw)/τ` |
| Parâmetros do FHN | a=0,7; b=0,8; ε=0,08; τ=1ms (literatura, sem treino) |
| Injeção do sinal | I(t) = ECG normalizado |
| Threshold/reset | v > 1 → spike → v = −1 |
| Método de integração | Runge-Kutta 4 |
| Features extraídas | 8 (nº spikes, ISI mean, ISI std, primeiro/último spike, duração, centroide, diferença entre metades) |
| Classificador final | Regressão Logística sobre as 8 features |
| Parâmetros aprendíveis | **8** (1/17 do modelo 1) |

---

## Resultados

| Métrica | Regressão Logística | FHN + Linear |
|---|---|---|
| **Acurácia** | **99,30%** | 93,90% |
| **F1-Score** | **0,9916** | 0,9248 |
| **ROC AUC** | **0,9969** | 0,9759 |
| Tempo de treino | 14,4 ms | 3,5 ms |
| Tempo de inferência (µs/amostra) | **0,42** | 274,3 |
| Parâmetros aprendíveis | 140 | **8** |

A LR vence em performance bruta; o FHN ganha em compactação (17,5× menos features), interpretabilidade biofísica, sensibilidade temporal e potencial energético em hardware neuromórfico (não medido aqui).

Veja o **relatório completo** em `relatorio/RELATORIO.md` e a **análise crítica detalhada** na seção 6 do notebook.

---

## Estrutura da Pasta

```
ir alem 2/
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências Python
├── build_notebook.py               # Script gerador do notebook (fonte de verdade)
├── notebook/
│   └── ecg_neuromorphic.ipynb      # Notebook principal (executado)
├── imagens/                        # Gráficos gerados
│   ├── 01_distribuicao_classes.png
│   ├── 02_exemplos_por_classe.png
│   ├── 03_normalizacao.png
│   ├── 04_lr_confusion_roc.png
│   ├── 05_lr_weights.png
│   ├── 06_fhn_sanity_check.png
│   ├── 07_fhn_confusion_roc.png
│   ├── 08_comparacao_metricas.png
│   └── 09_comparacao_roc.png
└── relatorio/
    └── RELATORIO.md                # Relatório comparativo de 2 páginas
```

---

## Como Executar

### Pré-requisitos
- Python 3.9 ou superior
- macOS / Linux / Windows
- ~500 MB de espaço (com venv e dependências)

### Passo a passo

1. **Clone o repositório e entre na pasta:**
```bash
cd "cardio-ia/fase 3/ir alem 2"
```

2. **Crie um ambiente virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Registre o kernel do Jupyter (opcional, recomendado):**
```bash
python -m ipykernel install --user --name cardio-fase3-iralem2 \
    --display-name "Python (CardioIA Fase3 IrAlem2)"
```

5. **Abra o JupyterLab:**
```bash
jupyter lab notebook/ecg_neuromorphic.ipynb
```

6. **Execute todas as células sequencialmente** (Run → Run All Cells).

> O download do ECG5000 é automático na primeira execução (~2 MB).
> Tempo total de execução do notebook: ~30 segundos em CPU.

### Reproduzir o notebook do zero

Se você quiser regenerar o notebook a partir do script-fonte:
```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute \
    notebook/ecg_neuromorphic.ipynb --output ecg_neuromorphic.ipynb
```

---

## Dependências Principais

| Lib | Versão | Uso |
|---|---|---|
| numpy | ≥1.24, <2.0 | Arrays e operações numéricas |
| pandas | ≥2.0 | Manipulação de dados tabulares |
| scikit-learn | ≥1.3 | Regressão Logística, métricas, scaler |
| **brian2** | **≥2.6** | Simulação do FHN |
| **aeon** | **≥0.9** | Carregamento do ECG5000 (sucessor do sktime) |
| matplotlib + seaborn | últimas | Visualização |
| jupyterlab | ≥4.0 | Interface do notebook |

> ⚠️ **Importante:** Brian2 2.6 ainda não é compatível com numpy 2.x — por isso fixamos `numpy<2.0` no `requirements.txt`.

---

## Citações

Se você usar esse trabalho como referência, cite:

```bibtex
@article{dau2019ucr,
  title={The UCR Time Series Archive},
  author={Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and others},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={6}, number={6}, pages={1293--1305}, year={2019}
}

@article{fitzhugh1961impulses,
  title={Impulses and physiological states in theoretical models of nerve membrane},
  author={FitzHugh, Richard},
  journal={Biophysical Journal},
  volume={1}, number={6}, pages={445--466}, year={1961}
}

@article{stimberg2019brian,
  title={Brian 2, an intuitive and efficient neural simulator},
  author={Stimberg, Marcel and Brette, Romain and Goodman, Dan FM},
  journal={eLife}, volume={8}, pages={e47314}, year={2019}
}
```

---

## Licença

Trabalho acadêmico desenvolvido no curso de Inteligência Artificial da FIAP.
Código sob licença MIT. Dataset ECG5000 sob ODC-By 1.0.
