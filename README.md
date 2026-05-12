# Ir Além 2 — IA em Séries Temporais de Saúde
CardioIA — Fase 3 | FIAP — Inteligência Artificial 
Documentação: Sistema Cardio IA 🩺

Este repositório contém a implementação de um sistema de monitoramento de sinais vitais automatizado, integrando comunicação **REST API**, lógica de detecção de anomalias clínicas e **RPA para alertas via e-mail**.

## 👥 Equipe (2TIAOA)
* **Giulia Bugatti Fonseca** – RM 562675
* **Mahmod Ahmad Issa** – RM 561426
* **Matheus Cardoso Oliveira Lima** – RM 565844
* **Silas Fernandes de Souza Fonseca** - RM 564246

---

## 🚀 Fluxo de Funcionamento

O sistema executa um ciclo completo de monitoramento:
1.  **Captura:** Simula a leitura de sensores de um paciente.
2.  **Sincronização:** Envia os dados para um servidor via protocolo **HTTP (API REST)**.
3.  **Análise:** Verifica se os parâmetros de saúde estão dentro dos limites seguros.
4.  **Ação (RPA):** Dispara automaticamente um e-mail de emergência caso riscos sejam detectados.

---

## 📸 Evidências de Funcionamento

### 1. Execução no Terminal
Abaixo, a captura de tela demonstrando a comunicação com a API e o processamento da lógica de risco em tempo real:

![Execução do Terminal](imagens\terminal.png)

### 2. Alerta Recebido (RPA)
O e-mail disparado automaticamente pelo sistema assim que a taquicardia ou febre foi identificada:

![E-mail de Alerta](imagens\email.png)

---

## 🛠️ Como Executar

1.  Certifique-se de ter o Python 3.x instalado.
2.  Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure suas credenciais no arquivo `monitor_vitals.py`.

    > ⚠️ **Importante:** Para o envio via Gmail, utilizamos o recurso de **"Senhas de App" do Google**, garantindo uma autenticação segura do script sem expor a senha principal da conta.

4.  Execute o monitoramento:
    ```bash
    python monitor_vitals.py
    ```

---

### 📋 Critérios Clínicos (LIMITES)
* **BPM:** > 100 (Taquicardia)
* **Temperatura:** > 37.8°C (Estado Febril)
* **Movimento:** 0 (Alerta de Queda/Inércia)
