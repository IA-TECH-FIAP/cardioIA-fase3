import requests
import smtplib
import time
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- CONFIGURAÇÕES ---
API_URL = "https://jsonplaceholder.typicode.com/posts" # Simulando endpoint REST
EMAIL_SENDER = "giuliabugatti02@gmail.com"
EMAIL_PASSWORD = "uhwmykteodxitkbb" # Use senhas de app do Google
EMAIL_RECEIVER = "giuliabugatti02@gmail.com"

# --- LÓGICA DE NEGÓCIO (Limites de Risco) ---
LIMITES = {
    "freq_cardiaca": 100,  # bpm (Taquicardia)
    "temperatura": 37.8,   # °C (Febre)
    "movimento": 1         # 0 para ausência de movimento
}

def verificar_riscos(dados):
    alertas = []
    if dados['bpm'] > LIMITES['freq_cardiaca']:
        alertas.append(f"Taquicardia detectada: {dados['bpm']} bpm")
    if dados['temp'] > LIMITES['temperatura']:
        alertas.append(f"Febre detectada: {dados['temp']}°C")
    if dados['mov'] < LIMITES['movimento']:
        alertas.append("Alerta: Ausência de movimento detectada")
    return alertas

def enviar_email_alerta(mensagens):
    corpo = "ALERTA DE SAÚDE CRÍTICO:\n\n" + "\n".join(mensagens)
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = "⚠️ Alerta de Monitoramento - Paciente 001"
    
    msg.attach(MIMEText(corpo, 'plain'))

    try:
        # Configuração simplificada para Gmail (SMTP)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print("✉️ E-mail de emergência enviado com sucesso!")
    except Exception as e:
        print(f"❌ Falha ao enviar e-mail: {e}")

def simular_sistema():
    print("🚀 Iniciando Monitoramento Digital...")
    
    # Simulação de dados recebidos de um sensor/API
    dados_paciente = {
        "bpm": 115,  # Exemplo de risco
        "temp": 38.5, # Exemplo de risco
        "mov": 0      # Exemplo de risco
    }

    # 1. Consumo/Envio via REST (Simulação)
    try:
        response = requests.post(API_URL, json=dados_paciente)
        if response.status_code == 201:
            print(f"✅ Dados sincronizados com o servidor (Status: {response.status_code})")
    except Exception as e:
        print(f"Erro na comunicação REST: {e}")

    # 2. Verificação de Risco
    alertas_encontrados = verificar_riscos(dados_paciente)

    # 3. Automação de Resposta (RPA)
    if alertas_encontrados:
        print(f"🚨 RISCO DETECTADO! Disparando protocolos...")
        enviar_email_alerta(alertas_encontrados)
    else:
        print("🟢 Sinais vitais dentro da normalidade.")

if __name__ == "__main__":
    simular_sistema()