# Use a imagem base oficial do Python
FROM python:3.10-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo requirements.txt para o container
COPY requirements.txt .

# Instala as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos da aplicação para o container
COPY . .

# Expõe a porta 5000
EXPOSE 5000

# Define o comando padrão para iniciar a aplicação
CMD ["python", "app.py"]