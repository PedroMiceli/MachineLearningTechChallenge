from flask import Flask, jsonify, render_template, request
from flask_caching import Cache
import yfinance as yf
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
from googletrans import Translator
from stocks import acoes

app = Flask(__name__)
translator = Translator()

modelo = joblib.load('modelo_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

# Configuração do Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Use um cache simples em memória para fins de demonstração

@app.route('/api/<symbol>')
@cache.cached(timeout=600)  # Cache válido por 10 minutos (ajuste conforme necessário)
def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        dados = get_infos(info)
        dados_scaled = scaler.transform([dados])
        previsao = modelo.predict(dados_scaled)

        return jsonify({'recomendado': int(previsao[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

def get_infos(stock_info):
    # Margem de lucro
    if 'profitMargins' in stock_info:
        margemDeLucro = stock_info['profitMargins']
    else:
        margemDeLucro = 0

    # dividendYield
    if 'dividendYield' in stock_info:
        dividendosPorAno = stock_info['dividendYield']
    else:
        dividendosPorAno = 0

    # Valor Patrimonial por ação
    if 'bookValue' in stock_info:
        valorPatrimonialPorAcao = stock_info['bookValue']
    else:
        valorPatrimonialPorAcao = 0

    # Divida total
    if 'totalDebt' in stock_info:
        dividaLiquida = stock_info['totalDebt']
    else:
        dividaLiquida = 9999999999

    # Preço da ação
    if 'currentPrice' in stock_info:
        precoAtual = stock_info['currentPrice']
    else:
        precoAtual = 0

    # Patrimonio total
    if 'sharesOutstanding' in stock_info:
        patrimonio = precoAtual * stock_info['sharesOutstanding']
    else:
        patrimonio = 0

    # Indice de endividamento
    if 'totalDebt' in stock_info and patrimonio > 0:
        indiceDeEndividamento = dividaLiquida / patrimonio
    else:
        indiceDeEndividamento = 0

    # Retorno sobre Ativos (ROA)
    if 'returnOnAssets' in stock_info:
        retornoSobreAtivos = stock_info['returnOnAssets']
    else:
        retornoSobreAtivos = 0

    return np.array([margemDeLucro, dividendosPorAno, valorPatrimonialPorAcao, dividaLiquida, precoAtual, patrimonio, indiceDeEndividamento, retornoSobreAtivos])

@app.route('/')
@cache.cached(timeout=600) 
def index():
    # Obtém os principais índices do mundo
    return render_template('index.html', acoes=acoes)

if __name__ == '__main__':
    app.run(debug=True)
