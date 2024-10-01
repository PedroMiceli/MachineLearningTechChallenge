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
        # Ajuste o símbolo
        # adjusted_symbol = adjust_symbol(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info

        # Margem de lucro
        if 'profitMargins' in info:
            margemDeLucro = info['profitMargins']
        else:
            margemDeLucro = 0

        # dividendYield
        if 'dividendYield' in info:
            dividendosPorAno = info['dividendYield']
        else:
            dividendosPorAno = 0

        # Valor Patrimonial por ação
        if 'bookValue' in info:
            valorPatrimonialPorAcao = info['bookValue']
        else:
            valorPatrimonialPorAcao = 0

        # Divida total
        if 'totalDebt' in info:
            dividaLiquida = info['totalDebt']
        else:
            dividaLiquida = 9999999999

        # Preço da ação
        if 'currentPrice' in info:
            precoAtual = info['currentPrice']
        else:
            precoAtual = 0

        # Patrimonio total
        if 'sharesOutstanding' in info:
            patrimonio = precoAtual * info['sharesOutstanding']
        else:
            patrimonio = 0

        # Indice de endividamento
        if 'totalDebt' in info and patrimonio > 0:
            indiceDeEndividamento = dividaLiquida / patrimonio
        else:
            indiceDeEndividamento = 0

        # Retorno sobre Ativos (ROA)
        if 'returnOnAssets' in info:
            retornoSobreAtivos = info['returnOnAssets']
        else:
            retornoSobreAtivos = 0


        dados = np.array([margemDeLucro,dividendosPorAno,valorPatrimonialPorAcao,dividaLiquida,precoAtual,patrimonio,indiceDeEndividamento,retornoSobreAtivos])
        dados_scaled = scaler.transform([dados])
        previsao = modelo.predict(dados_scaled)

        return jsonify({'recomendado': int(previsao[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
@cache.cached(timeout=600) 
def index():
    # Obtém os principais índices do mundo
    return render_template('index.html', acoes=acoes)

if __name__ == '__main__':
    app.run(debug=True)
