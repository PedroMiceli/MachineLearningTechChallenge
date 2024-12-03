from flask import Flask, jsonify, render_template
from flask_caching import Cache
import yfinance as yf
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel


from googletrans import Translator
from stocks import acoes

app = Flask(__name__)
translator = Translator()

loaded_model = load_model('lstm_stock_model.h5')
# Definir um scaler para normalizar/desnormalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))

# Estrutura para o payload da requisição
class PredictionRequest(BaseModel):
    historical_data: list


# Configuração do Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Use um cache simples em memória para fins de demonstração

@app.route('/api/<symbol>')
@cache.cached(timeout=600)  # Cache válido por 10 minutos
def get_stock_info(symbol):
    try:
        # Configurar as datas para o download dos dados
        start_date = '2018-01-01'
        end_date = '2024-12-01'

        # Baixar os dados históricos da ação
        df = yf.download(symbol, start=start_date, end=end_date)

        if df.empty:
            return jsonify({'error': f"Não foi possível obter dados para o símbolo {symbol}."}), 404

        # Utilizar somente a coluna 'Close'
        data = df[['Close']].values

        # Normalizar os dados históricos
        scaled_data = scaler.fit_transform(data)

        # Criar a sequência de entrada para o modelo
        time_steps = 60
        if len(scaled_data) < time_steps:
            return jsonify({'error': f"Não há dados suficientes para o símbolo {symbol} (mínimo de {time_steps} pontos necessários)."}), 400

        input_sequence = scaled_data[-time_steps:]  # Selecionar os últimos 60 pontos
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Adicionar dimensão para o modelo LSTM

        # Fazer a previsão com o modelo carregado
        predicted_normalized = loaded_model.predict(input_sequence)
        predicted_value = scaler.inverse_transform(predicted_normalized)

        print(predicted_value[0][0])
        # Retornar a previsão em formato JSON
        return jsonify({'symbol': symbol, 'predicted_price': predicted_value[0][0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/')
@cache.cached(timeout=600) 
def index():
    # Obtém os principais índices do mundo
    return render_template('index.html', acoes=acoes)

if __name__ == '__main__':
    app.run(debug=True)
