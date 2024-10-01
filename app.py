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




app = Flask(__name__)
translator = Translator()

modelo = joblib.load('modelo_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

# Configuração do Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Use um cache simples em memória para fins de demonstração
# Lista de ações brasileiras

acoes = ["ALOS3.SA", "ALPA4.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AZUL4.SA", "AZZA3.SA", "B3SA3.SA", "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA",
    "BRKM5.SA", "BRAV3.SA", "BRFS3.SA", "BPAC11.SA", "CXSE3.SA", "CRFB3.SA", "CCRO3.SA", "CMIG4.SA", "COGN3.SA", "CPLE6.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA",
    "CVCB3.SA", "CYRE3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "ENEV3.SA", "EGIE3.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA",
    "NTCO3.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "RENT3.SA", "LREN3.SA", "LWSA3.SA", "MGLU3.SA",
    "MRFG3.SA", "BEEF3.SA", "MRVE3.SA", "MULT3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "RECV3.SA", "PRIO3.SA", "PETZ3.SA", "RADL3.SA", "RAIZ4.SA", "RDOR3.SA",
    "RAIL3.SA", "SBSP3.SA", "SANB11.SA", "STBP3.SA", "SMTO3.SA", "CSNA3.SA", "SLCE3.SA", "SUZB3.SA", "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "TRPL4.SA",
    "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIVA3.SA", "WEGE3.SA", "YDUQ3.SA","CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DLTR","DASH","EA","EXC","FAST","FTNT","GEHC","GILD","GFS",
    "HON","IDXX","ILMN","INTC","INTU","ISRG","KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI","META","MCHP","MU","MSFT","MRNA","MDLZ","MDB","MNST","NFLX","NVDA",
    "NXPI","ORLY","ODFL","ON","PCAR","PANW","PAYX","PYPL","PDD","PEP","QCOM","REGN","ROP","ROST","SBUX","SMCI","SNPS","TTWO","TMUS","TSLA","TXN","TTD","VRSK","VRTX","WBD",
    "WDAY","XEL","ZS","MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AMTM","AEE","AEP",
    "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY",
    "AXON","BKR","BALL","BAC","BAX","BDX","BRK.B","BBY","TECH","BIIB","BLK","BX","BK","BA","BKNG","BWA","BSX","BMY","AVGO","BR","BRO","BF.B","BLDR","BG","BXP","CHRW","CDNS",
    "CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
    "CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS",
    "DHR","DRI","DVA","DAY","DECK","DE","DELL","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW",
    "EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ERIE","ESS","EL","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
    "FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY","GS",
    "HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD",
    "INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR",
    "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH",
    "MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX",
    "NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR",
    "PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR",
    "QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM",
    "SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN",
    "TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX",
    "VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"]


# Importe a lista de ações brasileiras do arquivo brazilian_stocks.py
from brazilian_stocks import brazilian_stocks

# Função para ajustar o símbolo com ".SA" para ações brasileiras
def adjust_symbol(symbol):
    symbol = symbol.upper()
    if symbol in brazilian_stocks:
        return f'{symbol}.SA'
    return symbol

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
