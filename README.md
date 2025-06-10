# PV & BESS Optimizer ⚡

Un'applicazione Streamlit per ottimizzare il dimensionamento di sistemi fotovoltaici (PV) e sistemi di accumulo a batteria (BESS) basata su dati di consumo reali e condizioni solari locali.

## 🚀 Caratteristiche

- **Ottimizzazione economica**: Trova la combinazione ottimale di PV e BESS per minimizzare il periodo di ammortamento
- **Dati solari reali**: Utilizza l'API PVGIS v5.2 (formato JSON) per ottenere dati di irraggiamento accurati
- **Simulazione dettagliata**: Simula 5 anni di operazione con intervalli di 15 minuti
- **Analisi finanziaria completa**: Calcola NPV, periodo di ammortamento, autosufficienza energetica
- **Degradazione realistica**: Modella la degradazione di PV e batterie nel tempo
- **Selezione rapida location**: Preset per città europee e mediterranee principali

## 📋 Requisiti

- Python 3.8+
- File CSV con dati di consumo (35.040 righe = 1 anno di dati a intervalli di 15 minuti)
- Connessione internet per accedere all'API PVGIS

## 🛠️ Installazione

1. Clona il repository:
```bash
git clone https://github.com/tuousername/pv-bess-optimizer.git
cd pv-bess-optimizer
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Avvia l'applicazione:
```bash
streamlit run app.py
```

## 📊 Formato Dati di Consumo

Il file CSV deve contenere:
- Una colonna chiamata `consumption_kWh`
- 35.040 righe (96 intervalli di 15 minuti × 365 giorni)
- Valori in kWh per intervallo di 15 minuti

Esempio:
```csv
consumption_kWh
0.125
0.130
0.128
...
```

## 🔧 Parametri Configurabili

### Vincoli di Progetto
- **Budget massimo**: Limite di spesa totale per l'installazione
- **Area disponibile**: Superficie disponibile per i pannelli solari (m²)

### Posizione
- **Latitudine/Longitudine**: Coordinate geografiche per i dati solari

### Parametri Avanzati
- **Prezzi elettricità**: Tariffe di acquisto e vendita dalla/alla rete
- **WACC**: Costo medio ponderato del capitale per l'analisi NPV
- **Parametri batteria**: DoD, C-rate, efficienza di carica/scarica

## 🐛 Risoluzione Problemi

### Errore API PVGIS
Se ricevi un errore dall'API PVGIS:
1. Verifica che le coordinate siano in Europa, Africa o Asia (usa il pulsante "Test Location")
2. Controlla la connessione internet
3. L'app proverà automaticamente database alternativi (SARAH2 → ERA5 → Default)
4. Usa i preset delle città per coordinate sicuramente funzionanti

### Aree Non Coperte
PVGIS **NON** copre:
- Nord e Sud America
- Australia e Oceania
- Regioni polari

### Problemi di Performance
Per dataset molto grandi o budget elevati, l'ottimizzazione potrebbe richiedere alcuni minuti. La barra di progresso mostra l'avanzamento.

## 📈 Output

L'applicazione fornisce:
- **Configurazione ottimale**: Dimensioni di PV (kWp) e batteria (kWh)
- **Metriche finanziarie**: CAPEX totale, NPV a 10 anni, periodo di ammortamento
- **Performance**: Tasso di autosufficienza, stato di salute della batteria dopo 5 anni
- **Grafici**: Progressione dei risparmi annuali, profilo di consumo giornaliero medio

## 🔄 Aggiornamenti Recenti

- **v3.0**: Passaggio a formato JSON per API PVGIS (più affidabile del CSV)
- **v3.0**: Aggiunta selezione rapida location con preset città
- **v3.0**: Test location integrato per verificare copertura PVGIS
- **v3.0**: Fallback automatico su database multipli (SARAH2 → ERA5 → Default)
- **v2.0**: Aggiornamento all'API PVGIS v5.2
- **v2.0**: Migliorato algoritmo di ricerca con step adattivi
- **v2.0**: Interfaccia utente migliorata con più visualizzazioni

## 📝 Licenza

MIT License

## 🤝 Contributi

I contributi sono benvenuti! Per favore:
1. Fork il repository
2. Crea un branch per la tua feature
3. Commit le modifiche
4. Push al branch
5. Apri una Pull Request
