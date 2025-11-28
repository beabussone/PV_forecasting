# Previsione della produzione di energia fotovoltaica

Questo progetto si propone di sviluppare un modello per prevedere la produzione di energia di un impianto fotovoltaico (PV) basandosi su dati meteorologici e altri parametri rilevanti. La previsione accurata della produzione di energia fotovoltaica è cruciale per l'ottimizzazione della gestione della rete elettrica, l'integrazione delle fonti rinnovabili e la pianificazione energetica.

Il set di dati utilizzato in questo progetto include informazioni meteorologiche come temperatura, punto di rugiada, pressione, umidità, velocità e direzione del vento, pioggia, copertura nuvolosa e descrizione del tempo, insieme a dati sulla radiazione solare (Dhi, Dni, Ghi) e la produzione di energia fotovoltaica corrispondente (kWp).

Attraverso l'analisi di questi dati, la pulizia e la preelaborazione, lo sviluppo di modelli di machine learning e la valutazione delle prestazioni, puntiamo a creare un modello di previsione robusto e affidabile.

**Obiettivi del progetto:**

*   Esplorare e comprendere la relazione tra i dati meteorologici, la radiazione solare e la produzione di energia fotovoltaica.
*   Preelaborare i dati per gestire valori mancanti, outlier e variabili categoriche.
*   Sviluppare e confrontare diversi modelli per la previsione della produzione di energia fotovoltaica.
*   Valutare le prestazioni dei modelli utilizzando metriche appropriate.
*   Fornire un modello che possa essere utilizzato per prevedere in modo affidabile la produzione di energia fotovoltaica.

Questo progetto contribuirà a una migliore comprensione dei fattori che influenzano la produzione di energia solare e fornirà uno strumento utile per la pianificazione e l'ottimizzazione dei sistemi fotovoltaici.

### Analisi Esplorativa dei Dati (EDA)  

In questa fase eseguiremo un'analisi esplorativa del dataset con l’obiettivo di comprendere la struttura, la qualità e le relazioni tra le variabili disponibili. L’EDA ci consentirà di identificare eventuali valori mancanti, outlier o anomalie nei dati, nonché di analizzare la distribuzione statistica delle variabili meteorologiche (temperatura, umidità, pressione, radiazione solare, ecc.) e la loro correlazione con la produzione di energia fotovoltaica (kWp). Verranno utilizzate tecniche di **statistica descrittiva**, **visualizzazioni grafiche** (istogrammi, boxplot, heatmap di correlazione) e **analisi temporali** per osservare l’andamento della produzione energetica rispetto alle condizioni climatiche. Questa fase fornirà una comprensione approfondita del dataset e guiderà le successive scelte di **preprocessing**, **feature engineering** e **modellazione predittiva**.

## **Osservazioni**

# Risultati sintetici EDA
- Missing: la sola variabile con vuoti rilevanti è `rain_1h` (~79%); le etichette PV non hanno missing. Per il modello imputiamo `rain_1h` a 0.
- Correlazioni numeriche con `kwp`: le componenti radiative sono le più informative (`Ghi` ≈ 0.95, `Dni` ≈ 0.79, `Dhi` ≈ 0.66), seguite da umidità e temperatura (≈0.43). Queste feature vanno conservate con attenzione.
- Categorie meteo: poche classi dominanti (`sky is clear`, `light rain`, `overcast clouds`) e molte classi rare; da qui la scelta di raggruppare e fare One-Hot Encoding con colonna `other`.
- Analisi temporale: il picco di `kwp` è concentrato nelle ore centrali del giorno, coerente con l’allineamento a fuso fisso e con i picchi di irraggiamento nei plot `eda_plots/time_series_kwp.png`.


# Ciclical Encoding

In questa sezione trasformiamo la colonna dt_iso per renderla digeribile dal modello, risolvendo due criticità fondamentali delle serie storiche:

**Rimozione dell'Ora Legale** (DST): abbiamo convertito tutti i timestamp in un fuso orario fisso (UTC+10 statico). Questo elimina i "salti" artificiali dell'ora legale e garantisce che il picco di irraggiamento solare avvenga sempre alla stessa ora "fisica" durante tutto l'anno, evitando che il modello veda il mezzogiorno solare oscillare tra le 12:00 e le 13:00.

**Encoding Ciclico** (Seno/Coseno): Le variabili cicliche come l'ora (0-23) e il mese (1-12) sono state trasformate in coordinate su un cerchio usando funzioni trigonometriche ($sin$ e $cos$). Questo permette alla rete neurale di comprendere correttamente la continuità temporale (es. capire che le ore 23:00 e 00:00 sono adiacenti) che andrebbe persa con una semplice rappresentazione numerica lineare.


## Feature engineering
Per migliorare le prestazioni e integrare conoscenza fisica senza introdurre artefatti (come quelli derivati dall’uso di POA senza tilt reale), sono state aggiunte due famiglie di feature:

### Solar features
| Feature           | Descrizione |
| ----------------- | ----------- |
| `solar_zenith`    | Angolo zenitale (90° = sole allo zenit), influenza la radiazione incidente. |
| `solar_azimuth`   | Direzione del sole (0° Nord, 180° Sud), distingue mattino/pomeriggio. |
| `clearness_index` | Rapporto tra GHI reale ed ETR (extraterrestrial irradiance), misura la limpidezza del cielo. |

### Effective irradiance
| Feature                | Formula                   | Significato |
| ---------------------- | ------------------------- | ----------- |
| `effective_irradiance` | `DNI * cos(zenith) + DHI` | Stima dell’energia effettivamente utile al pannello. |
| `direct_fraction`      | `DNI / (DNI + DHI)`       | Indica se prevale radiazione diretta o diffusa. |
| `clear_sky_index`      | `GHI / GHI_clear`         | Quanto la condizione reale differisce dal cielo ideale. |

Queste feature incorporano informazione fisica verificabile e sono estremamente predittive per modelli PV.

### Osservazioni
- Queste feature derivano da combinazioni non lineari → aggiungono informazione reale, non ridondanza.
- Sono altamente correlate con la produzione PV in modo fisicamente coerente.

## Output del preprocessing
- Dataset con feature: `data/processed/X_feat.csv`
- Target: `data/processed/y_processed.csv`