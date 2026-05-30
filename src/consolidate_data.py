"""
Simple consolidation of price and PV data into hourly format (Zgodność stref czasowych).
"""
import pandas as pd

# Load price data (2025) - Zakładamy, że giełda RDN jest w czasie lokalnym (Warszawa)
prices = pd.read_csv("../bquxjob_379ac21_19e273f4940.csv")
prices['datetime'] = pd.to_datetime(prices['datetime']).dt.tz_localize(None)
prices = prices[prices['datetime'].dt.year == 2025]

prices['time_key'] = prices['datetime'].dt.strftime('%m-%d %H:00')
prices = prices.drop_duplicates(subset=['time_key'], keep='first')
prices = prices[['datetime', 'fix1_price', 'time_key']]

# Load PV data (2025) - Dane w UTC
pv = pd.read_csv("../energy-charts_Öffentliche_Nettostromerzeugung_in_Polen_2025.csv", header=[0, 1])
pv.columns = ['datetime', 'pv_mw']

# 1. Wczytujemy z wymuszeniem UTC
pv['datetime'] = pd.to_datetime(pv['datetime'], utc=True)

# 2. PRZELICZENIE NA CZAS POLSKI i dopiero potem usunięcie strefy (tz-naive)
pv['datetime'] = pv['datetime'].dt.tz_convert('Europe/Warsaw').dt.tz_localize(None)

# 3. Teraz time_key będzie idealnie pasował do cen lokalnych
pv['time_key'] = pv['datetime'].dt.strftime('%m-%d %H:00')
pv = pv.groupby('time_key')['pv_mw'].mean().reset_index()

# Merge on time_key
result = prices.merge(pv, on='time_key', how='left')
result = result[['datetime', 'fix1_price', 'pv_mw']].sort_values('datetime')

# Zapis (Teraz plik zawiera czysty, wyrównany polski czas lokalny)
result.to_csv('../solar_base.csv', index=False)
print(f"✓ Saved {len(result)} rows to solar_base.csv")