import pandas as pd
import numpy as np

SCALE_PV_MW = 50.0   # Znamionowa moc farmy PV [MW]

def load_data(path: str) -> pd.DataFrame:
    """Wczytuje plik CSV, resampleuje do kroków 1-godzinnych i skaluje profil PV."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    # Resampling do pełnych godzin (średnia) – dla stabilności obliczeń
    df = df.resample("1h").mean().dropna(subset=["fix1_price"])
    df["pv_mw"] = df["pv_mw"].fillna(0.0)

    # Normalizacja profilu generacji PV do przedziału [0, SCALE_PV_MW]
    pv_max = df["pv_mw"].max()
    if pv_max > 0:
        df["pv_mw_scaled"] = df["pv_mw"] / pv_max * SCALE_PV_MW
    else:
        df["pv_mw_scaled"] = 0.0

    df = df.reset_index()
    df["date"] = df["datetime"].dt.date
    return df

def split_days(df: pd.DataFrame) -> list:
    """Dzieli roczny DataFrame na listę dobowych wektorów (date, price, pv)."""
    days = []
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("datetime")
        if len(grp) < 24:
            continue   # Pomijamy dni niepełne (np. zmiana czasu)
        price = grp["fix1_price"].values[:24].astype(float)
        pv = grp["pv_mw_scaled"].values[:24].astype(float)
        days.append((date, price, pv))
    return days

def calculate_pv_benchmark(days: list) -> float:
    """Oblicza roczny przychód ze sprzedaży energii bezpośrednio z PV (bez baterii)."""
    total_pv_profit = 0.0
    for _, price, pv in days:
        total_pv_profit += np.sum(price * pv * 1.0)  # dt = 1.0h
    return total_pv_profit