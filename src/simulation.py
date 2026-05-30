import numpy as np

# --- Parametry techniczne baterii ---
E_MAX = 100.0     # Pojemność maksymalna [MWh]
E_MIN = 10.0      # Pojemność minimalna (10% DoD Guard) [MWh]
E_INIT = 50.0     # Początkowy stan naładowania [MWh]
P_MAX = 50.0      # Maksymalna moc techniczna baterii [MW]
DT = 1.0          # Krok czasowy [h]

# --- Sprawności (Układ DC-Coupled) ---
ETA_PV = 0.96     # Wysoka sprawność ładowania bezpośrednio z paneli (DC/DC)
ETA_GRID = 0.90   # Niższa sprawność ładowania prądem z sieci (AC/DC)
ETA_D = 0.95      # Sprawność rozładowania baterii do sieci (DC/AC)

# --- Koszt zużycia sprzętu ---
BETA = 1.0       # Współczynnik kwadratowej degradacji ogniw [PLN/MW^2]

def simulate_day(u: np.ndarray, price: np.ndarray, pv: np.ndarray, 
                 e_init: float = E_INIT, penalize: bool = True) -> tuple:
    """
    Przeprowadza dobową symulację pracy układu hybrydowego.
    
    u: wektor decyzji (mocy) baterii dla 24h. u > 0 (ładowanie), u < 0 (rozładowanie)
    """
    T = len(price)
    soc = np.empty(T + 1)
    soc[0] = e_init
    total_profit = 0.0
    penalty = 0.0
    PENALTY_COEFF = 1e5  # Kara finansowa za złamanie ograniczeń SoC

    for t in range(T):
        ut = np.clip(u[t], -P_MAX, P_MAX)
        pvt = pv[t]

        # Finansowy bilans handlowy netto danej godziny (Sprzedaż PV + operacja baterii)
        profit_market = (pvt - ut) * price[t] * DT

        if ut >= 0:
            # ŁADOWANIE: Algorytm priorytetowo pobiera darmową energię z PV
            u_from_pv = min(ut, pvt)
            u_from_grid = ut - u_from_pv
            
            # Stan naładowania rośnie uwzględniając dwie różne sprawności
            delta_e = (u_from_pv * ETA_PV + u_from_grid * ETA_GRID) * DT
            new_soc = soc[t] + delta_e
        else:
            # ROZŁADOWANIE: Energia ucieka z baterii uwzględniając straty na falowniku
            delta_e = (ut / ETA_D) * DT  # ut jest ujemne, więc delta_e pomniejszy SoC
            new_soc = soc[t] + delta_e

        # Fizyczny, nieliniowy koszt zużycia ogniw (rośnie kwadratowo z mocą)
        cost_deg = BETA * (ut ** 2) * DT

        # Zysk godzinowy = Wynik rynkowy - koszt fizyczny zniszczenia ogniw
        total_profit += (profit_market - cost_deg)

        if penalize:
            if new_soc > E_MAX:
                penalty += PENALTY_COEFF * (new_soc - E_MAX) ** 2
                new_soc = E_MAX
            if new_soc < E_MIN:
                penalty += PENALTY_COEFF * (E_MIN - new_soc) ** 2
                new_soc = E_MIN
        else:
            new_soc = np.clip(new_soc, E_MIN, E_MAX)

        soc[t + 1] = new_soc

    return total_profit - penalty, soc

def fitness(u: np.ndarray, price: np.ndarray, pv: np.ndarray, e_init: float) -> float:
    """Funkcja celu przekazywana do optymalizatora (czysty zysk do maksymalizacji)."""
    profit, _ = simulate_day(u, price, pv, e_init=e_init, penalize=True)
    return profit