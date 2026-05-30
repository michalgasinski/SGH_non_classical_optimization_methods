"""
=============================================================================
BESS + PV  –  Optymalizacja harmonogramu 24h
Projekt studencki | Nieklasyczne metody optymalizacji
=============================================================================

Metody:
  1. Algorytm Genetyczny          (GA)   – DEAP
  2. Simulated Annealing          (SA)   – własna implementacja
  3. Particle Swarm Optimization  (PSO)  – własna implementacja
  4. SciPy SLSQP                  (REF)  – klasyczna metoda gradientowa (odniesienie)

Dane wejściowe:
  solar_base.csv  –  kolumny: datetime, fix1_price [PLN/MWh], pv_mw [kW]

Parametry baterii (domyślne):
  Pojemność   E_max  = 100 MWh
  Min. SoC    E_min  =  10 MWh  (10 % DoD guard)
  Moc ładow.  P_max  =  50 MW
  Sprawność   η      =  90 %   (round-trip → ładow. η_c=√0.9, rozład. η_d=√0.9)

Zmienna decyzyjna:
  u[t] ∈ [-P_max, +P_max]  (MW)
  u > 0 → ładowanie z sieci/PV do baterii
  u < 0 → rozładowanie baterii do sieci

Przepływ energii (dla gracza front-of-the-meter):
  - PV produkuje pv[t] MW;  gracz ZAWSZE sprzedaje całe PV na rynek.
  - Bateria jest dodatkowym aktyem:
      * ładowanie  (u>0): kupujemy u MW z rynku → koszt  = price[t] * u[t] * Δt
      * rozładow.  (u<0): sprzedajemy |u| MW     → przychód = price[t] * |u[t]| * Δt
  - SoC update: E[t+1] = E[t] + u[t]*η_c*Δt       (ładowanie)
                         E[t] + u[t]/η_d*Δt        (rozładowanie)
    (uproszczone: jeden η_rt jako sprawność cyklu)

Funkcja celu (MAXIMALIZACJA zysku baterii za rok):
  profit = Σ_days Σ_t ( -price[t] * u[t] * Δt )
           (ujemny koszt ładowania = zysk przy rozładowaniu po wyższej cenie)

Ograniczenia:
  E_min ≤ E[t] ≤ E_max   (pojemność)
  -P_max ≤ u[t] ≤ P_max  (moc)
  E[0] = E[T] (opcjonalne – SoC wraca do stanu początkowego)
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = "../solar_base.csv"         # ścieżka do pliku z danymi

# --- Parametry baterii -------------------------------------------------------
E_MAX   = 100.0   # MWh  – pojemność nominalna
E_MIN   =  10.0   # MWh  – minimalne SoC (DoD guard)
E_INIT  =  50.0   # MWh  – stan początkowy
P_MAX   =  50.0   # MW   – max moc ładowania / rozładowania
ETA_RT  =  0.90   # –    – sprawność round-trip
ETA_C   = np.sqrt(ETA_RT)   # sprawność ładowania
ETA_D   = np.sqrt(ETA_RT)   # sprawność rozładowania
DT      =  1.0    # h    – krok czasowy (dane godzinowe po resample)

# --- Skalowanie farmy PV -----------------------------------------------------
# Dane z Energy Charts podane są w kW dla konkretnej instalacji.
# Przyjmujemy skalowanie tak, aby szczyt produkcji odpowiadał SCALE_PV_MW MW.
SCALE_PV_MW = 50.0   # MW – znamionowa moc farmy PV

# --- Parametry algorytmów ----------------------------------------------------
SEED = 42

# GA
GA_POP      = 200
GA_NGEN     = 300
GA_CXPB     = 0.7
GA_MUTPB    = 0.2
GA_TOURN    = 5

# SA
SA_T_INIT   = 5000.0
SA_T_MIN    = 0.1
SA_ALPHA    = 0.995
SA_ITER_PER_T = 50

# PSO
PSO_N       = 100
PSO_ITER    = 300
PSO_W       = 0.7
PSO_C1      = 1.5
PSO_C2      = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# 2. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Wczytuje CSV, resampleuje do 1h, skaluje PV."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    # resample do pełnych godzin (mean) – dane mogą być 15-min lub 1h
    df = df.resample("1h").mean().dropna(subset=["fix1_price"])
    df["pv_mw"] = df["pv_mw"].fillna(0.0)

    # Skalowanie PV: normalizujemy do [0,1] względem maksimum, potem * SCALE_PV_MW
    pv_max = df["pv_mw"].max()
    if pv_max > 0:
        df["pv_mw_scaled"] = df["pv_mw"] / pv_max * SCALE_PV_MW
    else:
        df["pv_mw_scaled"] = 0.0

    df = df.reset_index()
    df["date"] = df["datetime"].dt.date
    return df


def split_days(df: pd.DataFrame):
    """Zwraca listę (price_array, pv_array) dla każdego dnia."""
    days = []
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("datetime")
        if len(grp) < 24:
            continue   # pomijamy niepełne dni
        price = grp["fix1_price"].values[:24].astype(float)
        pv    = grp["pv_mw_scaled"].values[:24].astype(float)
        days.append((date, price, pv))
    return days


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYMULACJA JEDNEGO DNIA
# ─────────────────────────────────────────────────────────────────────────────

def simulate_day(u: np.ndarray, price: np.ndarray, e_init: float = E_INIT,
                 penalize: bool = True) -> tuple:
    """
    Symuluje 24h pracy baterii.

    Parameters
    ----------
    u       : wektor decyzji [MW], len=24
              u>0 ładowanie, u<0 rozładowanie
    price   : ceny [PLN/MWh], len=24
    e_init  : SoC na początku dnia [MWh]
    penalize: czy doliczać kary za naruszenie ograniczeń

    Returns
    -------
    profit  : zysk baterii [PLN]
    soc     : trajektoria SoC [MWh], len=25
    """
    T = len(price)
    soc = np.empty(T + 1)
    soc[0] = e_init
    profit = 0.0
    penalty = 0.0
    PENALTY_COEFF = 1e5

    for t in range(T):
        ut = np.clip(u[t], -P_MAX, P_MAX)

        if ut >= 0:
            # ładowanie: zużywamy energię z rynku
            delta_e = ut * ETA_C * DT
            new_soc = soc[t] + delta_e
            profit -= price[t] * ut * DT        # płacimy za energię
        else:
            # rozładowanie: sprzedajemy energię na rynek
            delta_e = ut / ETA_D * DT           # ujemne
            new_soc = soc[t] + delta_e
            profit -= price[t] * ut * DT        # przychód (price*|u|*dt)

        if penalize:
            # kara za przekroczenie SoC
            if new_soc > E_MAX:
                penalty += PENALTY_COEFF * (new_soc - E_MAX) ** 2
                new_soc = E_MAX
            if new_soc < E_MIN:
                penalty += PENALTY_COEFF * (E_MIN - new_soc) ** 2
                new_soc = E_MIN
        else:
            new_soc = np.clip(new_soc, E_MIN, E_MAX)

        soc[t + 1] = new_soc

    return profit - penalty, soc


def annual_profit(solution_fn, days: list) -> float:
    """Oblicza roczny zysk stosując funkcję sterującą do każdego dnia."""
    total = 0.0
    e = E_INIT
    for date, price, pv in days:
        u = solution_fn(price, pv)
        profit, soc = simulate_day(u, price, e_init=e, penalize=False)
        total += profit
        e = soc[-1]   # SoC przechodzi na następny dzień
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTYMALIZACJA JEDNEGO DNIA – wspólny interfejs
# ─────────────────────────────────────────────────────────────────────────────

def fitness(u: np.ndarray, price: np.ndarray) -> float:
    """Zwraca zysk (do maksymalizacji)."""
    profit, _ = simulate_day(u, price, penalize=True)
    return profit


# ─────────────────────────────────────────────────────────────────────────────
# 4a.  ALGORYTM GENETYCZNY (GA)
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(price: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    Prosty GA z kodowaniem rzeczywistym.
    Chromosom: wektor 24 genów ∈ [-P_MAX, P_MAX].
    """
    rng = np.random.default_rng(seed)
    T = len(price)

    def random_individual():
        return rng.uniform(-P_MAX, P_MAX, T)

    def evaluate(ind):
        return fitness(ind, price)

    def tournament_select(pop, fits, k=GA_TOURN):
        idx = rng.choice(len(pop), k, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[best].copy()

    def crossover(p1, p2):
        alpha = rng.uniform(0, 1, T)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2

    def mutate(ind, sigma=P_MAX * 0.15):
        mask = rng.random(T) < 0.3
        ind[mask] += rng.normal(0, sigma, mask.sum())
        ind = np.clip(ind, -P_MAX, P_MAX)
        return ind

    # inicjalizacja
    pop = [random_individual() for _ in range(GA_POP)]
    fits = [evaluate(ind) for ind in pop]

    best_ind = pop[int(np.argmax(fits))].copy()
    best_fit = max(fits)

    for gen in range(GA_NGEN):
        new_pop = []
        while len(new_pop) < GA_POP:
            if rng.random() < GA_CXPB:
                p1 = tournament_select(pop, fits)
                p2 = tournament_select(pop, fits)
                c1, c2 = crossover(p1, p2)
            else:
                c1 = tournament_select(pop, fits).copy()
                c2 = tournament_select(pop, fits).copy()

            if rng.random() < GA_MUTPB:
                c1 = mutate(c1)
            if rng.random() < GA_MUTPB:
                c2 = mutate(c2)

            new_pop.extend([c1, c2])

        pop = new_pop[:GA_POP]
        fits = [evaluate(ind) for ind in pop]

        gen_best = max(fits)
        if gen_best > best_fit:
            best_fit = gen_best
            best_ind = pop[int(np.argmax(fits))].copy()

        # elityzm – wstawiamy najlepszego z powrotem
        worst_idx = int(np.argmin(fits))
        pop[worst_idx] = best_ind.copy()
        fits[worst_idx] = best_fit

    return best_ind


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  SIMULATED ANNEALING (SA)
# ─────────────────────────────────────────────────────────────────────────────

def run_sa(price: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    SA z schematem chłodzenia geometrycznego.
    Perturbacja: zmiana losowych godzin o wartość z N(0, σ).
    """
    rng = np.random.default_rng(seed)
    T = len(price)

    current = rng.uniform(-P_MAX, P_MAX, T)
    current_fit = fitness(current, price)
    best = current.copy()
    best_fit = current_fit

    temp = SA_T_INIT
    sigma = P_MAX * 0.3

    while temp > SA_T_MIN:
        for _ in range(SA_ITER_PER_T):
            # perturbacja jednej lub kilku godzin
            n_perturb = rng.integers(1, 5)
            idx = rng.choice(T, n_perturb, replace=False)
            candidate = current.copy()
            candidate[idx] += rng.normal(0, sigma, n_perturb)
            candidate = np.clip(candidate, -P_MAX, P_MAX)

            cand_fit = fitness(candidate, price)
            delta = cand_fit - current_fit

            if delta > 0 or rng.random() < np.exp(delta / temp):
                current = candidate
                current_fit = cand_fit

                if current_fit > best_fit:
                    best = current.copy()
                    best_fit = current_fit

        temp *= SA_ALPHA

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 4c.  PARTICLE SWARM OPTIMIZATION (PSO)
# ─────────────────────────────────────────────────────────────────────────────

def run_pso(price: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    Klasyczny PSO z inercją i akceleratorami poznawczym + społecznym.
    """
    rng = np.random.default_rng(seed)
    T = len(price)
    N = PSO_N

    # inicjalizacja pozycji i prędkości
    pos = rng.uniform(-P_MAX, P_MAX, (N, T))
    vel = rng.uniform(-P_MAX * 0.1, P_MAX * 0.1, (N, T))

    pbest_pos = pos.copy()
    pbest_fit = np.array([fitness(pos[i], price) for i in range(N)])

    gbest_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    w  = PSO_W
    c1 = PSO_C1
    c2 = PSO_C2

    for it in range(PSO_ITER):
        r1 = rng.random((N, T))
        r2 = rng.random((N, T))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        # ograniczenie prędkości
        vel = np.clip(vel, -P_MAX * 0.5, P_MAX * 0.5)

        pos = np.clip(pos + vel, -P_MAX, P_MAX)

        fits = np.array([fitness(pos[i], price) for i in range(N)])

        # aktualizacja pbest
        improved = fits > pbest_fit
        pbest_pos[improved] = pos[improved].copy()
        pbest_fit[improved] = fits[improved]

        # aktualizacja gbest
        best_idx = int(np.argmax(pbest_fit))
        if pbest_fit[best_idx] > gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        # liniowy spadek inercji
        w = PSO_W - (PSO_W - 0.4) * it / PSO_ITER

    return gbest_pos


# ─────────────────────────────────────────────────────────────────────────────
# 4d.  SCIPY SLSQP (klasyczna metoda gradientowa – punkt odniesienia)
# ─────────────────────────────────────────────────────────────────────────────

def run_slsqp(price: np.ndarray) -> np.ndarray:
    """
    Optymalizacja gradientowa z ograniczeniami nieliniowymi (SoC).
    Używa scipy.optimize.minimize z metodą SLSQP.
    """
    from scipy.optimize import minimize, Bounds

    T = len(price)

    def neg_profit(u):
        return -fitness(u, price)

    # ograniczenia SoC jako funkcje nierówności
    def soc_constraints():
        cons = []
        for t in range(T):
            def soc_min(u, t=t):
                soc = E_INIT
                for i in range(t + 1):
                    if u[i] >= 0:
                        soc += u[i] * ETA_C * DT
                    else:
                        soc += u[i] / ETA_D * DT
                return soc - E_MIN

            def soc_max(u, t=t):
                soc = E_INIT
                for i in range(t + 1):
                    if u[i] >= 0:
                        soc += u[i] * ETA_C * DT
                    else:
                        soc += u[i] / ETA_D * DT
                return E_MAX - soc

            cons.append({"type": "ineq", "fun": soc_min})
            cons.append({"type": "ineq", "fun": soc_max})
        return cons

    u0 = np.zeros(T)
    bounds = Bounds(-P_MAX, P_MAX)

    result = minimize(
        neg_profit, u0,
        method="SLSQP",
        bounds=bounds,
        constraints=soc_constraints(),
        options={"maxiter": 500, "ftol": 1e-6}
    )
    return result.x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTYMALIZACJA ROCZNA
# ─────────────────────────────────────────────────────────────────────────────

def optimize_annual(days: list, method: str) -> dict:
    """
    Optymalizuje każdy dzień osobno metodą `method`.
    Zwraca słownik z wynikami.
    """
    optimizer_map = {
        "GA":    run_ga,
        "SA":    run_sa,
        "PSO":   run_pso,
        "SLSQP": run_slsqp,
    }
    opt_fn = optimizer_map[method]

    total_profit = 0.0
    daily_profits = []
    daily_socs = []
    daily_actions = []

    e = E_INIT

    for i, (date, price, pv) in enumerate(days):
        if method == "SLSQP":
            u = opt_fn(price)
        else:
            u = opt_fn(price)

        profit, soc = simulate_day(u, price, e_init=e, penalize=False)
        total_profit += profit
        daily_profits.append(profit)
        daily_socs.append(soc)
        daily_actions.append(u)
        e = soc[-1]

        if (i + 1) % 30 == 0:
            print(f"  [{method}] Dzień {i+1}/{len(days)}  "
                  f"skumulowany zysk: {total_profit:,.0f} PLN")

    return {
        "method": method,
        "total_profit": total_profit,
        "daily_profits": daily_profits,
        "daily_socs": daily_socs,
        "daily_actions": daily_actions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  WIZUALIZACJA
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "GA":    "#e63946",
    "SA":    "#2a9d8f",
    "PSO":   "#f4a261",
    "SLSQP": "#457b9d",
}


def plot_comparison(results: list, days: list):
    """Wykres 1: porównanie skumulowanych zysków."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for res in results:
        cumsum = np.cumsum(res["daily_profits"])
        ax.plot(cumsum / 1000, label=res["method"],
                color=COLORS.get(res["method"], "gray"), linewidth=2)
    ax.set_title("Skumulowany zysk baterii – porównanie metod", fontsize=14)
    ax.set_xlabel("Dzień roku")
    ax.set_ylabel("Zysk skumulowany [tys. PLN]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_cumulative.png", dpi=150)
    plt.show()
    print("Zapisano: comparison_cumulative.png")


def plot_sample_day(results: list, days: list, day_idx: int = 0):
    """Wykres 2: szczegółowy widok wybranego dnia dla każdej metody."""
    date, price, pv = days[day_idx]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(len(results) + 1, 1, hspace=0.5)

    ax_price = fig.add_subplot(gs[0])
    hours = np.arange(24)
    ax_price.bar(hours, price, color="steelblue", alpha=0.7, label="Cena [PLN/MWh]")
    ax_price.axhline(0, color="black", linewidth=0.8)
    ax_price.set_title(f"Cena spot i harmonogram baterii – {date}", fontsize=13)
    ax_price.set_ylabel("PLN/MWh")
    ax_price.legend(loc="upper right")
    ax_price.grid(alpha=0.3)

    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[i + 1])
        u = res["daily_actions"][day_idx]
        soc = res["daily_socs"][day_idx]
        color = COLORS.get(res["method"], "gray")

        bars = ax.bar(hours, u, color=color, alpha=0.75, label=f"Działanie [MW]")
        ax2 = ax.twinx()
        ax2.plot(np.arange(25), soc, "k--", linewidth=1.5, label="SoC [MWh]")
        ax2.set_ylim(0, E_MAX * 1.1)
        ax2.set_ylabel("SoC [MWh]", fontsize=9)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Moc [MW]")
        ax.set_title(f"{res['method']}  |  zysk dnia: {res['daily_profits'][day_idx]:,.0f} PLN",
                     fontsize=11)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

    plt.savefig(f"sample_day_{day_idx}.png", dpi=150)
    plt.show()
    print(f"Zapisano: sample_day_{day_idx}.png")


def plot_profit_distribution(results: list):
    """Wykres 3: rozkład dziennych zysków (histogram)."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        profits_k = np.array(res["daily_profits"]) / 1000
        ax.hist(profits_k, bins=30, color=COLORS.get(res["method"], "gray"),
                alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(profits_k), color="black", linestyle="--", linewidth=1.5,
                   label=f"Śr.: {np.mean(profits_k):.1f} tys.")
        ax.set_title(res["method"])
        ax.set_xlabel("Zysk dzienny [tys. PLN]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Liczba dni")
    fig.suptitle("Rozkład dziennych zysków baterii", fontsize=13)
    plt.tight_layout()
    plt.savefig("profit_distribution.png", dpi=150)
    plt.show()
    print("Zapisano: profit_distribution.png")


def print_summary(results: list):
    print("\n" + "=" * 60)
    print(f"{'Metoda':<10} {'Zysk roczny [PLN]':>20} {'Zysk/dzień [PLN]':>18} {'Czas [s]':>10}")
    print("-" * 60)
    for res in results:
        print(f"{res['method']:<10} "
              f"{res['total_profit']:>20,.0f} "
              f"{np.mean(res['daily_profits']):>18,.0f} "
              f"{res.get('time', 0):>10.1f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BESS + PV  –  Optymalizacja harmonogramu")
    print("=" * 60)

    # --- wczytanie danych ---
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono pliku '{CSV_PATH}'.\n"
            "Upewnij się, że plik CSV jest w tym samym katalogu co skrypt."
        )

    print(f"\nWczytywanie danych z: {CSV_PATH}")
    df = load_data(CSV_PATH)
    days = split_days(df)
    print(f"  Łącznie pełnych dni: {len(days)}")
    print(f"  Zakres: {days[0][0]}  →  {days[-1][0]}")

    # --- wybór metod do uruchomienia ---
    # Możesz zakomentować metody których nie chcesz uruchamiać:
    METHODS = [
        "GA",
        "SA",
        "PSO",
        "SLSQP",   # ← zakomentuj jeśli chcesz szybszy test
    ]

    results = []
    for method in METHODS:
        print(f"\n{'─'*40}")
        print(f"  Uruchamiam: {method}")
        print(f"{'─'*40}")
        t0 = time.perf_counter()
        res = optimize_annual(days, method)
        res["time"] = time.perf_counter() - t0
        results.append(res)
        print(f"  → Zysk roczny: {res['total_profit']:,.0f} PLN  "
              f"(czas: {res['time']:.1f}s)")

    # --- podsumowanie ---
    print_summary(results)

    # --- wykresy ---
    print("\nGenerowanie wykresów...")
    plot_comparison(results, days)
    plot_sample_day(results, days, day_idx=0)
    plot_profit_distribution(results)

    # --- zapis wyników do CSV ---
    rows = []
    for res in results:
        for i, (date, _, _) in enumerate(days):
            rows.append({
                "date": date,
                "method": res["method"],
                "profit_pln": res["daily_profits"][i],
                "soc_end_mwh": res["daily_socs"][i][-1],
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv("results.csv", index=False)
    print("Zapisano wyniki do: results.csv")


if __name__ == "__main__":
    main()