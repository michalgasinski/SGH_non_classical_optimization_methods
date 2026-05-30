import numpy as np
from simulation import fitness, P_MAX

SA_T_INIT = 5000.0       
SA_T_MIN = 0.1           
SA_ALPHA = 0.995         
SA_ITER_PER_T = 40       

def run_sa(price: np.ndarray, pv: np.ndarray, e_init: float, seed: int = 42) -> tuple:
    """Poszukuje optymalnego profilu pracy baterii i zwraca (best_sol, history)."""
    rng = np.random.default_rng(seed)
    T = len(price)

    current_sol = rng.uniform(-P_MAX, P_MAX, T)
    current_fit = fitness(current_sol, price, pv, e_init)
    
    best_sol = current_sol.copy()
    best_fit = current_fit

    temp = SA_T_INIT
    sigma = P_MAX * 0.2  
    
    # Lista do zbierania historii zbieżności
    history = []
    history_current = []    
    history_u = []    

    while temp > SA_T_MIN:
        for _ in range(SA_ITER_PER_T):
            n_perturb = rng.integers(1, 5)
            hours_to_change = rng.choice(T, n_perturb, replace=False)
            
            candidate = current_sol.copy()
            candidate[hours_to_change] += rng.normal(0, sigma, n_perturb)
            candidate = np.clip(candidate, -P_MAX, P_MAX)

            cand_fit = fitness(candidate, price, pv, e_init)
            delta_profit = cand_fit - current_fit

            if delta_profit > 0 or rng.random() < np.exp(delta_profit / temp):
                current_sol = candidate
                current_fit = cand_fit

                if current_fit > best_fit:
                    best_sol = current_sol.copy()
                    best_fit = current_fit
        
        # Zapisujemy najlepszy dotychczasowy zysk na koniec każdego kroku temperatury
        history.append(best_fit)
        history_u.append(best_sol.copy())
        history_current.append(current_fit)
        temp *= SA_ALPHA

    return best_sol, history, history_current, history_u