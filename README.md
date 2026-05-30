# Business Case: PV + BESS Arbitrage Optimization

## Executive Summary

This repository demonstrates a market-driven optimization model for a utility-scale battery energy storage system (BESS) co-located with a photovoltaic (PV) plant. The solution is designed to increase asset value by maximizing energy trading profit through price arbitrage while maintaining technical battery constraints and prioritizing PV generation.

## Business Value

- **Revenue uplift:** Enables the battery to buy low and sell high on wholesale electricity markets while capturing solar output in a commercial dispatch strategy.
- **Risk mitigation:** Uses historical price and PV generation data to identify profitable storage schedules, reducing exposure to highly volatile spot prices.
- **Asset optimization:** Demonstrates how a hybrid PV+BESS portfolio can improve utilization and cash flow compared to a PV-only plant.

## Why this matters

Energy storage is a key enabler for renewable integration and merchant revenue streams. This project models the commercial potential of pairing solar generation with storage to:

- smooth cash flow across daily price cycles,
- increase capture of premium price periods,
- reduce curtailment risk for PV generation,
- and create a realistic planning tool for trading strategies.

## What this repository contains

- **Data ingestion and preprocessing:** Scripts to align market price data and PV production data into a clean hourly dataset.
- **Simulation engine:** A day-level PV+BESS dispatch model that respects state-of-charge limits, efficiency losses, and market-oriented charge/discharge decisions.
- **Optimization methods:** Implementations of non-classical metaheuristics to search for near-optimal daily schedules and annual revenue.
- **Business-oriented output:** Profit evaluation based on energy market prices and BESS operation rather than pure technical performance metrics.

## Key assets

- `src/main.py` — main executable for running the optimization and profit simulation.
- `src/simulation.py` — business-focused dispatch model using PV-first charge logic and revenue calculation.
- `src/data_utils.py` — data loading, preprocessing, and daily slicing utilities.
- `src/consolidate_data.py` — data consolidation from raw CSV sources into the project dataset.
- `src/optimizers/annealing.py` — example Simulated Annealing optimizer for schedule search.

## Business-focused metrics

- **Annual profit** based on hourly market prices and net energy flows.
- **Battery utilization** measured by charge/discharge hours and energy throughput.
- **Revenue capture** from PV generation plus arbitrage differential.
- **Technical compliance** enforced through SoC boundaries, max power limits, and efficiency factors.

## Analyses location

- The exploratory analyses, figures and narrative are in the notebook [src/analysis.ipynb](src/analysis.ipynb). Open that notebook in any Jupyter viewer to inspect charts and results.
## Exact environment — super-simple steps for non-programmers

This project uses `uv`, an extremely fast Python package and project manager. Thanks to the included `uv.lock` file, you are guaranteed to run the exact same dependencies used for the original analyses.

Follow these simple steps in your Windows PowerShell:

1) **Install `uv`** (if you don't have it already). Copy and paste this exact command into PowerShell and press Enter:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2) **Navigate to the project folder**. You must be inside the repository folder (where this README is located). You can do this by using the `cd` command, replacing the path with your own:
   ```powershell
   cd path\to\your\project\folder
   ```
   *(Tip: Alternatively, open the project folder in File Explorer, right-click in an empty space, and select "Open in Terminal").*

3) **Sync the environment**. Once you are inside the project folder, run this command. It will automatically download the correct Python version, create a virtual environment, and install all strictly locked packages:
   ```powershell
   uv sync
   ```

If any step fails, copy the full error text and share it — I will help you fix it.
## Final note (business framing)

This is a concise, business-oriented demonstration of how storage can add commercial value to PV. If you want, I can prepare a short one-page slide summarizing the revenue uplift and key assumptions.

***

