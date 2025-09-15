# Repository Guidelines

## Project Structure & Module Organization
- `extract_features.py`: Feature extraction from IMU windows; writes `data/{split}.csv`.
- `imu_pipeline.py`: RandomForest-based `IMUPipeline` with predict/proba over signal features.
- `visualization.py`: Utilities for exploring/raw signal viewing.
- `data/`: Pre-extracted CSVs and `raw/` NPZ files; `manual_annotation/` labels; large artifacts live here.
- `models/`: Trained model artifacts (e.g., `imu_pipeline.pkl`).
- `videos/`: Dashcam clips for Part 2.
- `outputs/`: Place analysis figures/exports you generate.
- Notebooks: `imu_proj_notebook.ipynb`, `ml_notebook.ipynb` for analysis.

## Build, Test, and Development Commands
- Setup: `pip install -r requirements.txt` (Python 3.8+).
- Generate features: `python -c "from extract_features import process_dataset; process_dataset('inference')"`.
- Quick model run: `python -c "import joblib, pandas as pd; m=joblib.load('models/imu_pipeline.pkl'); df=pd.read_csv('data/inference.csv'); print(m.predict(df)[:5])"`.
- Notebook: `jupyter notebook` then open the provided notebooks.

## Coding Style & Naming Conventions
- Style: PEP 8, 4-space indents, max line length ~100.
- Naming: `snake_case` for functions/variables/modules, `PascalCase` for classes (e.g., `IMUPipeline`).
- Types: Add type hints where clear; prefer explicit imports.
- Formatting: Prefer `black` and `isort` locally; do not reformat unrelated files.

## Testing Guidelines
- No formal test suite. Add lightweight checks in notebooks or scripts:
  - Sanity assertions on feature columns and shapes.
  - Deterministic seeds for model training (`random_state=42`).
- If adding tests, use `pytest` style `tests/test_*.py` and keep data-dependent tests small/mocked.

## Commit & Pull Request Guidelines
- Commits: Imperative, concise, scoped changes (e.g., "add feature extraction stats").
- PRs: Include purpose, summary of changes, sample commands, and any data paths touched. Add screenshots/figures if relevant.
- Link issues (if any) and note breaking changes or migration steps.

## Security & Data Tips
- Do not commit large raw data or private labels beyond samples already present. Keep outputs under `outputs/` and notebooks output-trimmed when possible.
- Use `.gitignore`d virtualenvs; avoid storing secrets in notebooks or code.
