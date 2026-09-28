# Data

Two ways to feed the project:

1. **Your own sales history** — any daily file with `date`, `item`, `quantity` (+ optional `store`, `unit_price`, `unit_cost`, `channel`, `lead_time_days`). Start from [`templates/sales_history_template.csv`](templates/sales_history_template.csv); the full contract is in [docs/07_data_dictionary.md §0](../docs/07_data_dictionary.md). Upload it in the app (*Data* page) or run `python -m src.pipeline --all --input your_file.csv`.
2. **Demo dataset (Kaggle)** — used for the README results. It is **not** in this repository (Kaggle terms + size). Download it yourself:

## Option A — Kaggle CLI

1. Create an API token on kaggle.com → *Settings → API → Create New Token*, and save `kaggle.json` to `~/.kaggle/`.
2. Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/rules).
3. Run:

```bash
kaggle competitions download -c demand-forecasting-kernels-only -p data/raw
unzip data/raw/demand-forecasting-kernels-only.zip -d data/raw
```

## Option B — manual
Download `train.csv` from the competition's *Data* tab and put it in `data/raw/train.csv`.

## Expected layout

```
data/
├── raw/train.csv                 # 913,000 rows — git-ignored
├── reference/item_prices.csv     # synthetic demo prices (generated, committed)
├── templates/sales_history_template.csv   # upload template (committed)
└── processed/
    ├── inventory.duckdb          # git-ignored, rebuilt by the pipeline
    └── artefacts/                # small Parquet results for the dashboard (committed)
```

Check: `train.csv` has the columns `date,store,item,sales` and 913,000 rows.

## No download? Synthetic data for testing

`src/synthetic.py` generates Kaggle-shaped data (trend, weekly and yearly seasonality, Poisson noise). The tests
and CI use it, and you can try the whole pipeline with it:

```bash
python -c "from src.synthetic import write_kaggle_like_csv; write_kaggle_like_csv('synthetic.csv')"
python -m src.pipeline --all --input synthetic.csv
```

Results on synthetic data are only for testing: the README reports the Kaggle demo.
