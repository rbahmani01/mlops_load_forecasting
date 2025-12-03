from __future__ import annotations

"""
Build df_mlf (load + holidays + weather) from the Kaggle
"jeanmidev/smart-meters-in-london" dataset and write it into Postgres.

All important logic (historical data window, missing ratio, imputation, grouping,
holidays, weather) is taken mostly from the EDA notebook and wrapped
in functions. Only DB-related code and a few safety checks are new.

Usage (from repo root):

    export KAGGLE_USERNAME=...
    export KAGGLE_KEY=...

    export DB_HOST=localhost
    export DB_PORT=5432
    export DB_NAME=energy
    export DB_USER=energy_user
    export DB_PASSWORD=change_me
    export DB_MLF_TABLE=df_mlf   # optional

    export MLF_N_FILES=10        # how many meter CSVs to use; None = all

    python -m scripts.kaggle_to_db_df_mlf
"""

import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "smart_meters"
METERS_DIR = DATA_ROOT / "halfhourly_dataset" / "halfhourly_dataset"
KAGGLE_DATASET = "jeanmidev/smart-meters-in-london"
MLF_N_FILES=20   # change later if you want
group_size = 20  # number of houses per group

def ensure_utc(s: pd.Series) -> pd.Series:
    """
    Make a datetime Series tz-aware in UTC.

    - If naive      -> tz_localize('UTC')
    - If tz-aware   -> tz_convert('UTC')
    """
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    else:
        return s.dt.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Kaggle download 
# ---------------------------------------------------------------------------

def download_kaggle_dataset() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The 'kaggle' package is not installed. "
            "Install it with `pip install kaggle`."
        ) from e

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DATA_ROOT),
        unzip=True,
    )

    return DATA_ROOT


# ---------------------------------------------------------------------------
# 1) half-hourly loading + cleaning
# ---------------------------------------------------------------------------

def load_halfhourly_smart_meters(n_files: int | None) -> pd.DataFrame:
    """
    logic:

    path_dataset = 'data/smart_meters'
    path_dataset_meters = 'data/smart_meters/halfhourly_dataset/halfhourly_dataset'
    N = 10
    ...
    halfhourly_dataset_clean = ...

    Returns: halfhourly_dataset_clean
    """
    import tqdm

    path_dataset = str(DATA_ROOT)
    path_dataset_meters = str(METERS_DIR)

    if not os.path.isdir(path_dataset_meters):
        raise FileNotFoundError(
            f"Expected smart meter CSVs under {path_dataset_meters}. "
            "Run the Kaggle download step first."
        )

    file_list = os.listdir(path_dataset_meters)
    if n_files is not None:
        file_list = file_list[:n_files]

    N = len(file_list)
    print(f"Using N={N} meter files from {path_dataset_meters}")

    halfhourly_dataset_list = []

    for file_name in tqdm.tqdm(file_list, desc="Loading smart meter CSVs"):
        df_temp = pd.read_csv(
            os.path.join(path_dataset_meters, file_name),
            index_col="tstp",
            parse_dates=True,
            low_memory=False,
        )
        df_temp["file_name"] = file_name.split(".")[0]
        df_temp = df_temp.replace("Null", np.nan).dropna()
        df_temp["energy(kWh/hh)"] = df_temp["energy(kWh/hh)"].astype(float)
        halfhourly_dataset_list.append(df_temp)

    if not halfhourly_dataset_list:
        raise RuntimeError(f"No CSV files loaded from {path_dataset_meters}")

    halfhourly_dataset = pd.concat(halfhourly_dataset_list, axis=0)

    # --- find bad LCLids & drop them ---
    bad_lclids = (
        halfhourly_dataset
        .groupby("LCLid")["energy(kWh/hh)"]
        .apply(lambda x: x.isna().any())
    )

    bad_lclids = bad_lclids[bad_lclids].index

    num_lclids_dropped = len(bad_lclids)

    halfhourly_dataset_clean = halfhourly_dataset[
        ~halfhourly_dataset["LCLid"].isin(bad_lclids)
    ]

    num_rows_dropped = len(halfhourly_dataset) - len(halfhourly_dataset_clean)

    print("LCLids dropped:", num_lclids_dropped)
    print("Rows dropped:", num_rows_dropped)

    return halfhourly_dataset_clean


# ---------------------------------------------------------------------------
# 2) historical window logic 
# ---------------------------------------------------------------------------

def find_best_365_day_window(
    halfhourly_dataset_clean: pd.DataFrame,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Index]:

    df = halfhourly_dataset_clean.copy()

    # daily_presence
    daily_presence = (
        df.groupby([pd.Grouper(freq="D"), "LCLid"])
          .size()
          .unstack(fill_value=0)
    )
    daily_presence = daily_presence > 0

    window = int(365 * 1.1)  #  modified window size

    dates = daily_presence.index

    best_start = None
    best_end = None
    best_meter_group = None
    max_meter_count = 0

    for i in range(len(dates) - window + 1):
        window_data = daily_presence.iloc[i:i+window]

        # Only meters that are present ALL historical days
        common_meters = window_data.all(axis=0)

        num_meters = common_meters.sum()

        if num_meters > max_meter_count:
            max_meter_count = num_meters
            best_start = dates[i]
            best_end = dates[i+window-1]
            best_meter_group = common_meters[common_meters].index

    if best_start is None or best_meter_group is None:
        raise RuntimeError("Could not find a valid historical window with any meters present.")

    print("Best historial data period:")
    print("Start:", best_start)
    print("End:  ", best_end)
    print("Meters available every day:", len(best_meter_group))

    return best_start, best_end, best_meter_group


# ---------------------------------------------------------------------------
# 3) missing-ratio + imputation logic
# ---------------------------------------------------------------------------

def filter_and_impute_year(
    halfhourly_dataset_clean: pd.DataFrame,
    best_start: pd.Timestamp,
    best_end: pd.Timestamp,
    best_meter_group: pd.Index,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Returns: imputed, expected_index
    """
    df = halfhourly_dataset_clean

    # restrict to year and best meters
    final_year_data = df[
        (df.index >= best_start) &
        (df.index <= best_end)
    ]

    final_year_data = final_year_data[
        final_year_data["LCLid"].isin(best_meter_group)
    ]

    # expected index + missing ratio
    freq = "30min"  # 
    expected_index = pd.date_range(
        start=best_start,
        end=best_end,
        freq=freq,
    )
    total_slots = len(expected_index)
    print("Total expected slots per meter in this window:", total_slots)

    # Count missing timestamps per LCLid (vectorized)
    missing_counts = (
        final_year_data
        .groupby("LCLid")
        .apply(lambda g: len(expected_index.difference(g.index)))
    )
    print("Missing counts head:")
    print(missing_counts.head())

    # Convert to ratio of missing points
    missing_ratio = missing_counts / total_slots

    print("\nMissing ratio summary:")
    print(missing_ratio.describe())

    # Choose a threshold: allow up to 1% missing points in the year
    threshold = 0.01

    good_meters = missing_ratio[missing_ratio <= threshold].index
    bad_meters  = missing_ratio[missing_ratio >  threshold].index

    print("\nGood meters:", len(good_meters))
    print("Bad meters :", len(bad_meters))

    # Keep only the good meters
    final_year_data = final_year_data[final_year_data["LCLid"].isin(good_meters)]

    # reindex_and_impute
    final_year_data_reset = final_year_data.reset_index().rename(columns={"index": "tstp"})

    def reindex_and_impute(g: pd.DataFrame) -> pd.DataFrame:
        # 1) put tstp on the index and reindex to full grid
        g = g.set_index("tstp").reindex(expected_index)

        # 2) work on the energy series
        s = g["energy(kWh/hh)"]

        # 2a) fill using the same time previous week: 7 days * 48 half-hour periods
        s = s.fillna(s.shift(48 * 7))

        # 2b) time interpolation
        s = s.interpolate(method="time")

        # 2c) forward/backward fill any remaining NaNs
        s = s.ffill().bfill()

        g["energy(kWh/hh)"] = s

        # 3) restore LCLid (it is constant for the group)
        g["LCLid"] = g["LCLid"].fillna(g["LCLid"].iloc[0])

        return g

    year_imputed = (
        final_year_data_reset
        .groupby("LCLid", group_keys=False)
        .apply(reindex_and_impute)
    )

    # 1) Give the index a name
    year_imputed.index.name = "tstp"
    year_imputed = year_imputed.sort_values(["LCLid", "tstp"])
    year_imputed["LCLid"] = year_imputed["LCLid"].astype(final_year_data["LCLid"].dtype)

    # safety check
    if year_imputed["energy(kWh/hh)"].isna().any():
        raise RuntimeError("Imputation failed: NaNs remain in 'energy(kWh/hh)'.")

    print("Any NaNs left in imputed year?",
          year_imputed["energy(kWh/hh)"].isna().any())
    print("Shape of year_imputed:", year_imputed.shape)
    print("Meters:", year_imputed["LCLid"].nunique())

    return year_imputed, expected_index


# ---------------------------------------------------------------------------
# 4) df_mlf creation 
# ---------------------------------------------------------------------------

def build_hourly_grouped_load(
    halfhourly_dataset_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    construct df_mlf from year_imputed.

    Returns:
        df_mlf      : hourly grouped load with (unique_id, ds, y)
        best_start  : start of the selected historical window (date)
        best_end    : end of the selected historical window (date)
    """
    best_start, best_end, best_meter_group = find_best_365_day_window(
        halfhourly_dataset_clean
    )
    year_imputed, expected_index = filter_and_impute_year(
        halfhourly_dataset_clean, best_start, best_end, best_meter_group
    )

    year_imputed = year_imputed.drop(columns=["file_name"])

    # 0) make sure tstp is a column (not index)
    df_tmp = year_imputed.reset_index()   # columns: tstp, LCLid, energy(kWh/hh)
    df_tmp["tstp"] = ensure_utc(df_tmp["tstp"])

    # 1) Aggregate from 30-min → hourly (sum 2 half-hours per hour, per LCLid)
    hourly_dataset = (
        df_tmp
          .groupby(
              ["LCLid", pd.Grouper(key="tstp", freq="h")]
          )["energy(kWh/hh)"]
          .sum()
          .reset_index()
    )

    # groups of ~20 houses
    unique_houses = hourly_dataset["LCLid"].unique()

    house_groups = [
        unique_houses[i:i + group_size]
        for i in range(0, len(unique_houses), group_size)
    ]

    aggregated_groups = []

    for idx, group in enumerate(house_groups):
        agg = (
            hourly_dataset[hourly_dataset["LCLid"].isin(group)]
                .groupby("tstp", as_index=False)["energy(kWh/hh)"]
                .sum()
        )
        agg["unique_id"] = f"group_{idx+1}"
        aggregated_groups.append(agg)

    # Combine all aggregated groups into one dataframe
    df_mlf = pd.concat(aggregated_groups)

    # rename + sort
    df_mlf = df_mlf.rename(columns={
        "LCLid": "unique_id",
        "tstp": "ds",
        "energy(kWh/hh)": "y",
    })

    df_mlf["ds"] = ensure_utc(df_mlf["ds"])
    df_mlf = df_mlf.sort_values(["unique_id", "ds"])

    return df_mlf, best_start, best_end



# ---------------------------------------------------------------------------
# 5) holidays & weather logic 
# ---------------------------------------------------------------------------

def attach_holidays(df_mlf: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook cells 39 and 40.
    """
    path_dataset = str(DATA_ROOT)

    holidays_path = os.path.join(path_dataset, "uk_bank_holidays.csv")
    if not os.path.isfile(holidays_path):
        raise FileNotFoundError(f"Expected {holidays_path} (from Kaggle dataset).")

    # df_holiday: raw bank holiday file
    df_holiday = pd.read_csv(holidays_path)
    df_holiday["Bank holidays"] = ensure_utc(df_holiday["Bank holidays"])
    df_holiday = df_holiday.rename(columns={"Bank holidays": "date"})

    df_holiday["is_holiday"] = 1
    df_holiday = df_holiday[["date", "is_holiday"]]

    # attach to df_mlf (hourly load)
    df = df_mlf.copy()
    df["ds"] = ensure_utc(df["ds"])
    df["date"] = df["ds"].dt.normalize()
    df = df.merge(df_holiday, on="date", how="left")

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df = df.drop(columns="date")

    return df


def attach_weather(df_mlf: pd.DataFrame) -> pd.DataFrame:

    path_dataset = str(DATA_ROOT)

    weather_path = os.path.join(path_dataset, "weather_hourly_darksky.csv")
    if not os.path.isfile(weather_path):
        raise FileNotFoundError(f"Expected {weather_path} (from Kaggle dataset).")

    # read full weather table once
    weather_hourly_darksky = pd.read_csv(
        weather_path,
        index_col="time",
        parse_dates=True,
    ).sort_index()

    # select only numeric weather features automatically
    WEATHER_FEATURES = weather_hourly_darksky.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    weather_cols = [c for c in WEATHER_FEATURES if c in weather_hourly_darksky.columns]

    weather_feats = (
        weather_hourly_darksky[weather_cols]
            .reset_index()
            .rename(columns={"time": "ds"})
    )

    # ensure both sides are UTC tz-aware
    weather_feats["ds"] = ensure_utc(weather_feats["ds"])

    df = df_mlf.copy()
    df["ds"] = ensure_utc(df["ds"])

    df = df.merge(weather_feats, on="ds", how="left")


    for col in weather_cols:
        df[col] = df[col].ffill().bfill()

    # same check as in notebook printing "False"
    if df.isnull().sum().any():
        raise RuntimeError("There are still NaNs left in df_mlf after merging weather.")

    return df


# ---------------------------------------------------------------------------
# 6) DB helpers 
# ---------------------------------------------------------------------------

def get_db_params() -> Tuple[str, int, str, str, str, str]:
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    name = os.getenv("DB_NAME", "energy")
    user = os.getenv("DB_USER", "energy_user")
    password = os.getenv("DB_PASSWORD", "change_me")
    table = os.getenv("DB_MLF_TABLE", "df_mlf")
    return host, port, name, user, password, table


def ensure_database_exists(host: str, port: int, user: str, password: str, db_name: str) -> None:
    """
     if the database doesn't exist, create it.
    """
    import psycopg2
    from psycopg2 import OperationalError

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=user,
            password=password,
        )
        conn.close()
        return
    except OperationalError as e:
        if f'database "{db_name}" does not exist' not in str(e):
            raise

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname="postgres",
        user=user,
        password=password,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {db_name};")
                print(f"Database '{db_name}' created.")
            else:
                print(f"Database '{db_name}' already exists (checked via postgres).")
    finally:
        conn.close()


def write_df_mlf_to_postgres(df: pd.DataFrame) -> None:
    """
    Create/replace table and insert df_mlf.
    """
    host, port, name, user, password, table = get_db_params()

    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError as e:
        raise RuntimeError(
            "The 'psycopg2' package is not installed. "
            "Install it with `pip install psycopg2-binary`."
        ) from e

    ensure_database_exists(host, port, user, password, name)

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=name,
        user=user,
        password=password,
    )

    try:
        with conn.cursor() as cur:
            cols = list(df.columns)

            col_defs = [
                'id SERIAL PRIMARY KEY',
                '"ds" TIMESTAMPTZ NOT NULL',
                '"unique_id" TEXT NOT NULL',
                '"y" DOUBLE PRECISION NOT NULL',
            ]

            for col in cols:
                if col in ("ds", "unique_id", "y"):
                    continue
                series = df[col]
                if pd.api.types.is_integer_dtype(series):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(series):
                    sql_type = "DOUBLE PRECISION"
                else:
                    sql_type = "TEXT"
                col_defs.append(f'"{col}" {sql_type}')

            create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)});"
            cur.execute(create_sql)

            # replace data
            cur.execute(f"TRUNCATE TABLE {table};")

            placeholders = ", ".join(["%s"] * len(cols))
            col_list_sql = ", ".join(f'"{c}"' for c in cols)
            sql = f"INSERT INTO {table} ({col_list_sql}) VALUES ({placeholders});"

            rows = list(df.itertuples(index=False, name=None))
            execute_batch(cur, sql, rows, page_size=1000)

        conn.commit()
        print(f"Inserted {len(df)} rows into table '{table}' on DB '{name}'.")
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# 7) Build a global exogenous dataframe (one row per ds)
# ---------------------------------------------------------------------------

def build_exogenous_df(
    best_start: pd.Timestamp,
    best_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a global exogenous dataframe (one row per ds) using
    the SAME attach_holidays + attach_weather functions as df_mlf.

    Time window: [best_start, best_end + 7 days]
    Columns (for this project):
      ds, is_holiday, <weather_features...>
    """
    horizon_end = best_end + pd.Timedelta(days=7)
    ds_index = pd.date_range(start=best_start, end=horizon_end, freq="h", tz="UTC")

    base = pd.DataFrame({"ds": ds_index})

    base = attach_holidays(base)   
    base = attach_weather(base)    

    exog_cols = [c for c in base.columns if c != "ds"]
    exog = base[["ds"] + exog_cols].copy()

    if exog.isnull().sum().any():
        raise RuntimeError(
            "There are NaNs left in the exogenous dataframe after holidays+weather."
        )

    exog = exog.sort_values("ds").reset_index(drop=True)
    return exog

# ---------------------------------------------------------------------------
# 8) Create/replace exogenous table (e.g. df_exog) and insert exogenous data.
# ---------------------------------------------------------------------------


def write_exogenous_to_postgres(exog: pd.DataFrame) -> None:
    """
    Create/replace exogenous table (e.g. df_exog) and insert exogenous data.

    Table schema:
      id SERIAL PRIMARY KEY
      ds TIMESTAMPTZ NOT NULL
      <one column per exogenous feature: DOUBLE PRECISION or INTEGER or TEXT>
    """
    host, port, name, user, password, _ = get_db_params()
    table = os.getenv("DB_EXOG_TABLE", "df_exog")

    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError as e:
        raise RuntimeError(
            "The 'psycopg2' package is not installed. "
            "Install it with `pip install psycopg2-binary`."
        ) from e

    ensure_database_exists(host, port, user, password, name)

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=name,
        user=user,
        password=password,
    )

    try:
        with conn.cursor() as cur:
            cols = list(exog.columns)

            col_defs = [
                'id SERIAL PRIMARY KEY',
                '"ds" TIMESTAMPTZ NOT NULL',
            ]

            for col in cols:
                if col == "ds":
                    continue
                series = exog[col]
                if pd.api.types.is_integer_dtype(series):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(series):
                    sql_type = "DOUBLE PRECISION"
                else:
                    sql_type = "TEXT"
                col_defs.append(f'"{col}" {sql_type}')

            create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)});"
            cur.execute(create_sql)

            # replace data
            cur.execute(f"TRUNCATE TABLE {table};")

            placeholders = ", ".join(["%s"] * len(cols))
            col_list_sql = ", ".join(f'"{c}"' for c in cols)
            insert_sql = f"INSERT INTO {table} ({col_list_sql}) VALUES ({placeholders});"

            df_insert = exog.copy()
            df_insert["ds"] = ensure_utc(df_insert["ds"])

            rows = list(df_insert.itertuples(index=False, name=None))
            execute_batch(cur, insert_sql, rows, page_size=1000)

        conn.commit()
        print(f"Inserted {len(exog)} rows into exogenous table '{table}' on DB '{name}'.")
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# 9) Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]

    n_files_env = os.getenv("MLF_N_FILES", str(MLF_N_FILES))
    if n_files_env.lower() in ("none", "", "all"):
        n_files: int | None = None
    else:
        n_files = int(n_files_env)

    print("1) Downloading Kaggle dataset (if needed)...")
    download_kaggle_dataset()
    print(f"Kaggle dataset ready under {DATA_ROOT}")

    print("2) Loading half-hourly smart meter data...")
    halfhourly_clean = load_halfhourly_smart_meters(n_files=n_files)
    print(f"Loaded {len(halfhourly_clean)} cleaned half-hourly rows.")

    print("3) Aggregating to hourly grouped load (df_mlf base) with historical days logic...")
    df_mlf, best_start, best_end = build_hourly_grouped_load(halfhourly_clean)
    print(df_mlf.head())
    print("Selected historical days window:", best_start, "to", best_end)

    print("4) Attaching bank holidays to df_mlf...")
    df_mlf = attach_holidays(df_mlf)

    print("5) Attaching weather features to df_mlf...")
    df_mlf = attach_weather(df_mlf)

    print("Final df_mlf sample:")
    print(df_mlf.head())
    print("Any NaNs left after merging weather?", df_mlf.isnull().sum().any())

    print("6) Writing df_mlf (load + exog) to Postgres...")
    write_df_mlf_to_postgres(df_mlf)

    print("7) Building exogenous dataframe (global) for best_start .. best_end+7d...")
    exog_df = build_exogenous_df(best_start=best_start, best_end=best_end)
    print(exog_df.head())
    print("Exogenous df shape:", exog_df.shape)

    print("8) Writing exogenous dataframe to Postgres exog table...")
    write_exogenous_to_postgres(exog_df)

    print("Done.")


if __name__ == "__main__":
    main()
