import os
import random
import warnings
warnings.filterwarnings('ignore')

import mlflow
SEED = 42
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED']        = str(SEED)
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, explained_variance_score)

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import (LSTM, GRU, SimpleRNN, Dense, Dropout,
                                          MultiHeadAttention, LayerNormalization,
                                          GlobalAveragePooling1D, Input)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available — using scikit-learn fallback.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed — skipping. Run: pip install shap")


# =============================================================================
# UPGRADE 1 — REAL-TIME DATA via NASA POWER API
# =============================================================================

FEATURE_COLS = ['T2M', 'RH2M', 'PRECTOTCORR', 'WS2M', 'PS',
                'T2M_MAX', 'T2M_MIN', 'WD10M', 'ALLSKY_SFC_LW_DWN']


def fetch_nasa_power(lat=28.6, lon=77.2, start='20100101', end='20231231'):
    params = ','.join(FEATURE_COLS)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={params}"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    print(f"\nFetching NASA POWER real-time data ...")
    print(f"  Location : lat={lat}, lon={lon}")
    print(f"  Period   : {start} -> {end}")
    print(f"  URL      : {url[:80]}...")
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        raw = resp.json()['properties']['parameter']
        df  = pd.DataFrame(raw)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df  = df.replace(-999.0, np.nan)[FEATURE_COLS]
        df  = df.apply(pd.to_numeric, errors='coerce')
        df  = df.ffill().bfill().fillna(df.mean(numeric_only=True))
        df  = df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        print(f"  Live data fetched: {len(df)} days  "
              f"({df.index.min().date()} -> {df.index.max().date()})")
        # Save data snapshot for DVC versioning
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/climate_data.csv')
        print("  Data snapshot saved -> data/climate_data.csv")
        return df
    except Exception as e:
        print(f"  NASA POWER API failed: {e}")
        print("  Falling back to CSV ...")
        return None


# =============================================================================
# STEP 1 — DATA LOADING & PREPROCESSING
# =============================================================================

def load_csv(filepath):
    print("=" * 65)
    print("STEP 1: DATA PREPROCESSING (CSV fallback)")
    print("=" * 65)
    columns = ['YEAR', 'MO', 'DY'] + FEATURE_COLS
    try:
        df = pd.read_csv(filepath)
        if not set(columns).issubset(df.columns):
            raise ValueError("Missing columns")
        df = df[columns].copy()
    except Exception:
        df = pd.read_csv(filepath, sep=r"\s+", comment="#",
                         names=columns, header=None, engine="python")
        df = df[pd.to_numeric(df['YEAR'], errors='coerce').notna()].copy()

    df = df.apply(pd.to_numeric, errors='coerce').replace(-999.0, np.nan)
    print(f"Raw records: {len(df)}  |  Missing before: {df.isnull().sum().sum()}")
    df = df.ffill().bfill().fillna(df.mean(numeric_only=True))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['DATE'] = pd.to_datetime(
        df[['YEAR', 'MO', 'DY']].rename(
            columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
    df = df.set_index('DATE').drop(columns=['YEAR', 'MO', 'DY']).sort_index()
    print(f"Missing after : {df.isnull().sum().sum()}")
    print(f"Date range    : {df.index.min().date()} -> {df.index.max().date()}  "
          f"({len(df)} days)")
    return df


def prepare_dataframe(df):
    df = df[FEATURE_COLS].copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace(-999.0, np.nan).replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(df.mean(numeric_only=True)).dropna()
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
    return df


def normalize_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    df_s   = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    print("\nNormalization  : DONE")
    print("Feature columns:", list(df.columns))
    return df_s, scaler


def create_sequences(data, target_cols, window_size=30):
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    for c in target_cols:
        if c not in data.columns:
            raise ValueError(f"Target '{c}' not found.")
    if len(data) <= window_size:
        raise ValueError(f"Not enough rows ({len(data)}) for window={window_size}.")

    target_idxs = [list(data.columns).index(c) for c in target_cols]
    values = data.values
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size, target_idxs])
    X, y = np.array(X), np.array(y)
    if len(target_cols) == 1:
        y = y.squeeze(-1)
    print(f"Windowing: window={window_size}, samples={len(X)}, targets={target_cols}")
    return X, y


# =============================================================================
# STEP 2 — MODELS
# =============================================================================

def build_lstm_model(input_shape, n_outputs=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2, seed=SEED),
        LSTM(32),
        Dropout(0.2, seed=SEED),
        Dense(16, activation='relu'),
        Dense(n_outputs)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def build_gru_model(input_shape, n_outputs=1):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2, seed=SEED),
        GRU(32),
        Dropout(0.2, seed=SEED),
        Dense(16, activation='relu'),
        Dense(n_outputs)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def build_rnn_model(input_shape, n_outputs=1):
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2, seed=SEED),
        SimpleRNN(32),
        Dropout(0.2, seed=SEED),
        Dense(16, activation='relu'),
        Dense(n_outputs)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def build_transformer_model(input_shape, n_outputs=1,
                             num_heads=4, ff_dim=64, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    x  = MultiHeadAttention(num_heads=num_heads,
                            key_dim=input_shape[-1])(inputs, inputs)
    x  = Dropout(dropout_rate)(x)
    x  = LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(input_shape[-1])(ff)
    ff = Dropout(dropout_rate)(ff)
    x  = LayerNormalization(epsilon=1e-6)(x + ff)
    x  = GlobalAveragePooling1D()(x)
    x  = Dense(32, activation='relu')(x)
    x  = Dropout(dropout_rate)(x)
    outputs = Dense(n_outputs)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


class SimpleARModel:
    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        n, w, f = X.shape
        self.reg = Ridge()
        self.reg.fit(X.reshape(n, w * f), y)
        return self

    def predict(self, X, **kwargs):
        n, w, f = X.shape
        return self.reg.predict(X.reshape(n, w * f))


# =============================================================================
# UPGRADE 6 — HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(X_train, y_train, model_type, input_shape, n_outputs):
    print("\n  [HyperTuning] Searching best lr & batch_size ...")
    lr_options    = [1e-3, 5e-4]
    batch_options = [32, 64]
    best_loss, best_params = np.inf, (1e-3, 32)

    val_split = int(len(X_train) * 0.9)
    Xt, yt = X_train[:val_split], y_train[:val_split]
    Xv, yv = X_train[val_split:], y_train[val_split:]

    builders = {'lstm': build_lstm_model, 'gru': build_gru_model,
                'rnn': build_rnn_model, 'transformer': build_transformer_model}
    builder  = builders.get(model_type, build_lstm_model)

    for lr in lr_options:
        for bs in batch_options:
            tf.random.set_seed(SEED)
            m = builder(input_shape, n_outputs)
            m.optimizer.learning_rate.assign(lr)
            m.fit(Xt, yt, epochs=5, batch_size=bs, shuffle=False, verbose=0)
            loss = m.evaluate(Xv, yv, verbose=0)[0]
            print(f"    lr={lr}, batch={bs}  -> val_loss={loss:.5f}")
            if loss < best_loss:
                best_loss, best_params = loss, (lr, bs)

    print(f"  Best: lr={best_params[0]}, batch={best_params[1]}")
    return best_params


# =============================================================================
# STEP 3 — DRIFT DETECTORS
# =============================================================================

class ADWINDetector:
    def __init__(self, delta=0.35, window_size=60):
        self.delta = delta; self.window_size = window_size
        self.errors = []; self.drift_points = []

    def update(self, error, idx):
        self.errors.append(abs(error))
        if len(self.errors) < self.window_size:
            return False
        window = np.array(self.errors[-self.window_size:])
        half   = self.window_size // 2
        w1, w2 = window[:half], window[half:]
        mean_diff = abs(w1.mean() - w2.mean())
        threshold = np.sqrt((1/(2*half)) * np.log(4*len(self.errors)/self.delta))
        if mean_diff > threshold:
            self.drift_points.append(idx)
            self.errors = list(window[half:])
            return True
        return False


class PageHinkleyDetector:
    def __init__(self, delta=0.002, lambda_=0.6):
        self.delta = delta; self.lambda_ = lambda_
        self.m_t = self.T_t = self.M_t = 0.0; self.n = 0
        self.drift_points = []

    def update(self, error, idx):
        self.n += 1
        self.m_t += (abs(error) - self.m_t) / self.n
        self.T_t += abs(error) - self.m_t - self.delta
        self.M_t  = min(self.M_t, self.T_t)
        if (self.T_t - self.M_t) > self.lambda_:
            self.drift_points.append(idx)
            self.m_t = self.T_t = self.M_t = 0.0; self.n = 0
            return True
        return False


class DDMDetector:
    def __init__(self, error_threshold=0.07, drift_scale=3.0, min_instances=40):
        self.error_threshold = error_threshold
        self.drift_scale = drift_scale; self.min_instances = min_instances
        self.n = 0; self.p = self.s = 0.0
        self.p_min = self.s_min = np.inf
        self.drift_points = []

    def update(self, error, idx):
        is_err = 1.0 if abs(error) > self.error_threshold else 0.0
        self.n += 1
        self.p += (is_err - self.p) / self.n
        self.s  = np.sqrt((self.p * (1-self.p)) / max(self.n, 1))
        if self.n < self.min_instances:
            return False
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p; self.s_min = self.s
        if self.p + self.s > self.p_min + self.drift_scale * self.s_min:
            self.drift_points.append(idx)
            self.n = 0; self.p = self.s = 0.0
            self.p_min = self.s_min = np.inf
            return True
        return False


# =============================================================================
# STEP 4 — ADAPTIVE RETRAINING
# =============================================================================

def adaptive_retrain(model, X_new, y_new, model_type):
    if TF_AVAILABLE and model_type in ('lstm', 'gru', 'rnn', 'transformer'):
        model.fit(X_new, y_new, epochs=5, batch_size=32,
                  shuffle=False, verbose=0,
                  callbacks=[EarlyStopping(patience=2,
                                           restore_best_weights=True)])
    else:
        model.fit(X_new, y_new)
    print("    -> Model retrained on recent window")
    return model


# =============================================================================
# UPGRADE 7 — EXTENDED METRICS
# =============================================================================

def smape(actual, predicted):
    return 100 * np.mean(
        2 * np.abs(predicted - actual) /
        (np.abs(actual) + np.abs(predicted) + 1e-8)
    )


def compute_metrics(trues, preds, label=""):
    mse  = mean_squared_error(trues, preds)
    mae  = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(trues, preds)
    ev   = explained_variance_score(trues, preds)
    mape = np.mean(np.abs((trues - preds) /
                           np.maximum(np.abs(trues), 1e-8))) * 100
    smap = smape(trues, preds)
    acc  = max(0.0, 100.0 - mape)

    tag = f" [{label}]" if label else ""
    print(f"\nFINAL METRICS (original scale){tag}")
    print("-" * 45)
    print(f"  MAE               : {mae:.4f}")
    print(f"  RMSE              : {rmse:.4f}")
    print(f"  MSE               : {mse:.4f}")
    print(f"  R2                : {r2:.4f}")
    print(f"  Explained Variance: {ev:.4f}")
    print(f"  MAPE              : {mape:.2f}%")
    print(f"  SMAPE             : {smap:.2f}%")
    print(f"  Accuracy(100-MAPE): {acc:.2f}%")

    return dict(mae=mae, rmse=rmse, mse=mse, r2=r2,
                ev=ev, mape=mape, smape=smap, accuracy=acc)


# =============================================================================
# UPGRADE 3 — SHAP FEATURE IMPORTANCE
# =============================================================================

def explain_with_shap(model, X_train, feature_names, model_type, n_samples=100):
    if not SHAP_AVAILABLE:
        print("\nSHAP not available — skipping. Run: pip install shap")
        return
    if not TF_AVAILABLE or model_type not in ('lstm', 'gru', 'rnn', 'transformer'):
        print("\nSHAP: skipped (requires TF model).")
        return

    print("\nComputing SHAP values using last-timestep features ...")
    print(f"  X_train shape  : {X_train.shape}  "
          f"(samples, timesteps, features)")

    X_last = X_train[:, -1, :]
    n_bg   = min(50, len(X_last))
    bg     = X_last[:n_bg]
    n_exp  = min(n_samples, len(X_last))
    X_exp  = X_last[:n_exp]

    print(f"  Background     : {bg.shape}  (2D — last timestep)")
    print(f"  Explain set    : {X_exp.shape}  (2D — last timestep)")

    window_size = X_train.shape[1]
    def predict_from_last_timestep(X_2d):
        n = X_2d.shape[0]
        X_3d = np.tile(X_2d[:, np.newaxis, :], (1, window_size, 1))
        preds = model.predict(X_3d, verbose=0)
        preds = np.atleast_2d(preds)
        return preds[:, 0]

    explainer   = shap.KernelExplainer(predict_from_last_timestep, bg)
    shap_values = explainer.shap_values(X_exp, nsamples=50, silent=True)

    sv         = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(sv, index=feature_names).sort_values(ascending=False)

    print("\nFeature Importance (SHAP — last timestep):")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    importance.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('SHAP Feature Importance (Last Timestep)')
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_xlabel('Feature')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/shap_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved -> outputs/shap_importance.png")


# =============================================================================
# STEP 5 — MAIN PIPELINE
# =============================================================================

def run_pipeline(df_raw, target_cols='T2M', window_size=30,
                 model_type='lstm', test_ratio=0.2,
                 tune=False, location_label=''):
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    n_outputs = len(target_cols)

    df_clean  = prepare_dataframe(df_raw)
    df_scaled, scaler = normalize_data(df_clean)

    X, y = create_sequences(df_scaled, target_cols, window_size)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("\n" + "=" * 65)
    print(f"STEP 2: MODEL — {model_type.upper()}  |  outputs={n_outputs}")
    print("=" * 65)

    # ── MLflow run starts here ──────────────────────────────────────
    mlflow.set_experiment("climate-drift-pipeline")
    with mlflow.start_run(run_name=f"{model_type}_{location_label}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("location", location_label)
        mlflow.log_param("target_cols", str(target_cols))

        if TF_AVAILABLE:
            tf.random.set_seed(SEED)
            input_shape = (X_train.shape[1], X_train.shape[2])

            best_lr, best_bs = (
                tune_hyperparameters(X_train, y_train, model_type, input_shape, n_outputs)
                if tune else (1e-3, 32)
            )

            builders = {'lstm': build_lstm_model, 'gru': build_gru_model,
                        'rnn': build_rnn_model, 'transformer': build_transformer_model}
            if model_type not in builders:
                raise ValueError(f"model_type must be one of {list(builders)}")

            tf.random.set_seed(SEED)
            model = builders[model_type](input_shape, n_outputs)
            model.optimizer.learning_rate.assign(best_lr)
            print(f"Model: {model_type.upper()}  lr={best_lr}  batch={best_bs}")

            mlflow.log_param("learning_rate", best_lr)
            mlflow.log_param("batch_size", best_bs)

            os.makedirs('outputs', exist_ok=True)
            ckpt_path = f'outputs/best_{model_type}_{location_label}.h5'
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0)
            ]
            history = model.fit(X_train, y_train,
                                validation_split=0.1,
                                epochs=30, batch_size=best_bs,
                                shuffle=False, callbacks=callbacks, verbose=1)
            print(f"Checkpoint saved -> {ckpt_path}")
        else:
            print("Model: Simple AR (TF not available)")
            model = SimpleARModel()
            model.fit(X_train, y_train)
            history = None

        print("\n" + "=" * 65)
        print("STEP 3: DRIFT DETECTION & ADAPTIVE RETRAINING")
        print("=" * 65)

        adwin = ADWINDetector()
        ph    = PageHinkleyDetector()
        ddm   = DDMDetector()

        MIN_VOTES  = 1
        MIN_GAP    = 60
        last_drift = -(10**9)
        RW         = 400

        predictions   = []
        true_values   = []
        errors        = []
        drift_events  = []
        retrain_count = 0
        detector_hits = {'ADWIN': 0, 'Page-Hinkley': 0, 'DDM': 0}

        print(f"Drift policy : min_votes={MIN_VOTES}, min_gap={MIN_GAP}")
        print("Running online evaluation ...")

        for i in range(len(X_test)):
            if i % 200 == 0:
                print(f"  {i}/{len(X_test)}  |  Retrains so far: {retrain_count}")

            x_i  = X_test[i:i+1]
            raw  = model.predict(x_i, verbose=0) if TF_AVAILABLE else model.predict(x_i)
            pred = np.asarray(raw).reshape(-1)
            true = np.asarray(y_test[i]).reshape(-1)

            err = float(true[0] - pred[0])
            predictions.append(pred)
            true_values.append(true)
            errors.append(err)

            fired = []
            if adwin.update(err, i): detector_hits['ADWIN'] += 1;        fired.append('ADWIN')
            if ph.update(err, i):    detector_hits['Page-Hinkley'] += 1; fired.append('Page-Hinkley')
            if ddm.update(err, i):   detector_hits['DDM'] += 1;          fired.append('DDM')

            if len(fired) >= MIN_VOTES and (i - last_drift) >= MIN_GAP:
                print(f"  [Drift @ step {i:4d}]  confirmed by: {', '.join(fired)}")
                drift_events.append(i)
                last_drift = i

                rX = np.concatenate([X_train[-RW:], X_test[max(0, i-RW):i]])
                ry = np.concatenate([y_train[-RW:], y_test[max(0, i-RW):i]])
                model = adaptive_retrain(model, rX, ry, model_type)
                retrain_count += 1
                print(f"  Total retrains so far: {retrain_count}")

        print(f"\nTotal drift events : {len(drift_events)}")
        print(f"Total retrains     : {retrain_count}")
        print(f"Detector hits      : ADWIN={detector_hits['ADWIN']}, "
              f"PH={detector_hits['Page-Hinkley']}, DDM={detector_hits['DDM']}")

        predictions = np.array(predictions)
        true_values = np.array(true_values)
        n_feat = df_scaled.shape[1]

        def inv(vals, col_name):
            idx   = list(df_scaled.columns).index(col_name)
            dummy = np.zeros((len(vals), n_feat))
            dummy[:, idx] = vals
            return scaler.inverse_transform(dummy)[:, idx]

        all_metrics = {}
        for k, tc in enumerate(target_cols):
            p_orig = inv(predictions[:, k], tc)
            t_orig = inv(true_values[:,  k], tc)
            m = compute_metrics(t_orig, p_orig,
                                label=f"{tc} | {location_label}")
            all_metrics[tc] = m

            # ── Log metrics to MLflow ──────────────────────────────
            mlflow.log_metric(f"{tc}_mae",      m['mae'])
            mlflow.log_metric(f"{tc}_rmse",     m['rmse'])
            mlflow.log_metric(f"{tc}_r2",       m['r2'])
            mlflow.log_metric(f"{tc}_accuracy", m['accuracy'])

            if k == 0 and TF_AVAILABLE:
                explain_with_shap(model, X_train,
                                  list(df_scaled.columns), model_type)

        # Log drift/retrain counts
        mlflow.log_metric("drift_events",  len(drift_events))
        mlflow.log_metric("retrain_count", retrain_count)

        # Save and log final model
        final_path = f'outputs/final_{model_type}_{location_label}.h5'
        if TF_AVAILABLE:
            model.save(final_path)
            print(f"Final model saved -> {final_path}")
            mlflow.log_artifact(final_path)(model, f"{model_type}_{location_label}_model")

        # Log plot artifact
        p0 = inv(predictions[:, 0], target_cols[0])
        t0 = inv(true_values[:,  0], target_cols[0])
        plot_results(t0, p0, errors, drift_events,
                     adwin, history, target_cols[0], model_type, location_label)

        plot_file = f'outputs/results_{model_type}_{location_label}.png'
        if os.path.exists(plot_file):
            mlflow.log_artifact(plot_file)

    # ── MLflow run ends here ────────────────────────────────────────

    return model, scaler, all_metrics, drift_events, retrain_count


# =============================================================================
# STEP 6 — VISUALISATION
# =============================================================================

def plot_results(trues, preds, errors, drift_events,
                 adwin, history, target_col, model_type, loc_label=''):

    fig = plt.figure(figsize=(18, 16))
    tag = f"{model_type.upper()} | {target_col}"
    if loc_label:
        tag += f" | {loc_label}"
    fig.suptitle(f'Adaptive Deep Learning Pipeline — {tag}',
                 fontsize=13, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(trues, label='Actual',    color='steelblue', lw=1.2, alpha=0.85)
    ax1.plot(preds, label='Predicted', color='orangered', lw=1.0,
             alpha=0.85, linestyle='--')
    for d in drift_events:
        ax1.axvline(d, color='red', alpha=0.3, lw=0.8)
    ax1.set_title('Actual vs Predicted'); ax1.set_xlabel('Test Step')
    ax1.set_ylabel(target_col); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(errors, color='purple', lw=0.8, alpha=0.7, label='Error')
    ax2.axhline(0, color='black', lw=0.8)
    for k, d in enumerate(drift_events):
        ax2.axvline(d, color='red', alpha=0.4, lw=0.8,
                    label='Drift' if k == 0 else '')
    ax2.set_title('Prediction Errors'); ax2.set_xlabel('Test Step')
    ax2.set_ylabel('Error'); ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 3)
    win  = 50
    rmae = [np.mean(np.abs(errors[max(0, i-win):i+1])) for i in range(len(errors))]
    ax3.plot(rmae, color='darkorange', lw=1.2)
    for d in drift_events:
        ax3.axvline(d, color='red', alpha=0.4, lw=0.8)
    ax3.set_title(f'Rolling MAE (w={win})'); ax3.set_xlabel('Test Step')
    ax3.set_ylabel('MAE'); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(adwin.errors, color='teal', lw=0.9, alpha=0.8)
    ax4.set_title('ADWIN Error Window')
    ax4.set_xlabel('Window steps'); ax4.set_ylabel('|Error|'); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 5)
    if drift_events:
        ax5.stem(drift_events, [1]*len(drift_events),
                 linefmt='r-', markerfmt='ro', basefmt='k-')
    ax5.set_title('Drift Detection Timeline')
    ax5.set_xlabel('Test Step'); ax5.set_yticks([]); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(3, 2, 6)
    if history and hasattr(history, 'history'):
        ax6.plot(history.history['loss'],     label='Train', color='blue')
        ax6.plot(history.history['val_loss'], label='Val',   color='orange')
        ax6.set_title('Training Loss'); ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MSE'); ax6.legend(); ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No training history',
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Training Loss')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('outputs', exist_ok=True)
    fname = (f'outputs/results_{model_type}_{loc_label}.png' if loc_label
             else f'outputs/results_{model_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Figure saved -> {fname}")
    plt.close(fig)


# =============================================================================
# UPGRADE 10 — LOCATION-BASED COMPARISON
# =============================================================================

LOCATIONS = {
    'New_Delhi_India': {'lat':  28.6, 'lon':  77.2},
    'New_York_USA'   : {'lat':  40.7, 'lon': -74.0},
    'Arctic_Norway'  : {'lat':  70.0, 'lon':  25.0},
}


def run_location_comparison(model_type='lstm', target_col='T2M'):
    print("\n" + "=" * 65)
    print("UPGRADE 10: LOCATION-BASED ANALYSIS (Real-time NASA POWER)")
    print("=" * 65)

    all_results = []
    for loc_name, coords in LOCATIONS.items():
        print(f"\n--- Location: {loc_name} ---")
        df_raw = fetch_nasa_power(lat=coords['lat'], lon=coords['lon'])
        if df_raw is None:
            print(f"  Skipping {loc_name} — no data available.")
            continue

        _, _, metrics, drift_events, retrain_count = run_pipeline(
            df_raw, target_cols=target_col,
            model_type=model_type, location_label=loc_name
        )
        m = metrics[target_col]
        all_results.append({
            'Location'  : loc_name,
            'Model'     : model_type.upper(),
            'MAE'       : m['mae'],
            'RMSE'      : m['rmse'],
            'R2'        : m['r2'],
            'SMAPE_%'   : m['smape'],
            'Accuracy_%': m['accuracy'],
            'Drifts'    : len(drift_events),
            'Retrains'  : retrain_count,
        })

    if all_results:
        df_loc = pd.DataFrame(all_results)
        print("\n" + "=" * 65)
        print("LOCATION COMPARISON")
        print("=" * 65)
        print(df_loc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print("Adaptive Deep Learning Pipeline for Climate Data")
    print("=" * 65)
    print(f"TensorFlow : {TF_AVAILABLE}"
          + (f"  (v{tf.__version__})" if TF_AVAILABLE else ""))
    print(f"SHAP       : {SHAP_AVAILABLE}")
    print(f"Seed       : {SEED}")

    df_raw = fetch_nasa_power(lat=28.6, lon=77.2,
                              start='20100101', end='20231231')
    if df_raw is None:
        raise RuntimeError("NASA POWER API failed — no data available.")

    models_to_run = (['lstm', 'gru', 'rnn', 'transformer']
                     if TF_AVAILABLE else ['simple_ar'])

    TARGET_COLS = ['T2M', 'RH2M', 'PS']

    all_results = []

    for model_name in models_to_run:
        print("\n" + "=" * 65)
        print(f"RUNNING MODEL: {model_name.upper()}")
        print("=" * 65)
        model, scaler, metrics, drift_events, retrain_count = run_pipeline(
            df_raw,
            target_cols    = TARGET_COLS,
            window_size    = 30,
            model_type     = model_name,
            test_ratio     = 0.2,
            tune           = False,
            location_label = 'India'
        )
        m = metrics[TARGET_COLS[0]]
        all_results.append({
            'Model'      : model_name.upper(),
            'MAE'        : m['mae'],
            'RMSE'       : m['rmse'],
            'MSE'        : m['mse'],
            'R2'         : m['r2'],
            'Expl_Var'   : m['ev'],
            'MAPE_%'     : m['mape'],
            'SMAPE_%'    : m['smape'],
            'Accuracy_%' : m['accuracy'],
            'Drifts'     : len(drift_events),
            'Retrains'   : retrain_count,
        })

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    results_df = pd.DataFrame(all_results).sort_values('RMSE')
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = results_df.iloc[0]
    print(f"\nBest model : {best['Model']}")
    print(f"  RMSE     : {best['RMSE']:.4f}")
    print(f"  Accuracy : {best['Accuracy_%']:.2f}%")
    print(f"  Retrains : {int(best['Retrains'])}")

    run_location_comparison(model_type='lstm', target_col='T2M')
    print("\nPipeline complete.")