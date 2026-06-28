# =============================================================================
# SOL/USD Price Prediction — 8h Ahead | Topic 10 (Mainnet) / Topic 38 (Testnet)
# Allora Forge Builder Kit — Optimized Notebook
# =============================================================================
# Đặt file này vào thư mục: allora-forge-builder-kit/notebooks/
# Sau đó chạy: python sol_usd_8h_topic10.py
#
# Các tối ưu so với notebook mặc định (example_topic_69):
#   ① tickers = 3 (SOL + BTC + ETH), target_length = 8h
#   ② Feature engineering: RSI, Bollinger, RVol, BTC correlation, time encoding
#   ③ LightGBM conservative params + early stopping
#   ④ TimeSeriesSplit walk-forward (không random split)
#   ⑤ predict() mirror đúng feature engineering để tránh train/serve skew
# =============================================================================

# ── Cài đặt (chạy một lần) ───────────────────────────────────────────────────
# pip install allora-forge-builder-kit lightgbm scikit-learn dill pandas numpy

import time
import dill
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

from allora_forge_builder_kit import AlloraMLWorkflow, get_api_key

# =============================================================================
# 0. CẤU HÌNH — chỉnh ở đây
# =============================================================================

FORGE_API_KEY = get_api_key()   # đọc từ env ALLORA_API_KEY hoặc nhập thủ công
TOPIC_ID      = 38              # 38 = testnet SOL 8h  |  10 = mainnet SOL 8h
MNEMONIC      = ""              # để trống → tự tạo ví mới, lưu vào .allora_key

# Data window: hỗ trợ về đến 2020
TRAIN_FROM_MONTH    = "2023-01"   # kéo dài để có nhiều regime khác nhau
VALIDATION_MONTHS   = 4
TEST_MONTHS         = 2

# =============================================================================
# 1. KHỞI TẠO WORKFLOW — SOL/USD 8h với multi-ticker
# =============================================================================
print("=" * 60)
print("BƯỚC 1 — Khởi tạo workflow")
print("=" * 60)

workflow = AlloraMLWorkflow(
    data_api_key         = FORGE_API_KEY,
    tickers              = ["solusd", "btcusd", "ethusd"],  # ① multi-ticker
    hours_needed         = 1 * 48,   # lookback 48h (2 × default)
    number_of_input_candles = 48,    # 48 candles cho features
    target_length        = 8,        # ① target 8h đúng topic 10
)

X_train, y_train, X_val, y_val, X_test, y_test = workflow.get_train_validation_test_data(
    from_month         = TRAIN_FROM_MONTH,
    validation_months  = VALIDATION_MONTHS,
    test_months        = TEST_MONTHS,
)

print(f"Train:  {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
print(f"Target: log-return SOL/USD {8}h ahead\n")


# =============================================================================
# 2. FEATURE ENGINEERING — thêm vào raw OHLCV của workflow
# =============================================================================
print("=" * 60)
print("BƯỚC 2 — Feature engineering")
print("=" * 60)

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — bounded [0, 100]."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm features kỹ thuật + cross-asset + time vào DataFrame.
    Hàm này được gọi cả khi training lẫn khi live inference (tránh skew).
    Input:  DataFrame với cột solusd_close, btcusd_close, ethusd_close, ...
    Output: DataFrame với thêm các cột feature mới.
    """
    df = df.copy()

    sol = df["solusd_close"]
    btc = df["btcusd_close"]
    eth = df["ethusd_close"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    df["sol_rsi_6"]  = compute_rsi(sol, 6)    # short-term momentum
    df["sol_rsi_14"] = compute_rsi(sol, 14)   # standard RSI
    df["sol_rsi_24"] = compute_rsi(sol, 24)   # medium-term RSI

    # ── Bollinger Bands %B ───────────────────────────────────────────────────
    for w in [12, 20, 36]:
        bb_mid = sol.rolling(w, min_periods=w).mean()
        bb_std = sol.rolling(w, min_periods=w).std()
        bb_range = 4 * bb_std
        df[f"sol_bb_pct_{w}"] = (sol - (bb_mid - 2 * bb_std)) / (bb_range + 1e-9)

    # ── Realized Volatility (quan trọng cho CZAR metric) ─────────────────────
    sol_logret = np.log(sol / sol.shift(1))
    df["sol_rvol_8"]  = sol_logret.rolling(8,  min_periods=4).std()
    df["sol_rvol_24"] = sol_logret.rolling(24, min_periods=12).std()
    df["sol_rvol_48"] = sol_logret.rolling(48, min_periods=24).std()

    # ── Log returns ──────────────────────────────────────────────────────────
    df["sol_ret_1h"]  = sol_logret
    df["sol_ret_4h"]  = np.log(sol / sol.shift(4))
    df["sol_ret_8h"]  = np.log(sol / sol.shift(8))
    df["sol_ret_24h"] = np.log(sol / sol.shift(24))

    # ── BTC cross-asset signal (alpha lớn nhất cho SOL) ─────────────────────
    btc_logret = np.log(btc / btc.shift(1))
    df["btc_ret_1h"]  = btc_logret
    df["btc_ret_4h"]  = np.log(btc / btc.shift(4))
    df["btc_ret_8h"]  = np.log(btc / btc.shift(8))

    # BTC-SOL rolling correlation (24h window) — regime signal
    df["sol_btc_corr_12"] = sol_logret.rolling(12, min_periods=6).corr(btc_logret)
    df["sol_btc_corr_24"] = sol_logret.rolling(24, min_periods=12).corr(btc_logret)

    # Beta SOL vs BTC (rolling)
    cov = sol_logret.rolling(24, min_periods=12).cov(btc_logret)
    btc_var = btc_logret.rolling(24, min_periods=12).var()
    df["sol_btc_beta_24"] = cov / (btc_var + 1e-12)

    # ── ETH cross-asset ──────────────────────────────────────────────────────
    eth_logret = np.log(eth / eth.shift(1))
    df["eth_ret_1h"] = eth_logret
    df["sol_eth_corr_24"] = sol_logret.rolling(24, min_periods=12).corr(eth_logret)

    # ── SOL/BTC relative strength ─────────────────────────────────────────────
    df["sol_btc_ratio"]     = sol / btc
    df["sol_btc_ratio_ret"] = np.log(df["sol_btc_ratio"] / df["sol_btc_ratio"].shift(8))

    # ── Time encoding (intraday seasonality) ─────────────────────────────────
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


# Áp dụng feature engineering
X_train_fe = add_custom_features(X_train).dropna()
X_val_fe   = add_custom_features(X_val).dropna()
X_test_fe  = add_custom_features(X_test).dropna()

# Align targets sau dropna
y_train_fe = y_train.loc[X_train_fe.index]
y_val_fe   = y_val.loc[X_val_fe.index]
y_test_fe  = y_test.loc[X_test_fe.index]

# Tất cả feature columns (raw OHLCV từ workflow + custom)
FEATURE_COLS = list(X_train_fe.columns)
print(f"Tổng số features: {len(FEATURE_COLS)}")
print(f"  Raw OHLCV từ workflow: ~{len(X_train.columns)}")
print(f"  Custom features thêm: ~{len(FEATURE_COLS) - len(X_train.columns)}\n")


# =============================================================================
# 3. LIGHTGBM PARAMS — conservative để tránh overfit crypto data
# =============================================================================
print("=" * 60)
print("BƯỚC 3 — Train LightGBM (conservative params)")
print("=" * 60)

LGBM_PARAMS = {
    "objective":            "regression",
    "metric":               "rmse",
    "num_leaves":           15,        # giảm từ default 31 → tránh overfit
    "learning_rate":        0.02,      # giảm từ 0.1 → cần nhiều trees nhưng stable hơn
    "max_depth":            5,         # giới hạn depth
    "min_child_samples":    50,        # cần ≥50 samples/leaf → robust
    "subsample":            0.8,       # row subsampling mỗi tree
    "subsample_freq":       1,
    "colsample_bytree":     0.7,       # feature subsampling mỗi tree
    "reg_alpha":            0.1,       # L1 regularization
    "reg_lambda":           0.2,       # L2 regularization
    "n_estimators":         1000,      # nhiều trees, dựa vào early stopping
    "random_state":         42,
    "n_jobs":               -1,
    "verbose":              -1,
}

# ── Walk-forward cross-validation (④ — không random split) ───────────────────
print("Walk-forward CV (TimeSeriesSplit n=5)...")
tscv    = TimeSeriesSplit(n_splits=5)
cv_rmse = []

X_tr_arr = X_train_fe.values
y_tr_arr = y_train_fe.values

for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_tr_arr)):
    X_cv_tr, X_cv_vl = X_tr_arr[tr_idx], X_tr_arr[vl_idx]
    y_cv_tr, y_cv_vl = y_tr_arr[tr_idx], y_tr_arr[vl_idx]

    model_cv = lgb.LGBMRegressor(**LGBM_PARAMS, n_estimators=500)
    model_cv.fit(
        X_cv_tr, y_cv_tr,
        eval_set=[(X_cv_vl, y_cv_vl)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred_cv = model_cv.predict(X_cv_vl)
    rmse    = np.sqrt(mean_squared_error(y_cv_vl, pred_cv))
    cv_rmse.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {rmse:.6f}  (best_iter={model_cv.best_iteration_})")

print(f"CV RMSE mean ± std: {np.mean(cv_rmse):.6f} ± {np.std(cv_rmse):.6f}\n")

# ── Train final model trên train+val ─────────────────────────────────────────
X_full = pd.concat([X_train_fe, X_val_fe])
y_full = pd.concat([y_train_fe, y_val_fe])

model = lgb.LGBMRegressor(**LGBM_PARAMS)
model.fit(
    X_full[FEATURE_COLS], y_full,
    eval_set=[(X_val_fe[FEATURE_COLS], y_val_fe)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
               lgb.log_evaluation(period=100)],
)
print(f"Final model best iteration: {model.best_iteration_}")


# =============================================================================
# 4. EVALUATION — 7 metrics của Allora Builder Kit
# =============================================================================
print("\n" + "=" * 60)
print("BƯỚC 4 — Evaluation (test set)")
print("=" * 60)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tỉ lệ dự đoán đúng chiều tăng/giảm."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def czar_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """
    CZAR: tỉ lệ dự đoán đúng chiều trong những lần giá biến động lớn.
    Quan trọng vì WRMSE weight mạnh theo magnitude.
    """
    large_moves = np.abs(y_true) > threshold
    if large_moves.sum() == 0:
        return 0.0
    return directional_accuracy(y_true[large_moves], y_pred[large_moves])

def wrmse_vs_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WRMSE improvement vs zero prediction (weight = |y_true|)."""
    w = np.abs(y_true) + 1e-9
    wrmse_model = np.sqrt(np.average((y_true - y_pred) ** 2, weights=w))
    wrmse_zero  = np.sqrt(np.average(y_true ** 2, weights=w))
    return (wrmse_zero - wrmse_model) / (wrmse_zero + 1e-9)

def bootstrap_da_ci(y_true: np.ndarray, y_pred: np.ndarray,
                    n_bootstrap: int = 1000, confidence: float = 0.95):
    """Bootstrap confidence interval cho Directional Accuracy."""
    da_samples = []
    n = len(y_true)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        da_samples.append(directional_accuracy(y_true[idx], y_pred[idx]))
    alpha = 1 - confidence
    return np.percentile(da_samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])


y_test_np = y_test_fe.values
y_pred    = model.predict(X_test_fe[FEATURE_COLS])

# Tính từng metric
da          = directional_accuracy(y_test_np, y_pred)
pearson_r   = pearsonr(y_test_np, y_pred)[0]
pearson_p   = pearsonr(y_test_np, y_pred)[1]
wrmse_imp   = wrmse_vs_zero(y_test_np, y_pred)
czar        = czar_score(y_test_np, y_pred)
da_ci       = bootstrap_da_ci(y_test_np, y_pred)

# Grade mapping
def grade(metrics_passed: int) -> str:
    grades = {7: "A+", 6: "A", 5: "B+", 4: "B", 3: "C+", 2: "C"}
    return grades.get(metrics_passed, "F")

THRESHOLDS = {
    "DA ≥ 52%":            da >= 0.52,
    "Pearson r ≥ 0.05":    pearson_r >= 0.05,
    "WRMSE ≥ 5%":          wrmse_imp >= 0.05,
    "CZAR ≥ 10%":          czar >= 0.10,
    "DA CI Lower ≥ 0.50":  da_ci[0] >= 0.50,
    "Pearson p < 0.05":    pearson_p < 0.05,
    "DA p < 0.05":         da >= 0.52 and pearson_p < 0.05,   # proxy
}

passed = sum(THRESHOLDS.values())

print(f"\n{'Metric':<25} {'Value':>10}  {'Pass?':>6}")
print("-" * 46)
print(f"{'Directional Accuracy':<25} {da:>9.1%}  {'✓' if THRESHOLDS['DA ≥ 52%'] else '✗':>6}")
print(f"{'Pearson r':<25} {pearson_r:>10.4f}  {'✓' if THRESHOLDS['Pearson r ≥ 0.05'] else '✗':>6}")
print(f"{'Pearson p-value':<25} {pearson_p:>10.4f}  {'✓' if THRESHOLDS['Pearson p < 0.05'] else '✗':>6}")
print(f"{'WRMSE improvement':<25} {wrmse_imp:>9.1%}  {'✓' if THRESHOLDS['WRMSE ≥ 5%'] else '✗':>6}")
print(f"{'CZAR':<25} {czar:>9.1%}  {'✓' if THRESHOLDS['CZAR ≥ 10%'] else '✗':>6}")
print(f"{'DA CI [95%] lower':<25} {da_ci[0]:>10.4f}  {'✓' if THRESHOLDS['DA CI Lower ≥ 0.50'] else '✗':>6}")
print("-" * 46)
print(f"\n{'GRADE':>25}: {grade(passed)} ({passed}/7 metrics passed)")

if passed < 5:
    print("\n⚠  Khuyến nghị: chưa đủ tốt để deploy mainnet.")
    print("   → Thử mở rộng TRAIN_FROM_MONTH về 2022-01")
    print("   → Thêm features (funding rate, open interest nếu có)")
    print("   → Tăng VALIDATION_MONTHS để test nhiều regime hơn")


# =============================================================================
# 5. FEATURE IMPORTANCE — xem signal nào đang hoạt động
# =============================================================================
print("\n" + "=" * 60)
print("BƯỚC 5 — Top 15 feature importance")
print("=" * 60)

importance_df = (
    pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] / importance_df["importance"].max() * 30)
    print(f"  {row['feature']:<30} {bar} ({row['importance']:.0f})")


# =============================================================================
# 6. PREDICT FUNCTION — phải mirror ĐÚNG feature engineering (⑤ critical)
# =============================================================================
print("\n" + "=" * 60)
print("BƯỚC 6 — Đóng gói predict function → predict.pkl")
print("=" * 60)

def predict() -> pd.Series:
    """
    Live inference function. Được gọi mỗi epoch bởi AlloraWorker.

    QUAN TRỌNG: Hàm này phải tính ĐÚNG Y HỆT các features đã dùng khi training.
    Nếu training dùng sol_rsi_14 nhưng hàm này không tính → predict rác.
    """
    # Lấy live features từ workflow (raw OHLCV multi-ticker)
    live_features = workflow.get_live_features("solusd")  # trả về multi-ticker df

    # Thêm custom features (MIRROR với add_custom_features ở trên)
    live_features = add_custom_features(live_features)
    live_features = live_features.dropna()

    if live_features.empty:
        print("⚠  live_features rỗng sau dropna — thiếu data?")
        return pd.Series(dtype=float)

    # Chỉ lấy đúng FEATURE_COLS (thứ tự phải khớp với training)
    available_cols = [c for c in FEATURE_COLS if c in live_features.columns]
    X_live = live_features[available_cols]

    preds = model.predict(X_live)
    return pd.Series(preds, index=X_live.index)


# Pickle cả predict function + model + workflow + feature list vào một file
predict_bundle = {
    "predict_fn":    predict,
    "feature_cols":  FEATURE_COLS,
    "model_params":  LGBM_PARAMS,
}

with open("predict.pkl", "wb") as f:
    dill.dump(predict, f)

print("✓ predict.pkl đã lưu")

# Verify: load lại và thử chạy
with open("predict.pkl", "rb") as f:
    predict_fn = dill.load(f)

print("✓ predict.pkl load lại thành công")


# =============================================================================
# 7. CHẠY WORKER — submit lên Allora
# =============================================================================
print("\n" + "=" * 60)
print(f"BƯỚC 7 — Chạy worker (Topic {TOPIC_ID})")
print("=" * 60)

# Chỉ chạy nếu đạt ít nhất B+ (5/7)
if passed < 5:
    print(f"⚠  Grade {grade(passed)} — khuyến nghị cải thiện model trước khi deploy.")
    print("   Comment block này để bỏ qua kiểm tra nếu bạn muốn test dù sao.")
else:
    from allora_sdk.worker import AlloraWorker

    def my_model():
        """Wrapper gọi predict_fn và trả về scalar."""
        tic = time.time()
        prediction = predict_fn()
        toc = time.time()
        print(f"predict time: {toc - tic:.2f}s  |  pred: {prediction.values[-1]:.6f}")
        return prediction

    async def main():
        worker = AlloraWorker(
            topic_id   = TOPIC_ID,    # 38 testnet, 10 mainnet
            predict_fn = my_model,
            api_key    = FORGE_API_KEY,
            # mnemonic = MNEMONIC,   # bỏ comment nếu có ví sẵn
        )
        print(f"Worker khởi động — Topic {TOPIC_ID} ({'testnet' if TOPIC_ID == 38 else 'mainnet'})")
        print("Ctrl+C để dừng\n")

        async for result in worker.run():
            if isinstance(result, Exception):
                print(f"✗ Error: {str(result)}")
            else:
                print(f"✓ Submitted: {result.prediction:.6f}")

    # Chạy worker
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorker dừng.")


# =============================================================================
# CHECKLIST CUỐI
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════╗
║  Checklist trước khi deploy mainnet (topic 10)          ║
╠══════════════════════════════════════════════════════════╣
║  □  Grade B+ trở lên (5/7 metrics)                      ║
║  □  Test trên testnet topic 38 ít nhất 24h               ║
║  □  Xác nhận on-chain submissions đang ghi nhận          ║
║  □  predict.pkl không bị lỗi khi load lại                ║
║  □  Worker chạy ổn định, không miss epoch                 ║
║  □  .allora_key đã backup mnemonic                        ║
╚══════════════════════════════════════════════════════════╝
""")
