import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# SABİTLER

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "prague3.csv"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_TARGET = "orders"

TEST_START = pd.Timestamp("2024-03-04")
TEST_END = pd.Timestamp("2024-03-10")

# Tahmin anında bilinmeyen değişkenler (precipitation, snow, user_activity)
# modelde kullanılmaz; yalnızca takvim ve geçmiş talep özellikleri kullanılır.
CALENDAR_FEATURES = ["holiday", "shops_closed", "winter_school_holidays"]
REQUIRED_RAW_COLUMNS = CALENDAR_FEATURES

# Grid Search ızgaraları (XGBoost için defterdeki 5 parametre; optimize edilen
# 400 / 0.01 / 3 / 0.8 / 0.8 kombinasyonu ızgara içinde yer alır)
PARAM_GRID_RF = {
    "n_estimators": [100, 200, 300, 400],
    "max_features": [0.3, 0.5, 0.7, 1.0],
    "min_samples_leaf": [1, 2, 3, 5],
}

PARAM_GRID_XGB = {
    "n_estimators": [200, 400],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Streamlit sürümüne göre genişlik parametresi
STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)


# YARDIMCI FONKSİYONLAR

def get_features(target):
    """Modelde kullanılan özellik listesi."""
    return [
        "year", "month", "day", "day_of_week", "week_of_year", "is_weekend",
        *CALENDAR_FEATURES,
        f"{target}_lag_1d", f"{target}_lag_7d", f"{target}_lag_14d",
        f"{target}_rolling_mean_7d", f"{target}_rolling_mean_14d",
    ]


def build_processed(data, prep):
    """Ön işleme: kronolojik sıralama, tarih / lag / rolling özellikleri."""
    date_col, target = prep["date_column"], prep["target"]

    out = data.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    history_cols = []

    if prep["use_date"]:
        out["year"] = out[date_col].dt.year
        out["month"] = out[date_col].dt.month
        out["day"] = out[date_col].dt.day
        out["day_of_week"] = out[date_col].dt.dayofweek
        out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    if prep["use_lag"]:
        for k in (1, 7, 14):
            col = f"{target}_lag_{k}d"
            out[col] = out[target].shift(k)
            history_cols.append(col)

    if prep["use_rolling"]:
        # shift(1): rolling hesabına o günün kendi değeri girmez (sızıntı önlemi)
        for k in (7, 14):
            col = f"{target}_rolling_mean_{k}d"
            out[col] = out[target].shift(1).rolling(k).mean()
            history_cols.append(col)

    if prep["remove_nan"]:
        out = out.dropna(subset=[target] + history_cols).reset_index(drop=True)

    return out


def prepare_split(raw_df, test_file, prep):
    """Kronolojik train/test ayrımı. Eğitim verisi test başlangıcından ÖNCEKİ günlerle sınırlıdır."""
    date_col, target = prep["date_column"], prep["target"]

    if test_file is None:
        proc = st.session_state["processed_df"]
        test_df = proc[(proc[date_col] >= TEST_START) & (proc[date_col] <= TEST_END)]
        test_start = TEST_START
        if test_df.empty:
            return None, None, (
                f"Varsayılan test haftası ({TEST_START:%d.%m.%Y} - {TEST_END:%d.%m.%Y}) "
                "veri setinde bulunamadı. Lütfen bir test verisi yükleyin."
            )
    else:
        test_file.seek(0)
        test_raw = pd.read_csv(test_file)

        missing_cols = [c for c in [date_col, target, *REQUIRED_RAW_COLUMNS] if c not in test_raw.columns]
        if missing_cols:
            return None, None, f"Test verisinde eksik sütunlar var: {missing_cols}"

        test_raw[date_col] = pd.to_datetime(test_raw[date_col])
        base = raw_df.copy()
        base[date_col] = pd.to_datetime(base[date_col])

        # Lag özelliklerinin hesaplanabilmesi için test verisi geçmişin devamı olarak eklenir
        combined = (
            pd.concat([base, test_raw], ignore_index=True)
            .drop_duplicates(subset=date_col, keep="last")
        )
        proc = build_processed(combined, prep)
        test_df = proc[proc[date_col].isin(test_raw[date_col])]
        test_start = test_raw[date_col].min()
        if test_df.empty:
            return None, None, "Test verisinden kullanılabilir satır elde edilemedi."

    train_df = proc[proc[date_col] < test_start]
    if train_df.empty:
        return None, None, "Test tarihinden önce eğitim verisi bulunamadı."

    return train_df.copy(), test_df.copy(), None


def make_model(model_name, params):
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    return XGBRegressor(random_state=42, objective="reg:squarederror", **params)


def compute_metrics(y_true, y_pred):
    """MAE, RMSE ve MAPE (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    return mae, rmse, mape


def run_grid_search(model_name, param_grid, X_train, y_train, val_days, progress):
    """
    Grid Search: eğitim dönemi kronolojik olarak eğitim ve doğrulama alt kümesine
    ayrılır (son `val_days` gün doğrulama). Test haftası aramaya hiç girmez.
    Seçim ölçütü doğrulama MAPE değeridir.
    """
    X_fit, y_fit = X_train.iloc[:-val_days], y_train.iloc[:-val_days]
    X_val, y_val = X_train.iloc[-val_days:], y_train.iloc[-val_days:]

    combos = list(ParameterGrid(param_grid))
    rows = []
    best_idx, best_mape = 0, float("inf")

    for i, combo in enumerate(combos):
        m = make_model(model_name, combo)
        m.fit(X_fit, y_fit)
        val_pred = np.round(m.predict(X_val)).astype(int)
        _, _, val_mape = compute_metrics(y_val, val_pred)

        rows.append({**combo, "Doğrulama MAPE (%)": round(val_mape, 3)})
        if val_mape < best_mape:
            best_mape, best_idx = val_mape, i

        progress.progress((i + 1) / len(combos), text=f"Grid Search: {i + 1} / {len(combos)} kombinasyon")

    table = pd.DataFrame(rows).sort_values("Doğrulama MAPE (%)").reset_index(drop=True)
    return combos[best_idx], table


# SAYFA AYARLARI

st.set_page_config(
    page_title="Talep Tahmini",
    page_icon="",
    layout="wide",
)

st.title("Talep Tahmini")
st.divider()

col1, col2 = st.columns(2)


# 1. VERİ YÜKLEME VE TARGET SEÇİMİ

with col1:

    st.subheader("1️⃣ Veri Yükleme ve Target Seçimi")

    uploaded_file = st.file_uploader("Veri seti yükleyin (CSV)", type=["csv"], key="train_upload")

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Yüklenen veri seti kullanılıyor.")
    elif DEFAULT_DATA_PATH.exists():
        df = pd.read_csv(DEFAULT_DATA_PATH)
        st.info("Varsayılan veri seti kullanılıyor.")
    else:
        st.error(f"Varsayılan veri seti bulunamadı: {DEFAULT_DATA_PATH.name}. Lütfen bir CSV yükleyin.")

    target = None
    if df is not None:
        st.write(f"Satır sayısı: **{len(df)}**")
        st.write(f"Sütun sayısı: **{len(df.columns)}**")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.error("Veri setinde sayısal sütun bulunamadı.")
        else:
            target_index = numeric_cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in numeric_cols else 0
            target = st.selectbox("Target değişkeni", numeric_cols, index=target_index)
            st.write(f"Seçilen Target: **{target}**")

        with st.expander("Veri Önizleme"):
            st.dataframe(df.head(10), **STRETCH)


# 2. VERİ ÖN İŞLEME

with col2:

    st.subheader("2️⃣ Veri Ön İşleme")

    if df is not None and target is not None:

        st.caption("Model için gerekli veri hazırlama işlemleri")

        columns = list(df.columns)
        date_index = columns.index("date") if "date" in columns else 0
        date_column = st.selectbox("Tarih sütunu", columns, index=date_index)

        use_date_features = st.checkbox("Tarih özelliklerini oluştur", value=True)
        use_lag_features = st.checkbox("Lag özelliklerini oluştur", value=True)
        use_rolling_features = st.checkbox("Rolling özelliklerini oluştur", value=True)
        remove_nan = st.checkbox("Eksik değerleri temizle", value=True)

        if st.button("Ön İşlemeyi Uygula", **STRETCH):

            prep = {
                "date_column": date_column,
                "target": target,
                "use_date": use_date_features,
                "use_lag": use_lag_features,
                "use_rolling": use_rolling_features,
                "remove_nan": remove_nan,
            }

            try:
                processed_df = build_processed(df, prep)
            except Exception as e:
                st.error(f"Ön işleme sırasında hata oluştu: {e}")
            else:
                st.session_state["processed_df"] = processed_df
                st.session_state["prep"] = prep

                st.success("Ön işleme başarıyla tamamlandı.")
                st.write(f"Önceki satır sayısı: {len(df)}")
                st.write(f"Sonraki satır sayısı: {len(processed_df)}")
                st.write(f"Oluşan sütun sayısı: {processed_df.shape[1]}")

                with st.expander("İşlenmiş Veri Önizleme"):
                    st.dataframe(processed_df.head(10), **STRETCH)

    else:
        st.info("Önce geçerli bir veri seti ve target gerekli.")


st.divider()

col3, col4 = st.columns(2)


# 4 (ÜST KISIM). TEST VERİSİ

with col4:

    st.subheader("4️⃣ Test Verisi ve Sonuçlar")

    test_file = st.file_uploader("Test verisi yükleyin (CSV, isteğe bağlı)", type=["csv"], key="test_upload")

    if test_file is None:
        st.info(
            f"Test verisi yüklenmediği için {TEST_START:%d.%m.%Y} - {TEST_END:%d.%m.%Y} "
            "test haftası otomatik olarak kullanılmaktadır."
        )
    else:
        st.success("Yüklenen test verisi kullanılacak.")


# 3. MODEL VE HİPERPARAMETRE AYARLARI

with col3:

    st.subheader("3️⃣ Model ve Hiperparametre Ayarları")

    model_name = st.selectbox("Model seçin", ["Random Forest", "XGBoost"])

    if model_name == "XGBoost" and not XGBOOST_AVAILABLE:
        st.error("XGBoost kurulu değil.")
        st.code("pip install xgboost")

    search_mode = st.radio(
        "Hiperparametre Yöntemi",
        ["Manuel", "Grid Search"],
        horizontal=True,
    )

    params = None
    val_days = 28

    if search_mode == "Manuel":

        if model_name == "Random Forest":

            st.write("Random Forest Hiperparametreleri")

            n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
            max_features = st.slider("max_features", 0.1, 1.0, 0.7, step=0.1)
            min_samples_leaf = st.slider("min_samples_leaf", 1, 5, 1)

            params = {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
            }

        else:

            st.write("XGBoost Hiperparametreleri")

            # Varsayılanlar: staj defterindeki optimize edilmiş yapılandırma
            xgb_n_estimators = st.slider("n_estimators", 50, 500, 400, step=50, key="xgb_n_estimators")
            xgb_learning_rate = st.slider("learning_rate", 0.01, 0.30, 0.01, step=0.01, key="xgb_learning_rate")
            xgb_max_depth = st.slider("max_depth", 2, 10, 3, key="xgb_max_depth")
            xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.1, key="xgb_subsample")
            xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.1, key="xgb_colsample")

            params = {
                "n_estimators": xgb_n_estimators,
                "learning_rate": xgb_learning_rate,
                "max_depth": xgb_max_depth,
                "subsample": xgb_subsample,
                "colsample_bytree": xgb_colsample,
            }

        st.write("Seçilen değerler")
        st.code("\n".join(f"{k} = {v}" for k, v in params.items()))

    else:

        param_grid = PARAM_GRID_RF if model_name == "Random Forest" else PARAM_GRID_XGB

        st.write("Grid Search en iyi kombinasyonu aşağıdaki ızgara içinde arar:")
        st.json(param_grid)

        val_days = st.slider(
            "Doğrulama süresi (gün)",
            min_value=14, max_value=56, value=28, step=7,
            help="Eğitim döneminin son günleri doğrulama için ayrılır; test haftası aramaya dahil edilmez.",
        )
        st.caption(
            f"Toplam {len(ParameterGrid(param_grid))} kombinasyon denenir. "
            "Eğitim dönemi kronolojik olarak eğitim ve doğrulama alt kümesine ayrılır."
        )

    if st.button("Modeli Eğit", type="primary", **STRETCH):

        prep = st.session_state.get("prep")

        if prep is None or "processed_df" not in st.session_state:
            st.warning("Önce 2. bölümden ön işlemeyi uygulayın.")

        elif df is None:
            st.warning("Önce geçerli bir veri seti yükleyin.")

        elif model_name == "XGBoost" and not XGBOOST_AVAILABLE:
            st.error("XGBoost kurulu değil. Önce `pip install xgboost` komutunu çalıştırın.")

        else:

            date_column = prep["date_column"]
            train_target = prep["target"]
            features = get_features(train_target)

            train_df, test_df, error = prepare_split(df, test_file, prep)

            if error:
                st.error(error)

            else:
                missing_features = [f for f in features if f not in train_df.columns]

                if missing_features:
                    st.error("Eksik model özellikleri (ön işleme seçeneklerini kontrol edin):")
                    st.write(missing_features)

                else:
                    X_train, y_train = train_df[features], train_df[train_target]
                    X_test, y_test = test_df[features], test_df[train_target]

                    best_params = None
                    search_table = None

                    try:
                        if search_mode == "Grid Search":
                            if len(X_train) <= val_days + 30:
                                raise ValueError(
                                    "Eğitim verisi doğrulama süresine göre çok az; doğrulama süresini azaltın."
                                )
                            progress = st.progress(0.0, text="Grid Search başlıyor...")
                            best_params, search_table = run_grid_search(
                                model_name, param_grid, X_train, y_train, val_days, progress
                            )
                            progress.empty()
                            final_params = best_params
                        else:
                            final_params = params

                        # En iyi / seçilen parametrelerle tüm eğitim verisi üzerinde final model
                        model = make_model(model_name, final_params)
                        with st.spinner(f"{model_name} modeli eğitiliyor..."):
                            model.fit(X_train, y_train)

                        # Sipariş sayısı tam sayıdır: tahminler tam sayıya yuvarlanır
                        y_pred = np.round(model.predict(X_test)).astype(int)
                    except Exception as e:
                        st.error(f"Model eğitilirken hata oluştu: {e}")
                    else:
                        mae, rmse, mape = compute_metrics(y_test, y_pred)

                        # Eğitilen model kaydedilir (uygulama her seferinde baştan eğitilmek zorunda kalmaz)
                        saved_name = None
                        try:
                            MODEL_DIR.mkdir(exist_ok=True)
                            file_name = "random_forest_model.joblib" if model_name == "Random Forest" else "xgboost_model.joblib"
                            joblib.dump(model, MODEL_DIR / file_name)
                            saved_name = file_name
                        except Exception:
                            pass

                        st.session_state["result"] = {
                            "model_name": model_name,
                            "search_mode": search_mode,
                            "params": final_params,
                            "search_table": search_table,
                            "dates": test_df[date_column].values,
                            "y_test": y_test.values,
                            "y_pred": y_pred,
                            "mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                        }

                        history = st.session_state.setdefault("history", {})
                        history[model_name] = {
                            "Model": model_name,
                            "Yöntem": search_mode,
                            "MAE": round(mae, 2),
                            "RMSE": round(rmse, 2),
                            "MAPE (%)": round(mape, 2),
                        }

                        st.success(f"{model_name} modeli ({search_mode}) başarıyla eğitildi.")
                        if best_params is not None:
                            st.write("En iyi parametreler:")
                            st.json(best_params)
                        st.write(f"Eğitim verisi: {len(X_train)} satır")
                        st.write(f"Test verisi: {len(X_test)} satır")
                        if saved_name:
                            st.caption(f"Model kaydedildi: models/{saved_name}")


# 4. TEST SONUÇLARI

with col4:

    result = st.session_state.get("result")

    if result is None:
        st.info("Hata metriklerini görmek için önce 3. bölümden modeli eğitin.")

    else:

        st.write("Hata Metrikleri")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("MAE", f"{result['mae']:.2f}")
        with metric_col2:
            st.metric("RMSE", f"{result['rmse']:.2f}")
        with metric_col3:
            st.metric("MAPE", f"{result['mape']:.2f}%")

        st.write(f"Model: **{result['model_name']}** ({result['search_mode']})")

        if result["search_table"] is not None:
            with st.expander("Grid Search — En İyi Parametreler"):
                st.json(result["params"])
                st.write("En iyi 10 kombinasyon (doğrulama MAPE):")
                st.dataframe(result["search_table"].head(10), hide_index=True, **STRETCH)

        # Sipariş sayıları tam sayı olarak gösterilir
        actual_int = np.round(result["y_test"]).astype(int)
        pred_int = np.asarray(result["y_pred"]).astype(int)
        test_dates = pd.to_datetime(result["dates"])

        results_df = pd.DataFrame({
            "Tarih": test_dates.strftime("%d.%m.%Y"),
            "Gerçek Değer": actual_int,
            "Tahmin": pred_int,
        })

        with st.expander("Tahmin Sonuçlarını Görüntüle"):
            st.dataframe(results_df, hide_index=True, **STRETCH)

        st.write("Gerçek ve Tahmin Edilen Değerler")

        chart_df = pd.DataFrame(
            {"Gerçek Değer": actual_int, "Tahmin": pred_int},
            index=test_dates,
        )
        st.line_chart(chart_df)

        history = st.session_state.get("history", {})
        if history:
            st.write("Model Performans Karşılaştırması")
            st.dataframe(pd.DataFrame(history.values()), hide_index=True, **STRETCH)
            if len(history) < 2:
                st.caption("Karşılaştırma için diğer modeli de eğitin.")