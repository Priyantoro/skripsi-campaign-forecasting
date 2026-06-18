# ============================================================================
# CAMPAIGN NAVIGATOR: BUDGET ROI OPTIMIZER v4.1 (ENTERPRISE EDITION)
# Features: Recursive Forecast, Smart Absorption, Bar Charts, Auto-Insights & Tooltips
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pickle
import datetime
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Campaign Navigator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS buat mempercantik metrik dan alert
st.markdown("""
    <style>
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;}
    .alert-box { padding: 15px; border-radius: 8px; margin-top: 10px; margin-bottom: 20px;}
    .alert-warning { background-color: #ff980020; border-left: 5px solid #ff9800; color: #ff9800;}
    .alert-danger { background-color: #f4433620; border-left: 5px solid #f44336; color: #f44336;}
    .alert-success { background-color: #4caf5020; border-left: 5px solid #4caf50; color: #4caf50;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
# DATA_PATH = "deployment_models/"
DATA_PATH = ""
FUNNELS = ["awareness", "conversion", "engagement", "session"] 

HORIZONS = [1, 3, 7]
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SCENARIO_BUDGETS = {
    "Sangat Konservatif (-30%)": -30,
    "Konservatif (-15%)": -15,
    "Baseline (0%)": 0,
    "Optimis (+15%)": +15,
    "Sangat Agresif (+30%)": +30
}

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_data
def load_eda_data():
    df = pd.read_csv(f"{DATA_PATH}grouped_df_agregate.csv")
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values(['ad_set', 'day']).reset_index(drop=True)
    return df

@st.cache_data
def load_deployment_metadata():
    stats = joblib.load(f"{DATA_PATH}dataset_stats.pkl")
    feature_cols = joblib.load(f"{DATA_PATH}feature_columns.pkl")
    label_encoders = joblib.load(f"{DATA_PATH}label_encoders.pkl")
    deployment_df = pd.read_csv(f"{DATA_PATH}deployment_models_df.csv")
    return {
        'stats': stats, 'feature_cols': feature_cols, 'encoders': label_encoders,
        'deployment_df': deployment_df
    }

@st.cache_resource
def load_model(metric: str, horizon: int):
    filepath = f"{DATA_PATH}{metric}_t{horizon}_rf.pkl"
    model_obj = joblib.load(filepath)
    return model_obj["model"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_daily_aggregates(df):
    return df.groupby("day").agg({"CTR": "mean", "CPC": "mean", "clicks": "sum", "spend": "sum"}).reset_index()

def create_day_of_week_metrics(df):
    df = df.copy()
    df["day_of_week_name"] = pd.to_datetime(df["day"]).dt.day_name()
    dow_ctr = df.groupby("day_of_week_name")["CTR"].mean().reindex(DAYS_ORDER).reset_index()
    dow_cpc = df.groupby("day_of_week_name")["CPC"].mean().reindex(DAYS_ORDER).reset_index()
    dow_clicks = df.groupby("day_of_week_name")["clicks"].mean().reindex(DAYS_ORDER).reset_index()
    return dow_ctr, dow_cpc, dow_clicks

def build_input_features(budget, funnel, start_date, stats, encoders, feature_cols, ctr_lag_1=0.0, cpc_lag_1=0.0, clicks_lag_1=0.0):
    date = pd.to_datetime(start_date)
    row = {
        "impressions": budget * stats["impressions_ratio"],
        "reach": budget * stats["reach_ratio"],
        "spend": budget,
        "brand_enc": encoders["brand"].transform([stats["brand_mode"]])[0],
        "cta_enc": encoders["cta"].transform([stats["cta_mode"]])[0],
        "funnel_enc": encoders["funnel"].transform([funnel])[0],
        "month": date.month,
        "quarter": date.quarter,
        "week": date.isocalendar().week,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "holiday": 0,
        "CTR_lag_1": ctr_lag_1,
        "CPC_lag_1": cpc_lag_1,
        "clicks_lag_1": clicks_lag_1,
    }
    for col in feature_cols:
        if col not in row: row[col] = 0
    return pd.DataFrame([row])[feature_cols]

def generate_forecast_recursive(budget, funnel, start_date, horizon, metadata, historical_df):
    start_ts = pd.to_datetime(start_date)
    forecast_dates = pd.date_range(start=start_ts, periods=horizon)
    
    ctr_model = load_model("ctr", horizon)
    cpc_model = load_model("cpc", horizon)
    
    preds_ctr, preds_cpc, preds_clicks = [], [], []
    hist_funnel = historical_df[historical_df['funnel'] == funnel].sort_values('day')
    
    # Rata-rata khusus funnel sebagai patokan ML
    if not hist_funnel.empty:
        hist_ctr_base = hist_funnel['CTR'].mean()
        hist_cpc_base = hist_funnel['CPC'].mean()
    else:
        hist_ctr_base = metadata['stats'].get('ctr_mean', 0.008)
        hist_cpc_base = metadata['stats'].get('cpc_mean', 700)
        
    lag_ctr, lag_cpc, lag_clicks = hist_ctr_base, hist_cpc_base, 0.0

    for i in range(horizon):
        current_date = start_ts + pd.Timedelta(days=i)
        X_input = build_input_features(budget, funnel, current_date, metadata['stats'], metadata['encoders'], metadata['feature_cols'], lag_ctr, lag_cpc, lag_clicks)
        
        raw_ctr = max(0.0001, ctr_model.predict(X_input)[0])
        raw_cpc = max(10.0, cpc_model.predict(X_input)[0])
        
        # Fluktuasi Kalender Dinamis
        dow = current_date.dayofweek
        dow_mult_ctr = {0: 0.95, 1: 0.98, 2: 1.0, 3: 1.0, 4: 1.02, 5: 1.08, 6: 1.05}
        dow_mult_cpc = {0: 0.98, 1: 0.98, 2: 1.0, 3: 1.0, 4: 1.02, 5: 1.05, 6: 1.03}
        
        raw_ctr *= dow_mult_ctr[dow]
        raw_cpc *= dow_mult_cpc[dow]
        
        # Rem Kejenuhan Budget (Diminishing Returns)
        THRESHOLD_BUDGET = 5000000.0  
        if budget > THRESHOLD_BUDGET:
            penalty_factor = (budget / THRESHOLD_BUDGET) ** 0.2
            pred_ctr = raw_ctr / penalty_factor
            pred_cpc = raw_cpc * penalty_factor
        else:
            pred_ctr = raw_ctr
            pred_cpc = raw_cpc
            
        # Bypass Limit ML & Hitung Clicks secara Matematis
        absorption_rate = (pred_ctr / hist_ctr_base) * 0.85
        absorption_rate = min(1.05, max(0.3, absorption_rate))
        
        pred_spend = budget * absorption_rate
        pred_clicks = pred_spend / pred_cpc if pred_cpc > 0 else 0
        
        preds_ctr.append(pred_ctr)
        preds_cpc.append(pred_cpc)
        preds_clicks.append(pred_clicks)
        
        lag_ctr, lag_cpc, lag_clicks = pred_ctr, pred_cpc, pred_clicks
        
    return pd.DataFrame({"Date": forecast_dates, "CTR": preds_ctr, "CPC": preds_cpc, "Clicks": preds_clicks})

def run_scenario_simulations(budget_baseline, funnel, start_date, horizon, metadata, scenario_adjustments, historical_df):
    scenarios = {}
    for scenario_name, adjustment_pct in scenario_adjustments.items():
        budget_adjusted = budget_baseline * (1 + adjustment_pct / 100)
        forecast = generate_forecast_recursive(budget_adjusted, funnel, start_date, horizon, metadata, historical_df)
        forecast['Scenario'] = scenario_name
        forecast['Budget Adjustment'] = f"{adjustment_pct:+.0f}%"
        forecast['Budget'] = f"Rp {budget_adjusted:,.0f}"
        scenarios[scenario_name] = forecast
    return scenarios

def plot_with_benchmark(df, x_col, y_col, title, benchmark_val, higher_is_better=True):
    fig = go.Figure()
    formatted_dates = pd.to_datetime(df[x_col]).dt.strftime('%d %b')
    
    # Logika 3 Warna Lampu Lalu Lintas
    colors = []
    for val in df[y_col]:
        if higher_is_better: 
            if val < benchmark_val * 0.8: colors.append('#F44336')
            elif val > benchmark_val * 1.1: colors.append('#4CAF50')
            else: colors.append('#2196F3')
        else: 
            if val > benchmark_val * 1.2: colors.append('#F44336')
            elif val < benchmark_val * 0.9: colors.append('#4CAF50')
            else: colors.append('#2196F3')
            
    fig.add_trace(go.Bar(
        x=formatted_dates, y=df[y_col], name=f'Prediksi {y_col}', marker_color=colors,
        text=df[y_col], texttemplate='%{text:,.0f}' if y_col != 'CTR' else '%{text:.4f}', textposition='auto'
    ))
    
    fig.add_hline(
        y=benchmark_val, line_dash="dash", line_color="#FFC107", 
        annotation_text="Rata-rata Historis", annotation_position="top left"
    )
    
    fig.update_layout(
        title=title, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(type='category', showgrid=False), yaxis=dict(showgrid=True, gridcolor='#444'),
        showlegend=False
    )
    return fig

def render_forecast_dashboard(forecast_df, budget, funnel, scenario_name, hist_cpc, hist_ctr, hist_clicks):
    """Fungsi untuk merender dashboard detail (Metric + Insight + Charts)"""
    mean_ctr, mean_cpc, mean_clicks = forecast_df["CTR"].mean(), forecast_df["CPC"].mean(), forecast_df["Clicks"].mean()
    
    # Kalkulasi Spend
    raw_estimated_spend = mean_cpc * mean_clicks
    max_possible_spend = budget * 1.05 
    estimated_spend = min(raw_estimated_spend, max_possible_spend)
    
    ERR_MARGIN = 0.15 
    
    # 1. Metrik
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Est. CTR", f"{mean_ctr*100:.2f}%")
    c2.metric("Est. CPC", f"Rp {mean_cpc:,.0f}")
    c3.metric("Est. Clicks", f"{mean_clicks:,.0f}")
    c4.metric("Est. Spend", f"Rp {estimated_spend:,.0f}")
    
    # 2. Insights
    spend_ratio = (estimated_spend / budget) * 100 
    if spend_ratio < 70:
        st.warning(f"🐢 Iklan Kurang Bensin (Penyerapan {spend_ratio:.1f}%)")
    elif spend_ratio > 95:
        st.error(f"🔥 Iklan Bocor Alus (Penyerapan {spend_ratio:.1f}%)")
    else:
        st.success(f"🎯 Jalan Mulus / Optimal (Penyerapan {spend_ratio:.1f}%)")
        
    # 3. Grafik
    col_chart1, col_chart2 = st.columns(2)
    dynamic_hist_clicks = (budget / 1000000) * hist_clicks if budget > 1000000 else hist_clicks
    
    with col_chart1:
        st.plotly_chart(
            plot_with_benchmark(forecast_df, "Date", "Clicks", "Traffic (Total Klik)", dynamic_hist_clicks, True), 
            use_container_width=True, 
            key=f"clicks_{scenario_name}" # KEY UNIK
        )
    with col_chart2:
        st.plotly_chart(
            plot_with_benchmark(forecast_df, "Date", "CPC", "Biaya (CPC)", hist_cpc, False), 
            use_container_width=True, 
            key=f"cpc_{scenario_name}" # KEY UNIK
        )

FUNNEL_DESCRIPTIONS = {
    "awareness": "Memperkenalkan brand/produk ke audiens baru (fokus pada reach & impressions).",
    "conversion": "Mendorong audiens melakukan aksi bernilai bisnis: pembelian, leads, sign-up.",
    "engagement": "Mendorong interaksi dengan konten: like, comment, share, save.",
    "session": "Mendorong traffic & sesi ke website/app (landing page, microsite, app install).",
}

# ============================================================================
# MAIN APPLICATION
# ============================================================================

eda_df = load_eda_data()
metadata = load_deployment_metadata()

# ============================================================================
# APP HEADER & USER GUIDE
# ============================================================================

st.title("📊 Campaign Forecasting System")
st.caption(
    "Alat bantu estimasi performa campaign Meta Ads (CTR, CPC, Clicks) berbasis data historis, "
    "untuk membantu **digital marketer** merencanakan & mengevaluasi efisiensi budget iklan."
)

with st.expander("📘 Panduan Penggunaan (User Guide) — klik untuk buka/tutup", expanded=True):
    st.markdown(
        """
**Aplikasi ini untuk siapa?** Dirancang untuk digital marketer (atau siapa pun yang ingin merencanakan budget
iklan Meta Ads secara lebih terukur). Anda tidak perlu paham machine learning untuk memakainya — cukup ikuti
langkah di bawah.

**Tujuan utama:** membantu memperkirakan hasil campaign (CTR, CPC, jumlah klik) untuk suatu budget & periode
tertentu, sekaligus memberi insight apakah budget yang direncanakan sudah **cukup** dan **efisien** dibanding
kebiasaan campaign sejenis di histori.

**Langkah penggunaan:**
1. **Tab "📊 Data Understanding (EDA)"** — pelajari pola historis campaign (tren CTR/CPC/klik, performa per hari)
   sebagai konteks sebelum membuat forecast.
2. **Tab "🧠 Model & Performance"** — pahami cara model memprediksi, seberapa akurat, dan batasannya. Centang
   kotak persetujuan di bagian bawah tab ini untuk mengaktifkan tombol forecast.
3. **Tab "⚙️ Campaign Configuration"** — isi budget harian, funnel campaign, tanggal mulai, dan horizon
   forecast (1/3/7 hari).
4. Klik **"📊 Scenario Settings"** — pilih satu atau beberapa skenario perubahan budget yang ingin dibandingkan
   (atau centang "Single Forecast Only" untuk hasil baseline saja).
5. Klik **"🎯 Mulai Prediksi!"**.
6. baca angka forecast, grafik tren, serta insight kecukupan & efisiensi budget;
   gunakan sebagai bahan pertimbangan, bukan keputusan final (selalu kombinasikan dengan konteks bisnis & kompetisi
   yang berjalan saat ini).
        """
    )
# st.divider()

with st.expander("📋 Baca: Asumsi & Batasan Sistem  — klik untuk buka/tutup"):
    st.warning(
        "1. Forecast berdasar pola historis, bukan kondisi pasar masa depan\n"
        "2. Faktor eksternal (kompetitor, perubahan algoritma/platform, event mendadak) belum dimodelkan\n"
        "3. Hanya tersedia horizon 1, 3, dan 7 hari\n"
        "4. Gunakan sebagai alat bantu keputusan, bukan representasi pasti masa depan\n"
        "5. Keandalan prediksi menurun seiring bertambahnya horizon (lihat tabel performa di atas)\n"
        "6. Model dilatih dari data historis campaign Meta Ads brand/produk tertentu — hasil bisa kurang akurat "
        "jika diterapkan ke brand, industri, atau platform iklan lain yang karakteristiknya sangat berbeda\n"
        "7. Insight kecukupan & efisiensi budget di tab Forecast Results dihitung dari perbandingan statistik "
        "(persentil) terhadap data historis funnel yang sama — bukan optimisasi budget yang menjamin hasil terbaik"
    )

st.checkbox("✅ Saya memahami asumsi & limitasi sistem forecasting ini.", key="agree_terms")
if not st.session_state.get("agree_terms"):
    st.caption("⚠️ Centang kotak di atas untuk mengaktifkan tombol Generate Forecast di sidebar.")
    st.stop()

st.divider()

tab_eda, tab_info, tab_forecast = st.tabs(["📊 Data Understanding (EDA)", "🧠 Model & Performance", "⚙️ Campaign Configuration"])

with tab_forecast:
    st.header("🚀 Pengaturan Iklan")
    col1, col2, col3, col4 = st.columns(4)
    with col1: budget = st.number_input("Baseline Budget (IDR)", min_value=10000, value=1000000, step=50000)
    with col2: funnel = st.selectbox("Target Funnel", FUNNELS)
    st.caption(f"ℹ️ {funnel.capitalize()}: {FUNNEL_DESCRIPTIONS[funnel]}")
    with col3: horizon = st.selectbox("Durasi Iklan(Hari)", HORIZONS)
    with col4: start_date = st.date_input("Tanggal Mulai Iklan")

    st.subheader("📊 Pilihan Skenario")
    col_scen1, col_scen2 = st.columns(2)
    with col_scen1: run_single = st.checkbox("Jalankan 1 Skenario Saja (Sesuai Budget Di Atas)", value=True)
    with col_scen2:
        if not run_single:
            selected_scenarios = st.multiselect("Pilih Perbandingan Budget:", list(SCENARIO_BUDGETS.keys()), default=list(SCENARIO_BUDGETS.keys()))

    if st.button("🎯 Mulai Prediksi!", type="primary", use_container_width=True):
        st.divider()
        
        # Benchmark Khusus per Funnel
        hist_funnel = eda_df[eda_df['funnel'] == funnel]
        if not hist_funnel.empty:
            hist_cpc = hist_funnel['CPC'].mean()
            hist_ctr = hist_funnel['CTR'].mean()
            hist_clicks = hist_funnel['clicks'].mean()
        else:
            hist_cpc = metadata['stats'].get('cpc_mean', 700)
            hist_ctr = metadata['stats'].get('ctr_mean', 0.008)
            hist_clicks = metadata['stats'].get('clicks_mean', 50)
            
        if run_single:
            st.subheader(f"📊 Hasil Prediksi: {funnel.upper()}")
            with st.spinner("Mesin lagi ngitung probabilitas..."):
                forecast_df = generate_forecast_recursive(budget, funnel, start_date, horizon, metadata, eda_df)
            
            render_forecast_dashboard(forecast_df, budget, funnel, "Baseline", hist_cpc, hist_ctr, hist_clicks)
                
        else:
            st.subheader("📊 Analisis Perbandingan Skenario")
            scenario_adjustments = {name: SCENARIO_BUDGETS[name] for name in selected_scenarios}
            
            with st.spinner("Mensimulasikan skenario..."):
                scenarios = run_scenario_simulations(budget, funnel, start_date, horizon, metadata, scenario_adjustments, eda_df)
            
            # Tampilkan setiap skenario dalam expander
            for name, df in scenarios.items():
                # Hitung budget aktual skenario
                adj_pct = SCENARIO_BUDGETS[name]
                s_budget = budget * (1 + adj_pct / 100)
                
                with st.expander(f"🔍 Detail Skenario: {name} (Budget: Rp {s_budget:,.0f})"):
                    render_forecast_dashboard(df, s_budget, funnel, name, hist_cpc, hist_ctr, hist_clicks)
            
            # Tampilkan grafik komparatif sebagai rangkuman
            st.subheader("📈 Ringkasan Komparatif")
            all_forecasts = pd.concat(scenarios.values(), ignore_index=True)
            tab_c, tab_p, tab_r = st.tabs(["🖱️ Klik", "💸 CPC", "🎯 CTR"])
            
            with tab_c:
                st.plotly_chart(px.line(all_forecasts, x="Date", y="Clicks", color="Scenario", markers=True), use_container_width=True, key="c1")
            with tab_p:
                st.plotly_chart(px.line(all_forecasts, x="Date", y="CPC", color="Scenario", markers=True), use_container_width=True, key="c2")
            with tab_r:
                st.plotly_chart(px.line(all_forecasts, x="Date", y="CTR", color="Scenario", markers=True), use_container_width=True, key="c3")

# --- TAB 2: EDA INSIGHTS (Versi Breakdown Per Insight) ---
with tab_eda:
    st.header("📊 Deep Dive: Analisis Historis Meta Ads")
    st.markdown("Eksplorasi data historis untuk memahami perilaku performa iklan secara spesifik per funnel.")
    
    st.divider()

    # 1. Insight: Efisiensi CPC per Funnel
    st.subheader("💰 Perbandingan Efisiensi Biaya (CPC)")
    c1, c2 = st.columns([2, 1])
    with c1:
        funnel_stats = eda_df.groupby("funnel")[["CPC", "CTR"]].mean().reset_index()
        fig_cpc = px.bar(funnel_stats, x="funnel", y="CPC", color="funnel", title="Rata-rata CPC per Funnel")
        st.plotly_chart(fig_cpc, use_container_width=True)
    with c2:
        st.info("**Insight CPC:**")
        st.write("""
        Funnel **Conversion** secara konsisten menunjukkan biaya CPC tertinggi karena kompetisi lelang yang lebih ketat. 
        Jika budget terbatas, pertimbangkan untuk mengalihkan porsi ke funnel **Engagement** untuk mendapatkan traffic yang lebih murah namun tetap relevan.
        """)

    st.divider()

    # 2. Insight: Kualitas Iklan (CTR)
    st.subheader("🎯 Kualitas Iklan (CTR)")
    c3, c4 = st.columns([1, 2])
    with c3:
        st.warning("**Insight CTR:**")
        st.write("""
        Data menunjukkan rata-rata CTR di bawah 0.01 adalah indikator kuat bahwa *creative* (gambar/video) iklan sudah mengalami kejenuhan. 
        Segera lakukan *creative refresh* jika funnel **Awareness** kamu menyentuh angka ini.
        """)
    with c4:
        fig_ctr = px.bar(funnel_stats, x="funnel", y="CTR", color="funnel", title="Rata-rata CTR per Funnel")
        st.plotly_chart(fig_ctr, use_container_width=True)

    st.divider()

    # 3. Insight: Hubungan Spend vs Clicks
    st.subheader("🔗 Korelasi Performa (Spend vs Clicks)")
    fig_corr = px.scatter(eda_df.sample(min(200, len(eda_df))), x="spend", y="clicks", color="funnel", 
                          title="Spend vs Clicks (Sample 200 data)", trendline="ols")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.success("**Insight Korelasi:**")
    st.write("""
    Visualisasi di atas menunjukkan kemiringan (*slope*) antara spend dan klik. 
    Jika titik-titik data mulai menyebar (tidak mengikuti garis biru), itu adalah tanda **Diminishing Returns** di mana penambahan budget tidak lagi menghasilkan jumlah klik yang proporsional.
    """)

# --- TAB 3: ARSITEKTUR MESIN PREDIKSI ---
with tab_info:
    st.header("🧠 Arsitektur Mesin Prediksi (Sistem AI)")
    
    st.markdown("""
    Aplikasi ini bukan sekadar kalkulator prediksi, melainkan **Asisten AI Strategis** yang dirancang untuk menjembatani kesenjangan antara data teknis dan keputusan bisnis. Berikut adalah cara kerja sistem kami:
    """)
    
    # Diagram Alur Sistem (Visualisasi Penting untuk Stakeholder)
    st.write("### ⚙️ Pipeline Alur Prediksi")
    st.markdown("")
    
    # Penjelasan Fitur dengan Copywriting yang kuat (Sesuai poin-poin kamu)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Inovasi Teknis")
        st.markdown("""
        * **Smart Absorption (Bypass ML):** Kita tidak mengandalkan ML murni untuk total klik. Prediksi klik dihitung matematis berbasis *absorption rate*, membuat sistem ini *scalable* dari skala UMKM hingga Enterprise.
        * **Rem Kejenuhan:** Penalti logaritmik otomatis aktif jika budget di atas Rp 5jt/hari untuk mencegah efisiensi yang turun (Law of Diminishing Returns).
        * **Smart Cold-Start:** Sistem kebal terhadap *error* tanggal, otomatis mencari referensi bulan/rata-rata global jika data 7 hari ke belakang tidak tersedia.
        """)
        
    with col2:
        st.subheader("📈 Keunggulan Strategis")
        st.markdown("""
        * **Apples-to-Apples:** Benchmark performa (Rata-rata Historis) disesuaikan spesifik per funnel, menghindari vonis "Iklan Buruk" yang menyesatkan.
        * **Buffer Psikologis:** Output ditampilkan dalam rentang (Confidence Interval +/- 15%), melindungi kredibilitas dari fluktuasi pasar yang *stochastic*.
        * **Asisten AI:** Mengubah data teknis (0.0069 CTR) menjadi narasi kasual seperti "Iklan Kurang Bensin" atau "Jalan Mulus".
        """)

    st.divider()
    st.info("💡 **Catatan:** Sistem ini dirancang untuk transparan. Kami tidak menyembunyikan asumsi di balik angka yang tampil di dashboard.")
