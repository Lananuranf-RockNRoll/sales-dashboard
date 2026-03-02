import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ================================
# Config
# ================================
st.set_page_config(page_title="Shoppers Intelligence", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* Background */
.stApp { background: #060d1f; }
section[data-testid="stSidebar"] { background: #0a1628 !important; border-right: 1px solid #1e3a5f; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060d1f; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2144 50%, #0a1f3d 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #f0f6ff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #3b82f6; }
.hero-sub { color: #64748b; font-size: 15px; margin: 0; font-weight: 400; }
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #3b82f6;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric Cards */
.metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 28px; }
.mcard {
    background: linear-gradient(145deg, #0d1f3c, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.mcard:hover { transform: translateY(-2px); border-color: #3b82f6; }
.mcard::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.mcard:hover::after { opacity: 1; }
.mcard-icon { font-size: 20px; margin-bottom: 10px; }
.mcard-label { font-size: 11px; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.mcard-value { font-size: 26px; font-weight: 800; color: #e2e8f0; letter-spacing: -0.5px; }
.mcard-value.blue { color: #3b82f6; }
.mcard-value.green { color: #10b981; }
.mcard-delta { font-size: 11px; color: #475569; margin-top: 4px; }

/* Section header */
.sec-header {
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 20px 0;
}
.sec-line { flex: 1; height: 1px; background: linear-gradient(90deg, #1e3a5f, transparent); }
.sec-title { color: #94a3b8; font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; white-space: nowrap; }

/* Chart container */
.chart-box {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 8px;
    margin-bottom: 16px;
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1a35);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.insight-icon { font-size: 20px; flex-shrink: 0; margin-top: 2px; }
.insight-text { color: #94a3b8; font-size: 14px; line-height: 1.6; }
.insight-text strong { color: #3b82f6; font-weight: 600; }

/* Predict box */
.predict-result {
    background: linear-gradient(135deg, #0d2144, #0a1f3d);
    border: 2px solid #3b82f6;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-top: 16px;
}
.predict-label { font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.predict-value { font-size: 40px; font-weight: 800; }
.predict-value.yes { color: #10b981; }
.predict-value.no { color: #ef4444; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1e3a5f !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: white !important;
}

/* Sidebar */
.sidebar-label { color: #475569; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 6px 0; }
div[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 13px !important; }
div[data-testid="stSidebar"] h2 { color: #3b82f6 !important; font-size: 18px !important; font-weight: 700 !important; }

/* Feature table */
.feat-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #1e3a5f;
}
.feat-rank { color: #3b82f6; font-weight: 700; font-size: 13px; width: 24px; }
.feat-name { color: #e2e8f0; font-size: 13px; font-weight: 500; flex: 1; }
.feat-bar-bg { width: 120px; height: 6px; background: #1e3a5f; border-radius: 3px; overflow: hidden; }
.feat-bar { height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 3px; }
.feat-pct { color: #64748b; font-size: 12px; width: 40px; text-align: right; }

/* Slider & input styling */
div[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 13px !important; }
div[data-testid="stNumberInput"] label { color: #94a3b8 !important; font-size: 13px !important; }
div[data-testid="stSelectbox"] label { color: #94a3b8 !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ================================
# Load & Cache Data
# ================================
@st.cache_data
def load_data():
    return pd.read_csv('online_shoppers_intention.csv')

@st.cache_resource
def train_models():
    df_ml = pd.read_csv('online_shoppers_intention.csv')
    le = LabelEncoder()
    df_ml['Month'] = le.fit_transform(df_ml['Month'])
    df_ml['VisitorType'] = le.fit_transform(df_ml['VisitorType'])
    df_ml['Weekend'] = df_ml['Weekend'].astype(int)
    df_ml['Revenue'] = df_ml['Revenue'].astype(int)

    X = df_ml.drop('Revenue', axis=1)
    y = df_ml['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cm = confusion_matrix(y_test, y_pred_lr)

    return rf, lr, scaler, acc_rf, acc_lr, feat_imp, cm, X.columns.tolist()

df = load_data()
rf_model, lr_model, scaler, acc_rf, acc_lr, feat_imp, cm, feature_cols = train_models()

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown("## 🛒 Navigator")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">📅 Filter Bulan</div>', unsafe_allow_html=True)
    month_order = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    available_months = [m for m in month_order if m in df['Month'].unique()]
    month_filter = st.multiselect("Bulan", options=available_months, default=available_months, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">👤 Tipe Visitor</div>', unsafe_allow_html=True)
    visitor_filter = st.multiselect("Visitor", options=df['VisitorType'].unique(), default=df['VisitorType'].unique(), label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">📆 Hari Kunjungan</div>', unsafe_allow_html=True)
    weekend_filter = st.radio("Hari", ["Semua", "Weekday", "Weekend"], label_visibility="collapsed")

    st.markdown("---")
    df_f = df[(df['Month'].isin(month_filter)) & (df['VisitorType'].isin(visitor_filter))]
    if weekend_filter == "Weekend": df_f = df_f[df_f['Weekend'] == True]
    elif weekend_filter == "Weekday": df_f = df_f[df_f['Weekend'] == False]

    st.markdown(f'<div style="color:#64748b;font-size:12px">📊 <strong style="color:#3b82f6">{len(df_f):,}</strong> baris terpilih</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    csv = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", csv, "filtered_data.csv", "text/csv", use_container_width=True)

# ================================
# Hero
# ================================
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">📊 Data Analytics Dashboard</div>
    <div class="hero-title">Online Shoppers <span>Intelligence</span></div>
    <p class="hero-sub">Analisis mendalam perilaku pengunjung e-commerce & prediksi konversi pembelian berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ================================
# Metrics
# ================================
conv = df_f['Revenue'].mean() * 100
c1,c2,c3,c4,c5 = st.columns(5)
def mcard(col, icon, label, value, cls=""):
    col.markdown(f"""
    <div class="mcard">
        <div class="mcard-icon">{icon}</div>
        <div class="mcard-label">{label}</div>
        <div class="mcard-value {cls}">{value}</div>
    </div>""", unsafe_allow_html=True)

mcard(c1, "👥", "Total Pengunjung", f"{len(df_f):,}")
mcard(c2, "💰", "Total Pembelian", f"{int(df_f['Revenue'].sum()):,}", "blue")
mcard(c3, "📈", "Konversi Rate", f"{conv:.1f}%", "green" if conv > 15 else "")
mcard(c4, "📅", "Weekend Visit", f"{int(df_f['Weekend'].sum()):,}")
mcard(c5, "📄", "Avg PageValues", f"{df_f['PageValues'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ================================
# Tabs
# ================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Analysis", "🤖 Machine Learning", "🎯 Prediksi Interaktif", "💡 Insights"])

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Plus Jakarta Sans'),
    title_font=dict(color='#e2e8f0', size=15, family='Plus Jakarta Sans'),
    margin=dict(t=40, b=20, l=10, r=10),
    xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
)

# ================================
# TAB 1: EDA
# ================================
with tab1:
    # Section 1
    st.markdown('<div class="sec-header"><div class="sec-title">Distribusi & Tren Pembelian</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        rev = df_f['Revenue'].value_counts().reset_index()
        rev.columns = ['Revenue','Count']
        rev['Revenue'] = rev['Revenue'].map({True:'Beli', False:'Tidak Beli'})
        fig = px.pie(rev, names='Revenue', values='Count', hole=0.55,
                     color_discrete_sequence=['#3b82f6','#1e3a5f'],
                     title='Distribusi Revenue')
        fig.update_layout(**CHART_LAYOUT)
        fig.update_traces(textfont_color='white')
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        monthly = df_f[df_f['Revenue']==True]['Month'].value_counts().reindex(available_months).reset_index()
        monthly.columns = ['Month','Count']
        fig2 = px.bar(monthly, x='Month', y='Count', title='Transaksi per Bulan',
                      color='Count', color_continuous_scale=[[0,'#1e3a5f'],[1,'#3b82f6']])
        fig2.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        wk = df_f.groupby('Weekend')['Revenue'].mean().reset_index()
        wk['Weekend'] = wk['Weekend'].map({True:'Weekend', False:'Weekday'})
        wk['Revenue'] = wk['Revenue'] * 100
        fig3 = px.bar(wk, x='Weekend', y='Revenue', title='Konversi: Weekday vs Weekend',
                      color='Weekend', color_discrete_sequence=['#3b82f6','#60a5fa'],
                      text=wk['Revenue'].apply(lambda x: f'{x:.1f}%'))
        fig3.update_traces(textposition='outside', textfont_color='#94a3b8')
        fig3.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2
    st.markdown('<div class="sec-header"><div class="sec-title">Perilaku Pengunjung</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        vis = df_f.groupby('VisitorType')['Revenue'].value_counts().reset_index()
        vis.columns = ['VisitorType','Revenue','Count']
        vis['Revenue'] = vis['Revenue'].map({True:'Beli', False:'Tidak Beli'})
        fig4 = px.bar(vis, x='VisitorType', y='Count', color='Revenue', barmode='group',
                      title='Visitor Type vs Revenue',
                      color_discrete_sequence=['#3b82f6','#1e3a5f'])
        fig4.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        sample = df_f.sample(min(800, len(df_f)))
        fig5 = px.scatter(sample, x='BounceRates', y='ExitRates',
                          color=sample['Revenue'].map({True:'Beli', False:'Tidak Beli'}),
                          title='Bounce Rate vs Exit Rate',
                          color_discrete_sequence=['#3b82f6','#ef4444'],
                          opacity=0.5, size_max=6)
        fig5.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3
    st.markdown('<div class="sec-header"><div class="sec-title">Page Values & Special Day</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)

    with col6:
        fig6 = px.box(df_f, x='Month', y='PageValues',
                      color=df_f['Revenue'].map({True:'Beli', False:'Tidak Beli'}),
                      category_orders={'Month': available_months},
                      title='PageValues per Bulan',
                      color_discrete_sequence=['#3b82f6','#1e3a5f'])
        fig6.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col7:
        sp = df_f.groupby('SpecialDay')['Revenue'].mean().reset_index()
        sp['Revenue'] = sp['Revenue'] * 100
        fig7 = px.line(sp, x='SpecialDay', y='Revenue',
                       title='Konversi Rate vs Special Day',
                       markers=True, color_discrete_sequence=['#3b82f6'])
        fig7.update_traces(line_width=2.5, marker_size=8)
        fig7.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap korelasi
    st.markdown('<div class="sec-header"><div class="sec-title">Korelasi Fitur</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    corr = df_f.select_dtypes(include='number').corr()
    fig8 = px.imshow(corr, text_auto='.2f', title='Heatmap Korelasi Antar Fitur',
                     color_continuous_scale=[[0,'#060d1f'],[0.5,'#1e3a5f'],[1,'#3b82f6']],
                     aspect='auto')
    fig8.update_layout(**CHART_LAYOUT)
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# TAB 2: ML
# ================================
with tab2:
    st.markdown('<div class="sec-header"><div class="sec-title">Model Performance</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    mcard(m1, "🌲", "Random Forest Accuracy", f"{acc_rf*100:.1f}%", "green")
    mcard(m2, "📈", "Logistic Regression Accuracy", f"{acc_lr*100:.1f}%", "blue")
    mcard(m3, "🏆", "Best Model", "Random Forest")
    mcard(m4, "📊", "Test Data Size", "2,466")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="sec-header"><div class="sec-title">Feature Importance</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
        # Chart
        fi_df = feat_imp.reset_index()
        fi_df.columns = ['Feature','Importance']
        fig9 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=[[0,'#1e3a5f'],[1,'#3b82f6']],
                      title='')
        layout9 = {**CHART_LAYOUT, 'yaxis': {'categoryorder': 'total ascending', 'gridcolor': '#1e3a5f'}}
        fig9.update_layout(**layout9)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-header"><div class="sec-title">Ranking Fitur Terpenting</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
        max_imp = feat_imp.max()
        rows_html = ""
        for i, (feat, imp) in enumerate(feat_imp.items(), 1):
            pct = int(imp / max_imp * 100)
            rows_html += f"""
            <div class="feat-row">
                <div class="feat-rank">#{i}</div>
                <div class="feat-name">{feat}</div>
                <div class="feat-bar-bg"><div class="feat-bar" style="width:{pct}%"></div></div>
                <div class="feat-pct">{imp:.3f}</div>
            </div>"""
        st.markdown(f'<div class="chart-box" style="padding:20px">{rows_html}</div>', unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<div class="sec-header"><div class="sec-title">Confusion Matrix — Logistic Regression</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    col3, col4 = st.columns([1,2])
    with col3:
        fig10 = px.imshow(cm, text_auto=True,
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Tidak Beli','Beli'], y=['Tidak Beli','Beli'],
                          color_continuous_scale=[[0,'#060d1f'],[1,'#3b82f6']])
        fig10.update_layout(**CHART_LAYOUT)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        tn,fp,fn,tp = cm.ravel()
        total = tn+fp+fn+tp
        st.markdown(f"""
        <div class="chart-box" style="padding:24px">
            <div class="feat-row"><div class="feat-name">✅ True Negative (Benar tidak beli)</div><div class="mcard-value blue" style="font-size:20px">{tn:,}</div></div>
            <div class="feat-row"><div class="feat-name">✅ True Positive (Benar beli)</div><div class="mcard-value green" style="font-size:20px">{tp:,}</div></div>
            <div class="feat-row"><div class="feat-name">❌ False Positive (Salah prediksi beli)</div><div class="mcard-value" style="font-size:20px;color:#ef4444">{fp:,}</div></div>
            <div class="feat-row" style="border:none"><div class="feat-name">❌ False Negative (Salah prediksi tidak beli)</div><div class="mcard-value" style="font-size:20px;color:#f59e0b">{fn:,}</div></div>
        </div>
        """, unsafe_allow_html=True)

# ================================
# TAB 3: Prediksi Interaktif
# ================================
with tab3:
    st.markdown('<div class="sec-header"><div class="sec-title">Prediksi Pembelian — Input Manual</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:13px;margin-bottom:20px">Masukkan data pengunjung, model Random Forest akan memprediksi apakah mereka akan melakukan pembelian.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📄 Aktivitas Halaman**")
        administrative = st.slider("Administrative Pages", 0, 27, 2)
        administrative_duration = st.slider("Administrative Duration (s)", 0.0, 3000.0, 80.0)
        informational = st.slider("Informational Pages", 0, 24, 1)
        informational_duration = st.slider("Informational Duration (s)", 0.0, 2500.0, 30.0)
        product_related = st.slider("Product Related Pages", 0, 705, 20)
        product_related_duration = st.slider("Product Duration (s)", 0.0, 63974.0, 1000.0)

    with col2:
        st.markdown("**📊 Metrik Halaman**")
        bounce_rates = st.slider("Bounce Rate", 0.0, 0.2, 0.02)
        exit_rates = st.slider("Exit Rate", 0.0, 0.2, 0.04)
        page_values = st.slider("Page Values", 0.0, 361.0, 10.0)
        special_day = st.slider("Special Day", 0.0, 1.0, 0.0)

    with col3:
        st.markdown("**👤 Info Pengunjung**")
        month = st.selectbox("Bulan", ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        operating_systems = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
        browser = st.selectbox("Browser", list(range(1,14)))
        region = st.selectbox("Region", list(range(1,10)))
        traffic_type = st.selectbox("Traffic Type", list(range(1,21)))
        visitor_type = st.selectbox("Visitor Type", ['Returning_Visitor','New_Visitor','Other'])
        weekend = st.radio("Weekend?", [False, True], format_func=lambda x: "Ya" if x else "Tidak")

    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        # Encode input
        month_map = {'Feb':0,'Mar':2,'May':3,'Jun':1,'Jul':4,'Aug':5,'Sep':6,'Oct':7,'Nov':8,'Dec':9}
        visitor_map = {'New_Visitor':0,'Other':1,'Returning_Visitor':2}

        input_data = np.array([[
            administrative, administrative_duration,
            informational, informational_duration,
            product_related, product_related_duration,
            bounce_rates, exit_rates, page_values, special_day,
            month_map.get(month, 0), operating_systems, browser,
            region, traffic_type, visitor_map.get(visitor_type, 2),
            int(weekend)
        ]])

        pred = rf_model.predict(input_data)[0]
        prob = rf_model.predict_proba(input_data)[0]

        if pred == 1:
            st.markdown(f"""
            <div class="predict-result">
                <div class="predict-label">Hasil Prediksi</div>
                <div class="predict-value yes">✅ KEMUNGKINAN BELI</div>
                <div style="color:#10b981;font-size:14px;margin-top:8px">Probabilitas: {prob[1]*100:.1f}%</div>
                <div style="color:#64748b;font-size:12px;margin-top:6px">Pengunjung ini diprediksi akan melakukan pembelian</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-result" style="border-color:#ef4444">
                <div class="predict-label">Hasil Prediksi</div>
                <div class="predict-value no">❌ KEMUNGKINAN TIDAK BELI</div>
                <div style="color:#ef4444;font-size:14px;margin-top:8px">Probabilitas beli: {prob[1]*100:.1f}%</div>
                <div style="color:#64748b;font-size:12px;margin-top:6px">Pengunjung ini diprediksi tidak melakukan pembelian</div>
            </div>""", unsafe_allow_html=True)

# ================================
# TAB 4: Insights
# ================================
with tab4:
    st.markdown('<div class="sec-header"><div class="sec-title">Key Business Insights</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    top_month = df[df['Revenue']==True]['Month'].value_counts().idxmax()
    top_visitor = df[df['Revenue']==True]['VisitorType'].value_counts().idxmax()
    weekend_conv = df[df['Weekend']==True]['Revenue'].mean()*100
    weekday_conv = df[df['Weekend']==False]['Revenue'].mean()*100
    top_feature = feat_imp.index[0]
    top2_feature = feat_imp.index[1]

    insights = [
        ("📅", f"Bulan <strong>{top_month}</strong> mencatat transaksi tertinggi — jadwalkan kampanye promosi & flash sale di bulan ini untuk maksimalkan konversi."),
        ("👤", f"<strong>{top_visitor}</strong> mendominasi pembelian. Strategi retensi pelanggan lama lebih efektif dibanding akuisisi pelanggan baru."),
        ("📆", f"Konversi <strong>Weekend ({weekend_conv:.1f}%)</strong> vs <strong>Weekday ({weekday_conv:.1f}%)</strong> — {'Fokuskan iklan berbayar di akhir pekan untuk ROI lebih tinggi.' if weekend_conv > weekday_conv else 'Hari kerja lebih produktif — alokasikan budget iklan di weekday.'}"),
        ("🏆", f"Fitur <strong>{top_feature}</strong> dan <strong>{top2_feature}</strong> adalah prediktor terkuat pembelian. Optimalkan kedua aspek ini di UI/UX website."),
        ("🤖", f"Model Random Forest mencapai akurasi <strong>{acc_rf*100:.1f}%</strong> — cukup andal untuk sistem rekomendasi atau scoring lead pelanggan."),
        ("⚡", f"Hanya <strong>15.5%</strong> pengunjung yang konversi — masih ada ruang optimasi besar melalui perbaikan checkout flow, personalisasi, dan retargeting."),
        ("📊", f"<strong>BounceRate & ExitRate</strong> tinggi berkorelasi negatif dengan pembelian — perbaiki landing page dan kecepatan loading untuk mengurangi bounce."),
    ]

    for icon, text in insights:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-icon">{icon}</div>
            <div class="insight-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header"><div class="sec-title">Raw Data Preview</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.dataframe(df_f.head(100), use_container_width=True)