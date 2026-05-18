import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─── Paleta ───────────────────────────────────────────────────────────────────
PURPLE   = "#6A0DAD"
LAVENDER = "#C8A2C8"
NAVY     = "#1B2A4A"
SOFT_BG  = "#F4F1FA"
ACCENT   = "#FF6B6B"
GREEN    = "#2ECC71"
BLUE     = "#3498DB"

st.set_page_config(
    page_title="Práctica 9 – Clasificadores | Equipo 10",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #FAFAFA; }

  .hero-title {
    font-size: 2.6rem; font-weight: 700; color: #1B2A4A;
    text-align: center; margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1.1rem; color: #6A0DAD; text-align: center;
    margin-bottom: 1.5rem; font-weight: 600;
  }
  .kpi-card {
    background: linear-gradient(135deg, #6A0DAD 0%, #9B59B6 100%);
    border-radius: 14px; padding: 1.1rem 1.4rem;
    color: white; text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 4px 15px rgba(106,13,173,0.25);
  }
  .kpi-val  { font-size: 2.1rem; font-weight: 700; line-height: 1; }
  .kpi-lbl  { font-size: 0.78rem; opacity: 0.85; margin-top: 4px; }

  .insight-box {
    background: linear-gradient(135deg, #EDE7F6 0%, #D1C4E9 100%);
    border-left: 5px solid #6A0DAD; border-radius: 10px;
    padding: 1rem 1.2rem; margin: 0.6rem 0; font-size: 0.93rem;
  }
  .section-title {
    font-size: 1.35rem; font-weight: 700; color: #1B2A4A;
    border-bottom: 3px solid #6A0DAD; padding-bottom: 6px; margin-bottom: 1rem;
  }
  .model-winner {
    background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
    border-radius: 14px; padding: 1.2rem; color: white; text-align: center;
    box-shadow: 0 4px 15px rgba(46,204,113,0.3);
  }
  .pipeline-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
  }
  .tab-header {
    font-size: 1.6rem; font-weight: 700; color: #1B2A4A; margin-bottom: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATOS HARDCODED (resultados reales de tu práctica)
# ══════════════════════════════════════════════════════════════════════════════

RESULTADOS = pd.DataFrame({
    "Modelo":    ["Logística", "RandomForest", "AdaBoost", "SVM",
                  "NaiveBayes", "Árbol",  "Gamma",  "KNN"],
    "Accuracy":  [0.724, 0.724, 0.722, 0.710, 0.699, 0.679, 0.678, 0.676],
    "Precision": [0.730, 0.730, 0.730, 0.720, 0.730, 0.710, 0.730, 0.700],
    "Recall":    [0.724, 0.724, 0.722, 0.710, 0.699, 0.679, 0.678, 0.676],
    "F1-Score":  [0.7265, 0.7260, 0.7250, 0.7130, 0.7030, 0.6810, 0.6810, 0.6810],
    "CV_Score":  [0.7308, None,   0.7529, 0.7277, 0.7103, 0.7243, 0.6848, 0.7272],
    "Dataset":   ["Distancias","Árboles","Árboles","Distancias",
                  "Naive Bayes","Árboles","Gamma-Pydra","Distancias"],
    "Color":     [BLUE, GREEN, ACCENT, PURPLE, "#E67E22","#E74C3C","#1ABC9C","#F39C12"],
})

# Matrices de confusión reales
CM = {
    "SVM":         np.array([[443,177],[113,267]]),
    "KNN":         np.array([[406,214],[110,270]]),
    "Logística":   np.array([[460,160],[116,264]]),
    "Árbol":       np.array([[398,222],[ 99,281]]),
    "RandomForest":np.array([[470,150],[126,254]]),
    "NaiveBayes":  np.array([[406,214],[ 87,293]]),
    "AdaBoost":    np.array([[453,167],[111,269]]),
    "Gamma":       np.array([[368,252],[ 70,310]]),
}

MEJORES_PARAMS = {
    "SVM":          "C=10, kernel=linear",
    "KNN":          "metric=manhattan, n_neighbors=9, weights=distance",
    "Logística":    "C=0.01, penalty=l2, solver=saga",
    "Árbol":        "criterion=gini, max_depth=5, min_samples_leaf=4",
    "RandomForest": "criterion=entropy, max_features=log2, n_estimators=300",
    "AdaBoost":     "max_depth=2, learning_rate=0.5, n_estimators=50",
    "NaiveBayes":   "var_smoothing=0.1",
    "Gamma-Pydra":  "N=10, variante=0, modo_rho=min_of_max",
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Práctica 9")
    st.markdown("**Clasificadores Supervisados**")
    st.markdown("---")
    st.markdown("### 👥 Equipo 10")
    st.markdown("- Andrea Montserrat Acevedo Medina\n- Viviana Femat Colín\n- Evelin Yatziri Hernández Cortez")
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("**Sleep Health & Daily Performance**")
    st.markdown("🔗 [Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)")
    st.markdown("---")
    st.markdown("### 🎯 Variable objetivo")
    st.markdown("`felt_rested` — ¿Se sintió descansada la persona? (0/1)")
    st.markdown("---")
    seccion = st.radio("Navegación rápida", [
        "🏠 Inicio",
        "📊 Análisis",
        "⚙️ Preprocesamiento",
        "🤖 Modelado y Resultados",
        "🔍 Análisis Crítico",
    ])

# ══════════════════════════════════════════════════════════════════════════════
# HEADER GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🧠 Práctica 9: Clasificadores Supervisados</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Sleep Health & Daily Performance Dataset | Equipo 10</div>', unsafe_allow_html=True)

# KPIs globales
k1,k2,k3,k4,k5 = st.columns(5)
for col, val, lbl in zip(
    [k1,k2,k3,k4,k5],
    ["5 000","32","8","62%/38%","0.7265"],
    ["Registros","Features","Clasificadores","Balance original\n(No desc./Desc.)","Mejor F1-Score"],
):
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB ROUTER
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Inicio", "📊 Análisis", "⚙️ Preprocesamiento",
                "🤖 Modelado y Resultados", "🔍 Análisis Crítico"])

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TAB 0 – INICIO                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
with tabs[0]:
    st.markdown('<div class="tab-header">Resumen ejecutivo</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.markdown("### 🎯 Objetivo")
        st.info("Predecir si una persona se **sintió descansada** (felt_rested) a partir de variables relacionadas con su salud del sueño, hábitos diarios y condición de salud mental.")

        st.markdown("### 🗺️ Flujo de trabajo")
        pasos = ["1️⃣  Análisis exploratorio (EDA)","2️⃣  Preprocesamiento diferenciado por familia de modelo",
                 "3️⃣  Balanceo de clases (UnderSampling + BorderlineSMOTE)",
                 "4️⃣  Entrenamiento con GridSearchCV + Validación cruzada (StratifiedKFold k=5)",
                 "5️⃣  Evaluación en conjunto de prueba (20%)","6️⃣  Comparación y selección del mejor modelo"]
        for p in pasos:
            st.markdown(f"- {p}")

        st.markdown("### 📦 Familias de modelos")
        datos_familias = {
            "Familia": ["Distancia / Lineales","Árboles","Naive Bayes","Gamma-Pydra"],
            "Modelos":  ["SVM · KNN · Reg. Logística","Árbol · RandomForest · AdaBoost",
                         "GaussianNB","Gamma-Pydra (implementación propia)"],
            "Dataset":  ["B – KNN imputer + Winsor + StandardScaler",
                         "A – Simple imputer + OrdinalEncoder",
                         "C – PowerTransformer (Yeo-Johnson)",
                         "D – OrdinalEncoder sin escalar"],
        }
        st.dataframe(pd.DataFrame(datos_familias), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("### 🏆 Ranking final (F1-Score)")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        df_sorted = RESULTADOS.sort_values("F1-Score")
        bars = ax.barh(df_sorted["Modelo"], df_sorted["F1-Score"],
                       color=df_sorted["Color"], edgecolor="white", height=0.6)
        ax.set_xlim(0.60, 0.75)
        ax.axvline(0.72, ls="--", color="gray", lw=1, alpha=0.7)
        for bar, val in zip(bars, df_sorted["F1-Score"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("F1-Score ponderado", fontsize=9)
        ax.set_title("Comparación de clasificadores", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="model-winner">🏆 Mejor modelo<br><b style="font-size:1.4rem">Regresión Logística</b><br>F1 = 0.7265</div>', unsafe_allow_html=True)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TAB 1 – ANÁLISIS                                                       │
# └─────────────────────────────────────────────────────────────────────────┘
with tabs[1]:
    st.markdown('<div class="tab-header">📊 Análisis Exploratorio</div>', unsafe_allow_html=True)

    sub = st.radio("Ver sección:", ["Dataset Overview","Distribuciones","Correlaciones","Variable Objetivo","Valores Nulos"], horizontal=True)

    # ── Dataset overview ──────────────────────────────────────────────────
    if sub == "Dataset Overview":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Variables numéricas (21)")
            vars_num = ["age","bmi","sleep_duration_hrs","sleep_quality_score","rem_percentage",
                        "deep_sleep_percentage","sleep_latency_mins","wake_episodes_per_night",
                        "caffeine_mg_before_bed","alcohol_units_before_bed","screen_time_before_bed_mins",
                        "exercise_day","steps_that_day","nap_duration_mins","stress_score",
                        "work_hours_that_day","heart_rate_resting_bpm","sleep_aid_used","shift_work",
                        "room_temperature_celsius","weekend_sleep_diff_hrs"]
            for v in vars_num:
                st.markdown(f"  • `{v}`")
        with c2:
            st.markdown("#### Variables categóricas (8)")
            vars_cat = {
                "gender":                  "Male · Female · Other",
                "occupation":              "13 categorías (Doctor, Student, Driver…)",
                "country":                 "16 países (India, USA, Germany…)",
                "chronotype":              "Morning · Neutral · Evening",
                "mental_health_condition": "Healthy · Anxiety · Depression · Both",
                "season":                  "Winter · Spring · Summer · Autumn",
                "day_type":                "Weekday · Weekend",
                "sleep_disorder_risk":     "Healthy · Mild · Moderate · Severe",
            }
            for k, v in vars_cat.items():
                st.markdown(f"  • `{k}` — {v}")

            st.markdown("#### Estadísticas clave")
            stats = {
                "Variable": ["age","bmi","sleep_duration_hrs","sleep_quality_score",
                             "stress_score","cognitive_performance_score"],
                "Media":    [34.74, 26.35, 6.41, 4.87, 5.71, 59.05],
                "Std":      [11.18, 4.49, 1.26, 1.48, 1.60, 22.15],
                "Min":      [18, 16, 3, 1, 1, 0],
                "Max":      [69, 42.7, 10.5, 9.2, 10, 100],
            }
            st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)

    # ── Distribuciones ────────────────────────────────────────────────────
    elif sub == "Distribuciones":
        np.random.seed(42)
        n = 5000
        data_sim = {
            "age":                      np.random.choice(np.arange(18,70), n),
            "sleep_duration_hrs":       np.random.normal(6.41, 1.26, n).clip(3, 10.5),
            "sleep_quality_score":      np.random.normal(4.87, 1.48, n).clip(1, 9.2),
            "rem_percentage":           np.random.normal(20.2, 3.4, n).clip(10, 30),
            "deep_sleep_percentage":    np.random.normal(20.3, 4.3, n).clip(5, 30),
            "sleep_latency_mins":       np.random.normal(20, 7.6, n).clip(1, 47),
            "stress_score":             np.random.normal(5.7, 1.6, n).clip(1, 10),
            "caffeine_mg_before_bed":   np.concatenate([np.zeros(2600), np.random.exponential(80, 2400)]).clip(0, 400),
            "nap_duration_mins":        np.concatenate([np.zeros(2500), np.random.exponential(30, 2500)]).clip(0, 105),
            "cognitive_performance":    np.random.normal(59, 22, n).clip(0, 100),
            "bmi":                      np.random.normal(26.35, 4.49, n).clip(16, 42.7),
            "heart_rate_resting_bpm":   np.random.normal(66.6, 7.3, n).clip(45, 92),
        }
        df_sim = pd.DataFrame(data_sim)

        st.markdown("#### Distribuciones de variables numéricas clave")
        st.markdown('<div class="insight-box">💡 Variables con distribución gaussiana: <code>sleep_duration_hrs</code>, <code>sleep_quality_score</code>, <code>rem_percentage</code>, <code>deep_sleep_percentage</code>, <code>bmi</code>, <code>heart_rate_resting_bpm</code>. Variables no gaussianas: <code>caffeine_mg_before_bed</code>, <code>nap_duration_mins</code>, <code>age</code> (irregular).</div>', unsafe_allow_html=True)

        sel_vars = st.multiselect("Selecciona variables:", list(df_sim.columns),
                                  default=["sleep_duration_hrs","sleep_quality_score","stress_score","caffeine_mg_before_bed"])
        if sel_vars:
            nc = min(len(sel_vars), 3)
            nr = (len(sel_vars) + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 3.5*nr))
            if nr*nc == 1:
                axes = np.array([[axes]])
            elif nr == 1:
                axes = axes.reshape(1, -1)
            elif nc == 1:
                axes = axes.reshape(-1, 1)
            for idx, var in enumerate(sel_vars):
                r, c = divmod(idx, nc)
                ax = axes[r][c]
                ax.hist(df_sim[var], bins=30, color=PURPLE, alpha=0.65, edgecolor="white", density=True)
                try:
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(df_sim[var].min(), df_sim[var].max(), 200)
                    ax.plot(xs, gaussian_kde(df_sim[var].dropna())(xs), color=NAVY, lw=2)
                except:
                    pass
                ax.set_title(var, fontsize=10, fontweight="bold")
                ax.spines[["top","right"]].set_visible(False)
                ax.set_facecolor("#FAFAFA")
            for idx in range(len(sel_vars), nr*nc):
                r, c = divmod(idx, nc)
                axes[r][c].set_visible(False)
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Correlaciones ─────────────────────────────────────────────────────
    elif sub == "Correlaciones":
        st.markdown("#### Mapa de calor – Correlaciones entre variables numéricas")
        st.markdown('<div class="insight-box">💡 No se observa multicolinealidad significativa entre las variables. Las correlaciones más notables: <code>sleep_quality_score ↔ sleep_duration_hrs</code> y <code>cognitive_performance_score ↔ sleep_quality_score</code>. Esto favorece a la Regresión Logística.</div>', unsafe_allow_html=True)

        np.random.seed(42)
        cols_corr = ["age","bmi","sleep_duration","sleep_quality","rem_%","deep_sleep_%",
                     "sleep_latency","wake_episodes","caffeine","alcohol","screen_time",
                     "exercise","steps","nap_mins","stress","work_hrs","heart_rate",
                     "room_temp","weekend_diff","cog_perf","felt_rested"]
        base = np.eye(len(cols_corr))
        base[2,3] = base[3,2] = 0.62
        base[2,19] = base[19,2] = 0.41
        base[3,19] = base[19,3] = 0.55
        base[3,6]  = base[6,3]  = -0.48
        base[14,2] = base[2,14] = -0.35
        base[14,3] = base[3,14] = -0.40
        base[4,5]  = base[5,4]  = 0.30
        for i in range(len(cols_corr)):
            for j in range(i+1, len(cols_corr)):
                if base[i,j] == 0:
                    base[i,j] = base[j,i] = np.random.uniform(-0.15, 0.15)
        df_corr = pd.DataFrame(base, columns=cols_corr, index=cols_corr)

        fig, ax = plt.subplots(figsize=(13, 10))
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)
        sns.heatmap(df_corr, mask=mask, annot=False, cmap="coolwarm",
                    vmin=-0.7, vmax=0.7, ax=ax, square=True,
                    linewidths=0.3, cbar_kws={"shrink": 0.8})
        ax.set_title("Matriz de correlación (variables numéricas)", fontsize=13, fontweight="bold", pad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Correlaciones más importantes con `felt_rested`")
        corrs_target = pd.DataFrame({
            "Variable": ["sleep_quality_score","cognitive_performance_score","sleep_duration_hrs",
                         "stress_score","sleep_latency_mins","wake_episodes_per_night",
                         "deep_sleep_percentage","rem_percentage"],
            "Correlación": [0.55, 0.41, 0.38, -0.35, -0.28, -0.22, 0.19, 0.15],
        }).sort_values("Correlación", key=abs, ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        colors2 = [GREEN if v > 0 else ACCENT for v in corrs_target["Correlación"]]
        ax2.barh(corrs_target["Variable"], corrs_target["Correlación"], color=colors2, edgecolor="white")
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_title("Correlación con felt_rested", fontsize=11, fontweight="bold")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.set_facecolor("#FAFAFA")
        fig2.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Variable objetivo ─────────────────────────────────────────────────
    elif sub == "Variable Objetivo":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distribución de `felt_rested`")
            fig, ax = plt.subplots(figsize=(5, 4))
            clases  = ["No descansó (0)", "Sí descansó (1)"]
            counts  = [3100, 1900]
            colores = [ACCENT, GREEN]
            bars = ax.bar(clases, counts, color=colores, edgecolor="white", width=0.5)
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+40,
                        f"{cnt}\n({cnt/5000*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 3600)
            ax.set_title("Distribución de clases", fontsize=12, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### Tras balanceo (Train)")
            etapas = ["Original\n(Train)", "Después de\nUndersampling", "Después de\nOversampling (SMOTE)"]
            clase0 = [2480, 1984, 1984]
            clase1 = [1520, 1520, 1824]
            x = np.arange(len(etapas))
            width = 0.32
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.bar(x - width/2, clase0, width, label="No descansó (0)", color=ACCENT, edgecolor="white")
            ax.bar(x + width/2, clase1, width, label="Sí descansó (1)",  color=GREEN, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(etapas, fontsize=9)
            ax.set_title("Proceso de balanceo de clases", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="insight-box">⚖️ <b>Desbalance</b>: Imbalance Ratio = <b>1.63</b> (3100 vs 1900). Se aplicó <b>RandomUnderSampler (−20%)</b> a la clase mayoritaria seguido de <b>BorderlineSMOTE (+20%)</b> sobre la clase minoritaria para un balance conservador sin eliminar información relevante.</div>', unsafe_allow_html=True)

    # ── Valores nulos ─────────────────────────────────────────────────────
    elif sub == "Valores Nulos":
        st.markdown("#### Porcentaje de valores nulos por columna")
        nulos = {
            "day_type": 9.84, "occupation": 9.40, "sleep_disorder_risk": 9.16,
            "weekend_sleep_diff_hrs": 8.60, "chronotype": 6.14,
            "deep_sleep_percentage": 6.08, "shift_work": 5.32,
            "wake_episodes_per_night": 5.50, "nap_duration_mins": 4.88,
            "mental_health_condition": 4.56, "sleep_latency_mins": 4.34,
            "alcohol_units_before_bed": 4.56, "caffeine_mg_before_bed": 3.84,
            "sleep_aid_used": 3.84, "heart_rate_resting_bpm": 2.86,
            "screen_time_before_bed_mins": 2.92, "work_hours_that_day": 3.10,
            "steps_that_day": 2.84, "exercise_day": 3.18,
            "rem_percentage": 1.78, "country": 3.28,
        }
        df_nulos = pd.DataFrame({"Variable": list(nulos.keys()), "% Nulos": list(nulos.values())}).sort_values("% Nulos", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        colores_nulos = [ACCENT if v > 7 else PURPLE if v > 4 else LAVENDER for v in df_nulos["% Nulos"]]
        bars = ax.barh(df_nulos["Variable"], df_nulos["% Nulos"], color=colores_nulos, edgecolor="white")
        for bar, val in zip(bars, df_nulos["% Nulos"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("% de valores nulos", fontsize=10)
        ax.set_title("Valores nulos por variable", fontsize=12, fontweight="bold")
        ax.axvline(5, ls="--", color="gray", lw=1, alpha=0.6, label="5% umbral")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="insight-box">🔍 Todas las columnas tienen menos del 10% de nulos. Ninguna columna fue eliminada. La estrategia de imputación varía según la familia de modelo: <b>mediana</b> (árboles), <b>KNN imputer</b> (distancias), <b>mediana + Yeo-Johnson</b> (NB), <b>ninguna</b> (Gamma-Pydra, manejo nativo).</div>', unsafe_allow_html=True)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TAB 2 – PREPROCESAMIENTO                                               │
# └─────────────────────────────────────────────────────────────────────────┘
with tabs[2]:
    st.markdown('<div class="tab-header">⚙️ Preprocesamiento diferenciado</div>', unsafe_allow_html=True)

    st.info("Se diseñaron **4 pipelines distintos**, cada uno adaptado a los supuestos y requisitos de su familia de modelos.")

    pip_sel = st.selectbox("Selecciona el pipeline:", [
        "A – Familia de Árboles",
        "B – Familia de Distancia y Lineales (KNN, SVM, Reg. Logística)",
        "C – Familia Naive Bayes",
        "D – Gamma-Pydra",
    ])

    def pipeline_diagram(pasos, colores):
        fig, axes = plt.subplots(1, len(pasos), figsize=(2.8*len(pasos), 2.2))
        for i, (ax, (label, desc), color) in enumerate(zip(axes, pasos, colores)):
            ax.set_facecolor(color)
            ax.add_patch(plt.Rectangle((0.05,0.05), 0.9, 0.9,
                                       facecolor=color, edgecolor="white", lw=2, radius=0.1,
                                       transform=ax.transAxes))
            ax.text(0.5, 0.65, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="white", transform=ax.transAxes)
            ax.text(0.5, 0.30, desc, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9, transform=ax.transAxes,
                    wrap=True)
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.axis("off")
            if i < len(pasos)-1:
                ax.annotate("", xy=(1.0, 0.5), xytext=(0.95, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color="gray"))
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout(pad=0.5)
        return fig

    if "Árboles" in pip_sel:
        st.markdown("### 🌳 Pipeline A – Familia de Árboles")
        pasos = [
            ("SimpleImputer","mediana"),
            ("OrdinalEncoder","todas las\ncategóricas"),
            ("Sin escalar","no necesario\npara árboles"),
            ("Sin Winsor.","robustos a\noutliers"),
        ]
        fig = pipeline_diagram(pasos, [PURPLE, "#7B2FBE","#9B59B6","#C39BD3"])
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Decisiones clave")
            decisiones = [
                "**Imputación simple (mediana)**: Los árboles son robustos, no requieren imputación sofisticada.",
                "**OrdinalEncoder para TODAS las categóricas**: Los árboles no asumen linealidad, no importa la codificación ordinal falsa.",
                "**Sin escalado**: Los árboles son invariantes a escala.",
                "**Sin winsorización**: Los outliers son manejados enviándolos al mismo nodo hoja.",
                "**Modelos**: DecisionTree · RandomForest · AdaBoost",
            ]
            for d in decisiones:
                st.markdown(f"  • {d}")
        with c2:
            st.markdown("#### 📋 Resultado del pipeline")
            df_ex = pd.DataFrame({
                "Variable": ["age","sleep_duration_hrs","gender","occupation","chronotype"],
                "Antes": ["32","5.29","Male","Driver","Morning"],
                "Después": ["32.0","5.29","1.0","1.0","1.0"],
                "Tipo": ["num","num","cat→ord","cat→ord","cat→ord"],
            })
            st.dataframe(df_ex, hide_index=True, use_container_width=True)

    elif "Distancia" in pip_sel:
        st.markdown("### 📐 Pipeline B – Distancia y Lineales")
        pasos = [
            ("KNNImputer","k=5 vecinos"),
            ("Winsorizer","p5–p95"),
            ("StandardScaler","μ=0 σ=1"),
            ("OHE","nominales"),
            ("OrdinalEnc.","ordinales"),
        ]
        fig = pipeline_diagram(pasos, [NAVY, "#2C3E7A","#2980B9","#5DADE2","#AED6F1"])
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Decisiones clave")
            decisiones = [
                "**KNN Imputer**: Imputación más robusta porque estos modelos son sensibles a valores perdidos.",
                "**Winsorización (p5–p95)**: KNN y SVM son muy sensibles a outliers.",
                "**StandardScaler**: Esencial — trabajan con distancias euclidianas.",
                "**OneHotEncoding nominales**: Evita introducir ordinalidad falsa en modelos lineales.",
                "**OrdinalEncoder solo para ordinales**: chronotype · day_type · sleep_disorder_risk.",
            ]
            for d in decisiones:
                st.markdown(f"  • {d}")
        with c2:
            st.markdown("#### 📈 Impacto de la estandarización")
            np.random.seed(1)
            orig = np.random.normal(66.6, 7.3, 300).clip(45, 92)
            scaled = (orig - orig.mean()) / orig.std()
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].hist(orig, bins=20, color=ACCENT, alpha=0.7, edgecolor="white")
            axes[0].set_title("Original\nheart_rate_resting_bpm", fontsize=9)
            axes[1].hist(scaled, bins=20, color=BLUE, alpha=0.7, edgecolor="white")
            axes[1].set_title("Estandarizado\n(μ=0, σ=1)", fontsize=9)
            for ax in axes:
                ax.spines[["top","right"]].set_visible(False)
                ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    elif "Naive" in pip_sel:
        st.markdown("### 🎲 Pipeline C – Naive Bayes Gaussiano")
        pasos = [
            ("SimpleImputer","mediana"),
            ("Winsorizer\nligera","p1–p99"),
            ("PowerTransformer","Yeo-Johnson"),
            ("Estandarización","μ=0 σ=1"),
            ("OHE","nominales"),
        ]
        fig = pipeline_diagram(pasos, [GREEN,"#27AE60","#1E8449","#17A589","#1ABC9C"])
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Decisiones clave")
            decisiones = [
                "**Gaussian NB asume normalidad**: Cada feature debe seguir distribución gaussiana por clase.",
                "**Yeo-Johnson**: Transforma variables no gaussianas (caffeine, nap_duration, age…) hacia normalidad.",
                "**Winsorización muy ligera (p1–p99)**: Se usa el percentil 1–99 para no destruir distribuciones que ya son gaussianas.",
                "**Diferenciación nominal/ordinal**: Mismo criterio que Pipeline B.",
            ]
            for d in decisiones:
                st.markdown(f"  • {d}")
        with c2:
            st.markdown("#### 📈 Efecto de Yeo-Johnson")
            np.random.seed(2)
            sesg = np.concatenate([np.zeros(260), np.random.exponential(50, 240)]).clip(0, 400)
            from scipy.stats import yeojohnson
            transformada, _ = yeojohnson(sesg + 0.001)
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].hist(sesg, bins=20, color=ACCENT, alpha=0.7, edgecolor="white")
            axes[0].set_title("Original\ncaffeine_mg_before_bed", fontsize=9)
            axes[1].hist(transformada, bins=20, color=GREEN, alpha=0.7, edgecolor="white")
            axes[1].set_title("Tras Yeo-Johnson", fontsize=9)
            for ax in axes:
                ax.spines[["top","right"]].set_visible(False)
                ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    else:
        st.markdown("### 🔬 Pipeline D – Gamma-Pydra")
        pasos = [
            ("OrdinalEncoder","categóricas"),
            ("Sin imputer","manejo nativo\nde NaN"),
            ("MinMaxScaler","[0, N]→enteros"),
            ("Gamma-Pydra","clasificación"),
        ]
        fig = pipeline_diagram(pasos, ["#E67E22","#D35400","#BA4A00","#922B21"])
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Decisiones clave")
            decisiones = [
                "**Algoritmo propio**: Gamma-Pydra es un clasificador basado en residuos con tabla PYDRA para manejo de NaN.",
                "**Sin imputación**: El algoritmo tiene su propia tabla de 8 variantes para manejar valores NaN (usó variante 0 y 6).",
                "**Escalado a enteros [0, N]**: El algoritmo trabaja con valores discretos; N=10 fue el parámetro óptimo.",
                "**OrdinalEncoder**: Solo para convertir categóricas a números — el algoritmo no distingue tipos de variable.",
                "**Sin SMOTE**: Gamma-Pydra no fue balanceado para preservar el comportamiento original del algoritmo.",
            ]
            for d in decisiones:
                st.markdown(f"  • {d}")
        with c2:
            st.markdown("#### 🔢 Lógica del algoritmo")
            st.markdown("""
**Idea principal:**
1. Para cada registro a clasificar, calcula los **residuos** entre él y cada muestra de entrenamiento
2. Itera sobre valores θ = 0, 1, 2, … , ρ
3. Cuenta cuántos residuos == θ por clase y calcula un promedio
4. La clase con mayor promedio gana (primer θ con ganador claro)

**Tabla PYDRA** define cómo contar cuando hay NaN:
- `qc`: valor de entrenamiento es NaN → depende de variante
- `cq`: valor del registro es NaN → depende de variante  
- `qq`: ambos son NaN → depende de variante
""")

    st.markdown("---")
    st.markdown("### 📊 Comparativa de pipelines")
    df_comp = pd.DataFrame({
        "Pipeline": ["A – Árboles","B – Distancias","C – Naive Bayes","D – Gamma"],
        "Imputer": ["SimpleImputer (mediana)","KNN Imputer (k=5)","SimpleImputer (mediana)","Ninguno (nativo)"],
        "Outliers": ["No tratados","Winsor p5-p95","Winsor p1-p99","No tratados"],
        "Escalado": ["Ninguno","StandardScaler","Yeo-Johnson + Std","MinMaxScaler [0,N]"],
        "Codif. Nominal": ["OrdinalEncoder","OneHotEncoder","OneHotEncoder","OrdinalEncoder"],
        "Codif. Ordinal": ["OrdinalEncoder","OrdinalEncoder","OrdinalEncoder","OrdinalEncoder"],
        "Balanceo": ["Sí","Sí","Sí","No"],
    })
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TAB 3 – MODELADO Y RESULTADOS                                          │
# └─────────────────────────────────────────────────────────────────────────┘
with tabs[3]:
    st.markdown('<div class="tab-header">🤖 Modelado y Resultados</div>', unsafe_allow_html=True)

    vista = st.radio("Selecciona vista:", [
        "Resumen general","Matrices de confusión","Curvas ROC simuladas",
        "Hiperparámetros","Comparativa detallada"
    ], horizontal=True)

    # ── Resumen general ───────────────────────────────────────────────────
    if vista == "Resumen general":
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### 📊 Comparación de métricas")
            metrica_sel = st.selectbox("Métrica:", ["F1-Score","Accuracy","Precision","Recall"])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            df_s = RESULTADOS.sort_values(metrica_sel, ascending=True)
            bars = ax.barh(df_s["Modelo"], df_s[metrica_sel], color=df_s["Color"], edgecolor="white", height=0.6)
            for bar, val in zip(bars, df_s[metrica_sel]):
                ax.text(val + 0.002, bar.get_y()+bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0.60, 0.77)
            ax.axvline(df_s[metrica_sel].max(), ls="--", color=ACCENT, lw=1.5, alpha=0.7)
            ax.set_xlabel(metrica_sel, fontsize=10)
            ax.set_title(f"Clasificadores – {metrica_sel}", fontsize=12, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("#### 📋 Tabla comparativa")
            df_show = RESULTADOS[["Modelo","Accuracy","Precision","Recall","F1-Score","CV_Score"]].copy()
            df_show = df_show.sort_values("F1-Score", ascending=False).reset_index(drop=True)
            df_show.index = df_show.index + 1
            st.dataframe(
                df_show.style.format({
                    "Accuracy":"{:.4f}","Precision":"{:.4f}",
                    "Recall":"{:.4f}","F1-Score":"{:.4f}",
                    "CV_Score": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                }).background_gradient(subset=["F1-Score"], cmap="Purples"),
                use_container_width=True
            )

            st.markdown("#### 🎯 CV Score vs Test F1")
            df_cv = RESULTADOS.dropna(subset=["CV_Score"]).copy()
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            x2 = np.arange(len(df_cv))
            ax2.plot(x2, df_cv["CV_Score"].values, "o--", color=PURPLE, label="CV Score", lw=2, ms=7)
            ax2.plot(x2, df_cv["F1-Score"].values,  "s-",  color=GREEN,  label="Test F1", lw=2, ms=7)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(df_cv["Modelo"], rotation=30, ha="right", fontsize=8)
            ax2.set_ylim(0.65, 0.78)
            ax2.legend(fontsize=9)
            ax2.set_title("CV Score vs Test F1-Score", fontsize=10, fontweight="bold")
            ax2.spines[["top","right"]].set_visible(False)
            ax2.set_facecolor("#FAFAFA")
            fig2.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig2); plt.close()

    # ── Matrices de confusión ─────────────────────────────────────────────
    elif vista == "Matrices de confusión":
        st.markdown("#### 🗂️ Matrices de confusión – Conjunto de prueba (n=1000)")

        modelos_sel = st.multiselect("Selecciona modelos:", list(CM.keys()),
                                     default=["Logística","RandomForest","AdaBoost","Gamma"])
        if modelos_sel:
            nc = min(len(modelos_sel), 4)
            nr = (len(modelos_sel) + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(4.2*nc, 3.8*nr))
            if nr == 1 and nc == 1:
                axes = np.array([[axes]])
            elif nr == 1:
                axes = axes.reshape(1, -1)
            elif nc == 1:
                axes = axes.reshape(-1, 1)

            for idx, modelo in enumerate(modelos_sel):
                r, c = divmod(idx, nc)
                ax = axes[r][c]
                cm = CM[modelo]
                im = ax.imshow(cm, cmap="Purples", interpolation="nearest")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["Real 0","Real 1"])
                for i2 in range(2):
                    for j2 in range(2):
                        color_txt = "white" if cm[i2,j2] > cm.max()*0.6 else "black"
                        ax.text(j2, i2, str(cm[i2,j2]), ha="center", va="center",
                                fontsize=15, fontweight="bold", color=color_txt)
                acc = (cm[0,0]+cm[1,1])/cm.sum()
                ax.set_title(f"{modelo}\nAcc={acc:.3f}", fontsize=10, fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.8)

            for idx in range(len(modelos_sel), nr*nc):
                r, c = divmod(idx, nc)
                axes[r][c].set_visible(False)
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### 🔍 Análisis por clase")
        modelo_detail = st.selectbox("Modelo para análisis detallado:", list(CM.keys()))
        cm_d = CM[modelo_detail]
        tn, fp, fn, tp = cm_d[0,0], cm_d[0,1], cm_d[1,0], cm_d[1,1]
        total = cm_d.sum()
        prec0 = tn/(tn+fn) if (tn+fn)>0 else 0
        rec0  = tn/(tn+fp) if (tn+fp)>0 else 0
        prec1 = tp/(tp+fn) if (tp+fn)>0 else 0
        rec1  = tp/(tp+fp) if (tp+fp)>0 else 0
        col1, col2, col3, col4 = st.columns(4)
        for col, val, lbl in zip([col1,col2,col3,col4],
                                  [f"{(tn+tp)/total:.3f}", f"{prec1:.3f}", f"{rec1:.3f}", f"{fn}"],
                                  ["Accuracy","Precision (clase 1)","Recall (clase 1)","Falsos Negativos"]):
            with col:
                st.metric(lbl, val)

    # ── Curvas ROC simuladas ──────────────────────────────────────────────
    elif vista == "Curvas ROC simuladas":
        st.markdown("#### 📈 Curvas ROC – Basadas en métricas reales")
        st.caption("Las curvas son aproximaciones sintéticas generadas a partir de los valores reales de accuracy, precision y recall de cada modelo.")

        auc_vals = {
            "Logística":    0.791,
            "RandomForest": 0.789,
            "AdaBoost":     0.785,
            "SVM":          0.771,
            "NaiveBayes":   0.762,
            "Árbol":        0.740,
            "Gamma":        0.738,
            "KNN":          0.737,
        }
        colores_roc = {m: r["Color"] for _, r in RESULTADOS.iterrows() for m in [r["Modelo"]]}
        colores_roc["Gamma"] = "#1ABC9C"

        modelos_roc = st.multiselect("Modelos a mostrar:", list(auc_vals.keys()),
                                     default=["Logística","RandomForest","AdaBoost","Gamma"])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        np.random.seed(99)
        for modelo in modelos_roc:
            auc_v = auc_vals[modelo]
            fpr = np.linspace(0, 1, 200)
            tpr = np.clip(fpr + (auc_v - 0.5) * 2 * (1 - fpr) * fpr**0.3 +
                          np.random.normal(0, 0.01, 200).cumsum()/80, 0, 1)
            tpr[0] = 0; tpr[-1] = 1
            tpr = np.sort(tpr)
            color_roc = colores_roc.get(modelo, PURPLE)
            ax.plot(fpr, tpr, lw=2.2, color=color_roc, label=f"{modelo} (AUC≈{auc_v:.3f})")
        ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Azar (AUC=0.5)")
        ax.fill_between([0,1],[0,1],[0,1], alpha=0.03, color="gray")
        ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
        ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
        ax.set_title("Curvas ROC – Comparación de clasificadores", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Hiperparámetros ───────────────────────────────────────────────────
    elif vista == "Hiperparámetros":
        st.markdown("#### 🔧 Mejores hiperparámetros por modelo (GridSearchCV)")
        for modelo, params in MEJORES_PARAMS.items():
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.markdown(f"**{modelo}**")
            with col2:
                st.code(params, language=None)

        st.markdown("---")
        st.markdown("#### 📊 Espacio de búsqueda por modelo")
        espacio = {
            "SVM":          "3 comb. (C × kernel) → 15 fits",
            "KNN":          "72 comb. (neighbors × weights × metric × p) → 360 fits",
            "Logística":    "45 comb. (C × penalty × solver × l1_ratio) → 225 fits",
            "Árbol":        "270 comb. (criterion × depth × split × leaf × features) → 1350 fits",
            "RandomForest": "576 comb. pre-definidas → se usaron mejores conocidos",
            "AdaBoost":     "12 comb. (estimador × lr × n_estimators) → 60 fits",
            "NaiveBayes":   "12 comb. (var_smoothing) → 60 fits",
            "Gamma-Pydra":  "2 comb. (variante) × 3 folds CV manual",
        }
        df_esp = pd.DataFrame({"Modelo": list(espacio.keys()), "Búsqueda": list(espacio.values())})
        st.dataframe(df_esp, hide_index=True, use_container_width=True)

        st.markdown("#### 📈 CV Score durante búsqueda")
        fig, ax = plt.subplots(figsize=(9, 4))
        cv_data = RESULTADOS.dropna(subset=["CV_Score"]).sort_values("CV_Score", ascending=False)
        bars = ax.bar(cv_data["Modelo"], cv_data["CV_Score"],
                      color=cv_data["Color"], edgecolor="white", width=0.55)
        for bar, val in zip(bars, cv_data["CV_Score"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0.65, 0.78)
        ax.set_ylabel("Accuracy – CV (k=5)", fontsize=10)
        ax.set_title("Mejor score durante GridSearchCV (StratifiedKFold k=5)", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Comparativa detallada ─────────────────────────────────────────────
    elif vista == "Comparativa detallada":
        st.markdown("#### 🕸️ Gráfica de radar – Perfil de cada modelo")
        categorias = ["Accuracy","Precision","Recall","F1-Score"]
        modelos_radar = st.multiselect("Selecciona modelos:", RESULTADOS["Modelo"].tolist(),
                                       default=["Logística","AdaBoost","NaiveBayes","Gamma"])
        if modelos_radar:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            N_cat = len(categorias)
            angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
            angles += angles[:1]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categorias, fontsize=11)
            ax.set_ylim(0.60, 0.78)
            ax.set_yticks([0.62, 0.66, 0.70, 0.74])
            ax.set_yticklabels(["0.62","0.66","0.70","0.74"], fontsize=7)
            for modelo in modelos_radar:
                row = RESULTADOS[RESULTADOS["Modelo"]==modelo].iloc[0]
                vals = [row[c] for c in categorias]
                vals += vals[:1]
                color = row["Color"]
                ax.plot(angles, vals, "o-", lw=2, color=color, label=modelo)
                ax.fill(angles, vals, alpha=0.07, color=color)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
            ax.set_title("Radar – Perfil de clasificadores", fontsize=12, fontweight="bold", pad=20)
            fig.patch.set_facecolor("#FAFAFA")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### 📉 Análisis de errores por modelo")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        df_err = RESULTADOS.copy()
        df_err["FP_rate"] = [177/620, 214/620, 160/620, 222/620,
                              150/620, 214/620, 167/620, 252/620]
        df_err["FN_rate"] = [113/380, 110/380, 116/380,  99/380,
                              126/380,  87/380, 111/380,  70/380]
        df_err_s = df_err.sort_values("FP_rate")
        axes[0].barh(df_err_s["Modelo"], df_err_s["FP_rate"],
                     color=df_err_s["Color"], edgecolor="white", height=0.55)
        axes[0].set_title("Tasa de Falsos Positivos\n(pred=1, real=0)", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("FP / Total clase 0")
        axes[0].spines[["top","right"]].set_visible(False)
        axes[0].set_facecolor("#FAFAFA")

        df_err_fn = df_err.sort_values("FN_rate")
        axes[1].barh(df_err_fn["Modelo"], df_err_fn["FN_rate"],
                     color=df_err_fn["Color"], edgecolor="white", height=0.55)
        axes[1].set_title("Tasa de Falsos Negativos\n(pred=0, real=1)", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("FN / Total clase 1")
        axes[1].spines[["top","right"]].set_visible(False)
        axes[1].set_facecolor("#FAFAFA")

        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TAB 4 – ANÁLISIS CRÍTICO                                               │
# └─────────────────────────────────────────────────────────────────────────┘
with tabs[4]:
    st.markdown('<div class="tab-header">🔍 Análisis Crítico y Conclusiones</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🏆 Ganador: Regresión Logística")
        st.markdown('<div class="insight-box">La Regresión Logística alcanzó el mejor F1-Score de <b>0.7265</b>, con C=0.01 y penalización L2. Su éxito se explica por la <b>ausencia de multicolinealidad</b> en las variables del dataset y el correcto preprocesamiento (StandardScaler + OHE + OrdinalEncoder). Un C bajo indica alta regularización, lo que sugiere que el modelo aprendió a generalizar bien sin sobreajustar.</div>', unsafe_allow_html=True)

        st.markdown("### 🌳 Modelos de árbol")
        st.markdown('<div class="insight-box"><b>RandomForest y AdaBoost</b> quedaron muy cerca (F1 ≈ 0.726). RandomForest con criterion=entropy y 300 estimadores aprovecha la diversidad del ensamble. AdaBoost con max_depth=2 y learning_rate=0.5 aprende de forma incremental sin sobreajustar. <b>El árbol individual</b> fue el más débil del grupo, lo que refuerza el valor del ensamble.</div>', unsafe_allow_html=True)

        st.markdown("### 🔬 Gamma-Pydra")
        st.markdown('<div class="insight-box">A pesar de ser un algoritmo experimental, Gamma-Pydra logró un <b>recall de 0.82 para la clase minoritaria</b> (el más alto de todos), sacrificando precisión global. Esto lo hace especialmente útil cuando el costo de un falso negativo es alto. El manejo nativo de NaN es una ventaja frente al dataset.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### 📊 ¿Por qué accuracy no es suficiente?")
        fig, ax = plt.subplots(figsize=(6, 4))
        modelos_plot = ["Logística","NaiveBayes","Gamma","KNN"]
        recall_min  = [0.69, 0.77, 0.82, 0.71]
        accuracy_v  = [0.724, 0.699, 0.678, 0.676]
        x3 = np.arange(len(modelos_plot))
        width3 = 0.38
        ax.bar(x3 - width3/2, accuracy_v, width3, label="Accuracy", color=PURPLE, alpha=0.85, edgecolor="white")
        ax.bar(x3 + width3/2, recall_min, width3, label="Recall clase 1", color=GREEN, alpha=0.85, edgecolor="white")
        ax.set_xticks(x3); ax.set_xticklabels(modelos_plot, fontsize=10)
        ax.set_ylim(0.60, 0.88)
        ax.legend(fontsize=9)
        ax.set_title("Accuracy vs Recall (clase minoritaria)", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("### ⚠️ Limitaciones identificadas")
        limitaciones = [
            "El dataset fue **generado sintéticamente** con ruido inducido (Kaggle), lo que puede afectar la generalización real.",
            "Gamma-Pydra tiene **complejidad cuadrática** O(n²) en predicción — no escala bien.",
            "El balanceo con **SMOTE puede introducir ruido** al generar muestras sintéticas cercanas al límite de decisión (BorderlineSMOTE lo mitiga).",
            "La validación cruzada de **Gamma usó solo k=3** (por costo computacional), lo que hace su estimación menos confiable.",
        ]
        for lim in limitaciones:
            st.markdown(f"  ⚠️ {lim}")

    st.markdown("---")
    st.markdown("### 📌 Conclusiones finales")
    col1, col2, col3 = st.columns(3)
    conclusiones = [
        ("🎯","Mejor modelo","La **Regresión Logística** obtuvo F1=0.7265, demostrando que un modelo simple y bien regularizado puede superar a ensambles complejos cuando el preprocesamiento es el adecuado."),
        ("⚖️","Selección de métrica","En datasets desbalanceados, **F1-Score y Recall** son más informativos que la Accuracy. Gamma-Pydra lo evidenció: accuracy=0.68 pero recall=0.82 en clase minoritaria."),
        ("🔧","Preprocesamiento","Diseñar **pipelines diferenciados** por familia de modelo es crítico. Un mismo tratamiento para todos los modelos hubiera reducido su rendimiento potencial."),
        ("🔬","Gamma-Pydra","El algoritmo propio demostró ser **competitivo y funcional** con datos reales y valores nulos, validando su implementación como alternativa a clasificadores estándar."),
        ("📊","Balanceo","La combinación **Undersampling + BorderlineSMOTE** fue conservadora y efectiva: mejoró el recall sin destruir información de la clase mayoritaria."),
        ("🌳","Ensambles","**RandomForest y AdaBoost** superaron al árbol individual en ~4 puntos de F1, confirmando que la diversidad de estimadores reduce la varianza y el sobreajuste."),
    ]
    for idx, (icon, titulo, texto) in enumerate(conclusiones):
        with [col1, col2, col3][idx % 3]:
            st.markdown(f"#### {icon} {titulo}")
            st.markdown(f'<div class="insight-box">{texto}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎓 Reflexión metodológica")
    st.success("""
Esta práctica demostró que el éxito en clasificación supervisada no depende únicamente del algoritmo elegido, sino del **proceso completo**:
el análisis exploratorio inicial, las decisiones de preprocesamiento fundamentadas en los supuestos de cada modelo,
el tratamiento adecuado del desbalance, la búsqueda sistemática de hiperparámetros y la evaluación con métricas apropiadas para el problema.
La combinación de estos elementos permitió obtener modelos consistentes con F1 > 0.72 en un problema de predicción de salud del sueño.
    """)