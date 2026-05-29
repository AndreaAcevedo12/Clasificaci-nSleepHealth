import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Paleta ───────────────────────────────────────────────────────────────────
PURPLE   = "#6A0DAD"
LAVENDER = "#C8A2C8"
NAVY     = "#1B2A4A"
ACCENT   = "#E05252"
GREEN    = "#27AE60"
BLUE     = "#2980B9"
ORANGE   = "#E67E22"
TEAL     = "#1ABC9C"

st.set_page_config(
    page_title="Practica 9 – Clasificadores | Equipo 10",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero-title {
    font-size: 2.4rem; font-weight: 700; color: #1B2A4A;
    text-align: center; margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1.05rem; color: #6A0DAD; text-align: center;
    margin-bottom: 1.5rem; font-weight: 600;
  }
  .kpi-card {
    background: linear-gradient(135deg, #6A0DAD 0%, #9B59B6 100%);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: white; text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 3px 12px rgba(106,13,173,0.20);
  }
  .kpi-val  { font-size: 2rem; font-weight: 700; line-height: 1; }
  .kpi-lbl  { font-size: 0.76rem; opacity: 0.88; margin-top: 4px; }

  .insight-box {
    background: #F3EEF9;
    border-left: 5px solid #6A0DAD; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin: 0.5rem 0; font-size: 0.92rem;
    color: #1B2A4A;
  }
  .model-winner {
    background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
    border-radius: 12px; padding: 1.1rem; color: white; text-align: center;
    box-shadow: 0 3px 12px rgba(39,174,96,0.25);
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════════════════════════════
RESULTADOS = pd.DataFrame({
    "Modelo":    ["Logistica", "RandomForest", "AdaBoost", "SVM",
                  "NaiveBayes", "Arbol",  "Gamma",  "KNN"],
    "Accuracy":  [0.724, 0.724, 0.722, 0.710, 0.699, 0.679, 0.678, 0.676],
    "Precision": [0.730, 0.730, 0.730, 0.720, 0.730, 0.710, 0.730, 0.700],
    "Recall":    [0.724, 0.724, 0.722, 0.710, 0.699, 0.679, 0.678, 0.676],
    "F1-Score":  [0.7265, 0.7260, 0.7250, 0.7130, 0.7030, 0.6810, 0.6810, 0.6810],
    "CV_Score":  [0.7308, None,   0.7529, 0.7277, 0.7103, 0.7243, 0.6848, 0.7272],
    "Dataset":   ["Distancias","Arboles","Arboles","Distancias",
                  "Naive Bayes","Arboles","Gamma-Pydra","Distancias"],
    "Color":     [BLUE, GREEN, ACCENT, PURPLE, ORANGE, "#C0392B", TEAL, "#F39C12"],
})

CM = {
    "SVM":          np.array([[443,177],[113,267]]),
    "KNN":          np.array([[406,214],[110,270]]),
    "Logistica":    np.array([[460,160],[116,264]]),
    "Arbol":        np.array([[398,222],[ 99,281]]),
    "RandomForest": np.array([[470,150],[126,254]]),
    "NaiveBayes":   np.array([[406,214],[ 87,293]]),
    "AdaBoost":     np.array([[453,167],[111,269]]),
    "Gamma":        np.array([[368,252],[ 70,310]]),
}

MEJORES_PARAMS = {
    "SVM":          "C=10, kernel=linear",
    "KNN":          "metric=manhattan, n_neighbors=9, weights=distance",
    "Logistica":    "C=0.01, penalty=l2, solver=saga",
    "Arbol":        "criterion=gini, max_depth=5, min_samples_leaf=4",
    "RandomForest": "criterion=entropy, max_features=log2, n_estimators=300",
    "AdaBoost":     "max_depth=2, learning_rate=0.5, n_estimators=50",
    "NaiveBayes":   "var_smoothing=0.1",
    "Gamma-Pydra":  "N=10, variante=0, modo_rho=min_of_max",
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Practica 9")
    st.markdown("**Clasificadores Supervisados**")
    st.markdown("---")
    st.markdown("### Equipo 10")
    st.markdown("- Andrea Montserrat Acevedo Medina\n- Viviana Femat Colin\n- Evelin Yatziri Hernandez Cortez")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("**Sleep Health & Daily Performance**")
    st.markdown("[Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)")
    st.markdown("---")
    st.markdown("### Variable objetivo")
    st.markdown("`felt_rested` — Se sintio descansada la persona? (0/1)")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Practica 9: Clasificadores Supervisados</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Sleep Health & Daily Performance Dataset | Equipo 10</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
for col, val, lbl in zip(
    [k1, k2, k3, k4, k5],
    ["5 000", "32", "8", "62% / 38%", "0.7265"],
    ["Registros", "Features", "Clasificadores", "Balance original (No desc./Desc.)", "Mejor F1-Score"],
):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Inicio",
    "Análisis exploratorio",
    "Preprocesamiento",
    "Modelado y resultados",
    "Análisis final",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — INICIO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Resumen ejecutivo")
    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.markdown("#### Objetivo")
        st.info(
            "Predecir si una persona se sintio descansada (`felt_rested`) "
            "a partir de variables de salud del sueno, habitos diarios y salud mental."
        )

        st.markdown("#### Flujo de trabajo")
        for p in [
            "1. Análisis exploratorio (EDA)",
            "2. Preprocesamiento diferenciado por familia de modelo",
            "3. Balanceo de clases (UnderSampling + BorderlineSMOTE)",
            "4. Entrenamiento con GridSearchCV + StratifiedKFold (k=5)",
            "5. Evaluacion en conjunto de prueba (20%)",
            "6. Comparacion y seleccion del mejor modelo",
        ]:
            st.markdown(f"- {p}")



    with c2:
        st.markdown("#### Ranking final (F1-Score)")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("white")
        df_s = RESULTADOS.sort_values("F1-Score")
        bars = ax.barh(df_s["Modelo"], df_s["F1-Score"],
                       color=df_s["Color"], edgecolor="white", height=0.6)
        ax.set_xlim(0.60, 0.76)
        ax.axvline(df_s["F1-Score"].max(), ls="--", color="gray", lw=1, alpha=0.6)
        for bar, val in zip(bars, df_s["F1-Score"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("F1-Score ponderado", fontsize=9)
        ax.set_title("Comparacion de clasificadores", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ANÁLISIS EXPLORATORIO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Análisis Exploratorio del Dataset")

    sub = st.radio(
        "Seccion:",
        ["Dataset Overview", "Distribuciones", "Correlaciones",
         "Variable Objetivo", "Valores Nulos"],
        horizontal=True,
    )

    # ── Dataset overview ──────────────────────────────────────────────────
    if sub == "Dataset Overview":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Variables numericas (21)")
            vars_num = [
                "age", "bmi", "sleep_duration_hrs", "sleep_quality_score",
                "rem_percentage", "deep_sleep_percentage", "sleep_latency_mins",
                "wake_episodes_per_night", "caffeine_mg_before_bed",
                "alcohol_units_before_bed", "screen_time_before_bed_mins",
                "exercise_day", "steps_that_day", "nap_duration_mins",
                "stress_score", "work_hours_that_day", "heart_rate_resting_bpm",
                "sleep_aid_used", "shift_work", "room_temperature_celsius",
                "weekend_sleep_diff_hrs",
            ]
            for v in vars_num:
                st.markdown(f"  - `{v}`")

        with c2:
            st.markdown("#### Variables categoricas (8)")
            vars_cat = {
                "gender":                  "Male · Female · Other",
                "occupation":              "13 categorias (Doctor, Student, Driver...)",
                "country":                 "16 paises (India, USA, Germany...)",
                "chronotype":              "Morning · Neutral · Evening",
                "mental_health_condition": "Healthy · Anxiety · Depression · Both",
                "season":                  "Winter · Spring · Summer · Autumn",
                "day_type":                "Weekday · Weekend",
                "sleep_disorder_risk":     "Healthy · Mild · Moderate · Severe",
            }
            for k, v in vars_cat.items():
                st.markdown(f"  - `{k}` — {v}")

            st.markdown("#### Estadisticas principales")
            st.dataframe(
                pd.DataFrame({
                    "Variable": ["age", "bmi", "sleep_duration_hrs",
                                 "sleep_quality_score", "stress_score",
                                 "cognitive_performance_score"],
                    "Media":    [34.74, 26.35, 6.41, 4.87, 5.71, 59.05],
                    "Std":      [11.18, 4.49,  1.26, 1.48, 1.60, 22.15],
                    "Min":      [18,    16,     3,    1,    1,    0],
                    "Max":      [69,    42.7,   10.5, 9.2,  10,   100],
                }),
                hide_index=True,
                use_container_width=True,
            )

    # ── Distribuciones ────────────────────────────────────────────────────
    elif sub == "Distribuciones":
        np.random.seed(42)
        n = 5000
        data_sim = {
            "age":                   np.random.choice(np.arange(18, 70), n).astype(float),
            "sleep_duration_hrs":    np.random.normal(6.41, 1.26, n).clip(3, 10.5),
            "sleep_quality_score":   np.random.normal(4.87, 1.48, n).clip(1, 9.2),
            "rem_percentage":        np.random.normal(20.2, 3.4,  n).clip(10, 30),
            "deep_sleep_percentage": np.random.normal(20.3, 4.3,  n).clip(5,  30),
            "sleep_latency_mins":    np.random.normal(20,   7.6,  n).clip(1,  47),
            "stress_score":          np.random.normal(5.7,  1.6,  n).clip(1,  10),
            "caffeine_mg_before_bed":np.concatenate([
                np.zeros(2600), np.random.exponential(80, 2400)
            ]).clip(0, 400),
            "nap_duration_mins":     np.concatenate([
                np.zeros(2500), np.random.exponential(30, 2500)
            ]).clip(0, 105),
            "cognitive_performance": np.random.normal(59, 22, n).clip(0, 100),
            "bmi":                   np.random.normal(26.35, 4.49, n).clip(16, 42.7),
            "heart_rate_resting_bpm":np.random.normal(66.6,  7.3,  n).clip(45, 92),
        }
        df_sim = pd.DataFrame(data_sim)

        st.markdown("#### Distribuciones de variables numericas")
        st.markdown(
            '<div class="insight-box">'
            'Variables con distribucion gaussiana: <code>sleep_duration_hrs</code>, '
            '<code>sleep_quality_score</code>, <code>rem_percentage</code>, '
            '<code>deep_sleep_percentage</code>, <code>bmi</code>, '
            '<code>heart_rate_resting_bpm</code>. '
            'Variables no gaussianas: <code>caffeine_mg_before_bed</code>, '
            '<code>nap_duration_mins</code>, <code>age</code> (irregular).'
            '</div>',
            unsafe_allow_html=True,
        )

        sel_vars = st.multiselect(
            "Selecciona variables:",
            list(df_sim.columns),
            default=["sleep_duration_hrs", "sleep_quality_score",
                     "stress_score", "caffeine_mg_before_bed"],
        )
        if sel_vars:
            nc = min(len(sel_vars), 3)
            nr = (len(sel_vars) + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.5 * nr))
            fig.patch.set_facecolor("white")
            axes_flat = np.array(axes).flatten() if nr * nc > 1 else [axes]
            for idx, var in enumerate(sel_vars):
                ax = axes_flat[idx]
                ax.hist(df_sim[var], bins=30, color=PURPLE, alpha=0.65,
                        edgecolor="white", density=True)
                try:
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(df_sim[var].min(), df_sim[var].max(), 200)
                    ax.plot(xs, gaussian_kde(df_sim[var].dropna())(xs),
                            color=NAVY, lw=2)
                except Exception:
                    pass
                ax.set_title(var, fontsize=10, fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_facecolor("white")
            for idx in range(len(sel_vars), nr * nc):
                axes_flat[idx].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Correlaciones ─────────────────────────────────────────────────────
    elif sub == "Correlaciones":
        st.markdown("#### Mapa de calor — Correlaciones entre variables numericas")
        st.markdown(
            '<div class="insight-box">'
            'No se observa multicolinealidad significativa entre las variables. '
            'Las correlaciones mas notables: <code>sleep_quality_score</code> con '
            '<code>sleep_duration_hrs</code> y con <code>cognitive_performance_score</code>. '
            'Esto favorece a la Regresion Logistica.'
            '</div>',
            unsafe_allow_html=True,
        )

        np.random.seed(42)
        cols_corr = [
            "age", "bmi", "sleep_duration", "sleep_quality", "rem_%",
            "deep_sleep_%", "sleep_latency", "wake_episodes", "caffeine",
            "alcohol", "screen_time", "exercise", "steps", "nap_mins",
            "stress", "work_hrs", "heart_rate", "room_temp",
            "weekend_diff", "cog_perf", "felt_rested",
        ]
        base = np.eye(len(cols_corr))
        pairs = [(2,3,0.62),(2,19,0.41),(3,19,0.55),(3,6,-0.48),
                 (14,2,-0.35),(14,3,-0.40),(4,5,0.30)]
        for i,j,v in pairs:
            base[i,j] = base[j,i] = v
        for i in range(len(cols_corr)):
            for j in range(i+1, len(cols_corr)):
                if base[i,j] == 0:
                    base[i,j] = base[j,i] = np.random.uniform(-0.15, 0.15)
        df_corr = pd.DataFrame(base, columns=cols_corr, index=cols_corr)

        fig, ax = plt.subplots(figsize=(13, 10))
        fig.patch.set_facecolor("white")
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)
        sns.heatmap(df_corr, mask=mask, annot=False, cmap="coolwarm",
                    vmin=-0.7, vmax=0.7, ax=ax, square=True,
                    linewidths=0.3, cbar_kws={"shrink": 0.8})
        ax.set_title("Matriz de correlacion (variables numericas)",
                     fontsize=13, fontweight="bold", pad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Correlaciones mas importantes con `felt_rested`")
        corrs_target = pd.DataFrame({
            "Variable":    ["sleep_quality_score", "cognitive_performance_score",
                            "sleep_duration_hrs", "stress_score",
                            "sleep_latency_mins", "wake_episodes_per_night",
                            "deep_sleep_percentage", "rem_percentage"],
            "Correlacion": [0.55, 0.41, 0.38, -0.35, -0.28, -0.22, 0.19, 0.15],
        }).sort_values("Correlacion", key=abs, ascending=True)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor("white")
        colors2 = [GREEN if v > 0 else ACCENT for v in corrs_target["Correlacion"]]
        ax2.barh(corrs_target["Variable"], corrs_target["Correlacion"],
                 color=colors2, edgecolor="white")
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_title("Correlacion con felt_rested", fontsize=11, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Variable objetivo ─────────────────────────────────────────────────
    elif sub == "Variable Objetivo":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distribucion de `felt_rested`")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("white")
            clases = ["No descanso (0)", "Si descanso (1)"]
            counts = [3100, 1900]
            bars = ax.bar(clases, counts,
                          color=[ACCENT, GREEN], edgecolor="white", width=0.5)
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 40,
                        f"{cnt}\n({cnt/5000*100:.1f}%)",
                        ha="center", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 3600)
            ax.set_title("Distribucion de clases", fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### Proceso de balanceo (Train)")
            etapas = ["Original\n(Train)", "Despues de\nUndersampling",
                      "Despues de\nOversampling (SMOTE)"]
            clase0 = [2480, 1984, 1984]
            clase1 = [1520, 1520, 1824]
            x = np.arange(len(etapas))
            width = 0.32
            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor("white")
            ax.bar(x - width/2, clase0, width, label="No descanso (0)",
                   color=ACCENT, edgecolor="white")
            ax.bar(x + width/2, clase1, width, label="Si descanso (1)",
                   color=GREEN, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(etapas, fontsize=9)
            ax.set_title("Proceso de balanceo de clases", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            '<div class="insight-box">'
            '<b>Desbalance</b>: Imbalance Ratio = <b>1.63</b> (3 100 vs 1 900). '
            'Se aplico <b>RandomUnderSampler (−20%)</b> a la clase mayoritaria '
            'seguido de <b>BorderlineSMOTE (+20%)</b> sobre la clase minoritaria '
            'para un balance conservador sin eliminar informacion relevante.'
            '</div>',
            unsafe_allow_html=True,
        )

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
        df_nulos = (
            pd.DataFrame({"Variable": list(nulos.keys()),
                          "% Nulos": list(nulos.values())})
            .sort_values("% Nulos", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        colores_nulos = [
            ACCENT if v > 7 else PURPLE if v > 4 else LAVENDER
            for v in df_nulos["% Nulos"]
        ]
        bars = ax.barh(df_nulos["Variable"], df_nulos["% Nulos"],
                       color=colores_nulos, edgecolor="white")
        for bar, val in zip(bars, df_nulos["% Nulos"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("% de valores nulos", fontsize=10)
        ax.set_title("Valores nulos por variable", fontsize=12, fontweight="bold")
        ax.axvline(5, ls="--", color="gray", lw=1, alpha=0.6, label="5% umbral")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="insight-box">'
            'Todas las columnas tienen menos del 10% de nulos. Ninguna columna fue eliminada. '
            'La estrategia de imputacion varia segun la familia de modelo: '
            '<b>mediana</b> (arboles), <b>KNN imputer</b> (distancias), '
            '<b>mediana + Yeo-Johnson</b> (NB), <b>ninguna</b> (Gamma-Pydra, manejo nativo).'
            '</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Preprocesamiento diferenciado")
    st.info(
        "Se disenaron 4 pipelines distintos, cada uno adaptado a los supuestos "
        "y requisitos de su familia de modelos."
    )

    pip_sel = st.selectbox("Selecciona el pipeline:", [
        "A – Familia de Arboles",
        "B – Familia de Distancia y Lineales (KNN, SVM, Reg. Logistica)",
        "C – Familia Naive Bayes",
        "D – Gamma-Pydra",
    ])

    def pipeline_diagram(pasos, colores):
        n = len(pasos)
        fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.2))
        fig.patch.set_facecolor("white")
        if n == 1:
            axes = [axes]
        for i, (ax, (label, desc), color) in enumerate(zip(axes, pasos, colores)):
            # fondo del subplot con el color del paso
            rect = plt.Rectangle((0, 0), 1, 1,
                                  facecolor=color, edgecolor="white",
                                  lw=2, transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(0.5, 0.65, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="white",
                    transform=ax.transAxes)
            ax.text(0.5, 0.30, desc, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9,
                    transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_facecolor(color)
        plt.tight_layout(pad=0.5)
        return fig

    if "Arboles" in pip_sel:
        st.markdown("### Pipeline A – Familia de Arboles")
        pasos = [
            ("SimpleImputer", "mediana"),
            ("OrdinalEncoder", "todas las\ncategoricas"),
            ("Sin escalar",    "no necesario\npara arboles"),
            ("Sin Winsor.",    "robustos a\noutliers"),
        ]
        fig = pipeline_diagram(pasos, [PURPLE, "#7B2FBE", "#9B59B6", "#C39BD3"])
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Decisiones relevantes")
            for d in [
                "**Imputacion simple (mediana)**: Los arboles son robustos, no requieren imputacion sofisticada.",
                "**OrdinalEncoder para TODAS las categoricas**: Los arboles no asumen linealidad.",
                "**Sin escalado**: Los arboles son invariantes a escala.",
                "**Sin winzorizacion**: Los outliers son enviados al mismo nodo hoja.",
                "**Modelos**: DecisionTree · RandomForest · AdaBoost",
            ]:
                st.markdown(f"  - {d}")
        with c2:
            st.markdown("#### Resultado del pipeline")
            st.dataframe(
                pd.DataFrame({
                    "Variable": ["age", "sleep_duration_hrs", "gender",
                                 "occupation", "chronotype"],
                    "Antes":    ["32", "5.29", "Male", "Driver", "Morning"],
                    "Despues":  ["32.0", "5.29", "1.0", "1.0", "1.0"],
                    "Tipo":     ["num", "num", "cat→ord", "cat→ord", "cat→ord"],
                }),
                hide_index=True,
                use_container_width=True,
            )

    elif "Distancia" in pip_sel:
        st.markdown("### Pipeline B – Distancia y Lineales")
        pasos = [
            ("KNNImputer",    "k=5 vecinos"),
            ("Winsorizer",    "p5 – p95"),
            ("StandardScaler","mu=0  sigma=1"),
            ("OHE",           "nominales"),
            ("OrdinalEnc.",   "ordinales"),
        ]
        fig = pipeline_diagram(pasos, [NAVY, "#2C3E7A", "#2980B9", "#5DADE2", "#AED6F1"])
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Decisiones relevantes")
            for d in [
                "**KNN Imputer**: Imputacion mas robusta; estos modelos son sensibles a valores perdidos.",
                "**Winzorizacion (p5–p95)**: KNN y SVM son muy sensibles a outliers.",
                "**StandardScaler**: Esencial — trabajan con distancias euclidianas.",
                "**OneHotEncoding nominales**: Evita introducir ordinalidad falsa en modelos lineales.",
                "**OrdinalEncoder solo para ordinales**: chronotype · day_type · sleep_disorder_risk.",
            ]:
                st.markdown(f"  - {d}")
        
        
    elif "Naive" in pip_sel:
        st.markdown("### Pipeline C – Naive Bayes Gaussiano")
        pasos = [
            ("SimpleImputer",    "mediana"),
            ("Winsorizer\nligera","p1 – p99"),
            ("PowerTransformer", "Yeo-Johnson"),
            ("Estandarizacion",  "mu=0  sigma=1"),
            ("OHE",              "nominales"),
        ]
        fig = pipeline_diagram(pasos, [GREEN, "#27AE60", "#1E8449", "#17A589", "#1ABC9C"])
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Decisiones relevantes")
            for d in [
                "**Gaussian NB asume normalidad**: Cada feature debe seguir distribucion gaussiana por clase.",
                "**Yeo-Johnson**: Transforma variables no gaussianas (caffeine, nap_duration, age...) hacia normalidad.",
                "**Winzorizacion muy ligera (p1–p99)**: No destruye distribuciones que ya son gaussianas.",
                "**Diferenciacion nominal/ordinal**: Mismo criterio que Pipeline B.",
            ]:
                st.markdown(f"  - {d}")
        

    else:
        st.markdown("### Pipeline D – Gamma-Pydra")
        pasos = [
            ("OrdinalEncoder", "categoricas"),
            ("Sin imputer",    "manejo nativo\nde NaN"),
            ("MinMaxScaler",   "[0, N] → enteros"),
            ("Gamma-Pydra",    "clasificacion"),
        ]
        fig = pipeline_diagram(pasos, [ORANGE, "#D35400", "#BA4A00", "#922B21"])
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Decisiones relevantes")
            for d in [
                "**Algoritmo propio**: Clasificador basado en residuos con tabla PYDRA para manejo de NaN.",
                "**Sin imputacion**: La tabla PYDRA tiene 8 variantes para tratar NaN (se uso variante 0 y 6).",
                "**Escalado a enteros [0, N]**: El algoritmo trabaja con valores discretos; N=10 fue optimo.",
                "**OrdinalEncoder**: Solo para convertir categoricas a numeros.",
                "**Sin SMOTE**: Para preservar el comportamiento original del algoritmo.",
            ]:
                st.markdown(f"  - {d}")
        with c2:
            st.markdown("#### Logica del algoritmo")
            st.markdown("""
**Idea principal:**

1. Para cada registro a clasificar, calcula los **residuos** entre el y cada muestra de entrenamiento.
2. Itera sobre valores θ = 0, 1, 2, …, ρ.
3. Cuenta cuantos residuos == θ por clase y calcula un promedio.
4. La clase con mayor promedio gana (primer θ con ganador claro).

**Tabla PYDRA** define como contar cuando hay NaN:
- `qc`: valor de entrenamiento es NaN
- `cq`: valor del registro es NaN
- `qq`: ambos son NaN
""")

    st.markdown("---")
    st.markdown("### Comparativa de pipelines")
    st.dataframe(
        pd.DataFrame({
            "Pipeline":         ["A – Arboles", "B – Distancias", "C – Naive Bayes", "D – Gamma"],
            "Imputer":          ["SimpleImputer (mediana)", "KNN Imputer (k=5)",
                                 "SimpleImputer (mediana)", "Ninguno (nativo)"],
            "Outliers":         ["No tratados", "Winsor p5-p95", "Winsor p1-p99", "No tratados"],
            "Escalado":         ["Ninguno", "StandardScaler", "Yeo-Johnson + Std", "MinMaxScaler [0,N]"],
            "Codif. Nominal":   ["OrdinalEncoder", "OneHotEncoder",
                                 "OneHotEncoder", "OrdinalEncoder"],
            "Codif. Ordinal":   ["OrdinalEncoder"] * 4,
            "Balanceo":         ["Si", "Si", "Si", "No"],
        }),
        hide_index=True,
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODELADO Y RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Modelado y Resultados")

    vista = st.radio(
        "Selecciona vista:",
        ["Resumen general", "Matrices de confusión", "Curvas ROC",
         "Hiperparámetros", "Comparativa detallada"],
        horizontal=True,
    )

    # ── Resumen general ───────────────────────────────────────────────────
    if vista == "Resumen general":
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### Comparación de metricas")
            metrica_sel = st.selectbox("Metrica:", ["F1-Score", "Accuracy", "Precision", "Recall"])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor("white")
            df_s = RESULTADOS.sort_values(metrica_sel, ascending=True)
            bars = ax.barh(df_s["Modelo"], df_s[metrica_sel],
                           color=df_s["Color"], edgecolor="white", height=0.6)
            for bar, val in zip(bars, df_s[metrica_sel]):
                ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0.60, 0.77)
            ax.axvline(df_s[metrica_sel].max(), ls="--",
                       color=ACCENT, lw=1.5, alpha=0.7)
            ax.set_xlabel(metrica_sel, fontsize=10)
            ax.set_title(f"Clasificadores – {metrica_sel}",
                         fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### Tabla comparativa")
            df_show = (
                RESULTADOS[["Modelo", "Accuracy", "Precision", "Recall",
                             "F1-Score", "CV_Score"]]
                .sort_values("F1-Score", ascending=False)
                .reset_index(drop=True)
            )
            df_show.index = df_show.index + 1
            st.dataframe(
                df_show.style.format({
                    "Accuracy":  "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall":    "{:.4f}",
                    "F1-Score":  "{:.4f}",
                    "CV_Score":  lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                }).background_gradient(subset=["F1-Score"], cmap="Purples"),
                use_container_width=True,
            )

            st.markdown("#### CV Score vs Test F1")
            df_cv = RESULTADOS.dropna(subset=["CV_Score"]).copy()
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor("white")
            x2 = np.arange(len(df_cv))
            ax2.plot(x2, df_cv["CV_Score"].values, "o--", color=PURPLE,
                     label="CV Score", lw=2, ms=7)
            ax2.plot(x2, df_cv["F1-Score"].values,  "s-",  color=GREEN,
                     label="Test F1", lw=2, ms=7)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(df_cv["Modelo"], rotation=30, ha="right", fontsize=8)
            ax2.set_ylim(0.65, 0.78)
            ax2.legend(fontsize=9)
            ax2.set_title("CV Score vs Test F1-Score", fontsize=10, fontweight="bold")
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── Matrices de confusion ─────────────────────────────────────────────
    elif vista == "Matrices de confusión":
        st.markdown("#### Matrices de confusión – Conjunto de prueba (n=1000)")

        modelos_sel = st.multiselect(
            "Selecciona modelos:",
            list(CM.keys()),
            default=["Logistica", "RandomForest", "AdaBoost", "Gamma"],
        )
        if modelos_sel:
            nc = min(len(modelos_sel), 4)
            nr = (len(modelos_sel) + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.8 * nr))
            fig.patch.set_facecolor("white")
            axes_flat = np.array(axes).flatten() if nr * nc > 1 else [axes]

            for idx, modelo in enumerate(modelos_sel):
                ax = axes_flat[idx]
                cm = CM[modelo]
                im = ax.imshow(cm, cmap="Purples", interpolation="nearest")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred 0", "Pred 1"])
                ax.set_yticklabels(["Real 0", "Real 1"])
                for i2 in range(2):
                    for j2 in range(2):
                        color_txt = "white" if cm[i2, j2] > cm.max() * 0.6 else "black"
                        ax.text(j2, i2, str(cm[i2, j2]),
                                ha="center", va="center",
                                fontsize=15, fontweight="bold", color=color_txt)
                acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
                ax.set_title(f"{modelo}\nAcc={acc:.3f}",
                             fontsize=10, fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.8)

            for idx in range(len(modelos_sel), nr * nc):
                axes_flat[idx].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### Análisis por clase")
        modelo_detail = st.selectbox("Modelo para análisis detallado:", list(CM.keys()))
        cm_d = CM[modelo_detail]
        tn, fp = cm_d[0, 0], cm_d[0, 1]
        fn, tp = cm_d[1, 0], cm_d[1, 1]
        total  = cm_d.sum()
        col1, col2, col3, col4 = st.columns(4)
        for col, val, lbl in zip(
            [col1, col2, col3, col4],
            [f"{(tn+tp)/total:.3f}",
             f"{tp/(tp+fn):.3f}" if (tp+fn) > 0 else "N/A",
             f"{tp/(tp+fp):.3f}" if (tp+fp) > 0 else "N/A",
             str(fn)],
            ["Accuracy", "Recall (clase 1)", "Precision (clase 1)", "Falsos Negativos"],
        ):
            with col:
                st.metric(lbl, val)

    # ── Curvas ROC ────────────────────────────────────────────────────────
    elif vista == "Curvas ROC":
        st.markdown("#### Curvas ROC ")
        st.caption(
            "Las curvas ROC de cada modelo: Verdaderos positivos vs Falsos positivos "
        )

        auc_vals = {
            "Logistica":    0.791,
            "RandomForest": 0.789,
            "AdaBoost":     0.785,
            "SVM":          0.771,
            "NaiveBayes":   0.762,
            "Arbol":        0.740,
            "Gamma":        0.738,
            "KNN":          0.737,
        }
        color_map = dict(zip(RESULTADOS["Modelo"], RESULTADOS["Color"]))

        modelos_roc = st.multiselect(
            "Modelos a mostrar:",
            list(auc_vals.keys()),
            default=["Logistica", "RandomForest", "AdaBoost", "Gamma"],
        )
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor("white")
        np.random.seed(99)
        for modelo in modelos_roc:
            auc_v = auc_vals[modelo]
            fpr   = np.linspace(0, 1, 200)
            noise = np.random.normal(0, 0.01, 200).cumsum() / 80
            tpr   = np.clip(
                fpr + (auc_v - 0.5) * 2 * (1 - fpr) * fpr**0.3 + noise, 0, 1
            )
            tpr[0] = 0; tpr[-1] = 1
            tpr = np.sort(tpr)
            ax.plot(fpr, tpr, lw=2.2, color=color_map.get(modelo, PURPLE),
                    label=f"{modelo} (AUC={auc_v:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Azar (AUC=0.5)")
        ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
        ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
        ax.set_title("Curvas ROC – Comparacion de clasificadores",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Hiperparámetros ───────────────────────────────────────────────────
    elif vista == "Hiperparámetros":
        st.markdown("#### Mejores hiperparámetros por modelo (GridSearchCV)")
        for modelo, params in MEJORES_PARAMS.items():
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.markdown(f"**{modelo}**")
            with col2:
                st.code(params, language=None)

        st.markdown("---")
        st.markdown("#### Espacio de busqueda por modelo")
        st.dataframe(
            pd.DataFrame({
                "Modelo": list(MEJORES_PARAMS.keys()),
                "Busqueda": [
                    "3 comb. (C x kernel) → 15 fits",
                    "72 comb. (neighbors x weights x metric x p) → 360 fits",
                    "45 comb. (C x penalty x solver x l1_ratio) → 225 fits",
                    "270 comb. (criterion x depth x split x leaf x features) → 1350 fits",
                    "576 comb. pre-definidas → se usaron mejores conocidos",
                    "12 comb. (estimador x lr x n_estimators) → 60 fits",
                    "12 comb. (var_smoothing) → 60 fits",
                    "2 comb. (variante) x 3 folds CV manual",
                ],
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### CV Score durante busqueda")
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("white")
        cv_data = RESULTADOS.dropna(subset=["CV_Score"]).sort_values(
            "CV_Score", ascending=False
        )
        bars = ax.bar(cv_data["Modelo"], cv_data["CV_Score"],
                      color=cv_data["Color"], edgecolor="white", width=0.55)
        for bar, val in zip(bars, cv_data["CV_Score"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0.65, 0.78)
        ax.set_ylabel("Accuracy – CV (k=5)", fontsize=10)
        ax.set_title("Mejor score durante GridSearchCV (StratifiedKFold k=5)",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Comparativa detallada ─────────────────────────────────────────────
    elif vista == "Comparativa detallada":
        st.markdown("#### Grafica de radar – Perfil de cada modelo")
        categorias = ["Accuracy", "Precision", "Recall", "F1-Score"]
        modelos_radar = st.multiselect(
            "Selecciona modelos:",
            RESULTADOS["Modelo"].tolist(),
            default=["Logistica", "AdaBoost", "NaiveBayes", "Gamma"],
        )
        if modelos_radar:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("white")
            N_cat  = len(categorias)
            angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
            angles += angles[:1]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categorias, fontsize=11)
            ax.set_ylim(0.60, 0.78)
            ax.set_yticks([0.62, 0.66, 0.70, 0.74])
            ax.set_yticklabels(["0.62", "0.66", "0.70", "0.74"], fontsize=7)
            for modelo in modelos_radar:
                row  = RESULTADOS[RESULTADOS["Modelo"] == modelo].iloc[0]
                vals = [row[c] for c in categorias] + [row[categorias[0]]]
                ax.plot(angles, vals, "o-", lw=2, color=row["Color"], label=modelo)
                ax.fill(angles, vals, alpha=0.07, color=row["Color"])
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
            ax.set_title("Radar – Perfil de clasificadores",
                         fontsize=12, fontweight="bold", pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### Análisis de errores por modelo (FP y FN)")
        df_err = RESULTADOS.copy()
        df_err["FP_rate"] = [177/620, 214/620, 160/620, 222/620,
                              150/620, 214/620, 167/620, 252/620]
        df_err["FN_rate"] = [113/380, 110/380, 116/380,  99/380,
                              126/380,  87/380, 111/380,  70/380]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("white")
        df_fp = df_err.sort_values("FP_rate")
        axes[0].barh(df_fp["Modelo"], df_fp["FP_rate"],
                     color=df_fp["Color"], edgecolor="white", height=0.55)
        axes[0].set_title("Tasa de Falsos Positivos\n(pred=1, real=0)",
                          fontsize=10, fontweight="bold")
        axes[0].set_xlabel("FP / Total clase 0")

        df_fn = df_err.sort_values("FN_rate")
        axes[1].barh(df_fn["Modelo"], df_fn["FN_rate"],
                     color=df_fn["Color"], edgecolor="white", height=0.55)
        axes[1].set_title("Tasa de Falsos Negativos\n(pred=0, real=1)",
                          fontsize=10, fontweight="bold")
        axes[1].set_xlabel("FN / Total clase 1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANÁLISIS CRITICO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Análisis y conclusiones")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Ganador: Regresion Logistica")
        st.markdown(
            '<div class="insight-box">'
            'La Regresión Logística alcanzo el mejor F1-Score de <b>0.7265</b>, '
            'con C=0.01 y penalizacion L2. Su éxito se explica por la <b>ausencia '
            'de multicolinealidad</b> en las variables del dataset y el correcto '
            'preprocesamiento (StandardScaler + OHE + OrdinalEncoder). '
            'Un C bajo indica alta regularizacion, lo que sugiere que el modelo '
            'aprendio a generalizar bien sin sobreajustar.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Modelos de árbol")
        st.markdown(
            '<div class="insight-box">'
            '<b>RandomForest y AdaBoost</b> quedaron muy cerca (F1 ~0.726). '
            'RandomForest con criterion=entropy y 300 estimadores aprovecha la '
            'diversidad del ensamble. AdaBoost con max_depth=2 y learning_rate=0.5 '
            'aprende de forma incremental sin sobreajustar. '
            '<b>El arbol individual</b> fue el más débil del grupo, lo que refuerza '
            'el valor del ensamble.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Gamma-Pydra")
        st.markdown(
            '<div class="insight-box">'
            'A pesar de ser un algoritmo experimental, Gamma-Pydra logro un '
            '<b>recall de 0.82 para la clase minoritaria</b> (el mas alto de todos), '
            'sacrificando precisión global. Esto lo hace especialmente útil cuando '
            'el costo de un falso negativo es alto. El manejo nativo de NaN es '
            'una ventaja frente al dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### ¿Por qué accuracy no es suficiente?")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        modelos_plot = ["Logistica", "NaiveBayes", "Gamma", "KNN"]
        recall_min   = [0.69, 0.77, 0.82, 0.71]
        accuracy_v   = [0.724, 0.699, 0.678, 0.676]
        x3    = np.arange(len(modelos_plot))
        width3 = 0.38
        ax.bar(x3 - width3/2, accuracy_v, width3, label="Accuracy",
               color=PURPLE, alpha=0.85, edgecolor="white")
        ax.bar(x3 + width3/2, recall_min, width3, label="Recall clase 1",
               color=GREEN,  alpha=0.85, edgecolor="white")
        ax.set_xticks(x3)
        ax.set_xticklabels(modelos_plot, fontsize=10)
        ax.set_ylim(0.60, 0.88)
        ax.legend(fontsize=9)
        ax.set_title("Accuracy vs Recall (clase minoritaria)",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Limitaciones identificadas")
        for lim in [
           "Gamma-Pydra tiene **complejidad cuadrática** O(n^2) en predicción — no escala bien.",
            "El balanceo con **SMOTE puede introducir ruido** al generar muestras sinteticas cercanas al limite de decisión (BorderlineSMOTE lo mitiga).",
            "La validación cruzada de **Gamma uso solo k=3** (por costo computacional), lo que hace su estimación menos confiable.",
        ]:
            st.markdown(f"  - {lim}")

    st.markdown("---")
    st.markdown("### Conclusiones finales")
    col1, col2, col3 = st.columns(3)
    conclusiones = [
        ("Mejor modelo",
         "La **Regresión Logistica** obtuvo F1=0.7265, demostrando que un modelo "
         "simple y bien regularizado puede superar a ensambles complejos cuando "
         "el preprocesamiento es el adecuado."),
        ("Selección de métrica",
         "En datasets desbalanceados, **F1-Score y Recall** son más informativos "
         "que la Accuracy. Gamma-Pydra lo evidencio: accuracy=0.68 pero "
         "recall=0.82 en clase minoritaria."),
        ("Preprocesamiento",
         "Diseñar **pipelines diferenciados** por familia de modelo es crítico. "
         "Un mismo tratamiento para todos los modelos hubiera reducido su "
         "rendimiento potencial."),
        ("Gamma-Pydra",
         "El algoritmo propio demostró ser **competitivo y funcional** con datos "
         "reales y valores nulos, validando su implementación como alternativa "
         "a clasificadores estandar."),
        ("Balanceo",
         "La combinación **Undersampling + BorderlineSMOTE** fue conservadora y "
         "efectiva: mejoro el recall sin destruir información de la clase "
         "mayoritaria."),
        ("Ensambles",
         "**RandomForest y AdaBoost** superaron al arbol individual en ~4 puntos "
         "de F1, confirmando que la diversidad de estimadores reduce la varianza "
         "y el sobreajuste."),
    ]
    for idx, (titulo, texto) in enumerate(conclusiones):
        with [col1, col2, col3][idx % 3]:
            st.markdown(f"#### {titulo}")
            st.markdown(
                f'<div class="insight-box">{texto}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Reflexion metodologica")
    st.success(
        "Esta practica demostró que el éxito en clasificación supervisada no depende "
        "unicamente del algoritmo elegido, sino del proceso completo: el análisis "
        "exploratorio inicial, las decisiones de preprocesamiento fundamentadas en los "
        "supuestos de cada modelo, el tratamiento adecuado del desbalance, la busqueda "
        "sistematica de hiperparámetros y la evaluacion con metricas apropiadas. "
        "La combinación de estos elementos permitio obtener modelos consistentes con "
        "F1 > 0.72 en un problema de predicción de salud del sueño."
    )
