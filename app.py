import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# K-Medoids (manual implementation)
def kmedoids_fit(X, k, max_iter=300, random_state=42):
    """Pure-numpy K-Medoids (PAM simplified)."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    medoid_idx = rng.choice(n, k, replace=False)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - X[medoid_idx][None, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_medoids = np.copy(medoid_idx)
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            intra = np.sum(np.linalg.norm(X[members][:, None] - X[members][None, :], axis=2), axis=1)
            new_medoids[c] = members[np.argmin(intra)]

        if np.all(new_medoids == medoid_idx):
            break
        medoid_idx = new_medoids

    dists = np.linalg.norm(X[:, None] - X[medoid_idx][None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    inertia = sum(
        np.sum(np.linalg.norm(X[labels == c] - X[medoid_idx[c]], axis=1) ** 2)
        for c in range(k)
    )
    return labels, medoid_idx, inertia


# Page Configuration
st.set_page_config(
    page_title="DataMining FD1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #f8fafc;
        --surface: #ffffff;
        --border: #e2e8f0;
        --primary: #2563eb;
        --primary-light: #dbeafe;
        --primary-dark: #1d4ed8;
        --success: #059669;
        --success-light: #d1fae5;
        --text: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --shadow: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-lg: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
        box-shadow: var(--shadow);
    }

    h1 {
        font-weight: 700;
        color: var(--text);
        font-size: 1.75rem;
        margin-bottom: 1.5rem;
    }

    h2 {
        font-weight: 600;
        color: var(--text);
        font-size: 1.25rem;
        margin-top: 2rem;
    }

    h3 {
        font-weight: 600;
        color: var(--text);
        font-size: 1.1rem;
    }

    /* Sidebar Styling */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary);
        padding: 1rem 0;
        border-bottom: 2px solid var(--primary-light);
        margin-bottom: 1rem;
    }

    .sidebar-info {
        font-size: 0.75rem;
        color: var(--text-secondary);
        line-height: 1.6;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        margin-top: 1rem;
    }

    /* Card Styling */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: var(--shadow);
        transition: box-shadow 0.2s ease;
    }

    .metric-card:hover {
        box-shadow: var(--shadow-lg);
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section Headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
    }

    /* Info Box */
    .info-box {
        background: var(--primary-light);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--text);
        font-size: 0.9rem;
    }

    .success-box {
        background: var(--success-light);
        border-left: 4px solid var(--success);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--text);
        font-size: 0.9rem;
    }

    /* Button Styling */
    .stButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow) !important;
    }

    .stButton > button:hover {
        background: var(--primary-dark) !important;
        box-shadow: var(--shadow-lg) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Form Labels */
    .stSelectbox label, .stSlider label, .stRadio label,
    .stNumberInput label, .stFileUploader label, .stMultiselect label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg);
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow);
    }

    /* Expander Styling */
    .stExpander {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Radio Button Navigation */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio > div > label {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--surface);
        transition: all 0.2s ease;
    }

    .stRadio > div > label:hover {
        background: var(--primary-light);
        border-color: var(--primary);
    }

    /* Hide hamburger and footer */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Session State
for key in ["df_raw", "df_clean", "df_norm", "target_col", "feature_cols",
            "norm_method", "cluster_labels_km", "cluster_labels_kmed",
            "cluster_labels_agnes", "cluster_labels_diana", "cluster_labels_dbscan"]:
    if key not in st.session_state:
        st.session_state[key] = None


# Sidebar Navigation
st.sidebar.markdown('<div class="sidebar-title">DataMining FD1</div>', unsafe_allow_html=True)

volet = st.sidebar.radio(
    "Navigation",
    ["Volet 1 - Pretraitement",
     "Volet 2 - Clustering",
     "Volet 3 - Classification"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div class="sidebar-info">
    <strong>Module : Fouille de Donnees 1</strong><br>
    Alem Mohamed<br>
</div>
""", unsafe_allow_html=True)


# Helper Functions
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def apply_light_style(fig):
    fig.patch.set_facecolor('#ffffff')
    for ax in fig.get_axes():
        ax.set_facecolor('#f8fafc')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.xaxis.label.set_color('#475569')
        ax.yaxis.label.set_color('#475569')
        ax.title.set_color('#1e293b')
        ax.title.set_fontsize(12)
        ax.title.set_fontweight('600')
        for spine in ax.spines.values():
            spine.set_edgecolor('#e2e8f0')
            spine.set_linewidth(1)
    return fig


# VOLET 1 - PRETRAITEMENT
if volet.startswith("Volet 1"):
    st.markdown("# Volet 1 - Pretraitement des Donnees")

    # Importation
    st.markdown('<div class="section-header">01 - Importation des Donnees</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger un fichier dataset",
        type=["csv", "xlsx", "xls", "data", "txt"],
        help="Formats supportes : CSV, Excel, TXT"
    )

    if uploaded:
        try:
            name = uploaded.name.lower()
            if name.endswith(".csv") or name.endswith(".data") or name.endswith(".txt"):
                for sep in [',', ';', '\\t', ' ']:
                    try:
                        df = pd.read_csv(uploaded, sep=sep, on_bad_lines='skip')
                        if df.shape[1] > 1:
                            break
                        uploaded.seek(0)
                    except Exception:
                        uploaded.seek(0)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df_raw = df.copy()
            st.session_state.df_clean = df.copy()
            st.markdown(f'<div class="success-box">Dataset charge avec succes : {df.shape[0]} lignes x {df.shape[1]} colonnes</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")

    df = st.session_state.df_clean

    if df is not None:
        # Exploration
        st.markdown('<div class="section-header">02 - Exploration des Donnees</div>', unsafe_allow_html=True)

        cols = st.columns(4)
        with cols[0]: metric_card("Instances", df.shape[0])
        with cols[1]: metric_card("Attributs", df.shape[1])
        with cols[2]: metric_card("Valeurs Manquantes", int(df.isnull().sum().sum()))
        with cols[3]: metric_card("Doublons", int(df.duplicated().sum()))

        with st.expander("Apercu du Dataset (10 premieres lignes)"):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Types des Attributs"):
            type_df = pd.DataFrame({
                "Colonne": df.columns,
                "Type": df.dtypes.values,
                "Non-nuls": df.notnull().sum().values,
                "Nuls": df.isnull().sum().values
            })
            st.dataframe(type_df, use_container_width=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            with st.expander("Statistiques Descriptives"):
                desc = df[num_cols].describe(percentiles=[.25, .5, .75]).T
                desc.columns = [c.capitalize() for c in desc.columns]
                desc["Mode"] = df[num_cols].mode().iloc[0]
                desc["Mediane"] = df[num_cols].median()
                st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

        # Nettoyage
        st.markdown('<div class="section-header">03 - Nettoyage des Donnees</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            miss_method = st.selectbox(
                "Strategie de gestion des valeurs manquantes",
                ["Moyenne", "Mediane", "Mode", "Supprimer les lignes"]
            )
            if st.button("Appliquer le Nettoyage"):
                df_c = st.session_state.df_clean.copy()
                df_c.drop_duplicates(inplace=True)
                for col in df_c.select_dtypes(include=np.number).columns:
                    if miss_method == "Moyenne":
                        df_c[col].fillna(df_c[col].mean(), inplace=True)
                    elif miss_method == "Mediane":
                        df_c[col].fillna(df_c[col].median(), inplace=True)
                    elif miss_method == "Mode":
                        df_c[col].fillna(df_c[col].mode()[0], inplace=True)
                    else:
                        df_c.dropna(inplace=True)
                        break
                st.session_state.df_clean = df_c
                df = df_c
                st.markdown(f'<div class="success-box">Nettoyage effectue : {df_c.shape[0]} lignes restantes</div>', unsafe_allow_html=True)

        with col_b:
            remaining = int(st.session_state.df_clean.isnull().sum().sum())
            metric_card("Valeurs Manquantes Restantes", remaining)

        # Normalisation
        st.markdown('<div class="section-header">04 - Normalisation</div>', unsafe_allow_html=True)

        norm_method = st.radio(
            "Methode de Normalisation",
            ["Min-Max Scaling", "Z-score (Standardisation)"],
            horizontal=True
        )

        if st.button("Normaliser les Donnees"):
            df_n = st.session_state.df_clean.copy()
            num = df_n.select_dtypes(include=np.number).columns
            if norm_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            df_n[num] = scaler.fit_transform(df_n[num])
            st.session_state.df_norm = df_n
            st.session_state.norm_method = norm_method
            st.markdown(f'<div class="success-box">Normalisation "{norm_method}" appliquee avec succes</div>', unsafe_allow_html=True)
            with st.expander("Apercu des Donnees Normalisees"):
                st.dataframe(df_n[num].head(), use_container_width=True)

        # Visualisation
        st.markdown('<div class="section-header">05 - Visualisation</div>', unsafe_allow_html=True)

        df_viz = st.session_state.df_norm if st.session_state.df_norm is not None else df
        num_cols = df_viz.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            tab_box, tab_scatter = st.tabs(["Boxplots", "Nuage de Points"])

            with tab_box:
                fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 1.2), 5))
                palette = sns.color_palette("Blues", len(num_cols))
                bp = ax.boxplot(
                    [df_viz[c].dropna() for c in num_cols],
                    labels=num_cols,
                    patch_artist=True,
                    medianprops=dict(color='#2563eb', linewidth=2)
                )
                for patch, color in zip(bp['boxes'], palette):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_title("Distribution des Attributs - Boxplots", fontsize=12, fontweight='600')
                plt.xticks(rotation=45, ha='right')
                apply_light_style(fig)
                st.pyplot(fig)
                plt.close()

            with tab_scatter:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("Axe X", num_cols, key="sx")
                y_col = c2.selectbox("Axe Y", num_cols, index=min(1, len(num_cols)-1), key="sy")

                cat_cols = df_viz.select_dtypes(exclude=np.number).columns.tolist()
                hue_col = cat_cols[-1] if cat_cols else None

                fig2, ax2 = plt.subplots(figsize=(7, 5))
                if hue_col:
                    groups = df_viz[hue_col].unique()
                    pal = sns.color_palette("viridis", len(groups))
                    for grp, col in zip(groups, pal):
                        mask = df_viz[hue_col] == grp
                        ax2.scatter(df_viz.loc[mask, x_col], df_viz.loc[mask, y_col],
                                    label=str(grp), color=col, alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
                    ax2.legend(fontsize=8, framealpha=0.9)
                else:
                    ax2.scatter(df_viz[x_col], df_viz[y_col],
                                color='#2563eb', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
                ax2.set_xlabel(x_col, fontweight='500')
                ax2.set_ylabel(y_col, fontweight='500')
                ax2.set_title(f"{x_col} vs {y_col}", fontsize=12, fontweight='600')
                apply_light_style(fig2)
                st.pyplot(fig2)
                plt.close()
        else:
            st.info("Aucune colonne numerique detectee pour la visualisation.")

    else:
        st.markdown('<div class="info-box">Veuillez charger un dataset pour commencer l\'analyse.</div>', unsafe_allow_html=True)


# VOLET 2 - CLUSTERING
elif volet.startswith("Volet 2"):
    st.markdown("# Volet 2 - Clustering")

    df_work = st.session_state.df_norm if st.session_state.df_norm is not None else st.session_state.df_clean
    if df_work is None:
        st.warning("Veuillez d'abord charger un dataset dans le Volet 1.")
        st.stop()

    num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("Aucune colonne numerique disponible.")
        st.stop()

    # Column selector
    with st.expander("Configuration - Selection des Features", expanded=True):
        selected_features = st.multiselect(
            "Colonnes utilisees pour le clustering",
            num_cols,
            default=num_cols
        )
    if not selected_features:
        st.warning("Veuillez selectionner au moins une colonne.")
        st.stop()

    X = df_work[selected_features].dropna().values

    # Courbe d'Elbow
    st.markdown('<div class="section-header">01 - Courbe d\'Elbow</div>', unsafe_allow_html=True)

    k_max = st.slider("Nombre maximum de clusters a tester", 2, 15, 10)
    if st.button("Tracer la Courbe d'Elbow"):
        inertias = []
        K_range = range(1, k_max + 1)
        with st.spinner("Calcul en cours..."):
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(K_range, inertias, 'o-', color='#2563eb', linewidth=2.5, markersize=8)
        ax.fill_between(K_range, inertias, alpha=0.15, color='#2563eb')
        ax.set_xlabel("Nombre de clusters (k)", fontweight='500')
        ax.set_ylabel("Inertie intra-cluster", fontweight='500')
        ax.set_title("Courbe d'Elbow - Methode du Coude", fontsize=12, fontweight='600')
        apply_light_style(fig)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="info-box"><b>Interpretation :</b> Recherchez le "coude" de la courbe ou la diminution de l\'inertie ralentit significativement.</div>', unsafe_allow_html=True)

    # K-Means
    st.markdown('<div class="section-header">02 - K-Means</div>', unsafe_allow_html=True)

    k_km = st.number_input("Nombre de clusters (K-Means)", 2, 20, 3, key="k_km")

    if st.button("Executer K-Means"):
        with st.spinner("K-Means en cours..."):
            km = KMeans(n_clusters=int(k_km), random_state=42, n_init=10)
            labels_km = km.fit_predict(X)
            st.session_state.cluster_labels_km = labels_km
            inertia_km = km.inertia_
            sil_km = silhouette_score(X, labels_km) if len(set(labels_km)) > 1 else 0

        cols = st.columns(3)
        with cols[0]: metric_card("Inertie", f"{inertia_km:.2f}")
        with cols[1]: metric_card("Silhouette", f"{sil_km:.4f}")
        with cols[2]: metric_card("Nombre de Clusters", int(k_km))

        pca = PCA(n_components=2)
        X2d = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5))
        palette = sns.color_palette("viridis", int(k_km))
        for c in range(int(k_km)):
            mask = labels_km == c
            ax.scatter(X2d[mask, 0], X2d[mask, 1], color=palette[c], alpha=0.7, s=35,
                      label=f"Cluster {c}", edgecolors='white', linewidth=0.5)
        centers2d = pca.transform(km.cluster_centers_)
        ax.scatter(centers2d[:, 0], centers2d[:, 1], c='white', marker='X', s=150,
                  zorder=5, label="Centroides", edgecolors='#1e293b', linewidth=1)
        ax.set_title(f"K-Means (k={int(k_km)}) - Projection PCA 2D", fontsize=12, fontweight='600')
        ax.legend(fontsize=8, framealpha=0.95)
        apply_light_style(fig)
        st.pyplot(fig)
        plt.close()

    # K-Medoids
    st.markdown('<div class="section-header">03 - K-Medoids</div>', unsafe_allow_html=True)

    k_kmed = st.number_input("Nombre de clusters (K-Medoids)", 2, 20, 3, key="k_kmed")

    if st.button("Executer K-Medoids"):
        with st.spinner("K-Medoids en cours (peut prendre quelques secondes)..."):
            labels_kmed, medoid_idx, inertia_kmed = kmedoids_fit(X, int(k_kmed))
            st.session_state.cluster_labels_kmed = labels_kmed
            sil_kmed = silhouette_score(X, labels_kmed) if len(set(labels_kmed)) > 1 else 0

        cols = st.columns(3)
        with cols[0]: metric_card("Inertie", f"{inertia_kmed:.2f}")
        with cols[1]: metric_card("Silhouette", f"{sil_kmed:.4f}")
        with cols[2]: metric_card("Nombre de Clusters", int(k_kmed))

        pca = PCA(n_components=2)
        X2d = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5))
        palette = sns.color_palette("viridis", int(k_kmed))
        for c in range(int(k_kmed)):
            mask = labels_kmed == c
            ax.scatter(X2d[mask, 0], X2d[mask, 1], color=palette[c], alpha=0.7, s=35,
                      label=f"Cluster {c}", edgecolors='white', linewidth=0.5)
        med2d = X2d[medoid_idx]
        ax.scatter(med2d[:, 0], med2d[:, 1], c='white', marker='D', s=130, zorder=5,
                  label="Medoides", edgecolors='#1e293b', linewidth=1)
        ax.set_title(f"K-Medoids (k={int(k_kmed)}) - Projection PCA 2D", fontsize=12, fontweight='600')
        ax.legend(fontsize=8, framealpha=0.95)
        apply_light_style(fig)
        st.pyplot(fig)
        plt.close()

    # AGNES & DIANA
    st.markdown('<div class="section-header">04 - Clustering Hierarchique (AGNES / DIANA)</div>', unsafe_allow_html=True)

    k_hier = st.number_input("Nombre de clusters (Hierarchique)", 2, 20, 3, key="k_hier")

    col_agnes, col_diana = st.columns(2)

    with col_agnes:
        if st.button("Executer AGNES"):
            with st.spinner("AGNES (agglomeratif) en cours..."):
                agnes = AgglomerativeClustering(n_clusters=int(k_hier), linkage='ward')
                labels_agnes = agnes.fit_predict(X)
                st.session_state.cluster_labels_agnes = labels_agnes
                sil_agnes = silhouette_score(X, labels_agnes) if len(set(labels_agnes)) > 1 else 0

            metric_card("Silhouette AGNES", f"{sil_agnes:.4f}")
            pca = PCA(n_components=2)
            X2d = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(5, 4))
            palette = sns.color_palette("viridis", int(k_hier))
            for c in range(int(k_hier)):
                mask = labels_agnes == c
                ax.scatter(X2d[mask, 0], X2d[mask, 1], color=palette[c], alpha=0.7, s=30,
                          label=f"C{c}", edgecolors='white', linewidth=0.5)
            ax.set_title(f"AGNES (k={int(k_hier)})", fontsize=11, fontweight='600')
            ax.legend(fontsize=7, framealpha=0.95)
            apply_light_style(fig)
            st.pyplot(fig)
            plt.close()

    with col_diana:
        if st.button("Executer DIANA"):
            with st.spinner("DIANA (divisif) en cours..."):
                diana = AgglomerativeClustering(n_clusters=int(k_hier), linkage='complete')
                labels_diana = diana.fit_predict(X)
                st.session_state.cluster_labels_diana = labels_diana
                sil_diana = silhouette_score(X, labels_diana) if len(set(labels_diana)) > 1 else 0

            metric_card("Silhouette DIANA", f"{sil_diana:.4f}")
            pca = PCA(n_components=2)
            X2d = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(5, 4))
            palette = sns.color_palette("plasma", int(k_hier))
            for c in range(int(k_hier)):
                mask = labels_diana == c
                ax.scatter(X2d[mask, 0], X2d[mask, 1], color=palette[c], alpha=0.7, s=30,
                          label=f"C{c}", edgecolors='white', linewidth=0.5)
            ax.set_title(f"DIANA (k={int(k_hier)})", fontsize=11, fontweight='600')
            ax.legend(fontsize=7, framealpha=0.95)
            apply_light_style(fig)
            st.pyplot(fig)
            plt.close()

    # DBSCAN
    st.markdown('<div class="section-header">05 - DBSCAN</div>', unsafe_allow_html=True)

    col_eps, col_min = st.columns(2)
    eps_val = col_eps.number_input("Epsilon (rayon de voisinage)", 0.01, 10.0, 0.5, step=0.05)
    min_pts = col_min.number_input("Points minimum", 1, 50, 5)

    if st.button("Executer DBSCAN"):
        with st.spinner("DBSCAN en cours..."):
            db = DBSCAN(eps=eps_val, min_samples=int(min_pts))
            labels_db = db.fit_predict(X)
            st.session_state.cluster_labels_dbscan = labels_db

        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise = int((labels_db == -1).sum())

        cols = st.columns(3)
        with cols[0]: metric_card("Clusters Trouves", n_clusters_db)
        with cols[1]: metric_card("Points de Bruit", n_noise)
        if n_clusters_db > 1:
            mask_valid = labels_db != -1
            sil_db = silhouette_score(X[mask_valid], labels_db[mask_valid]) if mask_valid.sum() > 1 else 0
            with cols[2]: metric_card("Silhouette", f"{sil_db:.4f}")

        pca = PCA(n_components=2)
        X2d = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5))
        unique_labels = sorted(set(labels_db))
        palette = sns.color_palette("viridis", max(len(unique_labels), 2))
        for i, lbl in enumerate(unique_labels):
            mask = labels_db == lbl
            color = '#94a3b8' if lbl == -1 else palette[i]
            label_str = "Bruit" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(X2d[mask, 0], X2d[mask, 1], color=color, alpha=0.7, s=30,
                      label=label_str, edgecolors='white', linewidth=0.5)
        ax.set_title(f"DBSCAN (eps={eps_val}, minPts={min_pts})", fontsize=12, fontweight='600')
        ax.legend(fontsize=8, framealpha=0.95)
        apply_light_style(fig)
        st.pyplot(fig)
        plt.close()

    # Comparaison
    st.markdown('<div class="section-header">06 - Comparaison des Methodes</div>', unsafe_allow_html=True)

    if st.button("Generer l'Histogramme Comparatif"):
        methods, inertias, silhouettes = [], [], []

        def compute_inertia_sil(labels, X):
            if labels is None or len(set(labels)) < 2:
                return None, None
            inertia = sum(
                np.sum(np.linalg.norm(X[labels == c] - X[labels == c].mean(axis=0), axis=1) ** 2)
                for c in set(labels) if c != -1
            )
            mask = labels != -1
            sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0
            return inertia, sil

        algo_map = {
            "K-Means": st.session_state.cluster_labels_km,
            "K-Medoids": st.session_state.cluster_labels_kmed,
            "AGNES": st.session_state.cluster_labels_agnes,
            "DIANA": st.session_state.cluster_labels_diana,
            "DBSCAN": st.session_state.cluster_labels_dbscan,
        }
        for name, labels in algo_map.items():
            iner, sil = compute_inertia_sil(labels, X)
            if iner is not None:
                methods.append(name)
                inertias.append(iner)
                silhouettes.append(sil)

        if methods:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = ['#2563eb', '#059669', '#7c3aed', '#ea580c', '#0891b2'][:len(methods)]

            axes[0].bar(methods, inertias, color=colors, edgecolor='#e2e8f0', linewidth=1)
            axes[0].set_title("Inertie par Methode", fontsize=12, fontweight='600')
            axes[0].set_ylabel("Inertie", fontweight='500')

            axes[1].bar(methods, silhouettes, color=colors, edgecolor='#e2e8f0', linewidth=1)
            axes[1].set_title("Score de Silhouette par Methode", fontsize=12, fontweight='600')
            axes[1].set_ylabel("Silhouette", fontweight='500')
            axes[1].set_ylim(0, 1)

            for ax in axes:
                for bar in ax.patches:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(inertias + [1]),
                        f"{bar.get_height():.2f}",
                        ha='center', va='bottom', fontsize=9, color='#1e293b', fontweight='500'
                    )

            apply_light_style(fig)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Veuillez d'abord executer au moins un algorithme de clustering.")


# VOLET 3 - CLASSIFICATION
elif volet.startswith("Volet 3"):
    st.markdown("# Volet 3 - Classification Supervisee")

    df_work = st.session_state.df_norm if st.session_state.df_norm is not None else st.session_state.df_clean
    if df_work is None:
        st.warning("Veuillez d'abord charger un dataset dans le Volet 1.")
        st.stop()

    # Configuration
    st.markdown('<div class="section-header">01 - Configuration</div>', unsafe_allow_html=True)

    all_cols = df_work.columns.tolist()
    c1, c2 = st.columns(2)
    target_col = c1.selectbox("Variable cible (classe)", all_cols, index=len(all_cols)-1)
    test_size = c2.slider("Taille de l'ensemble de test (%)", 10, 40, 20)

    feature_cols = [c for c in df_work.select_dtypes(include=np.number).columns if c != target_col]

    if not feature_cols:
        st.error("Aucune feature numerique disponible.")
        st.stop()

    st.markdown(f"**Features selectionnees :** `{', '.join(feature_cols)}`")

    # Modele
    st.markdown('<div class="section-header">02 - Selection du Modele</div>', unsafe_allow_html=True)

    model_name = st.selectbox("Algorithme de Classification", [
        "K-Nearest Neighbors (KNN)",
        "Arbre de Decision",
        "Naive Bayes",
        "SVM",
        "Regression Logistique"
    ])

    # Hyperparameters
    model_params = {}
    if model_name == "K-Nearest Neighbors (KNN)":
        model_params['n_neighbors'] = st.slider("Nombre de voisins (k)", 1, 20, 5)
    elif model_name == "Arbre de Decision":
        model_params['max_depth'] = st.slider("Profondeur maximale (0 = illimitee)", 0, 20, 5)
        if model_params['max_depth'] == 0:
            model_params['max_depth'] = None
    elif model_name == "SVM":
        model_params['C'] = st.slider("Parametre de regularisation C", 0.01, 10.0, 1.0)

    if st.button("Entrainer et Evaluer"):
        X = df_work[feature_cols].values
        y = df_work[target_col].values

        # Remove rows with NaN
        mask = ~(np.isnan(X).any(axis=1) | pd.isnull(y))
        X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        # Build model
        if model_name == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=model_params['n_neighbors'])
        elif model_name == "Arbre de Decision":
            model = DecisionTreeClassifier(max_depth=model_params.get('max_depth'), random_state=42)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "SVM":
            model = SVC(C=model_params.get('C', 1.0), random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        with st.spinner("Entrainement en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Metriques
        st.markdown('<div class="section-header">03 - Metriques de Performance</div>', unsafe_allow_html=True)

        acc = accuracy_score(y_test, y_pred)
        avg = 'binary' if len(set(y)) == 2 else 'weighted'
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: metric_card("Accuracy", f"{acc*100:.2f}%")
        with mc2: metric_card("Precision", f"{prec:.4f}")
        with mc3: metric_card("Recall", f"{rec:.4f}")
        with mc4: metric_card("F1-Score", f"{f1:.4f}")

        # Matrice de Confusion
        st.markdown('<div class="section-header">04 - Matrice de Confusion</div>', unsafe_allow_html=True)

        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y))
        fig, ax = plt.subplots(figsize=(min(8, len(labels)*1.5 + 2), min(6, len(labels)*1.2 + 2)))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='#e2e8f0',
            ax=ax, cbar_kws={'shrink': 0.8}
        )
        ax.set_xlabel("Prediction", fontweight='500')
        ax.set_ylabel("Reel", fontweight='500')
        ax.set_title(f"Matrice de Confusion - {model_name}", fontsize=12, fontweight='600')
        apply_light_style(fig)
        st.pyplot(fig)
        plt.close()

        # Rapport
        with st.expander("Rapport de Classification Complet"):
            report = classification_report(y_test, y_pred, zero_division=0)
            st.code(report, language="text")

        # Train/Test info
        st.markdown(f'<div class="info-box"><b>Details :</b> Ensemble d\'entrainement : {len(X_train)} instances | Ensemble de test : {len(X_test)} instances</div>', unsafe_allow_html=True)
