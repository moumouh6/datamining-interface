# DataMining FD1 - Application Streamlit

Application de fouille de donnees developpee avec Streamlit pour le module Data Mining (FD1). L'application propose des outils de pretraitement, de clustering non supervise et de classification supervisee.

---

## Table des matieres

1. [Installation](#installation)
2. [Lancement](#lancement)
3. [Fonctionnalites](#fonctionnalites)
4. [Workflow d'utilisation](#workflow-dutilisation)
5. [Details des algorithmes](#details-des-algorithmes)
6. [Structure du projet](#structure-du-projet)

---

## Installation

### Prerequis

- Python 3.8 ou superieur
- pip (gestionnaire de paquets Python)

### Etape 1 : Cloner ou telecharger le projet

```bash
cd Projet_fd
```

### Etape 2 : Creer un environnement virtuel (recommande)

```bash
python -m venv venv

# Activation Windows
venv\Scripts\activate

# Activation Linux/Mac
source venv/bin/activate
```

### Etape 3 : Installer les dependances

```bash
pip install -r Requirements.txt
```

**Contenu de Requirements.txt :**
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## Lancement

Pour demarrer l'application :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur a l'adresse `http://localhost:8501`.

---

## Fonctionnalites

L'interface est organisee en trois volets principaux :

### Volet 1 - Pretraitement

| Fonction | Description |
|----------|-------------|
| **Importation** | Chargement des datasets (CSV, Excel, TXT, .data) |
| **Exploration** | Apercu des donnees, types d'attributs, statistiques descriptives |
| **Nettoyage** | Gestion des valeurs manquantes (moyenne, mediane, mode, suppression) et doublons |
| **Normalisation** | Min-Max Scaling et Z-score (Standardisation) |
| **Visualisation** | Boxplots et nuages de points interactifs |

### Volet 2 - Clustering

Algorithmes de clustering implementes :

| Algorithme | Type | Particularites |
|------------|------|----------------|
| **K-Means** | Centroides | Rapide, sensible aux outliers |
| **K-Medoids** | Medoides | Plus robuste aux outliers |
| **AGNES** | Hierarchique ascendant | Linkage Ward |
| **DIANA** | Hierarchique descendant | Linkage Complete |
| **DBSCAN** | Densite | Detection automatique du nombre de clusters |

**Outils d'analyse :**
- Courbe d'Elbow (methode du coude)
- Projection PCA 2D pour visualisation
- Comparaison des inerties et silhouettes

### Volet 3 - Classification

Algorithmes de classification :

| Algorithme | Hyperparametres ajustables |
|------------|---------------------------|
| **K-Nearest Neighbors** | Nombre de voisins k |
| **Arbre de Decision** | Profondeur maximale |
| **Naive Bayes** | - (gaussien) |
| **SVM** | Parametre C |
| **Regression Logistique** | - (max_iter=1000) |

**Metriques calculees :**
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion (heatmap)
- Rapport de classification detaille

---

## Workflow d'utilisation

### Etape 1 : Preparation des donnees (Volet 1)
1. Chargez votre dataset
2. Explorez les statistiques descriptives
3. Nettoyez les valeurs manquantes
4. Normalisez si necessaire (recommande pour le clustering)

### Etape 2 : Analyse (Volet 2 ou 3)

**Pour le clustering :**
- Selectionnez les features numeriques
- Utilisez la courbe d'Elbow pour estimer k optimal
- Testez differents algorithmes
- Comparez les scores de silhouette

**Pour la classification :**
- Selectionnez la variable cible
- Ajustez la taille du test set
- Configurez les hyperparametres
- Analysez la matrice de confusion

---

## Details des algorithmes

### K-Medoids (Implementation personnalisee)
L'algorithme K-Medoids est implemente en pur NumPy sans dependance externe, selon la methode PAM (Partitioning Around Medoids) simplifiee. Contrairement a K-Means qui utilise les centroides (moyennes), K-Medoids selectionne des points reels comme centres, ce qui le rend plus robuste aux valeurs aberrantes.

### DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) detecte automatiquement :
- Le nombre de clusters
- Les points de "bruit" (outliers)
- Parametres : Epsilon (rayon de voisinage) et MinPts (points minimum)

### DIANA
DIANA (DIvisive ANAlysis) est simule via AgglomerativeClustering avec linkage='complete', ce qui produit un comportement similaire a l'approche descendante.

---

## Structure du projet

```
Projet_fd/
|
|-- app.py                  # Application principale Streamlit
|-- Readme.md              # Documentation (ce fichier)
|-- Requirements.txt     # Dependances Python
---

## Formats de fichiers supportes

- **CSV** : Separateur auto-detecte (`,`, `;`, `\t`, espace)
- **Excel** : `.xlsx`, `.xls`
- **Texte** : `.txt`, `.data`

---

## Notes importantes

1. **K-Medoids** peut etre plus lent sur de grands datasets (>5000 instances) car il calcule toutes les distances par paires
2. **Normalisation** est fortement recommandee avant le clustering pour les algorithmes bases sur les distances (K-Means, K-Medoids)
3. **DBSCAN** est particulierement adapte aux donnees avec des clusters de densites et formes variables
4. Les **matrices de confusion** s'adaptent automatiquement au nombre de classes



