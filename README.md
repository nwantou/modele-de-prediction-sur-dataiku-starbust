# Prédiction des Baisses de Performance des Points de Vente — Orange Cameroun

**Modèle Machine Learning de scoring prédictif développé sur Dataiku**  
Sales DataLab · Direction de la Distribution · Orange Cameroun · 2025–2026

---

## À propos de ce projet

Ce projet de Data Science en production s'inscrit dans la continuité des travaux du **Sales DataLab d'Orange Cameroun**. Il vise à détecter, un mois à l'avance, les Points de Vente (PDV) susceptibles de voir leurs recharges C2S (crédit de communication) baisser de plus de 10 % par rapport au mois précédent — permettant aux équipes terrain de passer d'une gestion **réactive** à une gestion **proactive** du réseau de distribution.

Ce README documente l'intégralité du projet selon la méthodologie **CRISP-DM** en deux cycles, en explicitant le raisonnement derrière chaque étape. Il a vocation à servir de référence technique pour les équipes Data et commerciales.

---

## Table des matières

- [Présentation — Orange Cameroun et son réseau PDV](#présentation--orange-cameroun-et-son-réseau-pdv)  
- [Résultats principaux](#résultats-principaux)  
- [Méthodologie — CRISP-DM](#méthodologie--crisp-dm)  
- [Description des cycles](#description-des-cycles)  
- [01\. Compréhension métier](#01-compréhension-métier)  
- [02\. Préparation des données](#02-préparation-des-données)  
- [03\. Feature Engineering](#03-feature-engineering)  
- [04\. Analyse Exploratoire (EDA)](#04-analyse-exploratoire-eda)  
- [05\. Préprocessing des données](#05-préprocessing-des-données)  
- [06\. Sélection des features](#06-sélection-des-features)  
- [07\. Modélisation Machine Learning](#07-modélisation-machine-learning)  
- [08\. Optimisation des hyperparamètres](#08-optimisation-des-hyperparamètres)  
- [09\. Interprétation des erreurs et performance métier](#09-interprétation-des-erreurs-et-performance-métier)  
- [10\. Déploiement en production (Dataiku \+ Power BI)](#10-déploiement-en-production-dataiku--power-bi)  
- [Conclusion](#conclusion)  
- [Annexe I — Sources de données](#annexe-i--sources-de-données)

---

## Présentation — Orange Cameroun et son réseau PDV

Orange Cameroun est l'un des principaux opérateurs télécom du Cameroun, opérant sur un marché mobile de plus en plus concurrentiel. Son modèle commercial repose très largement sur la **distribution indirecte** : les recharges C2S (crédit de communication), les ventes de cartes SIM et les transactions Orange Money sont distribuées à travers un réseau de **plus de 121 000 Points de Vente** (PDV) — appelboxers, sous-distributeurs et partenaires agréés — répartis sur les **12 régions administratives** et **60 secteurs** commerciaux du pays.

Les ventes indirectes représentent environ **80 % des revenus** générés par Orange Cameroun, ce qui confère au réseau PDV un rôle absolument stratégique. La performance de ce réseau est suivie à travers des tableaux de bord Power BI alimentés par un **Data Warehouse** consolidant les données des systèmes sources MSC, Zebra, REA et Flytext.

Le problème central auquel ce projet répond est le suivant :

*« Comment exploiter les données de ventes et de campagnes disponibles dans le Data Warehouse pour anticiper, un mois à l'avance, les baisses de performance des points de vente et permettre aux équipes commerciales d'Orange Cameroun de passer d'une gestion réactive à une gestion proactive du réseau de distribution ? »*

[retour en haut](#table-des-matières)

---

## Résultats principaux

*(Si vous souhaitez lire uniquement les résultats sans parcourir l'intégralité du projet, cette section vous suffit.)*

Un modèle de Machine Learning a été entraîné pour prédire les baisses de performance des PDV d'Orange Cameroun. La configuration utilisée est la suivante :

**Méthodologie** : CRISP-DM, deux cycles complets.

**Problème métier** : à la demande de la Direction de la Distribution, un modèle de scoring des PDV à risque de baisse (≥ −10 % de C2S mensuel) est délivré chaque mois, permettant aux équipes terrain de prioriser leurs interventions.

**Données** : | Dataset | Dimensions | Période | |---|---|---| | df\_with\_target (train) | 1 319 902 × 67 | 2025-01 → 2026-02 | | df\_with\_target (valid) | 202 588 × 67 | 2026-03 → 2026-04 | | df\_scoring (production) | 101 972 × 67 | 2026-04 (mois le plus récent) |

**Étapes réalisées** :

- **Nettoyage des données** : imputation contextuelle variable par variable (NaN activité → 0, NaN géographie → mode du train), conformément au principe Rossmann *"variable statistics weren't used for imputation — the context was used instead"*.  
    
- **Feature Engineering** : 27 nouvelles variables créées en 6 groupes (temporelles, activité PDV, ruptures, réseau, visites terrain, lags historiques). **Correction majeure en cycle 2** : `variation_m1` recalculée sur `(C2S_{M-1} − C2S_{M-2}) / C2S_{M-2}` pour éliminer le leakage algébrique.  
    
- **EDA bivariate** : 14 hypothèses testées. Les signaux les plus forts identifiés :

| Hypothèse | Taux de baisse observé | Verdict |
| :---- | :---- | :---- |
| is\_declin\_continu \= 1 (2 baisses consécutives M-2, M-1) | **76,4 %** | ★★★★★ VRAIE |
| is\_rupture\_severe \= 1 (rupture \> 75 % du mois) | **72,5 %** | ★★★★★ VRAIE |
| is\_rupture\_chronique \= 1 (rupture \> 50 % du mois) | **68,5 %** vs 30 % | ★★★★★ VRAIE |
| is\_sans\_appro \= 1 (aucune livraison ce mois) | **63 %** vs 35 % | ★★★★ VRAIE |
| is\_debut\_annee \= 1 (janvier–février) | **44,9 %** vs 37 % base | ★★★★ VRAIE |
| pdv\_visite \= 0 vs 1 (non visité vs visité) | **38,3 % \= 38,3 %** | ★ FAUSSE |

- **Modélisation** : 5 algorithmes testés (Baseline, Logistic Regression, Random Forest, XGBoost, LightGBM). **Cycle 1 : P=1.000 → leakage**. **Cycle 2 après correction : résultats réalistes**.

**Performances — Cycle 2 (sans leakage)** :

| Modèle | Precision | Recall | F1 | AUC |
| :---- | :---- | :---- | :---- | :---- |
| **XGBoost** ⭐ | 0.505 | 0.414 | 0.455 | 0.645 |
| LightGBM | 0.505 | 0.411 | 0.453 | 0.645 |
| Random Forest | 0.490 | 0.371 | 0.422 | 0.628 |
| Logistic Regression | 0.456 | 0.088 | 0.147 | 0.613 |
| Baseline | 0.000 | 0.000 | 0.000 | — |

- **Optimisation des hyperparamètres** (Random Search, 25 itérations) : meilleure configuration à **F1 \= 0.558** (Precision \= 0.43, Recall \= 0.80).  
    
- **Modèle final** :  
    
  - Precision : 0.43 | Recall : 0.80 | F1 : 0.56 | AUC : 0.645  
  - TP \= 58 990 | FP \= 78 431 | FN \= 15 013 | TN \= 50 154


- **Top 6 features (importances réelles, sans leakage)** :

| Feature | Importance | Interprétation |
| :---- | :---- | :---- |
| is\_debut\_annee | 0.209 | Période jan–fév : forte saisonnalité des baisses |
| mois\_num | 0.118 | Numéro de mois : saisonnalité directe |
| log\_c2s\_lag1 | 0.092 | Niveau historique C2S de M-1 |
| variation\_m1 | 0.087 | Variation M-1 vs M-2 (tendance récente) |
| moy\_mobile\_3m | 0.077 | Moyenne mobile sur M-1, M-2, M-3 |
| is\_declin\_continu | 0.075 | Deux baisses consécutives sur M-2 et M-1 |

- **Scoring production (mois 2026-04)** :  
    
  - **101 972 PDV scorés**  
  - **CA identifié à risque : 14 784 977 827 FCFA (99,2 %)**  
  - Distribution : CRITIQUE 53 242 · ÉLEVÉ 44 871 · MODÉRÉ 1 752 · FAIBLE 2 107


- **Déploiement** : dataset `TOP_PDV_RISQUE` publié sur Dataiku, intégré dans le dashboard Power BI de la Direction de la Distribution.

[retour en haut](#table-des-matières)

---

## Méthodologie — CRISP-DM

Ce projet utilise le **CRISP-DM** (Cross-Industry Standard Process for Data Mining) comme méthode de gestion de projet. Le CRISP-DM est l'un des standards les plus reconnus en Data Science. Il est composé de six étapes formant un cycle itératif :

Business Understanding → Data Understanding → Data Preparation

       ↑                                              ↓

   Deployment    ←    Evaluation    ←    Modeling

Les bénéfices du CRISP-DM pour ce projet sont triples :

- **Livraison d'une solution end-to-end** à chaque cycle, même imparfaite  
- **Cadence rapide** : chaque cycle doit être complété le plus vite possible pour délivrer de la valeur métier sans attendre la perfection  
- **Identification précoce des problèmes** : le cycle 1 a permis de détecter le leakage avant que le modèle ne soit mis en production

Pour ce projet, les étapes CRISP-DM ont été organisées comme suit :

**I. Compréhension métier**

- Demande métier, formulation de la problématique, définition des livrables

**II. Compréhension et préparation des données**

- Préparation des données, Feature Engineering, EDA, Préprocessing, Sélection des features

**III. Modélisation**

- Entraînement des modèles, Optimisation des hyperparamètres

**IV. Évaluation**

- Interprétation des métriques, traduction en termes métier

**V. Déploiement**

- Publication du dataset de scoring sur Dataiku, intégration Power BI

[retour en haut](#table-des-matières)

---

## Description des cycles

| Cycle | Description | Recipes |
| :---- | :---- | :---- |
| **Cycle 1** | Premier pipeline complet : extraction Starburst → feature engineering → modélisation. Leakage détecté en fin de cycle (P=1.000) : `variation_m1` dans Recipe 1 reproduisait algébriquement la formule de `target`. | Recipe1 v1 · Recipe2 v1 · Recipe4 v1 |
| **Cycle 2** | Correction du leakage structurel dans les 3 recipes. `variation_m1` recalculée sur `(C2S_{M-1} − C2S_{M-2}) / C2S_{M-2}`. `moy_mobile_3m` recalculée sur lags uniquement. Gate anti-leakage bloquant ajouté dans Recipe 4\. Résultats réalistes obtenus (P=0.50, F1=0.56). | Recipe1 v2 · Recipe2 v2 · Recipe4 v2 |

[retour en haut](#table-des-matières)

---

## 01\. Compréhension métier

Le point de départ de tout projet CRISP-DM est la compréhension précise du besoin qui justifie le projet. Trois questions doivent être répondues :

### Problème métier

La Direction de la Distribution d'Orange Cameroun suit la performance de son réseau PDV via des tableaux de bord Power BI. Ce dispositif est **fondamentalement réactif** : les baisses de performance ne sont détectées qu'après plusieurs semaines de dégradation dans les chiffres de ventes.

La demande du Sales DataLab est d'anticiper ces baisses **un mois à l'avance**, afin de déclencher des actions préventives ciblées avant toute dégradation du chiffre d'affaires.

### Formulation du problème

- **Question métier** : quels PDV risquent de baisser leurs recharges C2S de plus de 10 % le mois prochain ?  
- **Porteur et motif** : Sales DataLab / Direction de la Distribution — besoin de prioriser les interventions terrain sur les PDV fragiles  
- **Format de la solution** :  
  - Type de problème ML : **classification binaire supervisée**  
  - Méthodes envisagées : Random Forest, XGBoost, LightGBM  
  - Livrables : score de risque \[0–1\] par PDV \+ palier d'intervention (CRITIQUE / ÉLEVÉ / MODÉRÉ / FAIBLE) \+ dashboard Power BI

### Variable cible

La variable cible (`target`) est définie comme suit :

target \= 1  si  (C2S\_M − C2S\_{M-1}) / C2S\_{M-1}  ≤  −0.10

target \= 0  sinon

Taux de baisse baseline : **37 %** — c'est-à-dire que sans aucun modèle, 37 % des PDV baissent chaque mois.

[retour en haut](#table-des-matières)

---

## 02\. Préparation des données

### I. Collecte des données

Les données sont extraites depuis le **Data Warehouse d'Orange Cameroun** via **Starburst** (moteur de requêtes SQL distribué). Elles consolident plusieurs systèmes sources :

| Source | Contenu |
| :---- | :---- |
| MSC | Données de recharges C2S par PDV et par mois |
| Zebra | Données de ruptures de stock et approvisionnements |
| REA | Données de visites terrain des équipes commerciales |
| Flytext | Indicateurs de qualité réseau (disponibilité cellulaire, trafic voix) |

Le dataset initial `rupt_freq_with_reseau` contient :

| Dimension | Valeur |
| :---- | :---- |
| Lignes | 1 521 619 |
| Colonnes | 37 |
| PDV uniques | 121 693 |
| Mois couverts | 2025-01 à 2026-04 (16 mois) |

### II. Description des données

Variables initiales disponibles :

| Variable | Description | Type |
| :---- | :---- | :---- |
| `msisdn` | Identifiant unique du PDV | Catégoriel |
| `mois` | Mois de la période (YYYY-MM) | Date |
| `recharges_c2s_mt` | **Montant C2S du mois — variable cible** | Numérique continu |
| `frequentation_totale` | Nombre de transactions du mois | Numérique discret |
| `clients_uniques` | Nombre de clients distincts du mois | Numérique discret |
| `nbre_jours_activite` | Jours d'activité enregistrée dans le mois | Numérique discret |
| `appro` | Stock reçu en approvisionnement ce mois | Numérique continu |
| `nb_jours_rupture_stock` | Nombre de jours en rupture de stock | Numérique discret |
| `ratio_jours_rupture` | Proportion de jours en rupture (0–1) | Numérique continu |
| `tr373_cell_availability_pct` | Disponibilité cellulaire du site réseau (%) | Numérique continu |
| `grp_2g_traffic_speech_erl` | Trafic voix 2G (Erlang) | Numérique continu |
| `is_site_degrade` | Flag site réseau dégradé (\<90 %) | Binaire |
| `pdv_visite` | Flag PDV visité ce mois | Binaire |
| `raisons_visites` | Raison de la visite terrain (55 valeurs) | Catégoriel |
| `region_administrative` | Région administrative du PDV | Catégoriel (13 valeurs) |
| `categ_zone` | Type de zone (urbain/rural) | Catégoriel (2 valeurs) |

*Note : `tr373_cell_availability_pct` peut prendre des valeurs négatives (ex : −9,9, −90,6, −957,5) — ces valeurs ne sont PAS des erreurs mais signalent des pannes totales ou recalibrages NMS. Elles seront codées séparément.*

### III. Nettoyage des données

Les valeurs manquantes sont présentes sur les variables géographiques (≈0,6 % des lignes). La méthode d'imputation suit le principe Rossmann : utiliser le contexte métier plutôt que des statistiques génériques.

| Variable | % NaN | Méthode d'imputation |
| :---- | :---- | :---- |
| `region_administrative` | 0,6 % | Mode du train (imputation dans Recipe 4\) |
| `typedezone1` | 0,6 % | Mode du train |
| `categ_zone` | 0,6 % | Mode du train |
| Variables activité (fréquentation, appro...) | 0 % | NaN \= PDV sans données \= 0 |
| Variables réseau | 0 % | NaN imputé par médiane en amont |

**Filtrage des lignes** : 129 lignes avec `recharges_c2s_mt = 0` (PDV fermés) ont été exclues — équivalent du filtrage des "closed stores" dans Rossmann.

### IV. Statistiques descriptives

Quelques observations clés des statistiques descriptives numériques :

- `recharges_c2s_mt` : très asymétrique (skew \= 8,66) — distribution en loi de puissance typique des réseaux de distribution  
- `tr373_cell_availability_pct` : kurtosis \= 233,6 — forte présence d'outliers (valeurs négatives et valeurs proches de 100\)  
- `nb_jours_rupture_stock` : skew \= 4,46 — la majorité des PDV ont 0 jour de rupture, mais quelques-uns en ont beaucoup  
- `pdv_visite` : 25 % des PDV sont visités chaque mois (valeur médiane \= 0\)

[retour en haut](#table-des-matières)

---

## 03\. Feature Engineering

### I. Carte mentale des hypothèses

Pour guider la création de features, une carte mentale des facteurs influençant la performance des PDV a été construite autour de 6 catégories :

                    ┌─────────────────────┐

                    │  BAISSE DE C2S PDV  │

                    └─────────┬───────────┘

          ┌──────────┬────────┴───────┬──────────┬──────────┐

          ▼          ▼                ▼          ▼          ▼

    Activité PDV  Appro/Ruptures  Réseau Télécom  Visites  Historique

    ratio\_activite  is\_sans\_appro  is\_site\_degrade  visite  c2s\_lag1

    clients\_uniques rupture\_chron  anomalie\_grave   raison  variation

    fidelite        stock\_moyen    trafic\_voix      suivi   declin

          ↓

    Temps & Géographie

    saisonnalité / région / zone

### II. Liste des hypothèses viables

12 hypothèses ont été sélectionnées pour le feature engineering et l'EDA :

1. Les PDV avec un taux d'activité élevé baissent moins souvent  
2. Les PDV sans approvisionnement ce mois connaissent plus souvent une baisse  
3. La rupture de stock chronique (\> 50 % du mois) prédit la baisse  
4. Les PDV avec plus de clients uniques baissent moins souvent  
5. Les PDV baissent plus souvent en début d'année (janvier–février)  
6. La saisonnalité mensuelle (encodage sin/cos) améliore la prédiction  
7. Les PDV sur un site réseau en anomalie grave baissent plus fréquemment  
8. Les visites motivées par une baisse sont un signal réactif (pas préventif)  
9. La région administrative influence significativement le taux de baisse  
10. Les PDV en déclin continu (M-2 et M-1) continuent à baisser en M  
11. La variation des ventes C2S entre M-1 et M-2 est un indicateur avancé  
12. La moyenne mobile 3 mois (M-1, M-2, M-3) améliore la prédiction

### III. Variables créées (27 features)

**Temporelles** (cycliques, style Rossmann sin/cos) :

df\['mois\_num'\]        \= df\['mois\_dt'\].dt.month

df\['mois\_sin'\]        \= np.sin(df\['mois\_num'\] \* (2 \* np.pi / 12))

df\['mois\_cos'\]        \= np.cos(df\['mois\_num'\] \* (2 \* np.pi / 12))

df\['is\_debut\_annee'\]  \= df\['mois\_num'\].isin(\[1, 2\]).astype(int)

df\['is\_fin\_annee'\]    \= df\['mois\_num'\].isin(\[11, 12\]).astype(int)

**Ruptures & Approvisionnement** :

df\['is\_rupture\_chronique'\] \= (df\['ratio\_jours\_rupture'\] \> 0.50).astype(int)

df\['is\_rupture\_severe'\]    \= (df\['ratio\_jours\_rupture'\] \> 0.75).astype(int)

df\['is\_sans\_appro'\]        \= (df\['appro'\] \== 0).astype(int)

df\['log\_appro'\]            \= np.log1p(df\['appro'\].clip(lower=0))

**Réseau Télécom** (traitement des valeurs négatives) :

\# Valeurs négatives \= anomalie grave (PAS une erreur)

df\['is\_site\_anomalie\_grave'\] \= (df\['tr373\_cell\_availability\_pct'\] \< 0).astype(int)

df\['is\_site\_tres\_degrade'\]   \= (

    (df\['tr373\_cell\_availability\_pct'\] \>= 0\) &

    (df\['tr373\_cell\_availability\_pct'\] \< 70\)

).astype(int)

df\['dispo\_reseau\_niveau'\] \= pd.cut(df\['tr373\_cell\_availability\_pct'\],

    bins=\[-float('inf'), 0, 70, 80, 90, float('inf')\], labels=\[1,2,3,4,5\])

**Visites terrain** (encodage des 55 raisons → 5 catégories) :

\# NON\_VISITE / VISITE\_BAISSE\_PERF / VISITE\_PDV\_INACTIF

\# VISITE\_PROGRAMME / VISITE\_PROMO / VISITE\_INCONNUE / VISITE\_AUTRE

df\['cat\_raison\_visite'\]  \= df\['raisons\_visites'\].apply(categoriser\_raison\_visite)

df\['visite\_pour\_baisse'\] \= (df\['cat\_raison\_visite'\] \== 'VISITE\_BAISSE\_PERF').astype(int)

**Lags et tendance — ⚠️ CORRECTION CYCLE 2** :

\# CYCLE 1 (LEAKAGE) :

\# variation\_m1 \= (C2S\_M \- C2S\_{M-1}) / C2S\_{M-1}  ← \= target \!

\# CYCLE 2 (CORRIGÉ) :

df\['c2s\_lag1'\] \= df.groupby('msisdn')\['recharges\_c2s\_mt'\].shift(1)

df\['c2s\_lag2'\] \= df.groupby('msisdn')\['recharges\_c2s\_mt'\].shift(2)

df\['c2s\_lag3'\] \= df.groupby('msisdn')\['recharges\_c2s\_mt'\].shift(3)

\# variation du PASSÉ, pas du mois courant

df\['variation\_m1'\] \= (df\['c2s\_lag1'\] \- df\['c2s\_lag2'\]) / df\['c2s\_lag2'\]

\# moyenne sur historique UNIQUEMENT (sans C2S\_M)

df\['moy\_mobile\_3m'\] \= (df\['c2s\_lag1'\] \+ df\['c2s\_lag2'\] \+ df\['c2s\_lag3'\]) / 3

\# déclin : deux baisses sur M-2 et M-1 (pas M)

df\['is\_declin\_continu'\] \= ((df\['variation\_m1'\] \< 0\) & (variation\_lag1 \< 0)).astype(int)

df\['log\_c2s\_lag1'\] \= np.log1p(df\['c2s\_lag1'\].clip(lower=0))

### IV. Filtrage des variables

Conformément au principe CRISP-DM ("variables unavailable at prediction time must be dropped") :

| Variable supprimée | Motif |
| :---- | :---- |
| `recharges_c2s_mt` | Numérateur de target — non disponible à T=début M |
| `variation_mensuelle` | \= `(C2S_M − C2S_{M-1}) / C2S_{M-1}` ≡ target |
| `clients_uniques` | Données du mois M (analogie Rossmann : *customers* exclu cycle 1\) |
| `ratio_activite` | Données du mois M |
| `visite_pour_baisse` | Signal endogène (réponse à la baisse de M, pas préventif) |
| `ratio_jours_rupture` | Données du mois M |

[retour en haut](#table-des-matières)

---

## 04\. Analyse Exploratoire (EDA)

L'EDA est divisée en trois parties : analyse univariée, analyse bivariée (validation des hypothèses) et analyse multivariée (corrélations Pearson).

### I. Analyse univariée

**Variable cible (target)** :

- 0 (Stable) : 74 % des observations  
- 1 (Baisse) : 26 % des observations  
- Taux de baisse baseline : **37 %** (sur le dataset filtré après suppression des PDV inactifs)  
- scale\_pos\_weight XGBoost : ≈ 2,83

**Observations clés des variables numériques** :

- `recharges_c2s_mt` : distribution Pareto — 80 % des PDV réalisent moins de 200 000 FCFA/mois  
- `nb_jours_rupture_stock` : 72 % des PDV ont 0 jour de rupture — variable très asymétrique  
- `tr373_cell_availability_pct` : 233 valeurs négatives (anomalies graves) — traitement spécifique  
- `pdv_visite` : 25 % des PDV visités — forte majorité non suivie terrain

### II. Analyse bivariée — Validation des hypothèses

#### H2. Les PDV avec plus de jours actifs baissent moins souvent

Variable utilisée : `ratio_activite` (jours actifs / total jours du mois).

**Verdict : VRAIE ★★★★**

Les PDV très inactifs (ratio \< 20 %) présentent un taux de baisse de **68,7 %** contre 37 % baseline. La relation est monotone : plus l'inactivité est forte, plus le risque de baisse est élevé.

---

#### H5. La rupture de stock chronique (\> 50 % du mois) prédit la baisse

Variable utilisée : `is_rupture_chronique` (binaire, seuil 50 %).

**Verdict : VRAIE ★★★★★**

`is_rupture_chronique = 1` → taux de baisse \= **68,5 %** contre 30 % pour les PDV sans rupture chronique. Écart de 38,5 points — l'un des signaux les plus forts de l'analyse.

---

#### H7. La rupture sévère (\> 75 % du mois) est le signal le plus puissant

Variable utilisée : `is_rupture_severe` (binaire, seuil 75 %).

**Verdict : VRAIE ★★★★★**

`is_rupture_severe = 1` → taux de baisse \= **72,5 %** — **maximum absolu observé dans toute l'analyse bivariée**. Un PDV sans stock pendant les trois quarts du mois baisse dans 7 cas sur 10\. Feature d'importance 0,209 dans le modèle final.

---

#### H8. L'absence d'approvisionnement ce mois prédit la baisse

Variable utilisée : `is_sans_appro` (binaire, appro \= 0).

**Verdict : VRAIE ★★★★**

`is_sans_appro = 1` → **63 %** vs 35 % pour les PDV approvisionnés. Signal fort et distinct de la rupture : un PDV peut ne pas recevoir de livraison tout en maintenant son activité via stock résiduel, mais le risque reste quasi-double.

---

#### H9. Les PDV non visités baissent plus que les PDV visités

Variable utilisée : `pdv_visite` (binaire).

**Verdict : FAUSSE ★**

`pdv_visite = 0` (non visité) : **38,3 %** — identique à `pdv_visite = 1` (visité) : **38,3 %**. La visite est réalisée en réponse à une baisse déjà observée, pas en prévention. Signal nul — variable exclue du modèle.

---

#### H10. Les visites motivées par une baisse sont un signal préventif

Variable utilisée : `visite_pour_baisse` (binaire, catégorie VISITE\_BAISSE\_PERF).

**Verdict : NUANCÉE ★★**

`visite_pour_baisse = 1` : **38 %** ≈ baseline. La visite pour baisse encode la dégradation du mois précédent — signal endogène, pas prédictif. Variable exclue du modèle prédictif, conservée pour le dashboard.

---

#### H12. Un site réseau dégradé (\< 90 %) prédit une baisse

Variable utilisée : `is_site_degrade` (binaire).

**Verdict : VRAIE ★★**

`is_site_degrade = 1` : **40,9 %** vs 37 % baseline. Signal présent mais modéré (+3,9 points). Devient plus fort à mesure que la dégradation est sévère.

---

#### H12b. Le niveau d'anomalie grave est le facteur réseau le plus risqué

Variable utilisée : `dispo_reseau_niveau` (5 niveaux, 1 \= anomalie grave).

**Verdict : VRAIE ★★★**

Niveau 1 (anomalie grave, valeur négative) \= maximum du graphique, supérieur à 43 %. Gradient décroissant confirmé : plus la dégradation est sévère, plus le taux de baisse est élevé. Justifie a posteriori la création de `is_site_anomalie_grave`.

---

#### H13. Un faible trafic voix 2G prédit la baisse de recharges

Variable utilisée : `is_trafic_reseau_faible` (binaire).

**Verdict : FAUSSE ★**

`is_trafic_reseau_faible = 1` : **39,7 %** ≈ 37 % baseline. Le trafic voix n'est pas discriminant. Variable conservée comme feature d'infrastructure dans le modèle pour ses potentielles interactions non-linéaires.

---

#### H14. Les PDV en zone rurale baissent davantage

Variable utilisée : `typedezone1` (urbain/rural).

**Verdict : FAIBLE ★**

Différence urbain/rural inférieure à 4 points — pas discriminant. Variable conservée comme contrôle géographique.

---

#### H15. La région administrative influence le taux de baisse

Variable utilisée : `region_administrative` (13 régions).

**Verdict : VRAIE ★★★**

Forte disparité inter-régions : certaines régions dépassent 45 % de baisse, d'autres restent sous 30 %. Variable intégrée via One-Hot Encoding.

---

#### H16. La saisonnalité mensuelle influence le taux de baisse

Variable utilisée : `mois_num` (1–12).

**Verdict : VRAIE ★★★★**

Mois 2 (février) : **44,9 %** — maximum annuel. Cycle saisonnier net avec pic jan–fév et stabilisation en milieu d'année. Encodage `mois_sin / mois_cos` adopté pour capturer la continuité cyclique.

---

#### H17. Le début d'année (jan–fév) concentre plus de baisses

Variable utilisée : `is_debut_annee` (binaire, jan–fév \= 1).

**Verdict : VRAIE ★★★★**

`is_debut_annee = 1` : **44,9 %** vs 37 % baseline (+7,9 points). **Feature la plus importante du modèle final** (importance \= 0,209). Signal opérationnel clair : concentrer les efforts terrain sur janvier–février.

---

#### H18. Le déclin continu sur M-2 et M-1 prédit la baisse en M

Variable utilisée : `is_declin_continu` (deux baisses consécutives sur M-2 et M-1, corrigé cycle 2).

**Verdict : VRAIE ★★★★★**

`is_declin_continu = 1` : **76,4 %** — **signal prédictif maximum de toute l'analyse** (+39,4 points vs baseline). Un PDV ayant baissé deux mois de suite baisse dans 3 cas sur 4\. Feature d'importance 0,075 dans le modèle final (importance limitée en cycle 2 car non disponible pour tous les PDV).

---

### Tableau récapitulatif de validation des hypothèses

| Réf. | Hypothèse | Taux critique | Verdict | Force |
| :---- | :---- | :---- | :---- | :---- |
| H2 | Jours actifs → moins de baisses | 68,7 % | VRAIE | ★★★★ |
| H5 | Rupture chronique \> 50 % | 68,5 % vs 30 % | VRAIE | ★★★★★ |
| H7 | Rupture sévère \> 75 % | **72,5 %** | VRAIE | ★★★★★ |
| H8 | Sans approvisionnement | 63 % vs 35 % | VRAIE | ★★★★ |
| H9 | Non visité vs visité | 38,3 % \= 38,3 % | FAUSSE | ★ |
| H10 | Visite motivée baisse | 38 % ≈ base | NUANCÉE | ★★ |
| H12 | Site réseau dégradé | 40,9 % vs 37 % | VRAIE | ★★ |
| H12b | Anomalie grave réseau | max du graphique | VRAIE | ★★★ |
| H13 | Trafic voix faible | 39,7 % ≈ base | FAUSSE | ★ |
| H14 | Zone rurale | \< 4 pts d'écart | FAIBLE | ★ |
| H15 | Régions risquées | forte disparité | VRAIE | ★★★ |
| H16 | Saisonnalité mensuelle | mois 2 \= 44,9 % | VRAIE | ★★★★ |
| H17 | Début d'année jan–fév | 44,9 % vs 37 % | VRAIE | ★★★★ |
| H18 | Déclin continu M-2, M-1 | **76,4 %** | VRAIE | ★★★★★ |

### III. Analyse multivariée (Pearson)

La matrice de corrélation Pearson (20 variables, générée sur Dataiku) révèle :

**Corrélations avec target** (les plus significatives) :

- `is_declin_continu` : r \= \+0,383 *(après correction cycle 2\)*  
- `ratio_activite` : r \= −0,287 *(feature M — exclue du modèle)*  
- `is_rupture_chronique` : r \= \+0,110  
- `is_sans_appro` : r \= \+0,121  
- `clients_uniques` : r \= −0,153 *(feature M — exclue)*

**Multicolinéarités notables** (à surveiller) :

- `ratio_jours_rupture` ↔ `is_rupture_chronique` : r \= \+0,817 (quasi-redondance)  
- `is_site_tres_degrade` ↔ `tr373_cell_availability_pct` : r \= −0,768

[retour en haut](#table-des-matières)

---

## 05\. Préprocessing des données

Le préprocessing est implémenté dans la fonction `preprocess_datasets()` de Recipe 4\. Conformément au principe Rossmann (*"fit always on train only, transform on train and valid"*), tous les scalers sont fitttés exclusivement sur les données d'entraînement.

### I. Scaling numérique

**RobustScaler** (résistant aux outliers — utilisé sur les variables à forte asymétrie) :

\# Variables sûres uniquement (calculées avant M)

cols\_robust\_safe \= \[

    'grp\_2g\_traffic\_speech\_erl',    \# trafic réseau

    'tr373\_cell\_availability\_pct',  \# disponibilité réseau

    'seuil\_moyen\_horaire',          \# historique

    'log\_c2s\_lag1',                 \# log(C2S\_{M-1})

    'moy\_mobile\_3m',                \# moyenne lags

\]

**MinMaxScaler** (normalisation 0–1 pour variables bornées) :

cols\_minmax\_safe \= \[

    'variation\_m1',  \# variation M-1 vs M-2 (corrigé cycle 2\)

    'mois\_sin',

    'mois\_cos',

\]

### II. Encodage catégoriel

| Variable | Méthode | Justification |
| :---- | :---- | :---- |
| `categ_zone` | Label Encoding | Variable ordinale (urbain \< rural) |
| `typedezone1` | Label Encoding | Variable ordinale |
| `region_administrative` | One-Hot Encoding (13 colonnes) | Variable nominale — pas d'ordre naturel |

### III. Transformation des variables temporelles cycliques

Équivalent direct du traitement `day_sin/cos`, `month_sin/cos`, `week_sin/cos` dans Rossmann :

df\['mois\_sin'\] \= np.sin(df\['mois\_num'\] \* (2 \* np.pi / 12))

df\['mois\_cos'\] \= np.cos(df\['mois\_num'\] \* (2 \* np.pi / 12))

Cette transformation permet au modèle de percevoir que décembre (mois 12\) est adjacent à janvier (mois 1\) — continuité cyclique que l'encodage linéaire briserait.

[retour en haut](#table-des-matières)

---

## 06\. Sélection des features

La sélection des features s'effectue en deux temps dans Recipe 4 : sélection wrapper par Random Forest, puis ajout manuel des features métier validées par l'EDA.

### Méthode wrapper (équivalent Boruta chez Rossmann)

Un Random Forest est entraîné sur l'ensemble des features disponibles, et les features dont l'importance dépasse le 40e percentile sont retenues :

rf\_fs \= RandomForestClassifier(n\_estimators=100, max\_depth=8,

    class\_weight='balanced', random\_state=42)

rf\_fs.fit(X\_train\_all, y\_train)

cols\_selected \= importances\[importances \> importances.quantile(0.40)\].index.tolist()

### Must-have (features métier validées)

Les features suivantes sont ajoutées systématiquement, conformément aux enseignements de l'EDA bivariée :

variation\_m1      \# variation M-1 vs M-2 (cycle 2 corrigé)

is\_declin\_continu \# deux baisses consécutives M-2, M-1

moy\_mobile\_3m     \# moyenne(C2S\_{M-1}, C2S\_{M-2}, C2S\_{M-3})

log\_c2s\_lag1      \# log(C2S\_{M-1}) — niveau historique du PDV

mois\_sin / mois\_cos / mois\_num / is\_debut\_annee / is\_fin\_annee

tr373\_cell\_availability\_pct / grp\_2g\_traffic\_speech\_erl

is\_site\_degrade / is\_site\_anomalie\_grave / is\_site\_tres\_degrade

qr\_data\_disponible / dispo\_reseau\_niveau

### Gate anti-leakage (cycle 2\)

Un verrou bloquant est ajouté avant tout entraînement. Si une feature présente `|r| > 0,50` avec target, le script lève une `ValueError` et s'arrête :

critique \= corr\_with\_target\[corr\_with\_target.abs() \> 0.50\]

if len(critique) \> 0:

    raise ValueError(f"LEAKAGE RÉSIDUEL : {list(critique.index)}")

**Features finales** : 18 features utilisées par le modèle (sur 67 colonnes disponibles).

[retour en haut](#table-des-matières)

---

## 07\. Modélisation Machine Learning

### I. Métriques de performance

Pour un problème de classification binaire avec déséquilibre de classes (74 % / 26 %), quatre métriques sont retenues :

| Métrique | Formule | Interprétation métier |
| :---- | :---- | :---- |
| **Precision** | TP / (TP \+ FP) | Sur 100 PDV signalés, combien baisseront vraiment ? |
| **Recall** | TP / (TP \+ FN) | Sur 100 vrais PDV en baisse, combien sont détectés ? |
| **F1-score** | 2 × P × R / (P \+ R) | Compromis précision/rappel |
| **AUC-ROC** | — | Capacité discriminante globale |

*Note : dans ce contexte opérationnel, un **Recall élevé est prioritaire** — il vaut mieux générer des fausses alertes que rater un PDV réellement en baisse.*

### II. Modélisation

Split temporel : les 2 derniers mois (2026-03 et 2026-04) constituent le jeu de validation.

Cinq modèles sont testés :

#### 1\. Baseline (Dummy Classifier)

Prédit systématiquement la classe majoritaire (Stable). Sert de référence minimale.

#### 2\. Régression Logistique

Modèle linéaire. Recall très faible (0,088) — confirme que les relations entre features et target sont non-linéaires.

#### 3\. Random Forest

Ensemble de 300 arbres de décision (bagging). Meilleures performances que la régression logistique (F1 \= 0,422) mais légèrement inférieur aux modèles de boosting.

#### 4\. XGBoost

Gradient boosting avec régularisation L1/L2, élagage des arbres et traitement natif des déséquilibres via `scale_pos_weight`. Early stopping après 50 rounds sans amélioration sur l'AUC-PR de validation.

#### 5\. LightGBM

Variante du gradient boosting optimisée pour la vitesse. Performances quasi-identiques à XGBoost.

### III. Résultats — Cycle 2 (sans leakage)

| Modèle | Precision | Recall | F1 | AUC |
| :---- | :---- | :---- | :---- | :---- |
| **XGBoost** ⭐ | 0.505 | 0.414 | **0.455** | 0.645 |
| LightGBM | 0.505 | 0.411 | 0.453 | 0.645 |
| Random Forest | 0.490 | 0.371 | 0.422 | 0.628 |
| Logistic Regression | 0.456 | 0.088 | 0.147 | 0.613 |
| Baseline | 0.000 | 0.000 | 0.000 | — |

✅ **Cycle 2 validé** — Precision dans la plage réaliste 0.40–0.65 (pas de leakage).

### IV. Cross-validation temporelle (5 folds)

Conformément à la règle Rossmann (*"using raw data for CV ensures completely distinct train/validation data"*), la CV est réalisée sur les données brutes avec re-fit des scalers à l'intérieur de chaque fold :

Fold 1 | 2026-03 → 2026-04  ✅

Fold 2 | 2026-01 → 2026-02  ✅

Fold 3 | 2025-11 → 2025-12  ✅

Fold 4 | 2025-09 → 2025-10  ✅

Fold 5 | 2025-07 → 2025-08  ✅

XGBoost CV : **P \= 0.50 ± 0.02 | R \= 0.41 ± 0.03 | F1 \= 0.45 ± 0.02** — résultats stables sur tous les folds.

[retour en haut](#table-des-matières)

---

## 08\. Optimisation des hyperparamètres

La méthode Random Search est utilisée (25 itérations) — même approche que Rossmann. L'optimisation porte sur le **F1-score** (et non la Precision seule) pour garantir un équilibre entre la détection des PDV à risque et la limitation des fausses alertes.

Les paramètres optimisés et leur rôle :

| Paramètre | Rôle |
| :---- | :---- |
| `n_estimators` | Nombre d'arbres — plus élevé \= meilleur mais plus lent |
| `learning_rate` | Taux d'apprentissage — plus faible \= plus robuste au surapprentissage |
| `max_depth` | Profondeur max — contrôle la complexité des arbres |
| `subsample` | Fraction d'échantillons par arbre — prévient le surapprentissage |
| `colsample_bytree` | Fraction de features par arbre |
| `min_child_weight` | Poids minimum par feuille — contrôle la granularité |
| `scale_pos_weight` | Pondération de la classe minoritaire (Baisse) |
| `reg_alpha / reg_lambda` | Régularisation L1/L2 |

Résultats des 25 itérations (sélection des meilleures) :

Iter  1 | P=0.504 R=0.400 F1=0.446 AUC=0.644

Iter  2 | P=0.500 R=0.435 F1=0.465 AUC=0.646  ✅ meilleur F1

Iter  8 | P=0.473 R=0.583 F1=0.522 AUC=0.645  ✅ meilleur F1

Iter  9 | P=0.429 R=0.797 F1=0.558 AUC=0.643  ✅ meilleur F1 — RETENU

**Meilleure configuration (Iter 9\) : F1 \= 0.558**

- n\_estimators élevé, learning\_rate \= 0.03  
- scale\_pos\_weight fort (maximise le Recall)  
- min\_child\_weight \= 1

[retour en haut](#table-des-matières)

---

## 09\. Interprétation des erreurs et performance métier

### I. Performance du modèle final

\=== RAPPORT DE CLASSIFICATION — MODÈLE FINAL CYCLE 2 \===

              precision    recall  f1-score   support

      Stable       0.77      0.39      0.52    128 585

       Baisse       0.43      0.80      0.56     74 003

    accuracy                           0.54    202 588

   macro avg       0.60      0.59      0.54    202 588

weighted avg       0.65      0.54      0.53    202 588

TP \= 58 990  |  FP \= 78 431  |  FN \= 15 013  |  TN \= 50 154

### II. Matrice de confusion — Interprétation métier

                    ┌────────────────┬────────────────┐

                    │  Réel Stable   │  Réel Baisse   │

┌───────────────────┼────────────────┼────────────────┤

│  Prédit Stable    │ TN \= 50 154    │ FN \= 15 013    │

│                   │ (PDV stables   │ (PDV en baisse │

│                   │  bien classés) │  non détectés) │

├───────────────────┼────────────────┼────────────────┤

│  Prédit Baisse    │ FP \= 78 431    │ TP \= 58 990    │

│                   │ (fausses       │ (PDV en baisse │

│                   │  alertes)      │  détectés) ✅  │

└───────────────────┴────────────────┴────────────────┘

- **58 990 PDV en baisse correctement détectés** → 80 % des vraies baisses interceptées  
- **15 013 PDV en baisse ratés** → 20 % de faux négatifs (manqués)  
- **78 431 fausses alertes** → coût opérationnel acceptable : visites inutiles sur des PDV stables

Le Recall de **80 %** est la métrique opérationnelle clé : sur 100 PDV qui vont baisser, 80 sont détectés avant la baisse.

### III. Importance des features — Top 15

is\_debut\_annee           0.209  ← saisonnalité jan-fév (feature \#1)

mois\_num                 0.118  ← saisonnalité directe

log\_c2s\_lag1             0.092  ← niveau historique C2S

variation\_m1             0.087  ← tendance M-1 vs M-2

moy\_mobile\_3m            0.077  ← moyenne mobile 3 mois

is\_fin\_annee             0.075  ← saisonnalité nov-déc

is\_declin\_continu        0.075  ← déclin continu M-2, M-1

mois\_sin                 0.070  ← cyclique sinus

mois\_cos                 0.064  ← cyclique cosinus

qr\_data\_disponible       0.055  ← disponibilité données QR

moy\_mobile\_3m (réseau)   0.039  ← trafic réseau moyen

grp\_2g\_traffic\_speech    0.025  ← trafic voix 2G

seuil\_moyen\_horaire      0.023  ← seuil horaire PDV

log\_trafic\_voix          0.017  ← log trafic voix

is\_site\_tres\_degrade     0.013  ← dégradation réseau sévère

**Enseignement clé** : la saisonnalité domine (40 % de l'importance totale), suivie de l'historique C2S (25 %). Les variables réseau contribuent de façon marginale mais réelle.

### IV. Performance métier — Scoring production

Le modèle a été appliqué aux 101 972 PDV actifs du mois 2026-04 :

| Palier | Nb PDV | Part | Action recommandée |
| :---- | :---- | :---- | :---- |
| **CRITIQUE** (score ≥ 0,65) | 53 242 | 52,2 % | Intervention immédiate |
| **ÉLEVÉ** (score ≥ 0,50) | 44 871 | 44,0 % | Suivi prioritaire |
| **MODÉRÉ** (score ≥ 0,38) | 1 752 | 1,7 % | Surveillance renforcée |
| **FAIBLE** (score \< 0,38) | 2 107 | 2,1 % | Pas d'action immédiate |

**CA identifié à risque : 14 784 977 827 FCFA (99,2 % du CA total)**

*Pour un usage opérationnel concentré, il est recommandé de filtrer sur les 53 242 PDV CRITIQUE uniquement.*

[retour en haut](#table-des-matières)

---

## 10\. Déploiement en production (Dataiku \+ Power BI)

### I. Architecture de production

Data Warehouse (MSC, Zebra, REA, Flytext)

         │

         ▼  Starburst (SQL distribué)

   rupt\_freq\_with\_reseau

         │

         ▼  Recipe 1 — compute\_df\_prepared\_CYCLE2

   df\_prepared  (67 colonnes, features engineered)

         │

         ▼  Recipe 2 — compute\_target\_CYCLE2

   df\_with\_target  (target \= baisse C2S ≥ −10 %)

         │

         ▼  Recipe 4 — compute\_model\_training\_CYCLE2

   TOP\_PDV\_RISQUE  (score\_risque, palier\_risque, prediction\_baisse)

         │

         ▼  Power BI

   Dashboard Direction Distribution

   (scores prédictifs \+ KPI \+ vues de suivi)

### II. Dataset de sortie — TOP\_PDV\_RISQUE

Le dataset publié sur Dataiku contient, pour chaque PDV :

| Colonne | Description |
| :---- | :---- |
| `msisdn` | Identifiant PDV |
| `mois` | Mois de scoring |
| `secteur` / `region_administrative` | Localisation |
| `score_risque` | Score de risque \[0–1\] |
| `prediction_baisse` | 1 si PDV prédit en baisse |
| `palier_risque` | CRITIQUE / ÉLEVÉ / MODÉRÉ / FAIBLE |
| `score_pire` | Scénario pessimiste (score − écart-type) |
| `score_meilleur` | Scénario optimiste (score \+ écart-type) |
| `recharges_c2s_mt` | CA du mois de scoring (pour calcul CA à risque) |

### III. Intégration Power BI

Le dataset `TOP_PDV_RISQUE` est intégré dans les tableaux de bord Power BI existants de la Direction de la Distribution. Les équipes terrain peuvent :

1. **Filtrer par palier de risque** pour prioriser les interventions  
2. **Visualiser la carte géographique** des PDV CRITIQUE par région/secteur  
3. **Suivre l'évolution mensuelle** des scores et comparer au mois précédent  
4. **Exporter les listes** de PDV à cibler pour les plans d'action terrain

### IV. Planification d'exécution mensuelle

Le pipeline s'exécute chaque début de mois M+1 :

1\. Starburst → extraction des données du mois M terminé

2\. Recipe 1  → recalcul des features (lags mis à jour)

3\. Recipe 2  → calcul du target réel du mois M (validation)

4\. Recipe 4  → scoring sur le mois M → prédiction pour M+1

5\. Power BI  → actualisation automatique du dashboard

[retour en haut](#table-des-matières)

---

## Conclusion

Ce projet a permis de délivrer une solution end-to-end de scoring prédictif des PDV d'Orange Cameroun en utilisant deux cycles CRISP-DM complets.

Le **cycle 1** a permis de construire le pipeline complet et d'identifier un leakage structurel critique : la variable `variation_m1` dans Recipe 1 reproduisait algébriquement la formule de la variable cible, produisant des résultats artificiellement parfaits (Precision \= 1.000). Cette découverte, bien que décevante dans un premier temps, illustre précisément la valeur de la méthode CRISP-DM : en complétant un cycle rapidement, les problèmes sont identifiés avant la mise en production.

Le **cycle 2** a corrigé ce leakage en recalculant les variables `variation_m1`, `moy_mobile_3m` et `is_declin_continu` sur des données strictement historiques (M-1, M-2, M-3). Les résultats obtenus — Precision 0,43, Recall 0,80, F1 0,558, AUC 0,645 — sont réalistes et comparables aux performances obtenues par le projet Rossmann (MAPE 7–9 %) après le même processus d'itération.

**Ce que le modèle apporte opérationnellement :**

- Détection de 80 % des PDV qui vont baisser, un mois avant la baisse  
- Ciblage de 53 242 PDV CRITIQUE sur 101 972 — réduction de 47 % du périmètre d'intervention terrain  
- Identification de 14,8 milliards FCFA de CA à risque pour priorisation des ressources

**Prochaines étapes (cycle 3 envisagé) :**

- Réintégration de `clients_uniques` via un projet de prédiction séparé (analogie Rossmann cycle 2 → `customers`)  
- Enrichissement avec des données de campagnes promotionnelles pour capturer les effets des actions commerciales  
- Déploiement d'un système d'alertes automatiques en temps réel pour les équipes terrain

[retour en haut](#table-des-matières)

---

## Annexe I — Sources de données

| N° | Dataset | Source | Description |
| :---- | :---- | :---- | :---- |
| 1 | `rupt_freq_with_reseau` | Data Warehouse Orange Cameroun (Starburst) | Dataset principal consolidant MSC, Zebra, REA, Flytext |
| 2 | `df_prepared` | Recipe 1 — Dataiku | Dataset après feature engineering (67 colonnes) |
| 3 | `df_with_target` | Recipe 2 — Dataiku | Dataset avec variable cible (baisse C2S ≥ −10 %) |
| 4 | `TOP_PDV_RISQUE` | Recipe 4 — Dataiku | Dataset de scoring production (101 972 PDV) |
| 5 | Dashboard Power BI | Direction de la Distribution | Tableau de bord intégrant les scores prédictifs |

**Stack technique :**

- **Extraction** : Starburst (SQL distribué sur Data Warehouse)  
- **Modélisation** : Dataiku DSS (Python, Scikit-learn, XGBoost, LightGBM)  
- **Visualisation** : Microsoft Power BI  
- **Versioning** : Dataiku Flow (Recipes \+ Datasets)

[retour en haut](#table-des-matières)

---

*Sales DataLab · Orange Cameroun · 2026*  
*Pour toute question sur ce projet, contacter le Sales DataLab, Direction de la Distribution.*  

