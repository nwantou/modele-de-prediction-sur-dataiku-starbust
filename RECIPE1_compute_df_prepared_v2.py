# ══════════════════════════════════════════════════════════════════
# RECIPE 1 : compute_df_prepared — CYCLE 2
# Étapes 1 + 2 + 3 — Description + Feature Engineering + Filtrage
#
# CORRECTIONS CYCLE 2 vs CYCLE 1 :
#   [C1] variation_m1 = (C2S_M - C2S_{M-1}) / C2S_{M-1} = TARGET → ⛔
#        → Recalculé sur lags passés : (C2S_{M-1} - C2S_{M-2}) / C2S_{M-2}
#   [C2] moy_mobile_3m = rolling(3) incluait C2S_M → ⛔
#        → Recalculé sur moyenne(c2s_lag1, c2s_lag2, c2s_lag3)
#   [C3] is_declin_continu dépendait de variation_m1 ancien → ⛔
#        → Recalculé sur variations passées uniquement
#   [C4] log_c2s_mt = log(C2S_M) = log(numérateur de target) → ⛔
#        → Supprimé, remplacé par log_c2s_lag1
# ══════════════════════════════════════════════════════════════════
import dataiku
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Chargement ────────────────────────────────────────────────────
print("── Chargement ──")
df = dataiku.Dataset("rupt_freq_with_reseau").get_dataframe()
print(f"  Lignes : {df.shape[0]:,} | Colonnes : {df.shape[1]}")
print(f"  Mois   : {sorted(df['mois'].unique())}")
print(f"  PDV    : {df['msisdn'].nunique():,}")

# Normalisation colonnes
df.columns = [c.lower().strip() for c in df.columns]
df['msisdn'] = df['msisdn'].astype(str).str.strip()
df['mois']   = df['mois'].astype(str).str.strip()

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 1 — DESCRIPTION & IMPUTATION
# Rossmann 1.4 : imputation raisonnée variable par variable
# "Variable statistics weren't used for imputation —
#  the date/context of each entry was used instead"
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 1 — Description & Imputation ══")

df['mois_dt'] = pd.to_datetime(df['mois'] + '-01')

# Rapport NaN avant imputation
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"  NaN avant imputation :")
    for col, n in missing.items():
        print(f"    {col:45s} : {n:,} ({n/len(df)*100:.1f}%)")
else:
    print("  Aucun NaN détecté")

# ── Imputation par catégorie ──────────────────────────────────────
# Activité / Volume → NaN = PDV sans données = 0
for col in ['frequentation_totale', 'clients_uniques',
            'nbre_jours_activite', 'nbre_jours_inactivite',
            'recharges_c2s_mt', 'appro', 'nb_jours_rupture_stock',
            'ratio_jours_rupture', 'seuil_moyen_horaire', 'stock_moyen']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Qualité réseau → médiane si NaN (déjà imputé en amont par RECIPE0, sécurité)
# ⚠ RÈGLES D'INTERPRÉTATION DES VARIABLES RÉSEAU :
#   tr373_cell_availability_pct : borné [0, 100]    — % de disponibilité cellulaire
#   grp_2g_traffic_speech_erl   : positif sans borne — 2000+ Erl = trafic élevé, NORMAL
#   data_traffic_gb              : positif sans borne — 9000+ GB = trafic élevé, NORMAL
#   → Pour Voice et Data Traffic : seules les valeurs NÉGATIVES sont des erreurs
for col in ['grp_2g_traffic_speech_erl', 'tr373_cell_availability_pct',
            'data_traffic_gb', 'log_trafic_reseau', 'log_data_traffic']:
    if col in df.columns:
        med = df[col].median() if df[col].notna().any() else 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med)

# Indicateurs binaires → NaN = 0
# MISE À JOUR v2 : is_data_traffic_faible/fort + reseau_degrade_composite ajoutés
#                  is_site_anomalie_grave et is_site_tres_degrade SUPPRIMÉS
for col in ['qr_data_disponible', 'visite_data_disponible',
            'is_site_degrade',
            'is_trafic_reseau_faible',
            'is_data_traffic_faible',    # NOUVEAU
            'is_data_traffic_fort',      # NOUVEAU
            'reseau_degrade_composite']: # NOUVEAU
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Visites terrain → NaN = "PDV non visité" = 0
for col in ['nb_visites_mois', 'pdv_visite', 'pdv_non_visite', 'pdv_multi_visite']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

if 'raisons_visites' in df.columns:
    df['raisons_visites'] = df['raisons_visites'].fillna('Pas visité').astype(str).str.strip()

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
if len(missing_after) == 0:
    print("  ✅ Aucun NaN résiduel après imputation")
else:
    print(f"  ⚠️  NaN résiduels : {missing_after.to_dict()}")

# Statistiques descriptives
num_feat = df.select_dtypes(include=['int64', 'float64', 'int32'])
desc = pd.DataFrame({
    'min':      num_feat.min(),
    'max':      num_feat.max(),
    'mean':     num_feat.mean().round(2),
    'median':   num_feat.median(),
    'std':      num_feat.std().round(2),
    'skew':     num_feat.skew().round(2),
})
print("\n=== Statistiques descriptives ===")
print(desc.to_string())

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 2 — Feature Engineering ══")

# ── 2.1. Variables temporelles (Rossmann : sin/cos pour cycliques) ──
df['mois_num']        = df['mois_dt'].dt.month
df['annee']           = df['mois_dt'].dt.year
df['trimestre']       = df['mois_dt'].dt.quarter
df['is_debut_annee']  = df['mois_num'].isin([1, 2]).astype(int)
df['is_fin_annee']    = df['mois_num'].isin([11, 12]).astype(int)
df['is_milieu_annee'] = df['mois_num'].isin([5, 6, 7]).astype(int)
df['mois_sin']        = np.sin(df['mois_num'] * (2 * np.pi / 12)).round(6)
df['mois_cos']        = np.cos(df['mois_num'] * (2 * np.pi / 12)).round(6)
print(f"  ✅ Variables temporelles créées")

# ── 2.2. Variables d'activité PDV (calculées sur mois M) ─────────
# Ces variables décrivent l'état du PDV pendant M.
# USAGE : elles seront exclues du modèle par RECIPE4 (COLS_EXCLUDE)
# car non disponibles au début du mois M lors de la prédiction M+1.
# Elles sont conservées ici pour analyse EDA et Dashboard uniquement.
total_jours = df['nbre_jours_activite'] + df['nbre_jours_inactivite']
df['ratio_activite']    = np.where(
    total_jours > 0,
    (df['nbre_jours_activite'] / total_jours).round(4), 0
)
df['ratio_fidelite']    = np.where(
    df['frequentation_totale'] > 0,
    (df['clients_uniques'] / df['frequentation_totale']).round(4), 0
)
df['is_pdv_tres_inactif'] = (df['ratio_activite'] < 0.5).astype(int)
print(f"  ✅ Variables activité créées (usage EDA/Dashboard uniquement)")

# ── 2.3. Variables rupture de stock (calculées sur mois M) ───────
# Même remarque : exclues du modèle par RECIPE4, conservées pour EDA.
df['is_rupture_chronique'] = (df['ratio_jours_rupture'] > 0.50).astype(int)
df['is_rupture_severe']    = (df['ratio_jours_rupture'] > 0.75).astype(int)
df['is_sans_appro']        = (df['appro'] == 0).astype(int)
df['log_appro']            = np.log1p(df['appro'].clip(lower=0)).round(6)
df['log_stock_moyen']      = np.log1p(df['stock_moyen'].clip(lower=0)).round(6)
try:
    df['rupture_quintile'] = pd.qcut(
        df['ratio_jours_rupture'], q=5,
        labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(float).fillna(1)
except Exception:
    df['rupture_quintile'] = 1.0
print(f"  ✅ Variables rupture créées (usage EDA/Dashboard uniquement)")

# ── 2.4. Qualité réseau (v3 — seuils recalibrés sur données réelles) ─
# MISE À JOUR v3 — analyse 750 047 observations fichier Orange Cameroun
#
#  SEUILS JUSTIFIÉS PAR LES DONNÉES :
#   95% → is_site_degrade      : seuil SLA standard opérateur GSM/Orange
#                                 13.9% des obs en dessous
#   80% → is_site_critique     : point de rupture Voice Traffic dans les données
#                                 Voice passe de 389 Erl (100%) à 113 Erl (<80%)
#                                 8.1% des obs en dessous
#   50% → is_site_hors_service : panne grave, 4.6% des obs
#
#  SUPPRIMÉ : is_site_anomalie_grave (valeurs <0 = erreurs de collecte)
#  SUPPRIMÉ : is_site_tres_degrade   (seuil <70 non pertinent dans les données)
#  MODIFIÉ  : is_site_degrade        (90% → 95% standard télécom)
#  AJOUTÉ   : is_site_critique       (< 80% — point de rupture Voice)
#  AJOUTÉ   : is_site_hors_service   (< 50% — panne grave)
#  MODIFIÉ  : dispo_reseau_niveau    (3 paliers → 5 paliers calibrés)
#  MODIFIÉ  : reseau_degrade_composite    (seuil 90% → 95%)
#  AJOUTÉ   : reseau_critique_composite  (seuil 80% — point rupture)

# Filtre défensif : seules les valeurs NÉGATIVES sont des erreurs pour Voice et Data
# tr373 reste borné [0, 100] — Voice et Data n'ont pas de borne supérieure
if 'tr373_cell_availability_pct' in df.columns:
    n_hors = ((df['tr373_cell_availability_pct'] < 0) |
              (df['tr373_cell_availability_pct'] > 100)).sum()
    if n_hors > 0:
        print(f"  ⚠️  {n_hors} valeurs hors [0,100] dans tr373 → médiane")
        med_dispo = df.loc[
            (df['tr373_cell_availability_pct'] >= 0) &
            (df['tr373_cell_availability_pct'] <= 100),
            'tr373_cell_availability_pct'
        ].median()
        df.loc[
            (df['tr373_cell_availability_pct'] < 0) |
            (df['tr373_cell_availability_pct'] > 100),
            'tr373_cell_availability_pct'
        ] = med_dispo

# Voice Traffic : filtre uniquement les négatifs (2000+ Erl est valide)
if 'grp_2g_traffic_speech_erl' in df.columns:
    n_neg_v = (df['grp_2g_traffic_speech_erl'] < 0).sum()
    if n_neg_v > 0:
        print(f"  ⚠️  {n_neg_v} valeurs négatives Voice Traffic → médiane")
        med_v = df.loc[df['grp_2g_traffic_speech_erl'] >= 0,
                       'grp_2g_traffic_speech_erl'].median()
        df.loc[df['grp_2g_traffic_speech_erl'] < 0,
               'grp_2g_traffic_speech_erl'] = med_v

# Data Traffic : filtre uniquement les négatifs (9000+ GB est valide)
if 'data_traffic_gb' in df.columns:
    n_neg_d = (df['data_traffic_gb'] < 0).sum()
    if n_neg_d > 0:
        print(f"  ⚠️  {n_neg_d} valeurs négatives Data Traffic → médiane")
        med_d = df.loc[df['data_traffic_gb'] >= 0, 'data_traffic_gb'].median()
        df.loc[df['data_traffic_gb'] < 0, 'data_traffic_gb'] = med_d

# is_site_degrade : dispo < 95% (seuil SLA standard opérateur GSM)
# Remplace l'ancien seuil de 90% — justifié par l'analyse des données réelles
if 'is_site_degrade' not in df.columns and 'tr373_cell_availability_pct' in df.columns:
    df['is_site_degrade'] = (df['tr373_cell_availability_pct'] < 95.0).astype(int)

# is_site_critique : dispo < 80% (point de rupture Voice Traffic dans les données)
# À ce seuil, Voice Traffic chute de 389 Erl (normal) à 113 Erl — signal fort
if 'is_site_critique' not in df.columns and 'tr373_cell_availability_pct' in df.columns:
    df['is_site_critique'] = (df['tr373_cell_availability_pct'] < 80.0).astype(int)

# is_site_hors_service : dispo < 50% (panne grave — 4.6% des observations)
if 'is_site_hors_service' not in df.columns and 'tr373_cell_availability_pct' in df.columns:
    df['is_site_hors_service'] = (df['tr373_cell_availability_pct'] < 50.0).astype(int)

# dispo_reseau_niveau : 5 paliers calibrés sur distribution réelle Orange Cameroun
# Remplace les 3 paliers arbitraires [0-70-90-100] de v2
# 1 = hors_service [0–50%]  | 2 = critique [50–80%] | 3 = degrade [80–95%]
# 4 = sous_perf [95–99%]    | 5 = normal [99–100%]
if 'tr373_cell_availability_pct' in df.columns:
    df['dispo_reseau_niveau'] = pd.cut(
        df['tr373_cell_availability_pct'],
        bins=[-0.001, 50, 80, 95, 99, 100.999],
        labels=[1, 2, 3, 4, 5]  # 1=hors service → 5=normal
    ).astype(float).fillna(5)   # PDV sans données QR → niveau 5 (imputation prudente)

# log_trafic_voix : transformation log sur Voice Traffic (valeurs positives élevées)
# Note : pas de clip(upper=...) — 2000+ Erl est valide et attendu
if 'grp_2g_traffic_speech_erl' in df.columns:
    df['log_trafic_voix'] = np.log1p(
        df['grp_2g_traffic_speech_erl'].clip(lower=0)
    ).round(6)

# Data Traffic (GB) — NOUVELLE variable internet
# Note : pas de borne supérieure — 9000+ GB est valide et attendu
if 'data_traffic_gb' in df.columns:
    df['log_data_traffic'] = np.log1p(
        df['data_traffic_gb'].clip(lower=0)
    ).round(6)

    # Seuils relatifs P20/P80 (s'adaptent à la vraie distribution)
    p20_data = df.loc[df['qr_data_disponible'] == 1, 'data_traffic_gb'].quantile(0.20) \
               if 'qr_data_disponible' in df.columns else df['data_traffic_gb'].quantile(0.20)
    p80_data = df.loc[df['qr_data_disponible'] == 1, 'data_traffic_gb'].quantile(0.80) \
               if 'qr_data_disponible' in df.columns else df['data_traffic_gb'].quantile(0.80)

    if 'is_data_traffic_faible' not in df.columns:
        df['is_data_traffic_faible'] = (df['data_traffic_gb'] < p20_data).astype(int)
    if 'is_data_traffic_fort' not in df.columns:
        df['is_data_traffic_fort']   = (df['data_traffic_gb'] >= p80_data).astype(int)
    print(f"  ✅ data_traffic_gb — P20={p20_data:.0f} GB | P80={p80_data:.0f} GB")
    print(f"     (valeurs > 9000 GB sont normales — trafic internet élevé)")

# Composites réseau — recalibrés v3
# reseau_degrade_composite  : dispo<95% ET voice<P20 (alerte opérateur)
# reseau_critique_composite : dispo<80% ET voice<P20 (point rupture Voice Traffic)
if 'is_site_degrade' in df.columns and 'is_trafic_reseau_faible' in df.columns:
    if 'reseau_degrade_composite' not in df.columns:
        df['reseau_degrade_composite'] = (
            (df['is_site_degrade'] == 1) &          # dispo < 95%
            (df['is_trafic_reseau_faible'] == 1)    # voice < P20
        ).astype(int)
if 'is_site_critique' in df.columns and 'is_trafic_reseau_faible' in df.columns:
    if 'reseau_critique_composite' not in df.columns:
        df['reseau_critique_composite'] = (
            (df['is_site_critique'] == 1) &         # dispo < 80%
            (df['is_trafic_reseau_faible'] == 1)    # voice < P20
        ).astype(int)

# is_site_problematique : consolidation — regroupe tous les niveaux dégradés
df['is_site_problematique'] = df['is_site_degrade'].copy() \
    if 'is_site_degrade' in df.columns else pd.Series(0, index=df.index)

print(f"  ✅ Variables réseau v3 créées (seuils recalibrés sur données réelles)")
if 'is_site_degrade' in df.columns:
    n_deg  = df['is_site_degrade'].sum()
    n_crit = df['is_site_critique'].sum()    if 'is_site_critique'    in df.columns else 0
    n_hs   = df['is_site_hors_service'].sum() if 'is_site_hors_service' in df.columns else 0
    mn, mx = df['tr373_cell_availability_pct'].min(), df['tr373_cell_availability_pct'].max()
    print(f"     tr373 plage                    : [{mn:.2f}, {mx:.2f}]%")
    print(f"     is_site_degrade    (dispo<95%) : {n_deg:,}  ({n_deg/len(df)*100:.1f}%) ← SLA opérateur")
    print(f"     is_site_critique   (dispo<80%) : {n_crit:,}  ({n_crit/len(df)*100:.1f}%) ← rupture Voice")
    print(f"     is_site_hors_service(dispo<50%): {n_hs:,}   ({n_hs/len(df)*100:.1f}%) ← panne grave")
if 'grp_2g_traffic_speech_erl' in df.columns:
    mn2, mx2 = df['grp_2g_traffic_speech_erl'].min(), df['grp_2g_traffic_speech_erl'].max()
    print(f"     Voice Traffic plage            : [{mn2:.1f}, {mx2:.1f}] Erl (sans borne sup.)")
if 'data_traffic_gb' in df.columns:
    mn3, mx3 = df['data_traffic_gb'].min(), df['data_traffic_gb'].max()
    print(f"     Data Traffic plage             : [{mn3:.1f}, {mx3:.1f}] GB (sans borne sup.)")
if 'reseau_degrade_composite' in df.columns:
    n_comp  = df['reseau_degrade_composite'].sum()
    n_comp2 = df['reseau_critique_composite'].sum() if 'reseau_critique_composite' in df.columns else 0
    print(f"     reseau_degrade_composite       : {n_comp:,}  ({n_comp/len(df)*100:.1f}%) dispo<95%+voice<P20")
    print(f"     reseau_critique_composite      : {n_comp2:,}  ({n_comp2/len(df)*100:.1f}%) dispo<80%+voice<P20")
print(f"     is_site_anomalie_grave         : SUPPRIMÉ (valeurs <0 = erreurs)")
print(f"     is_site_tres_degrade           : SUPPRIMÉ (→ dispo_reseau_niveau)")
print(f"     Ancien seuil 90%               : REMPLACÉ par 95% (standard SLA)")

# ── 2.5. Visites terrain ─────────────────────────────────────────
def categoriser_raison_visite(raison):
    if pd.isna(raison):
        return 'NON_VISITE'
    r = str(raison).strip().upper()
    if 'PAS VISIT' in r or r == 'PAS VISITE':
        return 'NON_VISITE'
    elif any(x in r for x in ['BAISSE', 'PERFORMANCE']):
        return 'VISITE_BAISSE_PERF'
    elif any(x in r for x in ['INACTIF', 'INACTIFS']):
        return 'VISITE_PDV_INACTIF'
    elif any(x in r for x in ['KAABU', 'QUESTIONNAIRE', 'VISITE 2080',
                                'VISITE PDV', 'RETAIL', 'CREME']):
        return 'VISITE_PROGRAMME'
    elif any(x in r for x in ['VAS', 'OFFRE', 'NOUVELLES', 'COMMERCIAL',
                                'CHALLENGE', 'ORANGE MONEY', 'OM']):
        return 'VISITE_PROMO'
    elif 'RAISON INCONNUE' in r or 'INCONNU' in r:
        return 'VISITE_INCONNUE'
    else:
        return 'VISITE_AUTRE'

df['cat_raison_visite']   = df['raisons_visites'].apply(categoriser_raison_visite)
df['visite_pour_baisse']  = (df['cat_raison_visite'] == 'VISITE_BAISSE_PERF').astype(int)
df['visite_planifiee']    = (df['cat_raison_visite'] == 'VISITE_PROGRAMME').astype(int)
df['intensite_suivi_num'] = pd.cut(
    df['nb_visites_mois'], bins=[-1, 0, 1, 2, 9999], labels=[0, 1, 2, 3]
).astype(float).fillna(0).astype(int)
print(f"  ✅ Variables visites terrain créées")
print(f"     Distribution cat_raison_visite :")
for val, cnt in df['cat_raison_visite'].value_counts().items():
    print(f"       {val:30s} : {cnt:,} ({cnt/len(df)*100:.1f}%)")

# ── 2.6. LAGS ET TENDANCE — CORRECTION CYCLE 2 ───────────────────
# ─────────────────────────────────────────────────────────────────
# PROBLÈME CYCLE 1 CORRIGÉ :
#
#   ANCIEN variation_m1 = (C2S_M - C2S_{M-1}) / C2S_{M-1}
#                       = variation_mensuelle (Recipe 2)
#                       = TARGET → leakage parfait (r≈1.000)
#
#   ANCIEN moy_mobile_3m = rolling(3).mean() sur recharges_c2s_mt
#                        = (C2S_M + C2S_{M-1} + C2S_{M-2}) / 3
#                        → contient C2S_M = numérateur de target
#
#   ANCIEN is_declin_continu dépendait de variation_m1 ancien → leakage
#   ANCIEN log_c2s_mt = log(C2S_M) = log(numérateur de target)
#
# RÈGLE ROSSMANN APPLIQUÉE :
#   "Variables that won't be available at the moment of prediction
#    must be dropped." (Rossmann cycle 1 → customers exclue)
#
# HORIZON DE PRÉDICTION :
#   Le modèle est utilisé en FIN de mois M pour prédire M+1.
#   → C2S_M EST CONNU à ce moment.
#   → Mais target(M) = baisse_de_M_vs_M-1 utilise C2S_M directement.
#   → Donc toute variable calculée sur C2S_M encode target(M).
#   → On ne peut utiliser que C2S_{M-1}, C2S_{M-2}, C2S_{M-3}.
#
# NOUVEAU CALCUL :
#   c2s_lag1(M)    = C2S_{M-1}  ← dernier mois connu AVANT M
#   c2s_lag2(M)    = C2S_{M-2}
#   c2s_lag3(M)    = C2S_{M-3}
#   variation_m1   = (C2S_{M-1} - C2S_{M-2}) / C2S_{M-2}  ← variation passée
#   moy_mobile_3m  = (C2S_{M-1} + C2S_{M-2} + C2S_{M-3}) / 3  ← passé uniquement
#   is_declin_continu = deux baisses consécutives sur M-1 et M-2
# ─────────────────────────────────────────────────────────────────
print(f"\n  ── 2.6. Lags et tendance (CYCLE 2 — correction leakage) ──")

df = df.sort_values(['msisdn', 'mois']).copy()

# Lags sur C2S historique
df['c2s_lag1'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(1)
df['c2s_lag2'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(2)
df['c2s_lag3'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(3)

# CORRECTION [C1] : variation_m1 calculée sur lags passés uniquement
# = variation du mois M-1 par rapport à M-2 (information historique)
df['variation_m1'] = np.where(
    df['c2s_lag2'] > 0,
    ((df['c2s_lag1'] - df['c2s_lag2']) / df['c2s_lag2']).round(4),
    0
)

# CORRECTION [C2] : moy_mobile_3m = moyenne de M-1, M-2, M-3 uniquement
df['moy_mobile_3m'] = (
    df['c2s_lag1'].fillna(0) +
    df['c2s_lag2'].fillna(0) +
    df['c2s_lag3'].fillna(0)
) / 3

# CORRECTION [C3] : is_declin_continu sur variations passées uniquement
# variation_lag1(M) = (C2S_{M-2} - C2S_{M-3}) / C2S_{M-3}
variation_lag1 = np.where(
    df['c2s_lag3'] > 0,
    ((df['c2s_lag2'] - df['c2s_lag3']) / df['c2s_lag3']),
    0
)
df['is_declin_continu'] = (
    (df['variation_m1'] < 0) & (variation_lag1 < 0)
).astype(int)

# CORRECTION [C4] : log_c2s_lag1 remplace log_c2s_mt
# log(C2S_{M-1}) = information historique sûre
df['log_c2s_lag1'] = np.log1p(df['c2s_lag1'].fillna(0).clip(lower=0)).round(6)
# NE PAS créer log_c2s_mt = log(C2S_M) → encode target

# Imputation NaN des premières observations par PDV (médiane globale)
for col in ['c2s_lag1', 'c2s_lag2', 'c2s_lag3']:
    med = df[col].median()
    df[col] = df[col].fillna(med if not np.isnan(med) else 0)
df['variation_m1']      = df['variation_m1'].fillna(0)
df['moy_mobile_3m']     = df['moy_mobile_3m'].fillna(0)
df['is_declin_continu'] = df['is_declin_continu'].fillna(0).astype(int)
df['log_c2s_lag1']      = df['log_c2s_lag1'].fillna(0)

# Validation anti-leakage : vérifier que variation_m1 ≠ variation_mensuelle
print(f"  ✅ Lags et tendance recalculés (cycle 2)")
print(f"     variation_m1  = (C2S_{{M-1}} - C2S_{{M-2}}) / C2S_{{M-2}}")
print(f"     moy_mobile_3m = moyenne(C2S_{{M-1}}, C2S_{{M-2}}, C2S_{{M-3}})")
print(f"     is_declin_continu : {df['is_declin_continu'].sum():,} PDV en déclin "
      f"({df['is_declin_continu'].mean():.1%})")

# Vérification : variation_m1 ne doit PAS corréler à 1.0 avec recharges_c2s_mt
corr_check = df[['variation_m1', 'recharges_c2s_mt', 'c2s_lag1', 'c2s_lag2']].corr()
print(f"\n     Corrélation variation_m1 vs recharges_c2s_mt : "
      f"{corr_check.loc['variation_m1','recharges_c2s_mt']:.4f} "
      f"(doit être proche de 0, pas de 1)")

print(f"\n  Total colonnes après feature engineering : {df.shape[1]}")

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 3 — FILTRAGE
# Rossmann 3.1 : exclure stores avec sales=0 (fermés)
# Ici : exclure PDV avec recharges_c2s_mt=0
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 3 — Filtrage ══")

n_avant = len(df)
df = df[df['recharges_c2s_mt'] > 0].copy()
print(f"  Lignes supprimées (recharges=0) : {n_avant - len(df):,}")
print(f"  Lignes conservées               : {len(df):,}")

# Colonnes techniques à supprimer
cols_drop = [
    'mois_dt',          # datetime technique
    'site_name',        # redondant
    'partner_name',     # peu discriminant
    'raisons_visites',  # texte brut → encodé dans cat_raison_visite
    'nom',              # colonne technique QR
    # NE PAS supprimer recharges_c2s_mt → nécessaire pour Recipe 2 (calcul target)
]
cols_drop = [c for c in cols_drop if c in df.columns]
df = df.drop(columns=cols_drop)

print(f"  Colonnes supprimées : {cols_drop}")
print(f"  Colonnes conservées : {df.shape[1]}")
print(f"  PDV uniques         : {df['msisdn'].nunique():,}")
print(f"  Mois couverts       : {sorted(df['mois'].unique())}")

# ─────────────────────────────────────────────────────────────────
# ÉCRITURE
# ─────────────────────────────────────────────────────────────────
output = dataiku.Dataset("df_prepared")
output.write_with_schema(df)
print(f"\n✅ df_prepared (cycle 2) : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
print(f"   Corrections cycle 2 appliquées :")
print(f"   [C1] variation_m1   = (C2S_{{M-1}} - C2S_{{M-2}}) / C2S_{{M-2}}")
print(f"   [C2] moy_mobile_3m  = moyenne(lags M-1, M-2, M-3)")
print(f"   [C3] is_declin_continu basé sur variations passées uniquement")
print(f"   [C4] log_c2s_mt supprimé → remplacé par log_c2s_lag1")
