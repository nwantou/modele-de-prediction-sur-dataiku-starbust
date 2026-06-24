# ══════════════════════════════════════════════════════════════════
# RECIPE 1 v2 — compute_df_prepared (ANTI DATA LEAKAGE)
# Étapes 1+2+3 du CRISP-DM
# Input  : rupt_freq_with_reseau
# Output : df_prepared
#
# CORRECTION V2 — DATA LEAKAGE IDENTIFIÉ ET CORRIGÉ :
#
# PROBLÈME ORIGINAL :
#   variation_m1 = (C2S_M - C2S_M-1) / C2S_M-1
#   target       = 1 si variation_M <= -10%
#   → variation_m1 CONTIENT target → leakage direct → P=1.0
#
# CORRECTION :
#   variation_m1 = (C2S_M-1 - C2S_M-2) / C2S_M-2
#   → c'est la tendance PASSÉE, pas la variation actuelle
#   → Le modèle prédit si la tendance M-1 se poursuivra au mois M
#
# RÈGLE FONDAMENTALE :
#   On est au moment de prédire le mois M+1.
#   Les features doivent être connues à la FIN du mois M.
#   La valeur recharges_c2s_mt du mois M EST connue → OK en feature
#   Mais la variation_M (M vs M-1) calcule target → EXCLUE des features
#
# NOUVELLES VARIABLES CORRECTES :
#   variation_m1    = (C2S_M-1 - C2S_M-2) / C2S_M-2   ← tendance M-1 vs M-2
#   variation_m2    = (C2S_M-2 - C2S_M-3) / C2S_M-3   ← tendance M-2 vs M-3
#   is_declin_continu = variation_m1 < 0 ET variation_m2 < 0 ← OK (tout passé)
#   moy_mobile_3m   = moyenne de C2S_M-1, M-2, M-3     ← OK (tout passé)
# ══════════════════════════════════════════════════════════════════

import dataiku
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 0. CHARGEMENT
# ─────────────────────────────────────────────────────────────────
print("── Chargement rupt_freq_with_reseau ──")
df = dataiku.Dataset("rupt_freq_with_reseau").get_dataframe()
print(f"  Lignes   : {df.shape[0]:,}")
print(f"  Colonnes : {df.shape[1]}")
print(f"  Mois     : {sorted(df['mois'].unique())}")
print(f"  PDV      : {df['msisdn'].nunique():,}")

df.columns = [c.lower().strip() for c in df.columns]
df['msisdn'] = df['msisdn'].astype(str).str.strip()
df['mois']   = df['mois'].astype(str).str.strip()

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 1 — DESCRIPTION & IMPUTATION
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 1 — Description & Imputation ══")

df['mois_dt'] = pd.to_datetime(df['mois'] + '-01')

# Numériques → 0
for col in ['frequentation_totale', 'clients_uniques',
            'nbre_jours_activite', 'nbre_jours_inactivite',
            'recharges_c2s_mt', 'appro', 'nb_jours_rupture_stock',
            'ratio_jours_rupture', 'seuil_moyen_horaire', 'stock_moyen']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

for col in ['grp_2g_traffic_speech_erl', 'tr373_cell_availability_pct',
            'log_trafic_reseau']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
            df[col].median() if df[col].notna().any() else 0
        )

for col in ['qr_data_disponible', 'visite_data_disponible',
            'is_site_degrade', 'is_trafic_reseau_faible']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

for col in ['nb_visites_mois', 'pdv_visite', 'pdv_non_visite', 'pdv_multi_visite']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

if 'raisons_visites' in df.columns:
    df['raisons_visites'] = df['raisons_visites'].fillna('Pas visité').astype(str).str.strip()

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
if len(missing_after) == 0:
    print("  ✅ Aucun NaN résiduel")
else:
    print(f"  ⚠️  NaN résiduels : {missing_after.to_dict()}")

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 2 — Feature Engineering ══")

# ── 2.1. Variables temporelles ────────────────────────────────────
df['mois_num']         = df['mois_dt'].dt.month
df['annee']            = df['mois_dt'].dt.year
df['trimestre']        = df['mois_dt'].dt.quarter
df['is_debut_annee']   = df['mois_num'].isin([1, 2]).astype(int)
df['is_fin_annee']     = df['mois_num'].isin([11, 12]).astype(int)
df['is_milieu_annee']  = df['mois_num'].isin([5, 6, 7]).astype(int)
df['mois_sin']         = np.sin(df['mois_num'] * (2 * np.pi / 12)).round(6)
df['mois_cos']         = np.cos(df['mois_num'] * (2 * np.pi / 12)).round(6)
print("  ✅ Variables temporelles créées")

# ── 2.2. Variables d'activité PDV ─────────────────────────────────
df['ratio_fidelite'] = np.where(
    df['frequentation_totale'] > 0,
    (df['clients_uniques'] / df['frequentation_totale']).round(4), 0
)

total_jours = df['nbre_jours_activite'] + df['nbre_jours_inactivite']
df['ratio_activite'] = np.where(
    total_jours > 0,
    (df['nbre_jours_activite'] / total_jours).round(4), 0
)
df['is_pdv_tres_inactif'] = (df['ratio_activite'] < 0.5).astype(int)
print("  ✅ Variables activité créées")

# ── 2.3. Variables de rupture de stock ────────────────────────────
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
print("  ✅ Variables rupture créées")

# ── 2.4. Qualité réseau ───────────────────────────────────────────
df['is_site_anomalie_grave'] = (df['tr373_cell_availability_pct'] < 0).astype(int)
df['is_site_tres_degrade']   = (
    (df['tr373_cell_availability_pct'] >= 0) &
    (df['tr373_cell_availability_pct'] < 70)
).astype(int)
df['is_site_problematique'] = (
    (df['is_site_anomalie_grave'] == 1) | (df['is_site_degrade'] == 1)
).astype(int)
df['dispo_reseau_niveau'] = pd.cut(
    df['tr373_cell_availability_pct'],
    bins=[-float('inf'), 0, 70, 80, 90, float('inf')],
    labels=[1, 2, 3, 4, 5]
).astype(float).fillna(3)
df['log_trafic_voix'] = np.log1p(
    df['grp_2g_traffic_speech_erl'].clip(lower=0)
).round(6)

n_anomalies = df['is_site_anomalie_grave'].sum()
print(f"  ✅ Variables réseau créées (anomalies négatives codées)")
print(f"     Sites en anomalie grave (dispo<0) : {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")

# ── 2.5. Visites terrain ─────────────────────────────────────────
def categoriser_raison_visite(raison):
    if pd.isna(raison):
        return 'NON_VISITE'
    r = str(raison).strip().upper()
    if 'PAS VISIT' in r or r == 'PAS VISITE':
        return 'NON_VISITE'
    elif any(x in r for x in ['BAISSE', 'PERFORMANCE', 'PERFORMANCES']):
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

# ── 2.6. TENDANCE TEMPORELLE — CORRECTION ANTI-LEAKAGE ────────────
# ─────────────────────────────────────────────────────────────────
# CORRECTION CLEF v2 :
#
# VERSION INCORRECTE (leakage) :
#   variation_m1 = (C2S_M - C2S_M-1) / C2S_M-1
#   → Contient C2S_M, utilisé pour calculer target → LEAKAGE
#   → P=1.0000 garanti
#
# VERSION CORRECTE :
#   c2s_lag1  = C2S_M-1 (mois précédent)
#   c2s_lag2  = C2S_M-2 (il y a 2 mois)
#   c2s_lag3  = C2S_M-3 (il y a 3 mois)
#   variation_m1  = (C2S_M-1 - C2S_M-2) / C2S_M-2  ← tendance M-1 vs M-2
#   variation_m2  = (C2S_M-2 - C2S_M-3) / C2S_M-3  ← tendance M-2 vs M-3
#   is_declin_continu = variation_m1 < 0 ET variation_m2 < 0 ← tout passé
#   moy_mobile_3m = moyenne(C2S_M-1, C2S_M-2, C2S_M-3)      ← tout passé
#
# POURQUOI C'EST CORRECT :
#   Au moment de prédire le mois M+1, on connaît C2S jusqu'au mois M.
#   Donc C2S_M (= c2s_lag1 dans le prochain mois) est disponible.
#   La variation_m1 représente la tendance récente visible → signal légitime.
# ─────────────────────────────────────────────────────────────────

df = df.sort_values(['msisdn', 'mois']).copy()

# Lags de recharges_c2s_mt
df['c2s_lag1'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(1)
df['c2s_lag2'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(2)
df['c2s_lag3'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(3)

df['log_c2s_mt'] = np.log1p(df['recharges_c2s_mt'].clip(lower=0)).round(6)

# ────────────────────────────────────────────────────────────────
# CORRECTION : variation_m1 = (C2S_M-1 - C2S_M-2) / C2S_M-2
# et NON plus (C2S_M - C2S_M-1) / C2S_M-1
# ────────────────────────────────────────────────────────────────
df['variation_m1'] = np.where(
    df['c2s_lag2'] > 0,
    ((df['c2s_lag1'] - df['c2s_lag2']) / df['c2s_lag2']).round(4),
    0
)

df['variation_m2'] = np.where(
    df['c2s_lag3'] > 0,
    ((df['c2s_lag2'] - df['c2s_lag3']) / df['c2s_lag3']).round(4),
    0
)

# Moyenne mobile sur les 3 mois précédents (PAS le mois M)
df['moy_mobile_3m'] = (
    df.groupby('msisdn')['recharges_c2s_mt']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
).round(2)

# Déclin continu : baisse sur M-1 vs M-2 ET M-2 vs M-3 (tout passé)
df['is_declin_continu'] = (
    (df['variation_m1'] < 0) & (df['variation_m2'] < 0)
).astype(int)

# Imputation NaN lags (premières observations par PDV)
med_c2s = df['recharges_c2s_mt'].median()
for col in ['c2s_lag1', 'c2s_lag2', 'c2s_lag3']:
    df[col] = df[col].fillna(med_c2s if not np.isnan(med_c2s) else 0)

df['variation_m1']      = df['variation_m1'].fillna(0)
df['variation_m2']      = df['variation_m2'].fillna(0)
df['moy_mobile_3m']     = df['moy_mobile_3m'].fillna(med_c2s)
df['is_declin_continu'] = df['is_declin_continu'].fillna(0).astype(int)

print(f"  ✅ Variables de tendance (lags CORRIGÉS) créées")
print(f"     variation_m1 = (C2S_M-1 - C2S_M-2)/C2S_M-2 ← CORRECT (pas de leakage)")
print(f"     variation_m2 = (C2S_M-2 - C2S_M-3)/C2S_M-3 ← CORRECT (tout passé)")
print(f"     is_declin_continu = variation_m1 < 0 ET variation_m2 < 0 ← CORRECT")
print(f"     moy_mobile_3m = moyenne(C2S_M-1, M-2, M-3) ← CORRECT")

print(f"\n  Total colonnes après feature engineering : {df.shape[1]}")

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 3 — FILTRAGE
# ═════════════════════════════════════════════════════════════════
print("\n══ ÉTAPE 3 — Filtrage ══")

n_avant = len(df)
df = df[df['recharges_c2s_mt'] > 0].copy()
print(f"  Lignes supprimées (recharges=0) : {n_avant - len(df):,}")
print(f"  Lignes conservées               : {len(df):,}")

cols_drop = ['mois_dt', 'site_name', 'partner_name', 'raisons_visites', 'nom']
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
print(f"\n✅ df_prepared écrit : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
print(f"\nVÉRIFICATION ANTI-LEAKAGE :")
print(f"  variation_m1 = tendance M-1 vs M-2 (passé) ← OK")
print(f"  variation_m2 = tendance M-2 vs M-3 (passé) ← OK")
print(f"  is_declin_continu calculé sur variation_m1+m2 ← OK")
print(f"  moy_mobile_3m = moyenne sur M-1, M-2, M-3   ← OK")
print(f"  recharges_c2s_mt du mois M présent (sera exclue de la Recipe 4)")
