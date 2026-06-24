# ══════════════════════════════════════════════════════════════════
# RECIPE 2 : compute_target
# Étape 4 — Construction de la variable cible
# Input  : df_prepared
# Output : df_with_target
# ══════════════════════════════════════════════════════════════════

import dataiku
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("── Chargement df_prepared ──")
df = dataiku.Dataset("df_prepared").get_dataframe()
print(f"  {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
print(f"  Mois : {sorted(df['mois'].unique())}")

# ═════════════════════════════════════════════════════════════════
# ÉTAPE 4 — VARIABLE CIBLE
# Règle anti data leakage (Rossmann) :
# La cible du mois M est calculée à partir des données du mois M-1
# → On ne regarde pas le futur pour entraîner le modèle
# ═════════════════════════════════════════════════════════════════
SEUIL_BAISSE = -0.10  # -10% validé dans les analyses précédentes

print(f"\n══ ÉTAPE 4 — Variable cible (seuil = {SEUIL_BAISSE*100:.0f}%) ══")

df = df.sort_values(['msisdn', 'mois']).copy()

# Recharges du mois précédent
df['recharges_mois_prec'] = df.groupby('msisdn')['recharges_c2s_mt'].shift(1)

# Variation mensuelle M vs M-1
df['variation_mensuelle'] = np.where(
    df['recharges_mois_prec'] > 0,
    ((df['recharges_c2s_mt'] - df['recharges_mois_prec']) / df['recharges_mois_prec']).round(4),
    0
)

# Cible binaire
df['target'] = (df['variation_mensuelle'] <= SEUIL_BAISSE).astype(int)

# Supprimer lignes sans historique (première obs par PDV)
n_avant = len(df)
df = df.dropna(subset=['recharges_mois_prec'])
print(f"  Lignes supprimées (pas d'historique) : {n_avant - len(df):,}")

# Rapport
n_baisse = df['target'].sum()
n_stable = (df['target'] == 0).sum()
taux = df['target'].mean()
print(f"\n  0 (Stable) : {n_stable:,} ({1-taux:.1%})")
print(f"  1 (Baisse) : {n_baisse:,} ({taux:.1%})")
print(f"  scale_pos_weight XGBoost : {n_stable/n_baisse:.2f}")

# Distribution par mois
print("\n  Taux de baisse par mois :")
taux_mois = df.groupby('mois')['target'].agg(['mean','sum','count']).reset_index()
taux_mois.columns = ['mois','taux_baisse','nb_baisses','total']
print(taux_mois.to_string())

output = dataiku.Dataset("df_with_target")
output.write_with_schema(df)
print(f"\n✅ df_with_target : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
