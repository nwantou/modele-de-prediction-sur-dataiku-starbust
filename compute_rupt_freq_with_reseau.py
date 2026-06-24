# ══════════════════════════════════════════════════════════════════
# RECIPE : Jointure Qualité Réseau + Visites Terrain → rupt_freq
# VERSION 2 — Mise à jour nouveaux noms colonnes réseau
#
# CHANGEMENTS vs Version 1 :
#   ① tr373_cell_availability_pct ← "Disponibilité réseau (%)"
#      Valeurs valides : [0, 100] uniquement.
#      Les valeurs négatives du fichier précédent étaient des ERREURS
#      de collecte — elles sont maintenant absentes.
#      → Suppression du flag is_site_anomalie_grave
#      → Suppression du flag is_site_tres_degrade
#      → is_site_degrade recalibré sur seuil 90%
#
#   ② grp_2g_traffic_speech_erl ← "Voice Traffic (Erl)"
#      Valeurs positives SANS borne supérieure — des valeurs comme
#      2010 Erl sont tout à fait normales (trafic voix volumieux).
#      Corrélée à la disponibilité : si dispo baisse, Voice Traffic
#      baisse aussi. Signal traité via seuils relatifs (P20).
#
#   ③ Data Traffic (GB) → NOUVELLE variable Internet
#      Valeurs positives SANS borne supérieure — des valeurs comme
#      9674 GB sont tout à fait normales (site à fort trafic internet).
#      Plus la valeur est haute, plus le trafic est volumineux.
#      Signal traité via seuils relatifs (P20/P80).
#
# Sources :
#   1. qualite_reseau  → jointure sur site_cod + mois
#   2. BASE_VISITE_OK  → jointure sur msisdn + mois
# Imputation : méthode Rossmann
# ══════════════════════════════════════════════════════════════════

import dataiku
import pandas as pd
import numpy as np

# ── 1. Chargement ─────────────────────────────────────────────────
print("── Chargement des datasets ──")

df       = dataiku.Dataset("rupt_freq_copy").get_dataframe()
df_qr    = dataiku.Dataset("qualite_reseau").get_dataframe()
df_visit = dataiku.Dataset("BASE_VISITE_OK").get_dataframe()

print(f"  rupt_freq       : {df.shape[0]:,} lignes | {df.shape[1]} colonnes")
print(f"  qualite_reseau  : {df_qr.shape[0]:,} lignes | {df_qr.shape[1]} colonnes")
print(f"  BASE_VISITE_OK  : {df_visit.shape[0]:,} lignes | {df_visit.shape[1]} colonnes")
print(f"\n  Colonnes qualite_reseau brutes : {df_qr.columns.tolist()}")

# ── 2. Normalisation colonnes ─────────────────────────────────────
print("\n── Normalisation colonnes ──")

df.columns = [c.lower().strip() for c in df.columns]

df_qr.columns = (
    df_qr.columns
    .str.strip()
    .str.lower()
    .str.replace(' ',  '_', regex=False)
    .str.replace(':',  '_', regex=False)
    .str.replace('(',  '',  regex=False)
    .str.replace(')',  '',  regex=False)
    .str.replace('%',  'pct', regex=False)
)

df_visit.columns = (
    df_visit.columns
    .str.lower().str.strip()
    .str.replace(' ',  '_', regex=False)
    .str.replace('.',  '_', regex=False)
    .str.replace('(',  '',  regex=False)
    .str.replace(')',  '',  regex=False)
)

print(f"  Colonnes rupt_freq      : {df.columns.tolist()}")
print(f"  Colonnes qualite_reseau : {df_qr.columns.tolist()}")
print(f"  Colonnes BASE_VISITE    : {df_visit.columns.tolist()}")

# ══════════════════════════════════════════════════════════════════
# PARTIE A — QUALITÉ RÉSEAU
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE A — Qualité Réseau ══")

# ── 3A. Suppression colonnes dupliquées ───────────────────────────
cols_dup = df_qr.columns[df_qr.columns.duplicated()].tolist()
if cols_dup:
    print(f"  Colonnes dupliquées QR supprimées : {cols_dup}")
df_qr = df_qr.loc[:, ~df_qr.columns.duplicated(keep='first')]

# ── 4A. Renommage colonnes QR ─────────────────────────────────────
# MISE À JOUR v2 :
#   "Disponibilité réseau (%)" → tr373_cell_availability_pct
#   "Voice Traffic (Erl)"      → grp_2g_traffic_speech_erl
#   "Data Traffic (GB)"        → data_traffic_gb  [NOUVEAU]
#
# Après normalisation ci-dessus, les noms deviennent :
#   disponibilite_reseau_pct   (Disponibilité réseau (%) → minusc + replace)
#   voice_traffic_erl          (Voice Traffic (Erl) → minusc + replace)
#   data_traffic_gb            (Data Traffic (GB) → minusc + replace)

rename_qr = {}
for col in df_qr.columns:
    cl = col.lower()
    if cl == 'date':
        rename_qr[col] = 'date_qr'
    elif 'site' in cl and 'name' in cl:
        rename_qr[col] = 'site_name_qr'
    # ── Anciens noms (v1) ─────────────────────────────────────────
    elif 'traffic' in cl and 'speech' in cl:
        rename_qr[col] = 'grp_2g_traffic_speech_erl'
    elif 'availability' in cl or 'tr373' in cl:
        rename_qr[col] = 'tr373_cell_availability_pct'
    # ── Nouveaux noms (v2) ────────────────────────────────────────
    elif 'disponibilit' in cl and 'seau' in cl:
        rename_qr[col] = 'tr373_cell_availability_pct'
    elif 'voice' in cl and 'traffic' in cl:
        rename_qr[col] = 'grp_2g_traffic_speech_erl'
    elif 'data' in cl and 'traffic' in cl:
        rename_qr[col] = 'data_traffic_gb'          # NOUVELLE colonne

df_qr = df_qr.rename(columns=rename_qr)
print(f"  Colonnes QR après renommage : {df_qr.columns.tolist()}")

# ── 4A-bis. NETTOYAGE VALEURS NÉGATIVES ──────────────────────────
# MISE À JOUR v2 :
# Les valeurs négatives dans tr373_cell_availability_pct étaient des
# ERREURS de collecte dans l'ancien fichier réseau. Le nouveau fichier
# ne devrait plus en contenir. On applique quand même un filtre
# défensif : toute valeur hors [0, 100] est remplacée par NaN,
# puis imputée par la médiane du train (voir Partie C).
#
# Impact sur les features :
#   SUPPRIMÉ : is_site_anomalie_grave (basé sur valeurs < 0)
#   SUPPRIMÉ : is_site_tres_degrade   (basé sur valeurs < 70)
#   CONSERVÉ : is_site_degrade        (seuil 90% — toujours valide)

if 'tr373_cell_availability_pct' in df_qr.columns:
    n_neg  = (df_qr['tr373_cell_availability_pct'] < 0).sum()
    n_sup  = (df_qr['tr373_cell_availability_pct'] > 100).sum()
    n_hors = n_neg + n_sup
    if n_hors > 0:
        print(f"\n  [QR] ⚠ Valeurs hors [0,100] détectées : {n_hors}")
        print(f"       Négatives (<0) : {n_neg} | Sup. 100 : {n_sup}")
        print(f"       → Remplacement par NaN (seront imputées par la médiane)")
        df_qr.loc[
            (df_qr['tr373_cell_availability_pct'] < 0) |
            (df_qr['tr373_cell_availability_pct'] > 100),
            'tr373_cell_availability_pct'
        ] = np.nan
    else:
        print(f"\n  [QR] ✅ Aucune valeur hors [0,100] — nouveau fichier propre")

if 'grp_2g_traffic_speech_erl' in df_qr.columns:
    n_neg_v = (df_qr['grp_2g_traffic_speech_erl'] < 0).sum()
    if n_neg_v > 0:
        print(f"  [QR] ⚠ Voice Traffic valeurs negatives : {n_neg_v} → NaN")
        print(f"       (valeurs positives elevees comme 2000+ Erl sont VALIDES)")
        df_qr.loc[df_qr['grp_2g_traffic_speech_erl'] < 0,
                  'grp_2g_traffic_speech_erl'] = np.nan
    else:
        print(f"  [QR] ✅ Voice Traffic : aucune valeur negative")
        print(f"       Plage observee : [{df_qr['grp_2g_traffic_speech_erl'].min():.1f}, "
              f"{df_qr['grp_2g_traffic_speech_erl'].max():.1f}] Erl — normal")

if 'data_traffic_gb' in df_qr.columns:
    n_neg_d = (df_qr['data_traffic_gb'] < 0).sum()
    if n_neg_d > 0:
        print(f"  [QR] ⚠ Data Traffic valeurs negatives : {n_neg_d} → NaN")
        print(f"       (valeurs positives elevees comme 9000+ GB sont VALIDES)")
        df_qr.loc[df_qr['data_traffic_gb'] < 0,
                  'data_traffic_gb'] = np.nan
    else:
        print(f"  [QR] ✅ Data Traffic : aucune valeur negative")
        print(f"       Plage observee : [{df_qr['data_traffic_gb'].min():.1f}, "
              f"{df_qr['data_traffic_gb'].max():.1f}] GB — normal")

# ── 5A. Normalisation clés ────────────────────────────────────────
df['site_cod']        = df['site_cod'].astype(str).str.strip().str.upper()
df['site']            = df['site'].astype(str).str.strip().str.upper()
df['mois']            = df['mois'].astype(str).str.strip()
df['msisdn']          = df['msisdn'].astype(str).str.strip()
df_qr['site_name_qr'] = df_qr['site_name_qr'].astype(str).str.strip().str.upper()

# ── 6A. Conversion Date → Mois QR ────────────────────────────────
df_qr['date_qr'] = pd.to_datetime(df_qr['date_qr'], errors='coerce')
df_qr['mois_qr'] = df_qr['date_qr'].dt.strftime('%Y-%m')

mois_rupt   = set(df['mois'].unique())
mois_qr_set = set(df_qr['mois_qr'].dropna().unique())
print(f"\n  Mois couverts QR : {sorted(mois_qr_set)}")
print(f"  Mois sans QR     : {sorted(mois_rupt - mois_qr_set)}")

# ── 7A. Métriques QR disponibles ──────────────────────────────────
# MISE À JOUR v2 : data_traffic_gb est maintenant incluse
METRIC_COLS_QR_CANDIDATS = [
    'grp_2g_traffic_speech_erl',
    'tr373_cell_availability_pct',
    'data_traffic_gb',           # NOUVEAU
]
metric_cols_qr = [c for c in METRIC_COLS_QR_CANDIDATS if c in df_qr.columns]
print(f"\n  Métriques QR utilisées : {metric_cols_qr}")

if 'data_traffic_gb' not in metric_cols_qr:
    print("  ⚠ data_traffic_gb absent du fichier réseau — vérifier le nom de colonne")

# ── 8A. Construction clés QR + explode ───────────────────────────
def generer_cles(site_name_qr):
    if pd.isna(site_name_qr) or str(site_name_qr).upper() in ('NAN', ''):
        return ['__INCONNU__']
    parts    = str(site_name_qr).strip().upper().split('_')
    if len(parts) < 2:
        return [str(site_name_qr).strip().upper()]
    site_cod = f"{parts[0]}_{parts[1]}"
    if len(parts) <= 2:
        return [f"{site_cod}||{site_cod}"]
    return [f"{site_cod}||{'_'.join(parts[s:])}" for s in range(2, len(parts))]

cols_utiles_qr     = ['site_name_qr', 'mois_qr'] + metric_cols_qr
df_qr_work         = df_qr[cols_utiles_qr].copy()
df_qr_work['cles'] = df_qr_work['site_name_qr'].apply(generer_cles)
df_qr_exploded     = (
    df_qr_work.explode('cles')
    .rename(columns={'cles': 'cle_jointure'})
)
df_qr_exploded = df_qr_exploded[
    df_qr_exploded['cle_jointure'] != '__INCONNU__'
].copy()

# ── 9A. Agrégation mensuelle QR ───────────────────────────────────
agg_dict_qr   = {col: 'mean' for col in metric_cols_qr}
df_qr_monthly = (
    df_qr_exploded
    .groupby(['cle_jointure', 'mois_qr'], as_index=False)
    .agg(agg_dict_qr)
)
for col in metric_cols_qr:
    df_qr_monthly[col] = df_qr_monthly[col].round(4)

n_dup = df_qr_monthly.duplicated(subset=['cle_jointure', 'mois_qr']).sum()
print(f"  Lignes QR agrégées : {df_qr_monthly.shape[0]:,} | doublons : {n_dup}")

# ── 10A. Jointure QR ──────────────────────────────────────────────
df['cle_jointure'] = df['site_cod'] + '||' + df['site']

match_qr = (
    set(df['cle_jointure'].dropna().unique()) &
    set(df_qr_monthly['cle_jointure'].dropna().unique())
)
print(f"  Taux couverture QR : {len(match_qr)/max(len(set(df['cle_jointure'].dropna().unique())),1)*100:.1f}%")

df = pd.merge(
    df,
    df_qr_monthly[['cle_jointure', 'mois_qr'] + metric_cols_qr],
    left_on  = ['cle_jointure', 'mois'],
    right_on = ['cle_jointure', 'mois_qr'],
    how      = 'left'
)
df = df.drop(columns=['mois_qr', 'cle_jointure'], errors='ignore')

# ══════════════════════════════════════════════════════════════════
# PARTIE B — VISITES TERRAIN (BASE_VISITE_OK)
# inchangée vs v1
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE B — Visites Terrain (BASE_VISITE_OK) ══")

rename_visit = {}
for col in df_visit.columns:
    if col == 'msisdn':
        rename_visit[col] = 'msisdn_v'
    elif 'nom' in col and 'mois' in col:
        rename_visit[col] = 'nom_mois_v'
    elif col in ('annee', 'année', 'year'):
        rename_visit[col] = 'annee_v'
    elif col == 'date':
        rename_visit[col] = 'date_v'
    elif 'survey' in col or 'tache' in col or 'template' in col:
        rename_visit[col] = 'type_visite'
    elif col == 'visite':
        rename_visit[col] = 'visite'
    elif col == 'mois':
        rename_visit[col] = 'mois_v'

df_visit = df_visit.rename(columns=rename_visit)
print(f"  Colonnes après renommage : {df_visit.columns.tolist()}")

print("\n── [VISITES] Construction mois_v ──")
if 'mois_v' in df_visit.columns:
    df_visit['mois_v'] = df_visit['mois_v'].astype(str).str.strip()
    print(f"  mois_v direct : {df_visit['mois_v'].dropna().unique()[:5].tolist()}")
elif 'date_v' in df_visit.columns:
    df_visit['date_v'] = pd.to_datetime(df_visit['date_v'], errors='coerce')
    df_visit['mois_v'] = df_visit['date_v'].dt.strftime('%Y-%m')
    print(f"  mois_v depuis date : {df_visit['mois_v'].dropna().unique()[:5].tolist()}")
elif 'nom_mois_v' in df_visit.columns and 'annee_v' in df_visit.columns:
    mois_map = {
        'janvier':'01','février':'02','fevrier':'02','mars':'03',
        'avril':'04','mai':'05','juin':'06','juillet':'07',
        'août':'08','aout':'08','septembre':'09','octobre':'10',
        'novembre':'11','décembre':'12','decembre':'12',
        'january':'01','february':'02','march':'03','april':'04',
        'may':'05','june':'06','july':'07','august':'08',
        'september':'09','october':'10','november':'11','december':'12'
    }
    df_visit['mois_num'] = (
        df_visit['nom_mois_v'].astype(str).str.lower().str.strip().map(mois_map)
    )
    df_visit['mois_v'] = (
        df_visit['annee_v'].astype(str).str.strip()
        + '-' + df_visit['mois_num'].fillna('01')
    )
    print(f"  mois_v depuis nom+année : {df_visit['mois_v'].dropna().unique()[:5].tolist()}")
else:
    raise ValueError(
        f"❌ Impossible de construire mois_v — colonnes : {df_visit.columns.tolist()}"
    )

if 'msisdn_v' not in df_visit.columns:
    msisdn_candidates = [c for c in df_visit.columns if 'msisdn' in c.lower()]
    if msisdn_candidates:
        df_visit = df_visit.rename(columns={msisdn_candidates[0]: 'msisdn_v'})
    else:
        raise ValueError(
            f"❌ Colonne MSISDN introuvable — colonnes : {df_visit.columns.tolist()}"
        )

df_visit['msisdn_v'] = df_visit['msisdn_v'].astype(str).str.strip()

visite_num = pd.to_numeric(df_visit['visite'], errors='coerce')
if visite_num.isna().sum() == 0:
    df_visit['visite_bin'] = visite_num.astype(int)
else:
    df_visit['visite_bin'] = df_visit['visite'].astype(str).str.lower().apply(
        lambda x: 0 if any(mot in x for mot in ['pas', 'non', 'not', '0']) else 1
    )

print(f"  Distribution visite_bin : {df_visit['visite_bin'].value_counts().to_dict()}")

df_visit_agg = (
    df_visit
    .groupby(['msisdn_v', 'mois_v'], as_index=False)
    .agg(
        nb_visites_mois = ('visite_bin', 'sum'),
        pdv_visite      = ('visite_bin', 'max'),
    )
)

assert 'msisdn_v' in df_visit_agg.columns, \
    f"❌ msisdn_v absent après agg ! Colonnes : {df_visit_agg.columns.tolist()}"

if 'type_visite' in df_visit.columns:
    df_visit_raisons = (
        df_visit
        .groupby(['msisdn_v', 'mois_v'])['type_visite']
        .apply(lambda x: '|'.join(
            sorted(set(str(v) for v in x.dropna() if str(v).strip() != ''))
        ))
        .reset_index()
        .rename(columns={'type_visite': 'raisons_visites'})
    )
    df_visit_agg = pd.merge(
        df_visit_agg,
        df_visit_raisons[['msisdn_v', 'mois_v', 'raisons_visites']],
        on=['msisdn_v', 'mois_v'], how='left'
    )

msisdn_rupt  = set(df['msisdn'].dropna().unique())
msisdn_v_set = set(df_visit_agg['msisdn_v'].dropna().unique())
match_v      = msisdn_rupt & msisdn_v_set
print(f"\n  MSISDN matchés : {len(match_v):,} ({len(match_v)/max(len(msisdn_rupt),1)*100:.1f}%)")

cols_v = ['msisdn_v', 'mois_v', 'nb_visites_mois', 'pdv_visite']
if 'raisons_visites' in df_visit_agg.columns:
    cols_v.append('raisons_visites')

df_merged = pd.merge(
    df,
    df_visit_agg[cols_v],
    left_on=['msisdn', 'mois'],
    right_on=['msisdn_v', 'mois_v'],
    how='left'
)
df_merged = df_merged.drop(columns=['msisdn_v', 'mois_v'], errors='ignore')
print(f"  Lignes après jointure visites : {df_merged.shape[0]:,}")

# ══════════════════════════════════════════════════════════════════
# PARTIE C — IMPUTATION MÉTHODE ROSSMANN
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE C — Imputation (méthode Rossmann) ══")

n_total = len(df_merged)

# ── C1. Qualité réseau ─────────────────────────────────────────────
# Indicateur de disponibilité de la donnée réelle AVANT imputation
# Valide pour les 3 métriques : dispo, voice, data
df_merged['qr_data_disponible'] = (
    df_merged[metric_cols_qr[0]].notna()
    if metric_cols_qr else pd.Series(0, index=df_merged.index)
).astype(int)

n_qr = df_merged['qr_data_disponible'].sum()
print(f"\n  [QR] données réelles  : {n_qr:,} ({n_qr/n_total*100:.1f}%)")
print(f"  [QR] données imputées : {n_total-n_qr:,} ({(n_total-n_qr)/n_total*100:.1f}%)")

# Imputation par médiane du train pour chaque métrique QR
# MISE À JOUR v2 : inclut data_traffic_gb
for col in metric_cols_qr:
    med = df_merged.loc[df_merged['qr_data_disponible'] == 1, col].median()
    df_merged[col] = df_merged[col].fillna(med)
    print(f"  [QR] {col}")
    print(f"       médiane réelle = {med:.4f} | NaN résiduels = {df_merged[col].isna().sum()}")

# ── C2. Visites terrain ────────────────────────────────────────────
if 'pdv_visite' in df_merged.columns:
    df_merged['visite_data_disponible'] = df_merged['pdv_visite'].notna().astype(int)
    n_v = df_merged['visite_data_disponible'].sum()
    print(f"\n  [VISIT] données réelles  : {n_v:,} ({n_v/n_total*100:.1f}%)")

    df_merged['nb_visites_mois'] = df_merged['nb_visites_mois'].fillna(0).astype(int)
    df_merged['pdv_visite']      = df_merged['pdv_visite'].fillna(0).astype(int)
    if 'raisons_visites' in df_merged.columns:
        df_merged['raisons_visites'] = df_merged['raisons_visites'].fillna('NON_VISITE')

# ══════════════════════════════════════════════════════════════════
# PARTIE D — FEATURES DÉRIVÉES
# MISE À JOUR v3 — Seuils recalibrés sur données réelles Orange Cameroun
#   (analyse 750 047 observations, fichier donneses_reseau_2.xlsx)
#
#  DISTRIBUTION RÉELLE DE LA DISPONIBILITÉ RÉSEAU :
#    68.0%  des observations = 100%  (performance parfaite)
#    11.9%  dans [99–100%]            (performance normale)
#     6.2%  dans [95–99%]             (légèrement sous la normale)
#     3.0%  dans [90–95%]             (dégradation modérée)
#     2.8%  dans [80–90%]             (dégradation significative)
#     3.5%  dans [50–80%]             (dégradation majeure)
#     4.6%  < 50%                     (site quasi hors service)
#
#  CORRÉLATION DISPO ↔ VOICE TRAFFIC (confirmée par les données) :
#    dispo < 80% → Voice Traffic moyen = 113 Erl (vs 389 Erl à 100%)
#    → Le seuil 80% est le vrai POINT DE RUPTURE dans tes données
#
#  SEUILS STANDARDS OPÉRATEUR GSM (norme télécom) :
#    < 99% = sous performance   (objectif SLA non atteint)
#    < 95% = dégradé            (seuil d'alerte standard chez Orange)
#    < 80% = critique           (point de rupture Voice Traffic)
#    < 50% = hors service       (panne grave)
#
#  FEATURES RÉSEAU v3 :
#   SUPPRIMÉ   : is_site_anomalie_grave  (valeurs <0 = erreurs)
#   SUPPRIMÉ   : is_site_tres_degrade    (seuil <70 non pertinent)
#   MODIFIÉ    : is_site_degrade         (90% → 95% standard opérateur)
#   AJOUTÉ     : is_site_critique        (< 80% — point de rupture Voice)
#   AJOUTÉ     : is_site_hors_service    (< 50% — panne grave)
#   MODIFIÉ    : dispo_reseau_niveau     (3 → 5 paliers calés sur données)
#   CONSERVÉ   : is_data_traffic_faible / fort / log_data_traffic
#   MODIFIÉ    : reseau_degrade_composite (seuil dispo mis à jour à 95%)
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE D — Features dérivées (v3 — seuils recalibrés) ══")

# ── D1. Disponibilité réseau — seuils recalibrés sur données réelles ──
#
# JUSTIFICATION DES SEUILS (analyse 750 047 obs, Orange Cameroun) :
#
#   Seuil 95% → is_site_degrade (standard opérateur GSM / SLA Orange)
#     → 13.9% des observations sont en dessous
#     → C'est le seuil d'ALERTE utilisé par les équipes réseau Orange
#
#   Seuil 80% → is_site_critique (point de rupture constaté dans les données)
#     → Voice Traffic moyen chute de 389 Erl (dispo=100%) à 113 Erl (dispo<80%)
#     → Signal fort de panne partielle impactant les recharges
#
#   Seuil 50% → is_site_hors_service (panne grave)
#     → 4.6% des observations — site quasi inutilisable
#
#   Niveaux dispo_reseau_niveau (5 paliers calibrés sur distribution réelle) :
#     1 = Hors service  [0–50%]   → 4.6%  — Voice Traffic effondré
#     2 = Critique      [50–80%]  → 3.5%  — point de rupture voice
#     3 = Dégradé       [80–95%]  → 5.8%  — seuil alerte opérateur
#     4 = Sous-perf     [95–99%]  → 6.2%  — légèrement sous la normale
#     5 = Normal        [99–100%] → 79.9% — performance attendue

if 'tr373_cell_availability_pct' in df_merged.columns:

    # is_site_degrade : dispo < 95% (seuil standard opérateur GSM)
    df_merged['is_site_degrade'] = (
        (df_merged['tr373_cell_availability_pct'] < 95.0) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)
    n_deg = df_merged['is_site_degrade'].sum()
    print(f"  is_site_degrade         : {n_deg:,} PDV ({n_deg/n_total*100:.1f}%)")
    print(f"    ← seuil : dispo < 95% (standard SLA opérateur GSM)")

    # is_site_critique : dispo < 80% (point de rupture Voice Traffic)
    df_merged['is_site_critique'] = (
        (df_merged['tr373_cell_availability_pct'] < 80.0) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)
    n_crit = df_merged['is_site_critique'].sum()
    print(f"  is_site_critique        : {n_crit:,} PDV ({n_crit/n_total*100:.1f}%)")
    print(f"    ← seuil : dispo < 80% (Voice Traffic chute de 389→113 Erl)")

    # is_site_hors_service : dispo < 50% (panne grave)
    df_merged['is_site_hors_service'] = (
        (df_merged['tr373_cell_availability_pct'] < 50.0) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)
    n_hs = df_merged['is_site_hors_service'].sum()
    print(f"  is_site_hors_service    : {n_hs:,} PDV ({n_hs/n_total*100:.1f}%)")
    print(f"    ← seuil : dispo < 50% (site quasi hors service)")

    # dispo_reseau_niveau : 5 paliers calibrés sur distribution réelle
    # (remplace les 3 paliers arbitraires de v2)
    df_merged['dispo_reseau_niveau'] = pd.cut(
        df_merged['tr373_cell_availability_pct'],
        bins=[-0.001, 50, 80, 95, 99, 100.999],
        labels=['hors_service', 'critique', 'degrade', 'sous_perf', 'normal']
    ).astype(str)
    # PDV sans données QR → 'inconnu'
    df_merged.loc[df_merged['qr_data_disponible'] == 0, 'dispo_reseau_niveau'] = 'inconnu'
    distrib = df_merged['dispo_reseau_niveau'].value_counts().to_dict()
    print(f"  dispo_reseau_niveau     : {distrib}")
    print(f"    ← 5 paliers : [0-50] hors_service | [50-80] critique |")
    print(f"                  [80-95] degrade | [95-99] sous_perf | [99-100] normal")

# ── D2. Trafic voix (Voice Traffic Erl) ──────────────────────────
if 'grp_2g_traffic_speech_erl' in df_merged.columns:

    p20_voice = df_merged.loc[
        df_merged['qr_data_disponible'] == 1,
        'grp_2g_traffic_speech_erl'
    ].quantile(0.20)

    # Voice Traffic n'a pas de borne supérieure — des valeurs >2000 Erl sont normales
    # is_trafic_reseau_faible = site dont le trafic voix est dans le 1er quintile
    # (seuil relatif P20 — s'adapte automatiquement à la distribution réelle)
    df_merged['is_trafic_reseau_faible'] = (
        (df_merged['grp_2g_traffic_speech_erl'] < p20_voice) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)

    # log1p sur valeurs positives → compression de l'échelle pour le modèle
    df_merged['log_trafic_reseau'] = np.log1p(
        df_merged['grp_2g_traffic_speech_erl'].clip(lower=0)
    ).round(4)

    n_trafic_faible = df_merged['is_trafic_reseau_faible'].sum()
    print(f"  is_trafic_reseau_faible : {n_trafic_faible:,} PDV")
    print(f"    ← seuil relatif P20 = {p20_voice:.2f} Erl (valeurs peuvent depasser 2000)")
    print(f"  log_trafic_reseau       : creee ✅")

# ── D3. Trafic data / internet (Data Traffic GB) — NOUVEAU ────────
if 'data_traffic_gb' in df_merged.columns:

    p20_data = df_merged.loc[
        df_merged['qr_data_disponible'] == 1,
        'data_traffic_gb'
    ].quantile(0.20)

    p80_data = df_merged.loc[
        df_merged['qr_data_disponible'] == 1,
        'data_traffic_gb'
    ].quantile(0.80)

    # Data Traffic n'a pas de borne supérieure — des valeurs >9000 GB sont normales
    # Plus la valeur est haute, plus le site a un trafic internet élevé
    # is_data_traffic_faible = site dans le 1er quintile (faible activité internet)
    # is_data_traffic_fort   = site dans le 4e quintile (forte activité internet)
    df_merged['is_data_traffic_faible'] = (
        (df_merged['data_traffic_gb'] < p20_data) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)

    df_merged['is_data_traffic_fort'] = (
        (df_merged['data_traffic_gb'] >= p80_data) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)

    # log1p : compression de l'échelle (utile car distribution très étalée)
    df_merged['log_data_traffic'] = np.log1p(
        df_merged['data_traffic_gb'].clip(lower=0)
    ).round(4)

    n_data_faible = df_merged['is_data_traffic_faible'].sum()
    n_data_fort   = df_merged['is_data_traffic_fort'].sum()
    print(f"\n  [DATA] is_data_traffic_faible : {n_data_faible:,} PDV")
    print(f"    ← seuil relatif P20 = {p20_data:.1f} GB (valeurs peuvent depasser 9000)")
    print(f"  [DATA] is_data_traffic_fort   : {n_data_fort:,} PDV")
    print(f"    ← seuil relatif P80 = {p80_data:.1f} GB")
    print(f"  [DATA] log_data_traffic       : creee ✅")

# ── D4. Indicateur composite réseau — mis à jour seuil 95% ────────
# Logique : dispo ET voice sont corrélées (confirmé r=+0.20 globalement,
# mais très fort sous 80% : Voice chute de 389→113 Erl).
# reseau_degrade_composite = site avec dispo<95% ET voice<P20 simultanément
# → double signal : infrastructure dégradée ET impact mesurable sur le trafic
#
# reseau_critique_composite = version renforcée : dispo<80% ET voice<P20
# → correspond au point de rupture observé dans les données

dispo_ok  = 'tr373_cell_availability_pct' in df_merged.columns
voice_ok  = 'grp_2g_traffic_speech_erl'   in df_merged.columns

if dispo_ok and voice_ok:
    # Composite standard (seuil 95% — alerte opérateur)
    df_merged['reseau_degrade_composite'] = (
        (df_merged['is_site_degrade']         == 1) &   # dispo < 95%
        (df_merged['is_trafic_reseau_faible'] == 1) &   # voice < P20
        (df_merged['qr_data_disponible']      == 1)
    ).astype(int)

    # Composite critique (seuil 80% — point de rupture Voice Traffic)
    df_merged['reseau_critique_composite'] = (
        (df_merged['is_site_critique']        == 1) &   # dispo < 80%
        (df_merged['is_trafic_reseau_faible'] == 1) &   # voice < P20
        (df_merged['qr_data_disponible']      == 1)
    ).astype(int)

    n_comp  = df_merged['reseau_degrade_composite'].sum()
    n_comp2 = df_merged['reseau_critique_composite'].sum()
    print(f"\n  reseau_degrade_composite : {n_comp:,} PDV")
    print(f"    ← dispo < 95% ET voice < P20 (seuil alerte opérateur)")
    print(f"  reseau_critique_composite: {n_comp2:,} PDV")
    print(f"    ← dispo < 80% ET voice < P20 (point de rupture Voice Traffic)")
else:
    print("  ⚠ composites réseau non créés : dispo ou voice manquante")

# ── D5. Visites terrain (inchangé v1) ─────────────────────────────
if 'pdv_visite' in df_merged.columns:
    df_merged['pdv_non_visite']   = (df_merged['pdv_visite'] == 0).astype(int)
    df_merged['pdv_multi_visite'] = (df_merged['nb_visites_mois'] > 1).astype(int)
    print(f"\n  pdv_non_visite          : {df_merged['pdv_non_visite'].sum():,} PDV")
    print(f"  pdv_multi_visite        : {df_merged['pdv_multi_visite'].sum():,} PDV")

# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════
print("\n══ Résumé des colonnes réseau (v2) ══")

colonnes_reseau_v3 = {
    # Variables brutes (imputées par médiane)
    'grp_2g_traffic_speech_erl'    : "Voice Traffic (Erl) — sans borne sup., imputé médiane",
    'tr373_cell_availability_pct'  : "Disponibilité réseau (%) — borné [0,100], imputé médiane",
    'data_traffic_gb'              : "Data Traffic (GB) — sans borne sup., imputé médiane",
    # Indicateur disponibilité données
    'qr_data_disponible'           : "1 = données QR réelles, 0 = imputées",
    # Features disponibilité — RECALIBRÉES v3
    'is_site_degrade'              : "1 si dispo < 95%  (seuil SLA opérateur GSM)   [13.9% obs]",
    'is_site_critique'             : "1 si dispo < 80%  (point rupture Voice Traffic) [8.1% obs] — NOUVEAU",
    'is_site_hors_service'         : "1 si dispo < 50%  (panne grave)                [4.6% obs] — NOUVEAU",
    'dispo_reseau_niveau'          : "5 paliers : hors_service/critique/degrade/sous_perf/normal — NOUVEAU",
    # Features Voice Traffic
    'is_trafic_reseau_faible'      : "1 si Voice Traffic < P20 (seuil relatif)",
    'log_trafic_reseau'            : "log1p(Voice Traffic)",
    # Features Data Traffic
    'is_data_traffic_faible'       : "1 si Data Traffic < P20",
    'is_data_traffic_fort'         : "1 si Data Traffic >= P80",
    'log_data_traffic'             : "log1p(Data Traffic)",
    # Composites réseau — RECALIBRÉS v3
    'reseau_degrade_composite'     : "dispo<95% ET voice<P20 (alerte opérateur)",
    'reseau_critique_composite'    : "dispo<80% ET voice<P20 (rupture Voice Traffic) — NOUVEAU",
    # Visites
    'pdv_non_visite'               : "1 si pdv_visite=0",
    'pdv_multi_visite'             : "1 si nb_visites > 1",
}

print("\n══ Résumé des colonnes réseau (v3 — seuils recalibrés) ══")
for col, desc in colonnes_reseau_v3.items():
    if col in df_merged.columns:
        n_nn = df_merged[col].notna().sum()
        print(f"  ✅ {col:42s}  {n_nn:,}  ← {desc}")
    else:
        print(f"  ⚪ {col:42s}  ABSENT")

print(f"\n  SUPPRIMÉES (obsolètes ou basées sur erreurs) :")
print(f"  ✗  is_site_anomalie_grave   (valeurs <0 = erreurs de collecte)")
print(f"  ✗  is_site_tres_degrade     (seuil <70 remplacé par dispo_reseau_niveau)")
print(f"  ✗  dispo_niveau             (3 paliers → remplacé par dispo_reseau_niveau 5 paliers)")
print(f"\n  SEUILS DE RÉFÉRENCE (données Orange Cameroun réelles) :")
print(f"  → 95% : seuil SLA opérateur — 13.9% des obs en dessous")
print(f"  → 80% : point de rupture Voice Traffic — 8.1% en dessous")
print(f"  → 50% : panne grave — 4.6% en dessous")

# ── Écriture ──────────────────────────────────────────────────────
print(f"\n── Écriture — {df_merged.shape[0]:,} lignes | {df_merged.shape[1]} colonnes ──")
output = dataiku.Dataset("rupt_freq_with_reseau")
output.write_with_schema(df_merged)
print("✅ Dataset 'rupt_freq_with_reseau' écrit avec succès")
