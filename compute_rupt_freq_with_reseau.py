# ══════════════════════════════════════════════════════════════════
# RECIPE : Jointure Qualité Réseau + Visites Terrain → rupt_freq
# Fix : KeyError msisdn_v → named aggregation au lieu de dict multi-niveaux
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

# ── 2. Normalisation colonnes ─────────────────────────────────────
print("\n── Normalisation colonnes ──")

df.columns = [c.lower().strip() for c in df.columns]

df_qr.columns = (
    df_qr.columns
    .str.lower().str.strip()
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

print(f"  Colonnes rupt_freq     : {df.columns.tolist()}")
print(f"  Colonnes qualite_reseau: {df_qr.columns.tolist()}")
print(f"  Colonnes BASE_VISITE   : {df_visit.columns.tolist()}")
print(f"\n  Aperçu BASE_VISITE_OK :")
print(df_visit.head(3).to_string())

# ══════════════════════════════════════════════════════════════════
# PARTIE A — QUALITÉ RÉSEAU
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE A — Qualité Réseau ══")

# ── 3A. Suppression colonnes dupliquées ───────────────────────────
cols_dup = df_qr.columns[df_qr.columns.duplicated()].tolist()
print(f"  Colonnes dupliquées QR : {cols_dup}")
df_qr = df_qr.loc[:, ~df_qr.columns.duplicated(keep='first')]

# ── 4A. Renommage colonnes QR ─────────────────────────────────────
rename_qr = {}
for col in df_qr.columns:
    if col == 'date':
        rename_qr[col] = 'date_qr'
    elif 'site' in col and 'name' in col:
        rename_qr[col] = 'site_name_qr'
    elif 'traffic' in col and 'speech' in col:
        rename_qr[col] = 'grp_2g_traffic_speech_erl'
    elif 'availability' in col or 'tr373' in col:
        rename_qr[col] = 'tr373_cell_availability_pct'

df_qr = df_qr.rename(columns=rename_qr)
print(f"  Colonnes QR après renommage : {df_qr.columns.tolist()}")

# ── 5A. Normalisation ─────────────────────────────────────────────
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
print(f"  Mois couverts QR : {sorted(mois_qr_set)}")
print(f"  Mois sans QR     : {sorted(mois_rupt - mois_qr_set)}")

# ── 7A. Métriques QR ──────────────────────────────────────────────
metric_cols_qr = [c for c in ['grp_2g_traffic_speech_erl',
                               'tr373_cell_availability_pct']
                  if c in df_qr.columns]
print(f"  Métriques QR : {metric_cols_qr}")

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
# FIX : named aggregation au lieu de dict multi-niveaux
#       → évite le KeyError: msisdn_v
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE B — Visites Terrain (BASE_VISITE_OK) ══")

# ── 3B. Renommage colonnes visites ────────────────────────────────
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

# ── 4B. Construction mois_v ───────────────────────────────────────
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
        + '-'
        + df_visit['mois_num'].fillna('01')
    )
    print(f"  mois_v depuis nom+année : {df_visit['mois_v'].dropna().unique()[:5].tolist()}")
else:
    raise ValueError(
        f"❌ Impossible de construire mois_v — colonnes : {df_visit.columns.tolist()}"
    )

# ── 5B. Normalisation MSISDN ──────────────────────────────────────
if 'msisdn_v' not in df_visit.columns:
    # Chercher colonne MSISDN si renommage n'a pas fonctionné
    msisdn_candidates = [c for c in df_visit.columns if 'msisdn' in c.lower()]
    if msisdn_candidates:
        df_visit = df_visit.rename(columns={msisdn_candidates[0]: 'msisdn_v'})
    else:
        raise ValueError(
            f"❌ Colonne MSISDN introuvable — colonnes : {df_visit.columns.tolist()}"
        )

df_visit['msisdn_v'] = df_visit['msisdn_v'].astype(str).str.strip()

print(f"\n  MSISDN visites (5 ex) : {df_visit['msisdn_v'].dropna().unique()[:5].tolist()}")
print(f"  Mois visites          : {sorted(df_visit['mois_v'].dropna().unique().tolist())}")

# ── 6B. Colonne visite → binaire ──────────────────────────────────
print("\n── [VISITES] Normalisation colonne visite ──")

visite_sample = df_visit['visite'].dropna().unique()[:5]
print(f"  Valeurs visite (5 ex) : {visite_sample.tolist()}")

visite_num = pd.to_numeric(df_visit['visite'], errors='coerce')
if visite_num.isna().sum() == 0:
    df_visit['visite_bin'] = visite_num.astype(int)
else:
    df_visit['visite_bin'] = df_visit['visite'].astype(str).str.lower().apply(
        lambda x: 0 if any(mot in x for mot in ['pas', 'non', 'not', '0']) else 1
    )

print(f"  Distribution visite_bin : {df_visit['visite_bin'].value_counts().to_dict()}")

# ── 7B. Agrégation visites ────────────────────────────────────────
# FIX CRITIQUE : named aggregation → pas de colonnes multi-niveaux
# → msisdn_v et mois_v restent comme colonnes normales après reset_index
print("\n── [VISITES] Agrégation (named aggregation) ──")

# ÉTAPE 1 : nb_visites + pdv_visite avec named aggregation
df_visit_agg = (
    df_visit
    .groupby(['msisdn_v', 'mois_v'], as_index=False)
    .agg(
        nb_visites_mois = ('visite_bin', 'sum'),
        pdv_visite      = ('visite_bin', 'max'),
    )
)

# Vérification immédiate
assert 'msisdn_v' in df_visit_agg.columns, \
    f"❌ msisdn_v absent après agg ! Colonnes : {df_visit_agg.columns.tolist()}"
assert 'mois_v' in df_visit_agg.columns, \
    f"❌ mois_v absent après agg ! Colonnes : {df_visit_agg.columns.tolist()}"

print(f"  ✅ msisdn_v présent après agg")
print(f"  ✅ mois_v présent après agg")

# ÉTAPE 2 : raisons_visites (séparément pour éviter les multi-niveaux)
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
        on  = ['msisdn_v', 'mois_v'],
        how = 'left'
    )

print(f"  Colonnes agrégées : {df_visit_agg.columns.tolist()}")
print(f"  Lignes            : {df_visit_agg.shape[0]:,}")

# ── 8B. Diagnostic ────────────────────────────────────────────────
msisdn_rupt  = set(df['msisdn'].dropna().unique())
msisdn_v_set = set(df_visit_agg['msisdn_v'].dropna().unique())
match_v      = msisdn_rupt & msisdn_v_set

print(f"\n  MSISDN rupt_freq  : {len(msisdn_rupt):,}")
print(f"  MSISDN visites    : {len(msisdn_v_set):,}")
print(f"  MSISDN matchés    : {len(match_v):,}  ({len(match_v)/max(len(msisdn_rupt),1)*100:.1f}%)")
print(f"  Mois visites      : {sorted(df_visit_agg['mois_v'].dropna().unique().tolist())}")
print(f"  Mois en commun    : {sorted(set(df['mois'].unique()) & set(df_visit_agg['mois_v'].dropna().unique()))}")

# ── 9B. Jointure LEFT visites ─────────────────────────────────────
print("\n── [VISITES] Jointure LEFT ──")

cols_v = ['msisdn_v', 'mois_v', 'nb_visites_mois', 'pdv_visite']
if 'raisons_visites' in df_visit_agg.columns:
    cols_v.append('raisons_visites')

df_merged = pd.merge(
    df,
    df_visit_agg[cols_v],
    left_on  = ['msisdn', 'mois'],
    right_on = ['msisdn_v', 'mois_v'],
    how      = 'left'
)
df_merged = df_merged.drop(columns=['msisdn_v', 'mois_v'], errors='ignore')
print(f"  Lignes après jointure : {df_merged.shape[0]:,}")

# ══════════════════════════════════════════════════════════════════
# PARTIE C — IMPUTATION MÉTHODE ROSSMANN
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE C — Imputation (méthode Rossmann) ══")

n_total = len(df_merged)

# ── C1. Qualité réseau ─────────────────────────────────────────────
# Indicateur AVANT imputation
df_merged['qr_data_disponible'] = (
    df_merged[metric_cols_qr[0]].notna()
).astype(int) if metric_cols_qr else 0

n_qr = df_merged['qr_data_disponible'].sum()
print(f"\n  [QR] données réelles  : {n_qr:,} ({n_qr/n_total*100:.1f}%)")
print(f"  [QR] données imputées : {n_total-n_qr:,} ({(n_total-n_qr)/n_total*100:.1f}%)")

for col in metric_cols_qr:
    med = df_merged.loc[df_merged['qr_data_disponible'] == 1, col].median()
    df_merged[col] = df_merged[col].fillna(med)
    print(f"  [QR] {col}")
    print(f"       médiane réelle = {med:.4f} | NaN résiduels = {df_merged[col].isna().sum()}")

# ── C2. Visites terrain ────────────────────────────────────────────
# Règle Rossmann : NaN = information réelle
# Absence de visite = PDV non visité = 0
# Comme competition_distance=200000 = "pas de concurrent connu"
if 'pdv_visite' in df_merged.columns:
    # Indicateur AVANT imputation
    df_merged['visite_data_disponible'] = (
        df_merged['pdv_visite'].notna()
    ).astype(int)

    n_v = df_merged['visite_data_disponible'].sum()
    print(f"\n  [VISIT] données réelles  : {n_v:,} ({n_v/n_total*100:.1f}%)")
    print(f"  [VISIT] données imputées : {n_total-n_v:,} ({(n_total-n_v)/n_total*100:.1f}%)")

    # NaN = PDV non visité = 0 (valeur réelle, pas estimation)
    df_merged['nb_visites_mois'] = df_merged['nb_visites_mois'].fillna(0).astype(int)
    df_merged['pdv_visite']      = df_merged['pdv_visite'].fillna(0).astype(int)
    if 'raisons_visites' in df_merged.columns:
        df_merged['raisons_visites'] = df_merged['raisons_visites'].fillna('NON_VISITE')

    print(f"  [VISIT] nb_visites_mois → NaN = {df_merged['nb_visites_mois'].isna().sum()}")
    print(f"  [VISIT] pdv_visite      → NaN = {df_merged['pdv_visite'].isna().sum()}")

# ══════════════════════════════════════════════════════════════════
# PARTIE D — FEATURES DÉRIVÉES
# ══════════════════════════════════════════════════════════════════
print("\n══ PARTIE D — Features dérivées ══")

# Qualité réseau (sur données réelles uniquement)
if 'tr373_cell_availability_pct' in df_merged.columns:
    df_merged['is_site_degrade'] = (
        (df_merged['tr373_cell_availability_pct'] < 90.0) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)
    print(f"  is_site_degrade         : {df_merged['is_site_degrade'].sum():,} PDV")

if 'grp_2g_traffic_speech_erl' in df_merged.columns:
    p20 = df_merged.loc[
        df_merged['qr_data_disponible'] == 1,
        'grp_2g_traffic_speech_erl'
    ].quantile(0.20)
    df_merged['is_trafic_reseau_faible'] = (
        (df_merged['grp_2g_traffic_speech_erl'] < p20) &
        (df_merged['qr_data_disponible'] == 1)
    ).astype(int)
    df_merged['log_trafic_reseau'] = np.log1p(
        df_merged['grp_2g_traffic_speech_erl'].clip(lower=0)
    ).round(4)
    print(f"  is_trafic_reseau_faible : {df_merged['is_trafic_reseau_faible'].sum():,} PDV")
    print(f"  log_trafic_reseau       : créée ✅")

# Visites terrain
if 'pdv_visite' in df_merged.columns:
    df_merged['pdv_non_visite']   = (df_merged['pdv_visite'] == 0).astype(int)
    df_merged['pdv_multi_visite'] = (df_merged['nb_visites_mois'] > 1).astype(int)
    print(f"  pdv_non_visite          : {df_merged['pdv_non_visite'].sum():,} PDV")
    print(f"  pdv_multi_visite        : {df_merged['pdv_multi_visite'].sum():,} PDV")

# ── Résumé ────────────────────────────────────────────────────────
print("\n── Résumé des colonnes ajoutées ──")
nouvelles = [
    'grp_2g_traffic_speech_erl',
    'tr373_cell_availability_pct',
    'qr_data_disponible',
    'is_site_degrade',
    'is_trafic_reseau_faible',
    'log_trafic_reseau',
    'nb_visites_mois',
    'pdv_visite',
    'visite_data_disponible',
    'pdv_non_visite',
    'pdv_multi_visite',
    'raisons_visites',
]
for col in nouvelles:
    if col in df_merged.columns:
        n_nn = df_merged[col].notna().sum()
        print(f"  ✅ {col:42s} ({n_nn:,} non nuls)")
    else:
        print(f"  ⚪ {col} absente")

# ── Écriture ──────────────────────────────────────────────────────
print(f"\n── Écriture — {df_merged.shape[0]:,} lignes | {df_merged.shape[1]} colonnes ──")
output = dataiku.Dataset("rupt_freq_with_reseau")
output.write_with_schema(df_merged)
print("✅ Dataset 'rupt_freq_with_reseau' écrit avec succès")
