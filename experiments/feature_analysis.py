import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_oneway
import pandas as pd
import warnings
from src.features import compute_advanced_statistical_features
from src.data_loader import parse_flexible_waveform_data


circuit_files = {
    'RECNEW': 'data/RECNEW.txt',
    'DPKNEW1': 'data/DPKNEW.txt',
    'DCNEW': 'data/DCNEW.txt'
}

# ============================================================================
# PARSE ALL CIRCUITS
# ============================================================================

print("="*80)
print("STEP 1: PARSING WAVEFORMS FOR ALL CIRCUITS")
print("="*80)

circuits_waveforms = {}

for circuit_name, filepath in circuit_files.items():
    try:
        print(f"\nParsing: {circuit_name}...")
        X_wave, channels, wave_meta = parse_flexible_waveform_data(
            filepath, max_len=200, verbose=False
        )
        circuits_waveforms[circuit_name] = (X_wave, channels, wave_meta)
        print(f"  Success: {X_wave.shape[0]} samples, {len(channels)} channels")
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")

print(f"\n✓ Parsed {len(circuits_waveforms)} circuits successfully")

# ============================================================================
# EXTRACT FEATURES FOR ALL CIRCUITS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: EXTRACTING FEATURES FOR ALL CIRCUITS")
print("="*80)

circuits_features = {}

for circuit_name, (X_wave, channels, wave_meta) in circuits_waveforms.items():
    print(f"\nExtracting features: {circuit_name}...")

    X_stat, feature_names, stat_meta = compute_advanced_statistical_features(
        X_wave, channels,
        y_ft=wave_meta['y_fault_type'],
        y_deg=wave_meta['y_degradation'],
        runs=wave_meta['runs'],
        verbose=False
    )

    circuits_features[circuit_name] = (X_stat, feature_names, stat_meta)
    print(f"  Extracted {stat_meta['num_features']} features from {X_stat.shape[0]} samples")

print(f"\n Extracted features for {len(circuits_features)} circuits")

# ============================================================================
# ANALYZE FEATURE IMPORTANCE FOR EACH CIRCUIT
# ============================================================================

print("\n" + "="*80)
print("STEP 3: ANALYZING FEATURE IMPORTANCE")
print("="*80)

def analyze_circuit_robust(X_stat, feature_names, y_fault, y_deg, name, top_k=20):
    """
    ROBUST feature importance analysis with NaN handling.

    Fixes:
    - Removes constant features before analysis
    - Handles NaN/Inf in scores
    - Validates numerical stability
    - Provides detailed diagnostics
    """

    print(f"\n{'='*80}")
    print(f"CIRCUIT: {name}")
    print(f"{'='*80}")
    print(f"Samples: {X_stat.shape[0]} | Features: {X_stat.shape[1]}")

    # ========================================================================
    # DATA VALIDATION AND CLEANING
    # ========================================================================
    print("\nData validation...")


    nan_mask = np.isnan(X_stat).any(axis=0) | np.isinf(X_stat).any(axis=0)
    if nan_mask.any():
        print(f"  Warning: {nan_mask.sum()} features contain NaN/Inf - removing")
        valid_features = ~nan_mask
        X_stat = X_stat[:, valid_features]
        feature_names = [f for i, f in enumerate(feature_names) if valid_features[i]]


    variances = np.var(X_stat, axis=0)
    constant_mask = variances < 1e-10

    if constant_mask.any():
        print(f"   Warning: {constant_mask.sum()} constant features detected - removing")
        print(f"   Constant features: {[feature_names[i] for i, c in enumerate(constant_mask) if c][:5]}...")

        valid_features = ~constant_mask
        X_stat = X_stat[:, valid_features]
        feature_names = [f for i, f in enumerate(feature_names) if valid_features[i]]

    print(f"  Valid features: {len(feature_names)}")

    if len(feature_names) == 0:
        print("  ERROR: No valid features remaining!")
        return None


    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        X_scaled = scaler.fit_transform(X_stat)


    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # ========================================================================
    # RANDOM FOREST 
    # ========================================================================
    print("\n Random Forest...")

    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf.fit(X_scaled, y_fault)
        rf_scores = rf.feature_importances_

        rf_scores = np.nan_to_num(rf_scores, nan=0.0)
        print(f"  RF scores: min={rf_scores.min():.6f}, max={rf_scores.max():.6f}")
    except Exception as e:
        print(f"  RF failed: {str(e)[:100]}")
        rf_scores = np.zeros(len(feature_names))

    # ========================================================================
    # MUTUAL INFORMATION (30% weight)
    # ========================================================================
    print(" Mutual Information...")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mi_scores = mutual_info_classif(
                X_scaled, y_fault,
                random_state=42,
                n_neighbors=min(5, len(X_scaled) // 10)
            )

        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        print(f"  MI scores: min={mi_scores.min():.6f}, max={mi_scores.max():.6f}")
    except Exception as e:
        print(f"  MI failed: {str(e)[:100]}")
        mi_scores = np.zeros(len(feature_names))

    # ========================================================================
    # ANOVA F-STATISTIC 
    # ========================================================================
    print(" ANOVA F-test...")

    f_scores = []
    for i in range(X_scaled.shape[1]):
        try:
            groups = [X_scaled[y_fault == c, i] for c in np.unique(y_fault)]

            if any(np.std(g) < 1e-10 for g in groups):
                f_scores.append(0.0)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    f_stat, _ = f_oneway(*groups)
                    f_scores.append(f_stat if not np.isnan(f_stat) else 0.0)
        except:
            f_scores.append(0.0)

    f_scores = np.array(f_scores)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0)
    print(f"  F-scores: min={f_scores.min():.2f}, max={f_scores.max():.2f}")

    # ========================================================================
    # CORRELATION 
    # ========================================================================
    print(" Correlation...")

    corr_scores = []
    for i in range(X_scaled.shape[1]):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                corr_matrix = np.corrcoef(X_scaled[:, i], y_deg)
                corr = np.abs(corr_matrix[0, 1])
                corr_scores.append(corr if not np.isnan(corr) else 0.0)
        except:
            corr_scores.append(0.0)

    corr_scores = np.array(corr_scores)
    corr_scores = np.nan_to_num(corr_scores, nan=0.0)
    print(f"   Corr scores: min={corr_scores.min():.4f}, max={corr_scores.max():.4f}")

    # ========================================================================
    # CONSENSUS RANKING
    # ========================================================================
    print("\n Computing consensus...")


    def safe_normalize(scores):
        max_val = scores.max()
        if max_val < 1e-10:
            return scores
        return scores / max_val

    rf_norm = safe_normalize(rf_scores)
    mi_norm = safe_normalize(mi_scores)
    f_norm = safe_normalize(f_scores)
    corr_norm = safe_normalize(corr_scores)


    consensus = 0.4 * rf_norm + 0.3 * mi_norm + 0.2 * f_norm + 0.1 * corr_norm


    consensus = np.nan_to_num(consensus, nan=0.0)

    if consensus.max() < 1e-10:
        print("All consensus scores near zero - using RF scores only")
        consensus = rf_scores

    top_idx = np.argsort(consensus)[::-1][:top_k]

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"TOP {top_k} FEATURES")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Feature':<45} {'Score':<10} {'Type'}")
    print("-" * 80)

    for rank, idx in enumerate(top_idx, 1):
        feat = feature_names[idx]
        ftype = 'TD' if '_FD_' not in feat and 'SPICE' not in feat else \
               'FD' if '_FD_' in feat else 'SPICE'
        score = consensus[idx]
        print(f"{rank:<6} {feat:<45} {score:.6f}   {ftype}")

    # Category breakdown
    top_feats = [feature_names[i] for i in top_idx]
    td = sum(1 for f in top_feats if '_FD_' not in f and 'SPICE' not in f)
    fd = sum(1 for f in top_feats if '_FD_' in f)
    sp = sum(1 for f in top_feats if 'SPICE' in f)

    print(f"\n{'='*80}")
    print(f"Category Distribution:")
    print(f"  Time-domain: {td} ({td/top_k*100:.1f}%)")
    print(f"  Frequency-domain: {fd} ({fd/top_k*100:.1f}%)")
    print(f"  SPICE: {sp} ({sp/top_k*100:.1f}%)")

    # Diagnostic info
    print(f"\nDiagnostic Info:")
    print(f"  RF contribution: {np.mean(rf_norm):.4f}")
    print(f"  MI contribution: {np.mean(mi_norm):.4f}")
    print(f"  F-test contribution: {np.mean(f_norm):.4f}")
    print(f"  Correlation contribution: {np.mean(corr_norm):.4f}")
    print(f"{'='*80}\n")

    return {
        'top_features': top_feats,
        'top_scores': consensus[top_idx],
        'all_scores': consensus,
        'rf_scores': rf_scores,
        'mi_scores': mi_scores,
        'f_scores': f_scores,
        'corr_scores': corr_scores,
        'feature_names': feature_names
    }


# Analyze each circuit
all_results = {}
for circuit_name, (X_stat, feature_names, stat_meta) in circuits_features.items():
    results = analyze_circuit_robust(
        X_stat, feature_names,
        stat_meta['y_fault_type'],
        stat_meta['y_degradation'],
        circuit_name,
        top_k=20
    )
    all_results[circuit_name] = results

# ============================================================================
# COMPARE ACROSS CIRCUITS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CROSS-CIRCUIT COMPARISON")
print("="*80)

# Collect all top features
all_top = set()
for r in all_results.values():
    all_top.update(r['top_features'])

# Create comparison table
comparison_data = []
for feat in all_top:
    row = {'Feature': feat}
    ranks = []
    for circ_name, r in all_results.items():
        if feat in r['top_features']:
            rank = r['top_features'].index(feat) + 1
            score = r['top_scores'][rank - 1]
            row[f'{circ_name}_Rank'] = rank
            row[f'{circ_name}_Score'] = f"{score:.4f}"
            ranks.append(rank)
        else:
            row[f'{circ_name}_Rank'] = '-'
            row[f'{circ_name}_Score'] = '-'

    row['Avg_Rank'] = f"{np.mean(ranks):.1f}" if ranks else '-'
    row['Circuits'] = len(ranks)
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Avg_Rank')

print(f"\nFeatures appearing in multiple circuits:")
print(df_comparison.head(25).to_string(index=False))

# ============================================================================
# SAVE REPORTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING REPORTS")
print("="*80)
for circuit_name, results in all_results.items():
    filename = f'features_{circuit_name.lower()}.csv'
    df_circuit = pd.DataFrame({
        'Rank': range(1, len(results['top_features'])+1),
        'Feature': results['top_features'],
        'Score': results['top_scores']
    })
    df_circuit.to_csv(filename, index=False)
    print(f"   {filename}")

df_comparison.to_csv('features_comparison_all_circuits.csv', index=False)
print(f"   features_comparison_all_circuits.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAnalyzed {len(circuits_features)} circuits")
print(f"Generated {len(circuits_features) + 1} CSV reports")
print("="*80)