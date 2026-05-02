import numpy as np
from scipy import signal, stats
from typing import List, Tuple, Dict, Optional
from src.data_loader import parse_flexible_waveform_data


def parse_flexible_statistical_features(
    filepath: str,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Parse statistical features with flexible feature count.

    Args:
        filepath: Path to statistical features file (CSV or similar)
        feature_names: Optional list of feature names to extract
                      If None, extracts all numerical columns

    Returns:
        X_stat: Statistical features (N_samples, N_features)
        feature_names_out: List of feature names
        metadata: Dict with labels and other info
    """


    try:
        df = pd.read_csv(filepath, sep='\t')
    except:
        df = pd.read_csv(filepath)


    label_cols = ['fault_type', 'ft', 'Ft', 'degradation', 'deg', 'Deg', 'run', 'Run']


    y_ft = None
    y_deg = None
    runs = None

    for col in df.columns:
        if col.lower() in ['fault_type', 'ft']:
            y_ft = df[col].values
        elif col.lower() in ['degradation', 'deg']:
            y_deg = df[col].values
        elif col.lower() in ['run']:
            runs = df[col].values


    if feature_names is None:
        feature_cols = [col for col in df.columns
                       if col not in label_cols and df[col].dtype in [np.float64, np.float32, np.int64]]
        feature_names_out = feature_cols
    else:
        feature_cols = feature_names
        feature_names_out = feature_names

    X_stat = df[feature_cols].values
    num_features = X_stat.shape[1]

    print(f"Auto-detected {num_features} statistical features: {feature_names_out}")

    metadata = {
        'y_fault_type': y_ft,
        'y_degradation': y_deg,
        'runs': runs,
        'num_features': num_features,
        'feature_names': feature_names_out,
        'num_samples': len(X_stat)
    }

    print(f"Parsed {len(X_stat)} samples with {num_features} features")

    return X_stat, feature_names_out, metadata



def compute_advanced_statistical_features(
    X_wave: np.ndarray,
    channel_names: List[str],
    y_ft: np.ndarray = None,
    y_deg: np.ndarray = None,
    runs: np.ndarray = None,
    sampling_rate: float = 1000.0,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Compute comprehensive statistical features from waveform data.

    Features per channel:
    - Time-domain (TD): 10 features
      * RMS, Mean, Std, Var
      * Peak, PkPk (peak-to-peak), Min, Max
      * Energy, Power
      * Skewness, Kurtosis (3rd and 4th order cumulants)

    - Frequency-domain (FD): 7 features
      * Total Energy (sum of power spectrum)
      * Dominant Power (max peak in spectrum)
      * Dominant Frequency (frequency of max peak)
      * Spectral Centroid (center of mass of spectrum)
      * Spectral Entropy (randomness of spectrum)
      * Spectral Bandwidth (spread of spectrum)
      * Spectral Rolloff (frequency below which 85% of energy)

    Total: 17 features per channel

    For 3 channels: 3 × 17 = 51 features + 4 SPICE = 55 features

    Args:
        X_wave: Waveform data (N_samples, seq_len, N_channels)
        channel_names: List of channel names
        y_ft, y_deg, runs: Labels (optional)
        sampling_rate: Sampling rate in Hz (for frequency calculations)
        verbose: Print feature extraction progress

    Returns:
        X_stat: Statistical features (N_samples, N_features)
        feature_names: List of feature names
        metadata: Dict with feature info
    """

    n_samples, seq_len, n_channels = X_wave.shape

    if verbose:
        print(f"\\n{'='*70}")
        print(f"ADVANCED FEATURE EXTRACTION")
        print(f"{'='*70}")
        print(f"Input shape: {X_wave.shape}")
        print(f"Channels: {channel_names}")
        print(f"Extracting comprehensive features...")

    all_features = []
    all_feature_names = []


    for ch_idx, ch_name in enumerate(channel_names):
        ch_data = X_wave[:, :, ch_idx]

        if verbose:
            print(f"\\n  Processing {ch_name}...")

        # ================================================================
        # TIME-DOMAIN FEATURES
        # ================================================================

        # Basic statistics
        td_rms = np.sqrt(np.mean(ch_data**2, axis=1))
        td_mean = np.mean(ch_data, axis=1)
        td_std = np.std(ch_data, axis=1)
        td_var = np.var(ch_data, axis=1)

        # Peak features
        td_peak = np.max(np.abs(ch_data), axis=1)
        td_pkpk = np.ptp(ch_data, axis=1)
        td_min = np.min(ch_data, axis=1)
        td_max = np.max(ch_data, axis=1)

        # Energy and power
        td_energy = np.sum(ch_data**2, axis=1)
        td_power = td_energy / seq_len

        # Higher-order statistics (Cumulants)
        td_skewness = stats.skew(ch_data, axis=1)  # 3rd order cumulant
        td_kurtosis = stats.kurtosis(ch_data, axis=1)  # 4th order cumulant

        # Store TD features
        td_features = {
            f'{ch_name}_RMS': td_rms,
            f'{ch_name}_Mean': td_mean,
            f'{ch_name}_Std': td_std,
            f'{ch_name}_Var': td_var,
            f'{ch_name}_Peak': td_peak,
            f'{ch_name}_PkPk': td_pkpk,
            f'{ch_name}_Min': td_min,
            f'{ch_name}_Max': td_max,
            f'{ch_name}_Energy': td_energy,
            f'{ch_name}_Power': td_power,
            f'{ch_name}_TD_Cumulant3': td_skewness,
            f'{ch_name}_TD_Cumulant4': td_kurtosis,
        }

        # ================================================================
        # FREQUENCY-DOMAIN FEATURES
        # ================================================================

        fd_total_energy = []
        fd_dominant_power = []
        fd_dominant_freq = []
        fd_spectral_centroid = []
        fd_spectral_entropy = []
        fd_spectral_bandwidth = []
        fd_spectral_rolloff = []

        for sample_idx in range(n_samples):
            sample = ch_data[sample_idx]

            # Compute FFT
            fft_vals = np.fft.rfft(sample)
            power_spectrum = np.abs(fft_vals)**2
            freqs = np.fft.rfftfreq(seq_len, d=1.0/sampling_rate)

            # Normalize power spectrum
            power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)

            # Total energy
            total_energy = np.sum(power_spectrum)
            fd_total_energy.append(total_energy)

            # Dominant power and frequency
            dominant_idx = np.argmax(power_spectrum)
            dominant_power = power_spectrum[dominant_idx]
            dominant_freq = freqs[dominant_idx]
            fd_dominant_power.append(dominant_power)
            fd_dominant_freq.append(dominant_freq)

            # Spectral centroid (center of mass)
            spectral_centroid = np.sum(freqs * power_spectrum_norm)
            fd_spectral_centroid.append(spectral_centroid)

            # Spectral entropy
            spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-10))
            fd_spectral_entropy.append(spectral_entropy)

            # Spectral bandwidth (weighted std of frequencies)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * power_spectrum_norm))
            fd_spectral_bandwidth.append(spectral_bandwidth)

            # Spectral rolloff (frequency below which 85% of energy)
            cumsum_power = np.cumsum(power_spectrum_norm)
            rolloff_idx = np.where(cumsum_power >= 0.85)[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs[rolloff_idx[0]]
            else:
                spectral_rolloff = freqs[-1]
            fd_spectral_rolloff.append(spectral_rolloff)

        # Store FD features
        fd_features = {
            f'{ch_name}_FD_TotalEnergy': np.array(fd_total_energy),
            f'{ch_name}_FD_DominantPower': np.array(fd_dominant_power),
            f'{ch_name}_FD_DominantFreq': np.array(fd_dominant_freq),
            f'{ch_name}_FD_SpectralCentroid': np.array(fd_spectral_centroid),
            f'{ch_name}_FD_SpectralEntropy': np.array(fd_spectral_entropy),
            f'{ch_name}_FD_SpectralBandwidth': np.array(fd_spectral_bandwidth),
            f'{ch_name}_FD_SpectralRolloff': np.array(fd_spectral_rolloff),
        }

        # Combine TD and FD features for this channel
        channel_features = {**td_features, **fd_features}

        for feat_name, feat_values in channel_features.items():
            all_features.append(feat_values)
            all_feature_names.append(feat_name)

        if verbose:
            print(f"  Extracted {len(channel_features)} features")

    # ================================================================
    # SPICE MEASUREMENTS (from output channel if available)
    # ================================================================


    output_channel_idx = -1
    output_ch_name = "SPICE"

    output_data = X_wave[:, :, output_channel_idx]

    spice_rms = np.sqrt(np.mean(output_data**2, axis=1))
    spice_peak = np.max(np.abs(output_data), axis=1)
    spice_avg = np.mean(output_data, axis=1)
    spice_ripple = np.ptp(output_data, axis=1)

    spice_features = {
        f'{output_ch_name}_RMS': spice_rms,
        f'{output_ch_name}_Peak': spice_peak,
        f'{output_ch_name}_Avg': spice_avg,
        f'{output_ch_name}_Ripple': spice_ripple,
    }

    for feat_name, feat_values in spice_features.items():
        all_features.append(feat_values)
        all_feature_names.append(feat_name)

    # ================================================================
    # COMBINE ALL FEATURES
    # ================================================================

    X_stat = np.column_stack(all_features)

    if verbose:
        print(f"\\n{'='*70}")
        print(f"FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total features extracted: {len(all_feature_names)}")
        print(f"  - Time-domain: {12 * n_channels} ({n_channels} channels × 12 features)")
        print(f"  - Frequency-domain: {7 * n_channels} ({n_channels} channels × 7 features)")
        print(f"  - SPICE measurements: 4")
        print(f"\\nOutput shape: {X_stat.shape}")
        print(f"\\nFeature breakdown by channel:")

        current_idx = 0
        for ch_idx, ch_name in enumerate(channel_names):
            td_count = 12
            fd_count = 7
            total_ch = td_count + fd_count

            print(f"\\n  {ch_name}: {total_ch} features")
            print(f"    Time-domain (12):")
            for i in range(current_idx, current_idx + td_count):
                print(f"      {i+1:2d}. {all_feature_names[i]}")

            print(f"    Frequency-domain (7):")
            for i in range(current_idx + td_count, current_idx + total_ch):
                print(f"      {i+1:2d}. {all_feature_names[i]}")

            current_idx += total_ch

        print(f"\\n  SPICE measurements: 4 features")
        for i in range(current_idx, len(all_feature_names)):
            print(f"      {i+1:2d}. {all_feature_names[i]}")

        print(f"\\n{'='*70}\\n")

    metadata = {
        'y_fault_type': y_ft,
        'y_degradation': y_deg,
        'runs': runs,
        'num_features': len(all_feature_names),
        'feature_names': all_feature_names,
        'num_samples': n_samples,
        'num_channels': n_channels,
        'channel_names': channel_names,
        'feature_categories': {
            'time_domain': [f for f in all_feature_names if '_FD_' not in f and 'SPICE' not in f],
            'frequency_domain': [f for f in all_feature_names if '_FD_' in f],
            'spice_measurements': [f for f in all_feature_names if 'SPICE' in f]
        }
    }

    return X_stat, all_feature_names, metadata


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED FEATURE EXTRACTION - Test Mode")
    print("="*80)


    n_samples = 10
    seq_len = 200
    n_channels = 3
    X_wave = np.random.randn(n_samples, seq_len, n_channels)
    channel_names = ['V(in)', 'V(n1)', 'V(n2)']

    X_stat, features, meta = compute_advanced_statistical_features(
        X_wave,
        channel_names,
        verbose=True
    )

    print(f"\\nTest successful!")
    print(f"Input shape: {X_wave.shape}")
    print(f"Output shape: {X_stat.shape}")
    print(f"Number of features: {len(features)}")

    X_wave, channels, wave_meta = parse_flexible_waveform_data(
        '/content/drive/MyDrive/LTSPICE(SR)/PRO/TEST(PROLOG).txt',
        max_len=200
    )

    print(f"\nReady for training with:")
    print(f"  - Waveform channels: {wave_meta['num_channels']}")
    print(f"  - Sequence length: {wave_meta['sequence_length']}")
    #print(f"  - Statistical features: {stat_meta['num_features']}")
