from src.model import FlexibleEORN_LateFusionModel
from src.features import compute_advanced_statistical_features
from src.data_loader import parse_flexible_waveform_data, FlexibleOrdinalDataset
from src.loss import hybrid_evidential_ordinal_loss
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, mean_absolute_error
from datetime import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load the file
cfg = load_config()

# Accessing File Paths
circuit_files = cfg['data']['circuits']

# Accessing Training Config
# Note: We can dynamically check for CUDA here if desired
device = "cuda" if torch.cuda.is_available() and cfg['training']['device'] == "cuda" else "cpu"

training_config = {
    'epochs': cfg['training']['epochs'],
    'batch_size': cfg['training']['batch_size'],
    'lr': cfg['training']['lr'],
    'patience': cfg['training']['patience'],
    'k': cfg['training']['k_folds'],
    'save_dir': cfg['data']['save_dir'],
    'device': device
}

# Ensure output directory exists
os.makedirs(training_config['save_dir'], exist_ok=True)

# ============================================================================
# STEP 1: PARSE ALL CIRCUITS
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
# STEP 2: EXTRACT FEATURES FOR ALL CIRCUITS
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



def create_model_from_circuit_data(X_wave, X_stat, num_ft_classes=6,
                                   num_deg_levels=6, device='cpu'):
    n_samples, seq_len, wave_channels = X_wave.shape
    n_features = X_stat.shape[1]

    model = FlexibleEORN_LateFusionModel(
        wave_channels=wave_channels,
        stat_dim=n_features,
        num_ft_classes=num_ft_classes,
        num_deg_levels=num_deg_levels,
        seq_len=seq_len,
        dropout_rate=0.3
    ).to(device)

    return model

def predict_with_uncertainty(alpha, deg_levels=[0, 20, 40, 60, 80, 100]):
    S = alpha.sum(1, keepdim=True)
    prob = alpha / S
    pred_class = prob.argmax(1)

    deg_levels_tensor = torch.tensor(deg_levels, device=alpha.device, dtype=torch.float32)
    pred_value = (prob * deg_levels_tensor).sum(1)

    return {
        'pred_class': pred_class,
        'pred_value': pred_value,
        'prob': prob,
        'total_evidence': S.squeeze()
    }

def train_flexible_eorn(circuit_name, X_wave, X_stat, y_ft, y_deg_norm,
                       k=5, epochs=30, batch_size=16, lr=3e-4, patience=10,
                       save_dir='/content/drive/MyDrive/LTSPICE(SR)/',
                       device='cpu'):
    '''
    Enhanced training with FAULT ACCURACY and FAULT F1 tracking
    '''

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    print(f"\n{'='*80}")
    print(f"TRAINING: {circuit_name}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Waveforms: {X_wave.shape}, Features: {X_stat.shape}")
    print(f"{'='*80}\n")

    fold_results = []
    best_overall_mae = float('inf')
    best_overall_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_wave)):
        print(f"\nFOLD {fold+1}/{k}")
        print("-"*80)

        # Data split
        X_wave_train, X_wave_val = X_wave[train_idx], X_wave[val_idx]
        X_stat_train, X_stat_val = X_stat[train_idx], X_stat[val_idx]
        y_ft_train, y_ft_val = y_ft[train_idx], y_ft[val_idx]
        y_deg_train, y_deg_val = y_deg_norm[train_idx], y_deg_norm[val_idx]

        # Datasets
        train_ds = FlexibleOrdinalDataset(
            X_wave_train, X_stat_train, y_ft_train, y_deg_train, augment=True
        )
        val_ds = FlexibleOrdinalDataset(
            X_wave_val, X_stat_val, y_ft_val, y_deg_val,
            wave_mean=train_ds.wave_mean, wave_std=train_ds.wave_std,
            stat_mean=train_ds.stat_mean, stat_std=train_ds.stat_std
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model
        model = create_model_from_circuit_data(
            X_wave, X_stat, len(np.unique(y_ft)), 6, device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        ce_loss = nn.CrossEntropyLoss()

        best_val_mae = float('inf')
        best_val_ft_acc = 0.0
        best_val_ft_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            annealing_coef = min(1.0, epoch / (epochs * 0.3))

            # ===== TRAINING =====
            model.train()
            train_loss = 0
            train_total = 0
            train_ft_correct = 0

            for x_wave, x_stat, y_fb, y_db_class in train_loader:
                x_wave, x_stat = x_wave.to(device), x_stat.to(device)
                y_fb, y_db_class = y_fb.to(device), y_db_class.to(device)

                optimizer.zero_grad()
                out_ft, alpha = model(x_wave, x_stat)

                loss_ft = ce_loss(out_ft, y_fb)
                loss_deg, _ = hybrid_evidential_ordinal_loss(
                    alpha, y_db_class, annealing_coef=annealing_coef
                )

                loss = 0.3 * loss_ft + 0.7 * loss_deg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(y_fb)
                train_total += len(y_fb)

                # Track fault accuracy
                train_ft_correct += (out_ft.argmax(1) == y_fb).sum().item()

            train_loss /= train_total
            train_ft_acc = train_ft_correct / train_total

            # ===== VALIDATION =====
            model.eval()
            val_loss = 0
            val_ft_correct = 0
            val_ft_preds, val_ft_true = [], []
            val_deg_preds, val_deg_true = [], []

            with torch.no_grad():
                for x_wave, x_stat, y_fb, y_db_class in val_loader:
                    x_wave, x_stat = x_wave.to(device), x_stat.to(device)
                    y_fb, y_db_class = y_fb.to(device), y_db_class.to(device)

                    out_ft, alpha = model(x_wave, x_stat)

                    loss_ft = ce_loss(out_ft, y_fb)
                    loss_deg, _ = hybrid_evidential_ordinal_loss(
                        alpha, y_db_class, annealing_coef=annealing_coef
                    )
                    val_loss += (0.3 * loss_ft + 0.7 * loss_deg).item() * len(y_fb)

                    # Fault predictions
                    ft_pred = out_ft.argmax(1)
                    val_ft_correct += (ft_pred == y_fb).sum().item()
                    val_ft_preds.extend(ft_pred.cpu().numpy())
                    val_ft_true.extend(y_fb.cpu().numpy())

                    # Degradation predictions
                    predictions = predict_with_uncertainty(alpha)
                    val_deg_preds.extend(predictions['pred_value'].cpu().numpy())
                    val_deg_true.extend(y_db_class.cpu().numpy() * 20.0)

            val_loss /= train_total

            # Calculate metrics
            val_ft_acc = val_ft_correct / len(val_ds)
            val_ft_f1 = f1_score(val_ft_true, val_ft_preds, average='weighted')
            val_deg_mae = mean_absolute_error(val_deg_true, val_deg_preds)
            val_deg_rmse = np.sqrt(np.mean((np.array(val_deg_true) - np.array(val_deg_preds)) ** 2))

            scheduler.step()

            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d} | "
                      f"FtAcc={train_ft_acc:.3f}/{val_ft_acc:.3f} | "
                      f"FtF1={val_ft_f1:.3f} | "
                      f"MAE={val_deg_mae:.2f}° | "
                      f"RMSE={val_deg_rmse:.2f}°")

            # Early stopping based on MAE
            if val_deg_mae < best_val_mae:
                best_val_mae = val_deg_mae
                best_val_ft_acc = val_ft_acc
                best_val_ft_f1 = val_ft_f1
                best_val_deg_rmse = val_deg_rmse
                patience_counter = 0

                best_fold_state = {
                    'model_state_dict': model.state_dict(),
                    'circuit_name': circuit_name,
                    'fold': fold + 1,
                    'epoch': epoch,
                    # Fault metrics
                    'fault_accuracy': val_ft_acc,
                    'fault_f1': val_ft_f1,
                    # Degradation metrics
                    'deg_mae': val_deg_mae,
                    'deg_rmse': val_deg_rmse,
                    # Normalization params
                    'wave_mean': train_ds.wave_mean,
                    'wave_std': train_ds.wave_std,
                    'stat_mean': train_ds.stat_mean,
                    'stat_std': train_ds.stat_std,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Save fold model
        fold_path = os.path.join(save_dir, f'flex_eorn_{circuit_name}_fold{fold+1}.pth')
        torch.save(best_fold_state, fold_path)

        print(f"\n  ✓ Fold {fold+1} Results:")
        print(f"    Fault Accuracy: {best_val_ft_acc:.3f}")
        print(f"    Fault F1:       {best_val_ft_f1:.3f}")
        print(f"    Deg MAE:        {best_val_mae:.2f}°")
        print(f"    Deg RMSE:       {best_val_deg_rmse:.2f}°")
        print(f"    Saved: {fold_path}\n")

        fold_results.append({
            'fold': fold + 1,
            'fault_acc': best_val_ft_acc,
            'fault_f1': best_val_ft_f1,
            'deg_mae': best_val_mae,
            'deg_rmse': best_val_deg_rmse
        })

        if best_val_mae < best_overall_mae:
            best_overall_mae = best_val_mae
            best_overall_model = best_fold_state
            best_overall_fold = fold + 1

    # Save best overall model
    if best_overall_model:
        best_path = os.path.join(save_dir, f'flex_eorn_{circuit_name}_best.pth')
        torch.save(best_overall_model, best_path)

        print(f"\n{'='*80}")
        print(f"BEST MODEL: {circuit_name}")
        print(f"{'='*80}")
        print(f"Best Fold: {best_overall_fold}")
        print(f"Fault Accuracy: {best_overall_model['fault_accuracy']:.3f}")
        print(f"Fault F1 Score: {best_overall_model['fault_f1']:.3f}")
        print(f"Deg MAE:        {best_overall_model['deg_mae']:.2f}°")
        print(f"Deg RMSE:       {best_overall_model['deg_rmse']:.2f}°")
        print(f"Saved: {best_path}")
        print(f"{'='*80}")

    # Print summary across all folds
    print(f"\n{'='*80}")
    print(f"SUMMARY: {circuit_name} ({k}-Fold CV)")
    print(f"{'='*80}")
    print(f"Fault Accuracy: {np.mean([r['fault_acc'] for r in fold_results]):.3f} "
          f"± {np.std([r['fault_acc'] for r in fold_results]):.3f}")
    print(f"Fault F1 Score: {np.mean([r['fault_f1'] for r in fold_results]):.3f} "
          f"± {np.std([r['fault_f1'] for r in fold_results]):.3f}")
    print(f"Deg MAE:        {np.mean([r['deg_mae'] for r in fold_results]):.2f}° "
          f"± {np.std([r['deg_mae'] for r in fold_results]):.2f}°")
    print(f"Deg RMSE:       {np.mean([r['deg_rmse'] for r in fold_results]):.2f}° "
          f"± {np.std([r['deg_rmse'] for r in fold_results]):.2f}°")
    print(f"{'='*80}\n")

    return fold_results


# ==============================================================================
# STEP 6: PREPARE DATA AND TRAIN
# ==============================================================================

print("="*80)
print("PREPARING DATA FOR TRAINING")
print("="*80)

circuits_data = {}

for circuit_name in circuits_features.keys():
    X_wave, channels, wave_meta = circuits_waveforms[circuit_name]
    X_stat, feature_names, stat_meta = circuits_features[circuit_name]

    circuits_data[circuit_name] = {
        'X_wave': X_wave,
        'X_stat': X_stat,
        'y_ft': wave_meta['y_fault_type'],
        'y_deg_norm': wave_meta['y_degradation'] / 100.0,
    }

    print(f"✓ {circuit_name}: {X_wave.shape[0]} samples")

print(f"\n✓ Prepared {len(circuits_data)} circuits")
print("="*80)

# Training config
config = {
    'epochs': 50,
    'batch_size': 16,
    'lr': 3e-4,
    'patience': 10,
    'k': 5,
    'save_dir': 'outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

train_single = True
single_circuit = 'RECNEW'

if train_single:
    circuit_name = single_circuit
    data = circuits_data[circuit_name]

    results = train_flexible_eorn(
        circuit_name=circuit_name,
        X_wave=data['X_wave'],
        X_stat=data['X_stat'],
        y_ft=data['y_ft'],
        y_deg_norm=data['y_deg_norm'],
        **config
    )

    print(f"\n✓ Best MAE: {min([r['deg_mae'] for r in results]):.2f}°")