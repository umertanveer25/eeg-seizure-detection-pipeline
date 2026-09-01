import os
import mne
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pywt
import warnings

mne.set_log_level("ERROR")
warnings.filterwarnings('ignore')

def extract_advanced_features(window_data):
    mean = np.mean(window_data)
    var = np.var(window_data)
    std = np.std(window_data)
    line_length = np.sum(np.abs(np.diff(window_data)))
    zero_crossings = np.sum(np.diff(np.sign(window_data)) != 0)
    
    dy_dt = np.diff(window_data)
    dy2_dt2 = np.diff(dy_dt)
    activity = var
    mobility = np.std(dy_dt) / std if std > 0 else 0
    complexity = (np.std(dy2_dt2) / np.std(dy_dt)) / mobility if mobility > 0 else 0

    fft_vals = np.abs(np.fft.rfft(window_data))
    delta = np.sum(fft_vals[1:5])
    theta = np.sum(fft_vals[4:9])
    alpha = np.sum(fft_vals[8:14])
    beta = np.sum(fft_vals[13:31])
    gamma = np.sum(fft_vals[31:])
    
    coeffs = pywt.wavedec(window_data, 'db4', level=4)
    energy_dwt = [np.sum(np.square(c)) for c in coeffs]
    
    p = np.square(fft_vals)
    p_norm = p / np.sum(p) if np.sum(p) > 0 else np.zeros_like(p)
    spectral_entropy = -np.sum(p_norm[p_norm > 0] * np.log2(p_norm[p_norm > 0]))

    features = [mean, var, line_length, zero_crossings, activity, mobility, complexity, 
                delta, theta, alpha, beta, gamma, spectral_entropy] + energy_dwt
    return features

def process_single_file(row_tuple):
    _, row = row_tuple
    base_dir = r"D:\chb-mit-scalp-eeg-database-1.0.0"
    subj = row['Subject']
    fname = row['File']
    start = int(row['Start_Sec'])
    end = int(row['End_Sec'])
    
    edf_path = os.path.join(base_dir, subj, fname)
    if not os.path.exists(edf_path):
        return []
        
    local_data = []
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.pick(['FP1-F7'])
        data, _ = raw.copy().get_data(return_times=True)
        data = data[0] * 1e6
        sfreq = int(raw.info['sfreq'])
        
        for i in range(start, min(end, start+10)): 
            w = data[i*sfreq : (i+1)*sfreq]
            if len(w) == sfreq:
                local_data.append((extract_advanced_features(w), 1, subj))
            
        for i in range(10, 40):
            if i*sfreq < len(data):
                w = data[i*sfreq : (i+1)*sfreq]
                if len(w) == sfreq:
                    local_data.append((extract_advanced_features(w), 0, subj))
    except Exception:
        pass
    
    return local_data

def run_lopo():
    csv_path = r"C:\Users\umert\Downloads\WS Paper\Paper 3\chb_mit_seizure_metadata.csv"
    df = pd.read_csv(csv_path)
    
    rows = list(df.iterrows())
    with Pool(processes=cpu_count()) as pool:
        results_list = pool.map(process_single_file, rows)
        
    all_data = [item for sublist in results_list for item in sublist]
    
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data])
    groups = np.array([item[2] for item in all_data])
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logo = LeaveOneGroupOut()
    
    # We include the paper's exact models (Logistic Regression and ANN [128, 64])
    models = {
        'Logistic Regression (Paper model)': LogisticRegression(max_iter=1000),
        'ANN (Paper architecture)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300),
        'XGBoost (Advanced)': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss'),
        'LightGBM (Advanced)': lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, verbose=-1),
        'Random Forest (Advanced)': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, class_weight='balanced')
    }
    
    print("\n" + "="*80)
    print(f"{'Algorithm':<35} | {'LOPO Accuracy':<15} | {'Std Dev':<10}")
    print("-" * 80)
    
    for name, clf in models.items():
        accuracies = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"[OURS] {name:<28} | {mean_acc*100:>13.2f}% | ±{std_acc*100:.2f}%")
        
    print("=" * 80)

if __name__ == "__main__":
    run_lopo()
