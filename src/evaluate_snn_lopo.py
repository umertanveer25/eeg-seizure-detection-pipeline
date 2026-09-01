import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pywt
import warnings
from joblib import Parallel, delayed

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
    if not os.path.exists(edf_path): return []
        
    local_data = []
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
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

def train_eval_snn_fold(X_train_np, y_train_np, X_test_np, y_test_np, fold_id, test_patient):
    # Important: import torch INSIDE the worker to prevent Windows IPC deadlocks
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import snntorch as snn
    from snntorch import surrogate
    
    # Restrict internal multithreading so the 16 parallel processes don't spawn 16x16 threads
    torch.set_num_threads(1)
    
    class Net(nn.Module):
        def __init__(self, num_inputs):
            super().__init__()
            spike_grad = surrogate.fast_sigmoid(slope=25)
            beta = 0.9 
            
            self.fc1 = nn.Linear(num_inputs, 256)
            self.lif1 = snn.Leaky(beta=beta, threshold=1.0, spike_grad=spike_grad)
            self.fc2 = nn.Linear(256, 2)
            self.lif2 = snn.Leaky(beta=beta, threshold=1.0, spike_grad=spike_grad)

        def forward(self, x, num_steps=80):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec = []
            for step in range(num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
            return torch.stack(spk2_rec, dim=0)
            
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.int64)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.int64)
    
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    
    net = Net(num_inputs=X_train.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    net.train()
    for epoch in range(3): 
        for data, targets in train_loader:
            optimizer.zero_grad()
            spk_rec = net(data)
            loss = loss_fn(spk_rec.sum(dim=0), targets)
            loss.backward()
            optimizer.step()
        
    net.eval()
    with torch.no_grad():
        spk_rec = net(X_test)
        _, idx = spk_rec.sum(dim=0).max(1)
        acc = accuracy_score(y_test.numpy(), idx.numpy())
        
    print(f"[OK] Fold {fold_id} (Patient {test_patient}) Complete -> Acc: {acc*100:.2f}%", flush=True)
    return acc

def run_lopo_snn():
    csv_path = r"C:\Users\umert\Downloads\WS Paper\Paper 3\chb_mit_seizure_metadata.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Running parallel feature extraction using joblib...")
    rows = list(df.iterrows())
    results_list = Parallel(n_jobs=-1, backend="loky")(delayed(process_single_file)(r) for r in rows)
        
    all_data = [item for sublist in results_list for item in sublist]
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data], dtype=np.int64)
    groups = np.array([item[2] for item in all_data])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logo = LeaveOneGroupOut()
    
    print("\nStarting Fully Parallelized SNN LOPO Training Across All CPU Cores...")
    
    fold_tasks = []
    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_patient = groups[test_idx[0]]
        fold_tasks.append((X[train_idx], y[train_idx], X[test_idx], y[test_idx], i+1, test_patient))
        
    accuracies = Parallel(n_jobs=-1, backend="loky")(
        delayed(train_eval_snn_fold)(*task) for task in fold_tasks
    )
        
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print("\n" + "="*80)
    print(f"{'Algorithm':<35} | {'LOPO Accuracy':<15} | {'Std Dev':<10}")
    print("-" * 80)
    print(f"[OURS] SNN (Paper architecture)     | {mean_acc*100:>13.2f}% | ±{std_acc*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_lopo_snn()
