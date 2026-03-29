import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Column names (43 including difficulty)
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]

def load_data(path):
    df = pd.read_csv(path, names=columns)
    return df

def preprocess(df):
    df = df.copy()

    # ✅ Safe drop (fix for your error)
    if "difficulty" in df.columns:
        df.drop("difficulty", axis=1, inplace=True)

    # ✅ Encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # ✅ Map attack types
    def map_attack(label):
        if label == "normal":
            return "normal"
        elif label in ["neptune", "smurf", "back", "teardrop"]:
            return "dos"
        elif label in ["ipsweep", "nmap", "portsweep"]:
            return "probe"
        elif label in ["guess_passwd", "ftp_write"]:
            return "r2l"
        elif label in ["buffer_overflow", "rootkit"]:
            return "u2r"
        else:
            return "other"

    # If label exists (training case)
    if "label" in df.columns:
        df["label"] = df["label"].apply(map_attack)

        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"])

    return df