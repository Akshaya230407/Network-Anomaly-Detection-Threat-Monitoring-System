import pandas as pd
from functools import lru_cache
from pathlib import Path


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# PROCESSED DATASET DIRECTORY
# =========================================================

PROCESSED_DIR = (
    BASE_DIR
    / "datasets"
    / "processed_small"
)


# =========================================================
# UNSW-NB15 PROCESSED FILE
# =========================================================

UNSW_PROCESSED_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "UNSW_NB15_processed.csv"
)

# =========================================================
# UNSW-NB15 ORIGINAL TRAINING DATASET
# =========================================================

UNSW_TRAIN_PATH = (
    BASE_DIR
    / "datasets"
    / "UNSW-NB15"
    / "Training and Testing Sets"
    / "UNSW_NB15_training-set.csv"
)

# =========================================================
# CIC-IDS2017 PROCESSED FILES
# =========================================================

CIC_PROCESSED_FILES = list(
    PROCESSED_DIR.glob("*_processed.csv")
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_cic_files():
    files = [
        file
        for file in PROCESSED_DIR.glob("*_processed.csv")
        if "UNSW_NB15" not in file.name
    ]

    if not files:
        raise FileNotFoundError(
            "No processed CIC-IDS2017 files found."
        )

    return files

def load_cic_dataset():
    csv_files = get_cic_files()

    dataframes = []

    required_columns = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Label",
        "label",
    ]

    for file in csv_files:
        try:
            df = pd.read_csv(
                file,
                usecols=required_columns,
                low_memory=True
            )

            df.columns = df.columns.str.strip()
            dataframes.append(df)

        except Exception as e:
            print(f"Could not load {file.name}: {e}")

    if not dataframes:
        raise ValueError(
            "Could not load processed CIC-IDS2017 dataset."
        )

    return pd.concat(
        dataframes,
        ignore_index=True
    )

# =========================================================
# UNSW-NB15 SUMMARY
# =========================================================

def get_unsw_summary():

    if not UNSW_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed UNSW-NB15 dataset not found at: "
            f"{UNSW_PROCESSED_PATH}"
        )

    # Process the dataset in chunks to stay within Render's 512 MB RAM limit
    chunk_size = 10000

    total_records = 0
    normal_traffic = 0
    attack_traffic = 0

    attack_types = {}
    protocol_distribution = {}

    for chunk in pd.read_csv(
        UNSW_PROCESSED_PATH,
        chunksize=chunk_size,
        low_memory=False
    ):

        total_records += len(chunk)

        # Normal traffic
        normal_traffic += int(
            (chunk["label"] == 0).sum()
        )

        # Attack traffic
        attack_traffic += int(
            (chunk["label"] == 1).sum()
        )

        # Attack types
        if "attack_cat" in chunk.columns:

            attacks = (
                chunk.loc[chunk["label"] == 1, "attack_cat"]
                .astype(str)
                .str.strip()
            )

            for attack_type, count in attacks.value_counts().items():
                attack_types[attack_type] = (
                    attack_types.get(attack_type, 0) + int(count)
                )

        # Protocol distribution
        if "proto" in chunk.columns:

            protocols = (
                chunk["proto"]
                .astype(str)
                .str.strip()
            )

            for protocol, count in protocols.value_counts().items():
                protocol_distribution[protocol] = (
                    protocol_distribution.get(protocol, 0) + int(count)
                )

    attack_percentage = (
        round((attack_traffic / total_records) * 100, 2)
        if total_records > 0
        else 0
    )

    # Keep only the top 10 results
    attack_types = dict(
        sorted(
            attack_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    )

    protocol_distribution = dict(
        sorted(
            protocol_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    )

    return {
        "dataset": "UNSW-NB15",
        "total_records": total_records,
        "normal_traffic": normal_traffic,
        "attack_traffic": attack_traffic,
        "attack_percentage": attack_percentage,
        "top_attack_types": attack_types,
        "protocol_distribution": protocol_distribution
    }


# =========================================================
# NETWORK MONITORING
# =========================================================

def get_network_traffic(
    dataset="UNSW-NB15",
    limit=100
):

    # =====================================================
    # UNSW-NB15
    # =====================================================

    if dataset == "UNSW-NB15":

        if not UNSW_PROCESSED_PATH.exists():
            raise FileNotFoundError(
                f"Processed UNSW-NB15 dataset not found at: "
                f"{UNSW_PROCESSED_PATH}"
            )

        df = pd.read_csv(
            UNSW_PROCESSED_PATH,
            nrows=limit
        )

        print("MONITORING DATASET PATH:", UNSW_PROCESSED_PATH)
        print("UNSW MONITORING COLUMNS:")
        print(df.columns.tolist())

        # Columns useful for Network Monitoring
        columns = [
            "id",
            "proto",
            "service",
            "state",
            "dur",
            "spkts",
            "dpkts",
            "label",
            "attack_cat"
        ]
        # Keep only columns that actually exist
        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        # Get first 'limit' records
        traffic = df[
            available_columns
        ].head(limit)

        # Convert NaN values to None
        traffic = traffic.astype(object).where(
            pd.notnull(traffic),
            None
        )

        return traffic.to_dict(
            orient="records"
        )


    # =====================================================
    # CIC-IDS2017
    # =====================================================

    elif dataset == "CIC-IDS2017":

        csv_files = get_cic_files()

        if not csv_files:
            raise FileNotFoundError(
                "No processed CIC-IDS2017 files found."
            )

        # Use the first processed CIC file
        first_file = csv_files[0]

        df = pd.read_csv(
            first_file,
            low_memory=False
        )

        # Clean column names
        df.columns = (
            df.columns
            .str.strip()
        )

        columns = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Mean",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Label"
        ]

        # Keep only columns available in the file
        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        traffic = df[
            available_columns
        ].head(limit)

        # Convert NaN values to None
        traffic = traffic.astype(object).where(
            pd.notnull(traffic),
            None
        )

        return traffic.to_dict(
            orient="records"
        )


    # =====================================================
    # UNKNOWN DATASET
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================

def get_anomaly_summary(
    dataset="UNSW-NB15"
):

    # =====================================================
    # UNSW-NB15
    # =====================================================

    if dataset == "UNSW-NB15":

        if not UNSW_PROCESSED_PATH.exists():
            raise FileNotFoundError(
                "Processed UNSW-NB15 dataset not found."
            )

        df = pd.read_csv(
            UNSW_PROCESSED_PATH,
            low_memory=False
        )

        total_records = len(df)

        normal_records = int(
            (df["label"] == 0).sum()
        )

        anomalous_records = int(
            (df["label"] == 1).sum()
        )

        anomaly_percentage = round(
            (
                anomalous_records
                / total_records
            ) * 100,
            2
        ) if total_records > 0 else 0

        anomaly_types = {}

        if "attack_cat" in df.columns:

            anomaly_types = (
                df[
                    df["label"] == 1
                ]["attack_cat"]
                .astype(str)
                .str.strip()
                .value_counts()
                .head(10)
                .to_dict()
            )

        return {
            "dataset": "UNSW-NB15",
            "total_records": total_records,
            "normal_records": normal_records,
            "anomalous_records": anomalous_records,
            "anomaly_percentage": anomaly_percentage,
            "anomaly_types": anomaly_types
        }


    # =====================================================
    # CIC-IDS2017
    # =====================================================

    elif dataset == "CIC-IDS2017":

            total_records = 0
            normal_records = 0
            anomalous_records = 0
            anomaly_types = {}

            # Process CIC files one at a time in chunks
            for file in get_cic_files():

                for chunk in pd.read_csv(
                    file,
                    usecols=["Label"],
                    chunksize=10000,
                    low_memory=True
                ):

                    labels = (
                        chunk["Label"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                    )

                    total_records += len(labels)

                    normal_count = int(
                        (labels == "BENIGN").sum()
                    )

                    normal_records += normal_count

                    anomalous_records += (
                        len(labels) - normal_count
                    )

                    # Count attack types
                    attacks = labels[labels != "BENIGN"]

                    counts = attacks.value_counts()

                    for attack_type, count in counts.items():
                        anomaly_types[attack_type] = (
                            anomaly_types.get(attack_type, 0)
                            + int(count)
                        )

            anomaly_percentage = round(
                (
                    anomalous_records
                    / total_records
                ) * 100,
                2
            ) if total_records > 0 else 0

            anomaly_types = dict(
                sorted(
                    anomaly_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )

            return {
                "dataset": "CIC-IDS2017",
                "total_records": total_records,
                "normal_records": normal_records,
                "anomalous_records": anomalous_records,
                "anomaly_percentage": anomaly_percentage,
                "anomaly_types": anomaly_types
            }
    # =====================================================
    # UNKNOWN DATASET
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )


# =========================================================
# SECURITY ALERTS
# =========================================================

def get_security_alerts(
    dataset="UNSW-NB15"
):

    # =====================================================
    # UNSW-NB15
    # =====================================================

    if dataset == "UNSW-NB15":

        if not UNSW_PROCESSED_PATH.exists():
            raise FileNotFoundError(
                "Processed UNSW-NB15 dataset not found."
            )

        df = pd.read_csv(
            UNSW_PROCESSED_PATH,
            low_memory=False
        )


        # Only attack traffic
        attacks = df[
            df["label"] == 1
        ].copy()


        # Keep first 100 attacks
        attacks = attacks.head(100)


        alerts = []


        for index, row in attacks.iterrows():

            attack_type = str(
                row.get(
                    "attack_cat",
                    "Unknown"
                )
            )


            # Assign severity
            if attack_type in [
                "Generic",
                "Exploits",
                "DoS"
            ]:

                severity = "Critical"


            elif attack_type in [
                "Fuzzers",
                "Backdoor",
                "Shellcode"
            ]:

                severity = "High"


            else:

                severity = "Medium"


            alerts.append({

                "id": int(index) + 1,

                "severity": severity,

                "attack_type": attack_type,

                "srcip": str(
                    row.get(
                        "srcip",
                        "Unknown"
                    )
                ),

                "dstip": str(
                    row.get(
                        "dstip",
                        "Unknown"
                    )
                ),

                "proto": str(
                    row.get(
                        "proto",
                        "Unknown"
                    )
                )

            })


        return alerts


    # =====================================================
    # CIC-IDS2017
    # =====================================================

    elif dataset == "CIC-IDS2017":

        alerts = []
        alert_id = 1

        # Process CIC files one at a time
        for file in get_cic_files():

            for chunk in pd.read_csv(
                file,
                usecols=["Label"],
                chunksize=10000,
                low_memory=True
            ):

                chunk["Label"] = (
                    chunk["Label"]
                    .astype(str)
                    .str.strip()
                )

                attacks = chunk[
                    chunk["Label"].str.upper() != "BENIGN"
                ]

                for index, row in attacks.iterrows():

                    attack_type = str(row["Label"])

                    if attack_type.upper() in [
                        "DDOS",
                        "DOS HULK",
                        "DOS SLOWLORIS",
                        "DOS SLOWHTTPTEST"
                    ]:
                        severity = "Critical"

                    elif attack_type.upper() in [
                        "PORTSCAN",
                        "FTP-PATATOR",
                        "BOT"
                    ]:
                        severity = "High"

                    else:
                        severity = "Medium"

                    alerts.append({
                        "id": alert_id,
                        "severity": severity,
                        "attack_type": attack_type,
                        "srcip": "Unknown",
                        "dstip": "Unknown",
                        "proto": "Unknown"
                    })

                    alert_id += 1

                    if len(alerts) >= 100:
                        return alerts

        return alerts

    # =====================================================
    # UNKNOWN DATASET
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )


# =========================================================
# CIC-IDS2017 SUMMARY
# =========================================================

def get_cic_summary():

    df = load_cic_dataset().copy()


    # Clean Label column
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )


    # =====================================================
    # TRAFFIC COUNTS
    # =====================================================

    total_records = len(df)


    normal_traffic = int(
        (
            df["Label"]
            .str.upper()
            == "BENIGN"
        ).sum()
    )


    attack_traffic = (
        total_records
        - normal_traffic
    )


    attack_percentage = round(
        (
            attack_traffic
            / total_records
        ) * 100,
        2
    ) if total_records > 0 else 0


    # =====================================================
    # ATTACK TYPES
    # =====================================================

    attack_types = (
        df[
            df["Label"]
            .str.upper()
            != "BENIGN"
        ]["Label"]
        .value_counts()
        .head(10)
        .to_dict()
    )


    # =====================================================
    # PROTOCOL DISTRIBUTION
    # =====================================================

    protocol_distribution = {}


    if "Destination Port" in df.columns:

        protocol_distribution = (
            df["Destination Port"]
            .value_counts()
            .head(10)
            .to_dict()
        )


    # =====================================================
    # RETURN SUMMARY
    # =====================================================

    return {

        "dataset": "CIC-IDS2017",

        "total_records": total_records,

        "normal_traffic": normal_traffic,

        "attack_traffic": attack_traffic,

        "attack_percentage": attack_percentage,

        "top_attack_types": attack_types,

        "protocol_distribution": protocol_distribution

    }

def get_unsw_record(record_id):
    if not UNSW_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed UNSW-NB15 dataset not found at: "
            f"{UNSW_PROCESSED_PATH}"
        )

    df = pd.read_csv(
        UNSW_PROCESSED_PATH,
        low_memory=False
    )

    record = df[df["id"] == record_id]

    if record.empty:
        raise ValueError(
            f"UNSW-NB15 record with id {record_id} not found."
        )

    record = record.iloc[0]

    # Remove target / attack category
    excluded_columns = [
        "label",
        "attack_cat"
    ]

    model_data = {
        column: record[column]
        for column in df.columns
        if column not in excluded_columns
    }

    # Convert NumPy values to normal Python values
    for key, value in model_data.items():

        if pd.isna(value):
            model_data[key] = None

        elif hasattr(value, "item"):
            model_data[key] = value.item()

    return model_data

def get_cic_record(record_id):
    df = load_cic_dataset().copy()

    if df.empty:
        raise ValueError(
            "CIC-IDS2017 dataset is empty."
        )

    # CIC record_id uses the row position
    if record_id < 0 or record_id >= len(df):
        raise ValueError(
            f"CIC-IDS2017 record with id {record_id} not found."
        )

    record = df.iloc[record_id]

    # Remove target columns
    excluded_columns = [
        "Label",
        "label"
    ]

    model_data = {
        column: record[column]
        for column in df.columns
        if column not in excluded_columns
    }

    # Convert NumPy / pandas values to normal Python values
    for key, value in model_data.items():

        if pd.isna(value):
            model_data[key] = None

        elif hasattr(value, "item"):
            model_data[key] = value.item()

    return model_data