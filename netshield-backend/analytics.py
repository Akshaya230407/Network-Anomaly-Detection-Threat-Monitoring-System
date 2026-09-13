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
                usecols=[
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
                ],
                nrows=limit,
                low_memory=True
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

        total_records = 0
        normal_records = 0
        anomalous_records = 0
        anomaly_types = {}

        # Process UNSW in chunks to keep Render memory usage low
        for chunk in pd.read_csv(
            UNSW_PROCESSED_PATH,
            usecols=["label", "attack_cat"],
            chunksize=10000,
            low_memory=True
        ):

            labels = chunk["label"]

            total_records += len(labels)

            normal_records += int(
                (labels == 0).sum()
            )

            anomalous_records += int(
                (labels == 1).sum()
            )

            attacks = chunk[
                labels == 1
            ]["attack_cat"].astype(str).str.strip()

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

        alerts = []

        # Process UNSW-NB15 in chunks to reduce memory usage
        for chunk in pd.read_csv(
            UNSW_PROCESSED_PATH,
            usecols=[
                "label",
                "attack_cat",
                "proto"
            ],
            chunksize=10000,
            low_memory=True
        ):

            # Only attack traffic
            attacks = chunk[
                chunk["label"] == 1
            ]

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

                    "id": len(alerts) + 1,

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

                # Keep only the first 100 alerts
                if len(alerts) >= 100:
                    return alerts

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

    total_records = 0
    normal_traffic = 0
    attack_traffic = 0

    attack_types = {}
    protocol_distribution = {}

    # Process CIC files one at a time
    for file in get_cic_files():

        for chunk in pd.read_csv(
            file,
            usecols=[
                "Label",
                "Destination Port"
            ],
            chunksize=10000,
            low_memory=True
        ):

            # Clean Label column
            labels = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )

            total_records += len(labels)

            # Normal traffic
            normal_count = int(
                (
                    labels.str.upper()
                    == "BENIGN"
                ).sum()
            )

            normal_traffic += normal_count

            # Attack traffic
            attack_traffic += (
                len(labels)
                - normal_count
            )

            # Attack types
            attacks = labels[
                labels.str.upper()
                != "BENIGN"
            ]

            counts = attacks.value_counts()

            for attack_type, count in counts.items():

                attack_types[attack_type] = (
                    attack_types.get(
                        attack_type,
                        0
                    )
                    + int(count)
                )

            # Protocol / destination-port distribution
            port_counts = (
                chunk["Destination Port"]
                .value_counts()
            )

            for port, count in port_counts.items():

                protocol_distribution[port] = (
                    protocol_distribution.get(
                        port,
                        0
                    )
                    + int(count)
                )

    attack_percentage = round(
        (
            attack_traffic
            / total_records
        ) * 100,
        2
    ) if total_records > 0 else 0

    # Keep top 10 attack types
    attack_types = dict(
        sorted(
            attack_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    )

    # Keep top 10 destination ports
    protocol_distribution = dict(
        sorted(
            protocol_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    )

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

    # Read UNSW-NB15 in chunks to reduce memory usage
    for chunk in pd.read_csv(
        UNSW_PROCESSED_PATH,
        chunksize=10000,
        low_memory=True
    ):

        record = chunk[
            chunk["id"] == record_id
        ]

        if record.empty:
            continue

        record = record.iloc[0]

        # Remove target / attack category
        excluded_columns = [
            "label",
            "attack_cat"
        ]

        model_data = {
            column: record[column]
            for column in chunk.columns
            if column not in excluded_columns
        }

        # Convert NumPy values to normal Python values
        for key, value in model_data.items():

            if pd.isna(value):
                model_data[key] = None

            elif hasattr(value, "item"):
                model_data[key] = value.item()

        return model_data

    raise ValueError(
        f"UNSW-NB15 record with id {record_id} not found."
    )

def get_cic_record(record_id):

    if record_id < 0:
        raise ValueError(
            f"CIC-IDS2017 record with id {record_id} not found."
        )

    current_index = 0

    # Process CIC files one at a time
    for file in get_cic_files():

        for chunk in pd.read_csv(
            file,
            chunksize=10000,
            low_memory=True
        ):

            # Check whether the requested record
            # exists inside this chunk
            chunk_end = current_index + len(chunk)

            if current_index <= record_id < chunk_end:

                local_index = (
                    record_id - current_index
                )

                record = chunk.iloc[local_index]

                # Remove target columns
                excluded_columns = [
                    "Label",
                    "label"
                ]

                model_data = {
                    column: record[column]
                    for column in chunk.columns
                    if column not in excluded_columns
                }

                # Convert NumPy / pandas values
                # to normal Python values
                for key, value in model_data.items():

                    if pd.isna(value):
                        model_data[key] = None

                    elif hasattr(value, "item"):
                        model_data[key] = value.item()

                return model_data

            current_index = chunk_end

    raise ValueError(
        f"CIC-IDS2017 record with id {record_id} not found."
    )