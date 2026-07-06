"""IEEE-CIS Fraud Detection schema.

Single source of truth for column inventory, dtypes, and categorical vocabularies.
Used by the real-data loader, the synthetic fixture generator, and the feature
builder so all three agree on structure. Column set mirrors the public competition
schema: identity is present for only a subset of transactions and is left-joined on
``TransactionID``.

Reference: Kaggle IEEE-CIS Fraud Detection. The V1-V339 block is Vesta's engineered
feature set (anonymized); id_12-id_38 are treated as categorical per the competition
data description.
"""

from __future__ import annotations

# --- Keys / target / core columns ---------------------------------------------
ID_COL = "TransactionID"
TARGET = "isFraud"
TIME_COL = "TransactionDT"
AMT_COL = "TransactionAmt"

# --- Transaction block --------------------------------------------------------
PRODUCT_COL = "ProductCD"
CARD_COLS = [f"card{i}" for i in range(1, 7)]  # card1..card6
CARD_NUM_COLS = ["card1", "card2", "card3", "card5"]  # numeric cards
CARD_CAT_COLS = ["card4", "card6"]  # network / type
ADDR_COLS = ["addr1", "addr2"]
DIST_COLS = ["dist1", "dist2"]
EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]
C_COLS = [f"C{i}" for i in range(1, 15)]  # C1..C14 counting features
D_COLS = [f"D{i}" for i in range(1, 16)]  # D1..D15 timedelta features
M_COLS = [f"M{i}" for i in range(1, 10)]  # M1..M9 match features
V_COLS = [f"V{i}" for i in range(1, 340)]  # V1..V339 Vesta features

# --- Identity block (left-joined; present for a subset) -----------------------
ID_NUM_COLS = [f"id_{i:02d}" for i in range(1, 12)]  # id_01..id_11 numeric
ID_CAT_COLS = [f"id_{i:02d}" for i in range(12, 39)]  # id_12..id_38 categorical
DEVICE_COLS = ["DeviceType", "DeviceInfo"]

# --- Canonical categorical feature list (competition data description) --------
CATEGORICAL_FEATURES = [
    PRODUCT_COL,
    *CARD_COLS,
    *ADDR_COLS,
    *EMAIL_COLS,
    *M_COLS,
    *DEVICE_COLS,
    *ID_CAT_COLS,
]

# --- Entities for causal aggregate features -----------------------------------
# Chosen because they identify a repeat actor over time (card, buyer email, region).
ENTITY_COLS = ["card1", "P_emaildomain", "addr1"]

# --- Categorical vocabularies (for the synthetic fixture; real data is a superset)
PRODUCT_VOCAB = ["W", "C", "R", "H", "S"]
CARD4_VOCAB = ["visa", "mastercard", "american express", "discover"]
CARD6_VOCAB = ["debit", "credit", "debit or credit", "charge card"]
EMAIL_VOCAB = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "outlook.com",
    "icloud.com",
    "comcast.net",
]
M_BINARY_VOCAB = ["T", "F"]
M4_VOCAB = ["M0", "M1", "M2"]
DEVICE_TYPE_VOCAB = ["desktop", "mobile"]
DEVICE_INFO_VOCAB = ["Windows", "iOS Device", "MacOS", "Android", "SAMSUNG", "Trident/7.0"]
ID_30_VOCAB = ["Windows 10", "iOS 11.1.2", "Mac OS X 10_13", "Android 7.0", "Windows 7"]
ID_31_VOCAB = ["chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop", "firefox 57.0"]
ID_33_VOCAB = ["1920x1080", "1366x768", "2208x1242", "1334x750", "2436x1125"]


def transaction_columns() -> list[str]:
    """Ordered column list for train_transaction (matches the raw CSV header order)."""
    cols = [ID_COL, TARGET, TIME_COL, AMT_COL, PRODUCT_COL, *CARD_COLS, *ADDR_COLS, *DIST_COLS]
    cols += [EMAIL_COLS[0]] + C_COLS + D_COLS + M_COLS + [EMAIL_COLS[1]]
    cols += V_COLS
    return cols


def identity_columns() -> list[str]:
    """Ordered column list for train_identity."""
    return [ID_COL, *ID_NUM_COLS, *ID_CAT_COLS, *DEVICE_COLS]


def numeric_feature_columns() -> list[str]:
    """Numeric feature columns (excludes keys/target)."""
    return [
        AMT_COL,
        *CARD_NUM_COLS,
        *ADDR_COLS,
        *DIST_COLS,
        *C_COLS,
        *D_COLS,
        *V_COLS,
        *ID_NUM_COLS,
    ]


def all_feature_columns() -> list[str]:
    """Every modelling column (numeric + categorical), excluding keys/target/time."""
    return numeric_feature_columns() + CATEGORICAL_FEATURES
