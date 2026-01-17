# generate_can_dataset.py
# ==============================================================
# 📌 FILE PURPOSE: Generate synthetic CAN-Bus data
# - Normal traffic + 4 attack types (Spoof, Replay, Unauthorized, Corruption)
# - Output: can_data.csv (timestamp, can_id, dlc, b0-b7, label)
# ==============================================================

# Required libraries
import csv, random, time      # random: random data generation
import numpy as np            # numpy: scientific computations
import pandas as pd           # pandas: data processing and CSV writing
from datetime import datetime, timedelta  # timestamp creation

# -------------------- HYPERPARAMETERS --------------------
OUT_CSV = "can_data.csv"     # Output file name
NUM_NORMAL = 20000           # Number of normal messages
NUM_SPIKE_EVENTS = 4         # Number of attack blocks
RANDOM_SEED = 42             # Seed value for reproducibility

# Initialize random number generator (same seed = same result)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# CAN ID definitions
# normal_ids: Legitimate ECU IDs seen in the vehicle (engine, brake, etc.)
normal_ids = [0x100, 0x101, 0x200, 0x300, 0x350, 0x7FF]
# unauth_ids: IDs used by attacker devices (outside normal set)
unauth_ids = [0x9AA, 0x9AB]

def random_payload():
    """
    Generate random CAN payload between 1-8 bytes, padded to 8 bytes.
    CAN messages carry max 8 bytes of data, DLC specifies the length.
    """
    length = random.randint(1, 8)                    # Actual data length
    b = [random.randint(0,255) for _ in range(length)]  # Random bytes
    b += [0]*(8-length)                              # Pad the rest with zeros
    return b

# Timestamp creation
start = datetime.now()  # Start time
# ts: Generate timestamp for each message with 5ms interval (in milliseconds)
ts = lambda idx: (start + timedelta(milliseconds=idx*5)).timestamp() * 1000.0

rows = []  # List to hold all messages

# -------------------- 1) GENERATE NORMAL TRAFFIC --------------------
# Generate NUM_NORMAL clean (non-anomalous) messages
for i in range(NUM_NORMAL):
    t = ts(i)                            # Timestamp
    cid = random.choice(normal_ids)      # Select normal ID
    b = random_payload()                 # Random payload
    dlc = 8                              # Data Length Code
    rows.append({
        "timestamp": t,
        "can_id": cid,
        "dlc": dlc,
        **{f"b{k}": b[k] for k in range(8)},  # b0-b7 columns
        "label": 0                            # 0 = normal (not anomaly)
    })

# -------------------- 2) INSERT ATTACK BLOCKS --------------------
total = len(rows)
# Attack positions: Random selection excluding first 100 and last 500 messages
insert_positions = sorted(random.sample(range(100, total-500), NUM_SPIKE_EVENTS))

def insert_spoof(pos):
    """
    SPOOFING ATTACK: Send abnormal payload with normal ID.
    Attacker impersonates legitimate ECU but leaves signature in payload (b0=0xFF).
    """
    for j in range(50):  # 50 spoofed messages
        t = rows[pos + j]["timestamp"] + j*1.0
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": random.choice(normal_ids),  # Normal-looking ID
            "dlc": 8,
            # b0=0xFF: Anomaly signature (model learns this)
            **{f"b{k}": (0xFF if k==0 else random.randint(0,255)) for k in range(8)},
            "label": 1  # Anomaly
        })

def insert_replay(pos):
    """
    REPLAY ATTACK: Copy and quickly resend past messages.
    Attacker records previous traffic and replays it.
    """
    seg_start = max(0, pos - 200)        # Start of segment to copy
    segment = rows[seg_start:seg_start+50]
    base_time = rows[pos]["timestamp"]
    for idx, s in enumerate(segment):
        t = base_time + idx*1.0          # Fast replay (1ms interval)
        new = s.copy()
        new["timestamp"] = t
        new["label"] = 1                 # Replayed message = anomaly
        rows.insert(pos+idx, new)

def insert_unauthorized(pos):
    """
    UNAUTHORIZED DEVICE ATTACK: Message flood with IDs outside normal set.
    Simulates an unknown device connected to the CAN bus.
    """
    for j in range(80):  # 80 unauthorized messages
        t = rows[pos + j]["timestamp"] + j*2.0
        cid = random.choice(unauth_ids)  # Suspicious ID (0x9AA or 0x9AB)
        b = [random.randint(0,255) for _ in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": cid,
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

def insert_corruption(pos):
    """
    CORRUPTION ATTACK: Corrupt payload bytes using bitwise NOT.
    Simulates physical interference or malicious data manipulation.
    """
    for j in range(60):  # 60 corrupted messages
        t = rows[pos + j]["timestamp"] + j*3.0
        orig = rows[pos + j]
        # 50% chance to invert byte (~x), other half stays original
        b = [((~orig.get(f"b{k}",0)) & 0xFF) if random.random()<0.5 else orig.get(f"b{k}",0)
             for k in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": orig["can_id"],
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

# Apply different attack types sequentially to each position
for i, pos in enumerate(insert_positions):
    # Adjust position since insert operations extend the list
    current_pos = min(len(rows)-200, pos + i*100)
    if i % 4 == 0:
        insert_spoof(current_pos)
    elif i % 4 == 1:
        insert_replay(current_pos)
    elif i % 4 == 2:
        insert_unauthorized(current_pos)
    else:
        insert_corruption(current_pos)

# Sort by timestamp (insert operations may disrupt ordering)
rows = sorted(rows, key=lambda r: r["timestamp"])

# Normalize timestamps to a regular incrementing format
first_ts = rows[0]["timestamp"]
for i,r in enumerate(rows):
    r["timestamp"] = first_ts + i*5.0  # Regular series with 5ms interval

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")
