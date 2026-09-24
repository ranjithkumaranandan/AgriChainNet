# ============================================================
# PRIORITY 4 - BLOCKCHAIN INTEGRATION
# AgriChainNet - Tamper-proof Disease Record Storage
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================
# This script SIMULATES blockchain integration
# for the paper without needing actual Ethereum network.
# It demonstrates:
# 1. Image hashing (SHA256)
# 2. Transaction recording
# 3. Smart contract logic
# 4. Tamper detection
# 5. Performance metrics (TPS, latency)
# ============================================================

import os
import json
import time
import hashlib
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH  = "D:/September"
SAVE_PATH  = os.path.join(BASE_PATH, "models/blockchain")
LOG_PATH   = os.path.join(BASE_PATH, "models/blockchain/logs")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

print("=" * 60)
print("  PRIORITY 4 - BLOCKCHAIN INTEGRATION")
print("  AgriChainNet — Tamper-proof Disease Storage")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 60)

# ============================================================
# SIMULATED AI RESULTS (from our trained models)
# ============================================================
SAMPLE_RESULTS = [
    {
        "farmer_id"   : "TN_FARM_001",
        "location"    : "Kanchipuram, Tamil Nadu",
        "crop"        : "Chilli",
        "detection"   : {"disease": "Bacterial Blight",
                         "confidence": 0.89, "model": "YOLOv8m"},
        "segmentation": {"dice_score": 0.9948, "area_pct": 34.5,
                         "model": "U-Net"},
        "classification":{"disease": "Bacterial Leaf Blight",
                          "confidence": 0.9921, "severity": "High",
                          "model": "AgriAttention-Net"},
    },
    {
        "farmer_id"   : "TN_FARM_002",
        "location"    : "Chengalpattu, Tamil Nadu",
        "crop"        : "Banana",
        "detection"   : {"disease": "Sigatoka", "confidence": 0.82,
                         "model": "YOLOv8m"},
        "segmentation": {"dice_score": 0.9876, "area_pct": 22.3,
                         "model": "U-Net"},
        "classification":{"disease": "Black Sigatoka",
                          "confidence": 0.9834, "severity": "Medium",
                          "model": "AgriAttention-Net"},
    },
    {
        "farmer_id"   : "TN_FARM_003",
        "location"    : "Krishnagiri, Tamil Nadu",
        "crop"        : "Rice",
        "detection"   : {"disease": "Rice Blast",
                         "confidence": 0.94, "model": "YOLOv8m"},
        "segmentation": {"dice_score": 0.9912, "area_pct": 45.1,
                         "model": "U-Net"},
        "classification":{"disease": "Magnaporthe oryzae",
                          "confidence": 0.9956, "severity": "High",
                          "model": "AgriAttention-Net"},
    },
    {
        "farmer_id"   : "TN_FARM_004",
        "location"    : "Kanchipuram, Tamil Nadu",
        "crop"        : "Groundnut",
        "detection"   : {"disease": "Early Leaf Spot",
                         "confidence": 0.78, "model": "YOLOv8m"},
        "segmentation": {"dice_score": 0.9789, "area_pct": 18.7,
                         "model": "U-Net"},
        "classification":{"disease": "Cercospora arachidicola",
                          "confidence": 0.9712, "severity": "Low",
                          "model": "AgriAttention-Net"},
    },
    {
        "farmer_id"   : "TN_FARM_005",
        "location"    : "Chengalpattu, Tamil Nadu",
        "crop"        : "Cauliflower",
        "detection"   : {"disease": "Black Rot",
                         "confidence": 0.91, "model": "YOLOv8m"},
        "segmentation": {"dice_score": 0.9934, "area_pct": 28.9,
                         "model": "U-Net"},
        "classification":{"disease": "Xanthomonas campestris",
                          "confidence": 0.9878, "severity": "Medium",
                          "model": "AgriAttention-Net"},
    },
]

# ============================================================
# BLOCK STRUCTURE
# ============================================================
class Block:
    def __init__(self, index, timestamp, data,
                 previous_hash, nonce=0):
        self.index         = index
        self.timestamp     = timestamp
        self.data          = data
        self.previous_hash = previous_hash
        self.nonce         = nonce
        self.hash          = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index"        : self.index,
            "timestamp"    : self.timestamp,
            "data"         : self.data,
            "previous_hash": self.previous_hash,
            "nonce"        : self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index"        : self.index,
            "timestamp"    : self.timestamp,
            "data"         : self.data,
            "previous_hash": self.previous_hash,
            "nonce"        : self.nonce,
            "hash"         : self.hash
        }

# ============================================================
# BLOCKCHAIN CLASS
# ============================================================
class AgriBlockchain:
    def __init__(self, difficulty=2):
        self.chain      = []
        self.difficulty = difficulty
        self.pending_tx = []
        self.gas_costs  = []
        self.latencies  = []

        # Create genesis block
        genesis = Block(0, str(datetime.now()),
                        {"type": "GENESIS",
                         "message": "AgriChainNet Genesis Block"},
                        "0" * 64)
        self.chain.append(genesis)
        print(f"\n  ✅ Genesis block created")
        print(f"     Hash: {genesis.hash[:20]}...")

    def proof_of_work(self, block):
        """Simple PoW for demonstration"""
        block.nonce = 0
        while not block.hash.startswith('0' * self.difficulty):
            block.nonce += 1
            block.hash   = block.compute_hash()
        return block

    def add_disease_record(self, result):
        """
        Add a disease detection result to the blockchain.
        This is what happens when AgriChainNet detects a disease.
        """
        start_time = time.time()

        # Generate image hash (simulated)
        img_content = json.dumps(result, sort_keys=True).encode()
        image_hash  = hashlib.sha256(img_content).hexdigest()

        # Smart contract logic
        smart_contract = self.execute_smart_contract(result)

        # Create transaction record
        transaction = {
            "tx_id"         : hashlib.md5(
                f"{result['farmer_id']}{time.time()}".encode()
            ).hexdigest()[:16],
            "farmer_id"     : result["farmer_id"],
            "location"      : result["location"],
            "crop"          : result["crop"],
            "timestamp"     : str(datetime.now()),
            "image_hash"    : image_hash,
            "detection"     : result["detection"],
            "segmentation"  : result["segmentation"],
            "classification": result["classification"],
            "smart_contract": smart_contract,
        }

        # Create new block
        prev_hash = self.chain[-1].hash
        new_block = Block(
            len(self.chain),
            transaction["timestamp"],
            transaction,
            prev_hash
        )

        # Mine block (PoW)
        new_block = self.proof_of_work(new_block)
        self.chain.append(new_block)

        # Record metrics
        latency  = (time.time() - start_time) * 1000  # ms
        gas_cost = random.randint(21000, 50000)  # Simulated gas
        self.latencies.append(latency)
        self.gas_costs.append(gas_cost)

        return new_block, transaction, latency, gas_cost

    def execute_smart_contract(self, result):
        """
        Smart contract logic:
        - Automatically triggers alerts/actions based on severity
        """
        severity   = result["classification"]["severity"]
        confidence = result["classification"]["confidence"]
        area_pct   = result["segmentation"]["area_pct"]
        crop       = result["crop"]

        contract_result = {
            "contract_version": "AgriChainNet_SC_v1.0",
            "executed"        : True,
        }

        # Rule 1: High severity → trigger insurance
        if severity == "High" and area_pct > 30:
            payout = round(area_pct * 500, 2)
            contract_result["insurance_triggered"] = True
            contract_result["payout_inr"]          = payout
            contract_result["action"] = \
                f"Auto-insurance payout ₹{payout:,} triggered"

        # Rule 2: Medium severity → send alert
        elif severity == "Medium":
            contract_result["insurance_triggered"] = False
            contract_result["alert_sent"]          = True
            contract_result["action"] = \
                "Alert sent to farmer and agriculture dept"

        # Rule 3: Low severity → log only
        else:
            contract_result["insurance_triggered"] = False
            contract_result["action"] = \
                "Disease logged. Monitor recommended."

        # Rule 4: Notify government if confidence > 95%
        if confidence > 0.95:
            contract_result["govt_notified"] = True
            contract_result["dept"] = \
                "Tamil Nadu Agriculture Department"

        return contract_result

    def verify_integrity(self):
        """Verify the entire blockchain is tamper-proof"""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]

            # Check hash validity
            if curr.hash != curr.compute_hash():
                return False, f"Block {i} hash invalid!"

            # Check chain linkage
            if curr.previous_hash != prev.hash:
                return False, f"Block {i} chain broken!"

        return True, "Blockchain integrity verified ✅"

    def simulate_tamper(self, block_index):
        """Demonstrate tamper detection"""
        original = self.chain[block_index].data.copy()
        self.chain[block_index].data["classification"]["disease"] = \
            "TAMPERED DATA"
        valid, msg = self.verify_integrity()
        self.chain[block_index].data = original
        return valid, msg

    def get_metrics(self):
        return {
            "total_blocks"  : len(self.chain),
            "avg_latency_ms": np.mean(self.latencies),
            "min_latency_ms": np.min(self.latencies),
            "max_latency_ms": np.max(self.latencies),
            "avg_gas_cost"  : np.mean(self.gas_costs),
            "throughput_tps": 1000 / np.mean(self.latencies),
        }

# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_blockchain_metrics(blockchain):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Block structure visualization
    ax = axes[0, 0]
    colors = ['#2ecc71' if i == 0 else '#3498db'
              for i in range(len(blockchain.chain))]
    for i, (block, color) in enumerate(
            zip(blockchain.chain, colors)):
        rect = mpatches.FancyBboxPatch(
            (i * 1.8, 0.3), 1.5, 0.4,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(i * 1.8 + 0.75, 0.5,
                f"Block {block.index}",
                ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')
        if i < len(blockchain.chain) - 1:
            ax.annotate('', xy=((i+1)*1.8, 0.5),
                        xytext=(i*1.8+1.5, 0.5),
                        arrowprops=dict(arrowstyle='->',
                                        color='gray', lw=2))
    ax.set_xlim(-0.2, len(blockchain.chain) * 1.8)
    ax.set_ylim(0, 1)
    ax.set_title('AgriChainNet Blockchain Structure',
                 fontweight='bold')
    ax.axis('off')

    # 2. Transaction latency
    ax = axes[0, 1]
    ax.bar(range(1, len(blockchain.latencies)+1),
           blockchain.latencies,
           color='#3498db', edgecolor='white')
    ax.axhline(y=np.mean(blockchain.latencies),
               color='#e74c3c', linestyle='--',
               label=f'Avg: {np.mean(blockchain.latencies):.1f}ms')
    ax.set_title('Transaction Latency (ms)',
                 fontweight='bold')
    ax.set_xlabel('Transaction #')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Smart contract actions
    ax = axes[1, 0]
    actions = defaultdict(int)
    for block in blockchain.chain[1:]:
        sc = block.data.get('smart_contract', {})
        if sc.get('insurance_triggered'):
            actions['Insurance\nTriggered'] += 1
        elif sc.get('alert_sent'):
            actions['Alert\nSent'] += 1
        else:
            actions['Logged\nOnly'] += 1

    colors_sc = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(list(actions.keys()),
                  list(actions.values()),
                  color=colors_sc[:len(actions)],
                  edgecolor='white')
    for bar, val in zip(bars, actions.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                str(val), ha='center', fontweight='bold')
    ax.set_title('Smart Contract Actions Triggered',
                 fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Disease severity distribution
    ax = axes[1, 1]
    severities = defaultdict(int)
    for block in blockchain.chain[1:]:
        sev = block.data.get('classification', {}).get('severity', 'Unknown')
        severities[sev] += 1

    colors_sev = {'High': '#e74c3c', 'Medium': '#f39c12',
                  'Low': '#2ecc71', 'Unknown': '#95a5a6'}
    wedge_colors = [colors_sev.get(s, '#95a5a6')
                    for s in severities.keys()]
    ax.pie(severities.values(),
           labels=severities.keys(),
           colors=wedge_colors,
           autopct='%1.0f%%',
           startangle=90)
    ax.set_title('Disease Severity Distribution\non Blockchain',
                 fontweight='bold')

    plt.suptitle('AgriChainNet Blockchain Performance Metrics\n'
                 'Blockchain in Agriculture - Q1 Paper',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH, 'blockchain_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Blockchain metrics chart saved!")

# ============================================================
# SAVE RESULTS
# ============================================================
def save_blockchain_results(blockchain, metrics):
    # Save full chain to JSON
    chain_data = [block.to_dict() for block in blockchain.chain]
    with open(os.path.join(LOG_PATH, 'blockchain_records.json'),
              'w') as f:
        json.dump(chain_data, f, indent=2)

    # Save metrics report
    with open(os.path.join(LOG_PATH, 'blockchain_metrics.txt'),
              'w') as f:
        f.write("AgriChainNet Blockchain Metrics\n")
        f.write("Blockchain in Agriculture - Q1 Paper\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Blocks      : {metrics['total_blocks']}\n")
        f.write(f"Avg Latency       : "
                f"{metrics['avg_latency_ms']:.2f} ms\n")
        f.write(f"Min Latency       : "
                f"{metrics['min_latency_ms']:.2f} ms\n")
        f.write(f"Max Latency       : "
                f"{metrics['max_latency_ms']:.2f} ms\n")
        f.write(f"Throughput        : "
                f"{metrics['throughput_tps']:.2f} TPS\n")
        f.write(f"Avg Gas Cost      : "
                f"{metrics['avg_gas_cost']:.0f} units\n")
        f.write("\nSmart Contract Actions:\n")
        for block in blockchain.chain[1:]:
            sc = block.data.get('smart_contract', {})
            f.write(f"  Block {block.index}: "
                    f"{sc.get('action', 'N/A')}\n")

    print(f"  ✅ Blockchain records saved!")
    print(f"  ✅ Metrics report saved!")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Create blockchain
    print("\n🔗 Initializing AgriChainNet Blockchain...")
    blockchain = AgriBlockchain(difficulty=2)

    # Add disease records
    print("\n📝 Adding disease detection records to blockchain...")
    print("-" * 60)

    for i, result in enumerate(SAMPLE_RESULTS):
        block, tx, latency, gas = \
            blockchain.add_disease_record(result)

        print(f"\n  Block {block.index}:")
        print(f"  Farmer   : {result['farmer_id']} "
              f"({result['location']})")
        print(f"  Crop     : {result['crop']}")
        print(f"  Disease  : "
              f"{result['classification']['disease']}")
        print(f"  Severity : "
              f"{result['classification']['severity']}")
        print(f"  Action   : "
              f"{tx['smart_contract']['action']}")
        print(f"  Hash     : {block.hash[:25]}...")
        print(f"  Latency  : {latency:.2f}ms | "
              f"Gas: {gas:,} units")

    # Verify integrity
    print("\n🔐 Verifying Blockchain Integrity...")
    valid, msg = blockchain.verify_integrity()
    print(f"  {msg}")

    # Test tamper detection
    print("\n🚨 Testing Tamper Detection...")
    tampered, tamper_msg = blockchain.simulate_tamper(2)
    print(f"  Tamper attempt on Block 2:")
    print(f"  Detected: {'YES ✅' if not tampered else 'NO ❌'}")
    print(f"  Message : {tamper_msg}")

    # Get metrics
    metrics = blockchain.get_metrics()
    print(f"\n📊 Blockchain Performance Metrics:")
    print(f"  Total Blocks  : {metrics['total_blocks']}")
    print(f"  Avg Latency   : {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Throughput    : {metrics['throughput_tps']:.2f} TPS")
    print(f"  Avg Gas Cost  : {metrics['avg_gas_cost']:.0f} units")

    # Plot
    plot_blockchain_metrics(blockchain)

    # Save
    save_blockchain_results(blockchain, metrics)

    print(f"\n{'='*60}")
    print(f"  BLOCKCHAIN INTEGRATION COMPLETE!")
    print(f"  Total Blocks  : {metrics['total_blocks']}")
    print(f"  Avg Latency   : {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Throughput    : {metrics['throughput_tps']:.2f} TPS")
    print(f"  Integrity     : {'VERIFIED ✅' if valid else 'FAILED ❌'}")
    print(f"  Saved to      : {LOG_PATH}")
    print(f"{'='*60}")
    print(f"\n✅ Priority 4 Done! Next → Priority 5: Ablation Study")
