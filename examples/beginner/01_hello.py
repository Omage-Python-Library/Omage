"""
Hello Omage! — Your first AI model
"""

import sys
sys.path.insert(0, r"C:\Users\Zomorrod\OneDrive\Desktop\Omage library")

import omage as og
import pandas as pd
import numpy as np

print("=" * 50)
print("🚀 Hello Omage!")
print("=" * 50)

# Check GPU
print(f"\n💻 Device: {og.get_device()}")
print(f"🎮 GPU: {'Yes!' if og.is_gpu() else 'No (CPU)'}")

# Create dummy data
np.random.seed(42)
df = pd.DataFrame({
    "feature1": np.random.randn(100),
    "feature2": np.random.randn(100),
    "label": np.random.randint(0, 3, 100),
})

# Save to CSV
df.to_csv("dummy_data.csv", index=False)
print("\n📊 Dummy data created!")

# 1. Load data
print("\n📊 Loading data...")
data = og.load("dummy_data.csv")
data.clean().normalize().split(0.8, 0.2)

# 2. Build model (auto GPU if available!)
print("\n🏗️ Building model...")
model = og.model(
    type="classifier",
    layers=[
        og.dense(16),
        og.dense(3, activation="softmax"),
    ],
    optimizer=og.adam(lr=0.001),
    loss=og.cross_entropy(),
    device="auto"  # "auto", "cuda", "cpu", "mps"
)

# 3. Train
print("\n🚀 Training...")
history = model.train(data, epochs=3, batch_size=16)

# 4. Evaluate
print("\n📈 Evaluating...")
metrics = model.evaluate(data)

# 5. Predict
print("\n🔮 Predicting...")
result = model.predict([[0.5, -0.3]])

# 6. Save
print("\n💾 Saving...")
model.save("my_first_model.omg")

print("\n" + "=" * 50)
print("✅ Done! Welcome to Omage!")
print("=" * 50)