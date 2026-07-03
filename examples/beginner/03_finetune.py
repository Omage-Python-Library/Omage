"""
Fine-tune Example — Using pre-trained model on custom data
"""

import sys
sys.path.insert(0, r"C:\Users\Zomorrod\OneDrive\Desktop\Omage library")

import omage as og
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("=" * 60)
print("🔧 Fine-tune Demo")
print("=" * 60)

# 1. Create dummy image data (simulation)
print("\n📊 Creating dummy image data...")
np.random.seed(42)

# 100 images, 3x32x32 (small for demo)
n_samples = 100
X = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)
y = np.random.randint(0, 10, n_samples)  # 10 classes

# Save as DataFrame-like structure
data_dict = {
    "image": list(X),
    "label": y
}

print(f"  Samples: {n_samples}")
print(f"  Image shape: {X[0].shape}")
print(f"  Classes: 10")

# 2. Build simple CNN (like ResNet but smaller)
print("\n🏗️ Building model...")

model = og.model(
    type="classifier",
    layers=[
        og.conv2d(16, kernel_size=3),  # 16 filters
        og.pool(pool_size=2),
        og.conv2d(32, kernel_size=3),
        og.pool(pool_size=2),
        og.flatten(),
        og.dense(64),
        og.dropout(0.3),
        og.dense(10, activation="softmax"),  # 10 classes
    ],
    optimizer=og.adam(lr=0.001),
    loss=og.cross_entropy(),
)

# 3. Create custom dataset
class ImageDataset:
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def get_train_loader(self, batch_size=16):
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(self.X, self.y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_test_loader(self, batch_size=16):
        from torch.utils.data import DataLoader, TensorDataset
        # Simple split: 80% train, 20% test
        n = len(self.X)
        train_size = int(n * 0.8)
        test_dataset = TensorDataset(self.X[train_size:], self.y[train_size:])
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

data = ImageDataset(X, y)

# 4. Fine-tune (train from scratch for demo)
print("\n🚀 Training...")
history = model.train(data, epochs=5, batch_size=16)

# 5. Evaluate
print("\n📈 Evaluating...")
metrics = model.evaluate(data)

# 6. Predict on new image
print("\n🔮 Predicting...")
new_image = np.random.randn(1, 3, 32, 32).astype(np.float32)
result = model.predict(new_image)
predicted_class = np.argmax(result)

print(f"  Predicted class: {predicted_class}")

# 7. Save fine-tuned model
print("\n💾 Saving fine-tuned model...")
model.save("finetuned_model.omg")

print("\n" + "=" * 60)
print("✅ Fine-tune complete!")
print("=" * 60)