"""
Model Zoo Example — Using pre-trained models
"""

import sys
sys.path.insert(0, r"C:\Users\Zomorrod\OneDrive\Desktop\Omage library")

import omage as og

print("=" * 60)
print("🦁 Omage Model Zoo Demo")
print("=" * 60)

# 1. List available models
print("\n📋 Available models:")
og.list_models()

# 2. List only classifiers
print("\n📋 Classification models:")
og.list_models(filter_type="classifier")

# 3. Load a pre-trained model (would need torchvision)
# model = og.load_model("resnet-18", num_classes=10)

# 4. For now, show how it would work
print("\n💡 Example usage:")
print("  model = og.load_model('resnet-18', num_classes=10)")
print("  model.fine_tune(data, epochs=5)")
print("  model.predict(image)")

print("\n" + "=" * 60)
print("✅ Model Zoo ready!")
print("=" * 60)