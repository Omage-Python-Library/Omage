"""
Omage — Advanced Example: Custom Model Training
Dataset: Breast Cancer (569 samples, 30 features, binary classification)
"""

import omage as og

print("=" * 55)
print("  Omage — Advanced Custom Model Example")
print("  Dataset: Breast Cancer (binary classification)")
print("=" * 55)

# ------------------------------------------------------------------ #
# 1. Load & prepare data
# ------------------------------------------------------------------ #
print("\n📂 Loading dataset...")
data = og.load("tests/breast_cancer.csv")
print(f"   Loaded: {data}")

data.set_target("target")  # explicit target column
data.clean()
data.normalize(0, 1)
data.shuffle(seed=42)
data.batch(32)
data.split(0.8)
print("   Preprocessed: clean → normalize → shuffle → split(80/20)")

# ------------------------------------------------------------------ #
# 2. Build model
# ------------------------------------------------------------------ #
print("\n🧠 Building model...")
ai = og.model(
    type="classifier",
    layers=[
        og.dense(128),
        og.dropout(0.3),
        og.dense(64),
        og.dropout(0.2),
        og.dense(32),
        og.dense(2, activation="softmax"),   # 2 classes: benign / malignant
    ],
    optimizer=og.adam(lr=0.001),
    loss=og.cross_entropy(),
)
print(f"   {ai}")

# ------------------------------------------------------------------ #
# 3. Register event callbacks
# ------------------------------------------------------------------ #
ai.on("onTrain",  lambda: print("\n🚀 Training started!"))
ai.on("onFinish", lambda: print("\n✅ Training complete!"))
ai.on("onError",  lambda: print("\n❌ An error occurred!"))

# ------------------------------------------------------------------ #
# 4. Train — with checkpoints + early stopping
# ------------------------------------------------------------------ #
history = ai.train(
    data,
    epochs=30,
    checkpoint_path="examples/advanced/checkpoints/breast_cancer.omg",
    checkpoint_every=10,
    early_stopping_patience=5,
)

# ------------------------------------------------------------------ #
# 5. Evaluate
# ------------------------------------------------------------------ #
print("\n📊 Evaluation:")
metrics = ai.evaluate(data)
print(f"   Accuracy : {metrics['accuracy']:.2f}%")
print(f"   Loss     : {metrics['loss']:.4f}")

# ------------------------------------------------------------------ #
# 6. Save final model
# ------------------------------------------------------------------ #
ai.save("breast_cancer_model.omg")
print("\n💾 Model saved → breast_cancer_model.omg")

# ------------------------------------------------------------------ #
# 7. Training summary
# ------------------------------------------------------------------ #
print("\n📈 Training Summary:")
print(f"   Epochs trained  : {len(history['loss'])}")
print(f"   Best train loss : {min(history['loss']):.4f}")
print(f"   Best accuracy   : {max(history['accuracy']):.2f}%")

if history["val_loss"]:
    print(f"   Best val loss   : {min(history['val_loss']):.4f}")
    print(f"   Best val acc    : {max(history['val_accuracy']):.2f}%")

print("\n" + "=" * 55)
print("  Done! 🎉")
print("=" * 55)
