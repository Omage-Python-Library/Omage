# Changelog

All notable changes to Omage will be documented here.

## [0.1.1-alpha] — 2025

### Added
- Core model builder (`og.model`)
- Data pipeline (`og.load`, `.clean()`, `.normalize()`, `.split()`)
- Layers: `dense`, `dropout`, `conv2d`, `flatten`, `pool`
- Optimizers: `adam`, `sgd`, `rmsprop`
- Losses: `cross_entropy`, `mse`, `bce`, `nll`
- AI Loops: `polish`, `seek`, `flow`, `grow`
- Model Zoo: ResNet-18/50, MobileNetV3, GPT-2, BERT, DistilBERT, YOLOv8
- CLI: `omage version`, `omage run`, `omage compile`, `omage zoo list`
- Progress bars with `tqdm`
- Model checkpointing and early stopping
- Full model save/load (weights + config)
- Validation during training
- GPU auto-detection
- 61 tests on real datasets (Iris, Wine, Breast Cancer)
- GitHub Actions CI/CD
- PyPI-ready packaging

### Known Limitations (coming soon)
- `.omg` transpiler syntax — v0.2.0
- ONNX export — v0.2.0
- Distributed training — v0.3.0
- TensorBoard integration — v0.2.0

## [0.1.0-alpha] — Initial Release
- Basic structure and concept
