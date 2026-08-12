# Fruit Classification ML Model

A deep learning image classifier that identifies fruits (Apple, Banana, Orange, Strawberry) from photos, built with transfer learning and deployed as an interactive Streamlit dashboard.

## Key Features
- **Model:** EfficientNetV2B0 (transfer learning, fine-tuned)
- **Classes:** Apple, Banana, Orange, Strawberry
- **Test Accuracy:** 100% on held-out test split
- **Generalization Focus:** Background-diversification augmentation pipeline (background removal + synthetic background compositing + rotation/lighting/blur augmentation) to overcome the common failure mode of models trained on clean/white-background datasets collapsing on real-world photos.
- **Dashboard:** Streamlit app for live image upload and prediction with confidence scores.

## Project Structure
├── deploy/
│ ├── app.py # Streamlit dashboard
│ ├── fruit_classifier.keras # Trained model
│ └── class_labels.json # Class label mapping
├── fruit_classifier.keras
├── class_labels.json
├── training_curves.png # Accuracy/loss curves
└── README.md


## Why Background Augmentation?
Standard fruit datasets (e.g. Fruits-360) are collected in controlled studio settings with plain white backgrounds. Models trained directly on this data tend to overfit to background cues and fail on real-world photos (colorful backgrounds, varied lighting, different angles). This project addresses that by:
1. Removing the original background from training images (rembg / U2-Net)
2. Compositing the isolated fruit onto diverse synthetic backgrounds (solid colors, gradients, noise, wood textures, kitchen-like surfaces)
3. Applying rotation, brightness/contrast, and blur augmentation

## How to Run the Dashboard
```bash
pip install streamlit tensorflow pillow numpy
cd deploy
streamlit run app.py
```

## Model Architecture
- Backbone: EfficientNetV2B0 (ImageNet pretrained)
- Two-phase training: frozen backbone head training, then fine-tuning top layers at low learning rate
- Output: Softmax over 4 classes with confidence scores
