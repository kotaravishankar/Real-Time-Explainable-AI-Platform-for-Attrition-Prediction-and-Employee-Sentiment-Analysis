---
title: "Explainable Attrition and Sentiment Platform"
emoji: "📊"
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Real-Time Explainable Deep Learning Platform

Employee **attrition prediction** and **sentiment analysis** in one app.

| Task | Models | Framework |
|------|--------|-----------|
| Attrition | FT-Transformer, Tabular ResNet | PyTorch |
| Sentiment | BiLSTM, CNN-BiGRU-Attention | TensorFlow / Keras |

Attrition uses 12 features from the HR Analytics job-change dataset, label-encoded
and standard-scaled. Sentiment cleans text (lowercase, URL/emoji strip, lemmatise,
stopword removal), tokenises to a vocabulary of 20,000 and pads to length 80.

Select a model, enter inputs, and the app returns class probabilities.
