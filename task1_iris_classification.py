# ============================================================
# TASK 1: Iris Flower Classification
# CodeAlpha Data Science Internship
# Student: Zoha Yousaf | ID: CA/DF1/83644
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('Iris.csv')
print("=" * 55)
print("        IRIS FLOWER CLASSIFICATION")
print("=" * 55)
print(f"\nDataset Shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['Species'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── 2. PREPARE FEATURES ──────────────────────────────────────
# Drop the Id column (not a feature)
df = df.drop(columns=['Id'])

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 3. TRAIN MODEL ───────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 4. EVALUATE ──────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 55}")
print(f"  Model Accuracy : {acc * 100:.2f}%")
print(f"{'=' * 55}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=le.classes_))

# ── 5. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Flower Classification — Analysis', fontsize=16, fontweight='bold')

colors = {'Iris-setosa': '#e74c3c',
          'Iris-versicolor': '#2ecc71',
          'Iris-virginica': '#3498db'}

# Plot 1 – Sepal scatter
ax = axes[0, 0]
for species, color in colors.items():
    mask = df['Species'] == species
    ax.scatter(df.loc[mask, 'SepalLengthCm'],
               df.loc[mask, 'SepalWidthCm'],
               label=species, color=color, alpha=0.7, s=60)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_title('Sepal Length vs Width')
ax.legend(fontsize=8)

# Plot 2 – Petal scatter
ax = axes[0, 1]
for species, color in colors.items():
    mask = df['Species'] == species
    ax.scatter(df.loc[mask, 'PetalLengthCm'],
               df.loc[mask, 'PetalWidthCm'],
               label=species, color=color, alpha=0.7, s=60)
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Petal Length vs Width')
ax.legend(fontsize=8)

# Plot 3 – Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix  (Accuracy: {acc*100:.2f}%)')
ax.tick_params(axis='x', rotation=15)

# Plot 4 – Feature importance
ax = axes[1, 1]
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=ax, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
ax.set_title('Feature Importance')
ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('iris_classification_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] iris_classification_results.png")

# ── 6. SAMPLE PREDICTION ─────────────────────────────────────
print("\n--- Sample Prediction ---")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred   = model.predict(sample)
print(f"Input  : SepalLen=5.1, SepalWid=3.5, PetalLen=1.4, PetalWid=0.2")
print(f"Predicted Species : {le.inverse_transform(pred)[0]}")
print("\nTask 1 Complete!")
