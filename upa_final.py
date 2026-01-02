# upa_final.py
# پروژه UPA - نسخه ساده و قابل اجرا

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("UPA PROJECT - Breast Cancer Wisconsin")
print("=" * 60)

# 1. بارگذاری داده
try:
    df = pd.read_csv('data.csv')
    print("✓ Dataset loaded")
    print(f"  Samples: {len(df)}, Features: {len(df.columns) - 2}")
except:
    print("✗ Error: data.csv not found!")
    exit()

# 2. آماده‌سازی داده
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"\nClass distribution:")
print(f"  Benign (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
print(f"  Malignant (1): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

# 3. تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. آموزش مدل
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 5. ارزیابی
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Sensitivity (Recall): {sensitivity:.3f}")
print(f"Specificity:          {specificity:.3f}")
print(f"Accuracy:             {accuracy:.3f}")
print(f"AUC:                  {auc:.3f}")
print(f"\nConfusion Matrix:")
print(f"[[{tn:3d}  {fp:3d}]")
print(f" [{fn:3d}  {tp:3d}]]")

# 6. پیاده‌سازی ساده UPA
print("\n" + "=" * 60)
print("UPA SIMULATION")
print("=" * 60)


class SimpleUPA:
    def __init__(self):
        self.mapping = None

    def align_predictions(self, source_preds, target_preds):
        # هیستوگرام ساده
        source_sorted = np.sort(source_preds)
        target_sorted = np.sort(target_preds)

        # نگاشت صدک‌ها
        percentiles = np.linspace(0, 1, 100)
        source_quantiles = np.quantile(source_sorted, percentiles)
        target_quantiles = np.quantile(target_sorted, percentiles)

        # ایجاد تابع نگاشت
        self.mapping = np.polyfit(target_quantiles, source_quantiles, 3)

    def apply_alignment(self, predictions):
        if self.mapping is None:
            return predictions

        # استفاده از چندجمله‌ای درجه ۳ برای نگاشت
        p = np.poly1d(self.mapping)
        aligned = p(predictions)
        return np.clip(aligned, 0, 1)


# شبیه‌سازی shift
print("\nSimulating acquisition shift...")
X_test_shifted = X_test_scaled * 1.2 + 0.1  # تغییر ساده
y_pred_proba_shifted = model.predict_proba(X_test_shifted)[:, 1]
y_pred_shifted = (y_pred_proba_shifted >= 0.5).astype(int)

cm_shifted = confusion_matrix(y_test, y_pred_shifted)
tn_s, fp_s, fn_s, tp_s = cm_shifted.ravel()

sens_shifted = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
spec_shifted = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0

print(f"\nAfter shift:")
print(f"  Sensitivity: {sens_shifted:.3f} (Δ: {sens_shifted - sensitivity:+.3f})")
print(f"  Specificity: {spec_shifted:.3f} (Δ: {spec_shifted - specificity:+.3f})")

# اعمال UPA
print("\nApplying UPA correction...")
upa = SimpleUPA()
upa.align_predictions(y_pred_proba, y_pred_proba_shifted)
y_pred_aligned = upa.apply_alignment(y_pred_proba_shifted)
y_pred_corrected = (y_pred_aligned >= 0.5).astype(int)

cm_corrected = confusion_matrix(y_test, y_pred_corrected)
tn_c, fp_c, fn_c, tp_c = cm_corrected.ravel()

sens_corrected = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
spec_corrected = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0

print(f"\nAfter UPA:")
print(f"  Sensitivity: {sens_corrected:.3f} (Δ: {sens_corrected - sens_shifted:+.3f})")
print(f"  Specificity: {spec_corrected:.3f} (Δ: {spec_corrected - spec_shifted:+.3f})")

# 7. نمودارها
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# نمودار ۱: توزیع پیش‌بینی‌ها
axes[0, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Benign', density=True)
axes[0, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Malignant', density=True)
axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0, 0].set_xlabel('Probability')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Original Predictions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# نمودار ۲: بعد از shift
axes[0, 1].hist(y_pred_proba_shifted[y_test == 0], bins=30, alpha=0.6, label='Benign', density=True, color='orange')
axes[0, 1].hist(y_pred_proba_shifted[y_test == 1], bins=30, alpha=0.6, label='Malignant', density=True, color='orange')
axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0, 1].set_xlabel('Probability')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('After Acquisition Shift')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# نمودار ۳: بعد از UPA
axes[0, 2].hist(y_pred_aligned[y_test == 0], bins=30, alpha=0.6, label='Benign', density=True, color='green')
axes[0, 2].hist(y_pred_aligned[y_test == 1], bins=30, alpha=0.6, label='Malignant', density=True, color='green')
axes[0, 2].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0, 2].set_xlabel('Probability')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('After UPA Correction')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# نمودار ۴: مقایسه Sensitivity/Specificity
labels = ['Original', 'After Shift', 'After UPA']
sensitivities = [sensitivity, sens_shifted, sens_corrected]
specificities = [specificity, spec_shifted, spec_corrected]

x = np.arange(len(labels))
width = 0.35

axes[1, 0].bar(x - width / 2, sensitivities, width, label='Sensitivity', color='blue', alpha=0.7)
axes[1, 0].bar(x + width / 2, specificities, width, label='Specificity', color='green', alpha=0.7)
axes[1, 0].set_xlabel('Scenario')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Sensitivity vs Specificity')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# نمودار ۵: بهبود عملکرد
improvement_sens = [0, sens_shifted - sensitivity, sens_corrected - sens_shifted]
improvement_spec = [0, spec_shifted - specificity, spec_corrected - spec_shifted]

axes[1, 1].bar(labels, improvement_sens, color=['gray', 'red', 'blue'], alpha=0.7)
axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].set_ylabel('Δ Sensitivity')
axes[1, 1].set_title('Sensitivity Changes')
axes[1, 1].grid(True, alpha=0.3)

# نمودار ۶: ماتریس اشتباه
axes[1, 2].text(0.3, 0.9, 'Confusion Matrices', fontsize=12, ha='center')
axes[1, 2].text(0.2, 0.6, f'Original:\nTN={tn}\nFP={fp}\nFN={fn}\nTP={tp}', fontsize=10)
axes[1, 2].text(0.5, 0.6, f'After Shift:\nTN={tn_s}\nFP={fp_s}\nFN={fn_s}\nTP={tp_s}', fontsize=10)
axes[1, 2].text(0.8, 0.6, f'After UPA:\nTN={tn_c}\nFP={fp_c}\nFN={fn_c}\nTP={tp_c}', fontsize=10)
axes[1, 2].axis('off')

plt.suptitle('UPA: Automatic Correction of Performance Drift\nBreast Cancer Wisconsin Dataset', fontsize=13,
             fontweight='bold')
plt.tight_layout()
plt.savefig('upa_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Results saved to 'upa_results.png'")

# 8. گزارش نهایی
print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"\nDataset: Breast Cancer Wisconsin (Diagnostic)")
print(f"Model: Random Forest (n_estimators=100)")
print(f"Test Samples: {len(X_test)}")

print(f"\n{'Metric':<15} {'Original':<10} {'After Shift':<12} {'After UPA':<10} {'Improvement':<12}")
print("-" * 60)
print(
    f"{'Sensitivity':<15} {sensitivity:<10.3f} {sens_shifted:<12.3f} {sens_corrected:<10.3f} {sens_corrected - sens_shifted:+.3f}")
print(
    f"{'Specificity':<15} {specificity:<10.3f} {spec_shifted:<12.3f} {spec_corrected:<10.3f} {spec_corrected - spec_shifted:+.3f}")
print(
    f"{'Accuracy':<15} {accuracy:<10.3f} {(tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s):<12.3f} {(tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c):<10.3f} {((tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)) - ((tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s)):+.3f}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

plt.show()