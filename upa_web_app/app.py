# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # برای سرور بدون GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import io
import base64
import os
from datetime import datetime
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ایجاد پوشه‌های لازم
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)


class UPAWebSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_dataset(self):
        """بارگذاری دیتاست"""
        try:
            self.df = pd.read_csv('data.csv')
            print(f"Dataset loaded: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def preprocess_data(self):
        """پیش‌پردازش داده"""
        if self.df is None:
            return False

        # حذف ستون‌های غیرضروری
        self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

        # کدگذاری هدف
        self.df['diagnosis'] = self.df['diagnosis'].map({'B': 0, 'M': 1})

        # جدا کردن ویژگی‌ها و هدف
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']

        # تقسیم داده
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # استانداردسازی
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return True

    def train_model(self):
        """آموزش مدل"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(self.X_train, self.y_train)
        return True

    def predict_single(self, features):
        """پیش‌بینی برای یک نمونه"""
        if self.model is None or self.scaler is None:
            return None

        # تبدیل به آرایه و استانداردسازی
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        # پیش‌بینی
        probability = self.model.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        diagnosis = "Malignant" if prediction == 1 else "Benign"

        return {
            'probability': float(probability),
            'prediction': int(prediction),
            'diagnosis': diagnosis,
            'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
        }

    def simulate_upa_experiment(self, shift_severity=0.4):
        """شبیه‌سازی آزمایش UPA"""
        if self.model is None:
            return None

        # پیش‌بینی روی داده تست اصلی
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # محاسبه متریک‌های اولیه
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        original_metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'auc': roc_auc_score(self.y_test, y_pred_proba)
        }

        # شبیه‌سازی shift
        X_test_shifted = self.X_test * (1 + shift_severity) + shift_severity * 0.2
        y_pred_proba_shifted = self.model.predict_proba(X_test_shifted)[:, 1]
        y_pred_shifted = (y_pred_proba_shifted >= 0.5).astype(int)

        cm_shifted = confusion_matrix(self.y_test, y_pred_shifted)
        tn_s, fp_s, fn_s, tp_s = cm_shifted.ravel()

        shifted_metrics = {
            'sensitivity': tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0,
            'specificity': tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0,
            'accuracy': (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s),
            'auc': roc_auc_score(self.y_test, y_pred_proba_shifted)
        }

        # شبیه‌سازی UPA (ساده)
        # در واقعیت اینجا الگوریتم UPA کامل پیاده‌سازی می‌شود
        correction_factor = shift_severity * 0.8
        y_pred_proba_corrected = y_pred_proba_shifted * (1 - correction_factor)
        y_pred_corrected = (y_pred_proba_corrected >= 0.5).astype(int)

        cm_corrected = confusion_matrix(self.y_test, y_pred_corrected)
        tn_c, fp_c, fn_c, tp_c = cm_corrected.ravel()

        corrected_metrics = {
            'sensitivity': tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0,
            'specificity': tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0,
            'accuracy': (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c),
            'auc': roc_auc_score(self.y_test, y_pred_proba_corrected)
        }

        # ایجاد نمودار
        fig = self.create_performance_plot(original_metrics, shifted_metrics, corrected_metrics)

        return {
            'original': original_metrics,
            'shifted': shifted_metrics,
            'corrected': corrected_metrics,
            'plot': fig,
            'confusion_matrices': {
                'original': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                'shifted': {'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s)},
                'corrected': {'tn': int(tn_c), 'fp': int(fp_c), 'fn': int(fn_c), 'tp': int(tp_c)}
            }
        }

    def create_performance_plot(self, original, shifted, corrected):
        """ایجاد نمودار عملکرد"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # تنظیمات کلی
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        labels = ['Original', 'After Shift', 'After UPA']

        # نمودار 1: Sensitivity & Specificity
        ax1 = axes[0, 0]
        x = np.arange(2)
        width = 0.25

        ax1.bar(x - width, [original['sensitivity'], original['specificity']],
                width, label='Original', color=colors[0], alpha=0.8)
        ax1.bar(x, [shifted['sensitivity'], shifted['specificity']],
                width, label='After Shift', color=colors[1], alpha=0.8)
        ax1.bar(x + width, [corrected['sensitivity'], corrected['specificity']],
                width, label='After UPA', color=colors[2], alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels(['Sensitivity', 'Specificity'])
        ax1.set_ylabel('Value')
        ax1.set_title('Sensitivity & Specificity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # نمودار 2: Accuracy & AUC
        ax2 = axes[0, 1]
        metrics_acc = [original['accuracy'], shifted['accuracy'], corrected['accuracy']]
        metrics_auc = [original['auc'], shifted['auc'], corrected['auc']]

        x2 = np.arange(len(labels))
        ax2.bar(x2 - 0.2, metrics_acc, 0.4, label='Accuracy', color='#4ECDC4', alpha=0.8)
        ax2.bar(x2 + 0.2, metrics_auc, 0.4, label='AUC', color='#FF6B6B', alpha=0.8)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Value')
        ax2.set_title('Accuracy & AUC Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # نمودار 3: Improvement
        ax3 = axes[1, 0]
        improvement_sens = corrected['sensitivity'] - shifted['sensitivity']
        improvement_spec = corrected['specificity'] - shifted['specificity']
        improvement_acc = corrected['accuracy'] - shifted['accuracy']

        improvements = [improvement_sens, improvement_spec, improvement_acc]
        improvement_labels = ['Δ Sensitivity', 'Δ Specificity', 'Δ Accuracy']
        colors_improve = ['green' if x > 0 else 'red' for x in improvements]

        ax3.bar(improvement_labels, improvements, color=colors_improve, alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Improvement')
        ax3.set_title('Improvement from UPA')
        ax3.grid(True, alpha=0.3)

        # نمودار 4: Text Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = (
            f"UPA Performance Analysis\n"
            f"=======================\n"
            f"Shift Recovery:\n"
            f"  • Sensitivity: {improvement_sens:+.3f}\n"
            f"  • Specificity: {improvement_spec:+.3f}\n"
            f"  • Accuracy: {improvement_acc:+.3f}\n\n"
            f"Final Performance:\n"
            f"  • Sensitivity: {corrected['sensitivity']:.3f}\n"
            f"  • Specificity: {corrected['specificity']:.3f}\n"
            f"  • AUC: {corrected['auc']:.3f}"
        )

        ax4.text(0.1, 0.5, summary_text, fontsize=10,
                 verticalalignment='center', fontfamily='monospace')

        plt.suptitle('UPA: Automatic Performance Drift Correction', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # تبدیل نمودار به عکس
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()

        # کدگذاری base64
        plot_url = base64.b64encode(img.getvalue()).decode()

        return f"data:image/png;base64,{plot_url}"


# ایجاد نمونه سیستم
upa_system = UPAWebSystem()


@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')


@app.route('/about')
def about():
    """صفحه درباره"""
    return render_template('about.html')


@app.route('/api/dataset_info', methods=['GET'])
def get_dataset_info():
    """اطلاعات دیتاست"""
    if upa_system.df is None:
        if not upa_system.load_dataset():
            return jsonify({'error': 'Failed to load dataset'}), 500

    info = {
        'samples': int(upa_system.df.shape[0]),
        'features': int(upa_system.df.shape[1] - 1),  # minus target
        'benign_count': int((upa_system.df['diagnosis'] == 'B').sum()),
        'malignant_count': int((upa_system.df['diagnosis'] == 'M').sum()),
        'features_list': upa_system.df.columns.tolist()[:10]  # اولین 10 ویژگی
    }

    return jsonify(info)


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """مقداردهی اولیه سیستم"""
    try:
        # بارگذاری دیتاست
        if not upa_system.load_dataset():
            return jsonify({'error': 'Failed to load dataset'}), 500

        # پیش‌پردازش
        if not upa_system.preprocess_data():
            return jsonify({'error': 'Failed to preprocess data'}), 500

        # آموزش مدل
        if not upa_system.train_model():
            return jsonify({'error': 'Failed to train model'}), 500

        return jsonify({
            'success': True,
            'message': 'System initialized successfully',
            'training_samples': len(upa_system.X_train),
            'test_samples': len(upa_system.X_test)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """پیش‌بینی برای داده ورودی"""
    try:
        data = request.json

        # بررسی وجود مدل
        if upa_system.model is None:
            return jsonify({'error': 'Model not trained. Please initialize system first.'}), 400

        # استخراج ویژگی‌ها
        if 'features' in data:
            features = data['features']
        else:
            # اگر ویژگی‌ها جداگانه ارسال شده
            features = []
            for i in range(1, 31):  # 30 ویژگی
                feature_name = f'feature_{i}'
                if feature_name in data:
                    features.append(float(data[feature_name]))
                else:
                    features.append(0.0)  # مقدار پیش‌فرض

        # پیش‌بینی
        result = upa_system.predict_single(features)

        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # اضافه کردن زمان
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """اجرای آزمایش UPA"""
    try:
        data = request.json
        shift_severity = data.get('shift_severity', 0.4)

        # بررسی وجود مدل
        if upa_system.model is None:
            return jsonify({'error': 'Model not trained. Please initialize system first.'}), 400

        # اجرای آزمایش
        results = upa_system.simulate_upa_experiment(shift_severity)

        if results is None:
            return jsonify({'error': 'Experiment failed'}), 500

        # اضافه کردن اطلاعات اضافی
        results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results['shift_severity'] = shift_severity

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """آپلود دیتاست جدید"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.endswith('.csv'):
            # ذخیره فایل
            filename = 'data.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # بارگذاری دیتاست جدید
            upa_system.df = pd.read_csv(filepath)

            return jsonify({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'samples': int(upa_system.df.shape[0]),
                'features': int(upa_system.df.shape[1])
            })
        else:
            return jsonify({'error': 'Only CSV files are allowed'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results_page():
    """صفحه نتایج"""
    return render_template('results.html')


if __name__ == '__main__':
    print("Starting UPA Web Application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)