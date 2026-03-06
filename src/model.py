# Modern sklearn importları
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import pandas as pd
import numpy as np

class NslKddClassifier:
    def __init__(self):
        print("--> NİHAİ ÇÖZÜM: Stacking (Neural Network + Boosting)...")
        
        # 1. Uzman: Histogram Gradient Boosting (Ağaç Tabanlı)
        # Kategorik verilerle harika çalışır.
        learner_1 = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=10,
            l2_regularization=1.0,
            random_state=42
        )
        
        # 2. Uzman: Yapay Sinir Ağı (Beyin Tabanlı)
        # Ağaçların kaçırdığı karmaşık ilişkileri yakalar.
        learner_2 = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=300,
            random_state=42
        )

        # PATRON: Lojistik Regresyon (Final Karar Verici)
        final_learner = LogisticRegression()

        # STACKING SİSTEMİ
        self.model = StackingClassifier(
            estimators=[
                ('hgb', learner_1),
                ('mlp', learner_2)
            ],
            final_estimator=final_learner,
            n_jobs=-1 # Tüm işlemcileri kullan
        )
        self.is_trained = False

    def train(self, X_train, y_train):
        print(f"--> Stacking Eğitimi Başladı (Temizlenmiş Veri: {X_train.shape})...")
        print("    (Bu işlem birden fazla modeli eğittiği için 1-2 dakika sürebilir)")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("--> EĞİTİM TAMAMLANDI!")

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            print("[HATA] Model eğitilmedi.")
            return

        print("--> Test analizi yapılıyor...")
        
        # Tahminleri al
        y_pred = self.model.predict(X_test)
        
        # Skoru hesapla
        acc = accuracy_score(y_test, y_pred)
        
        print("\n" + "★"*50)
        print(f"★  STACKING DOĞRULUK ORANI: %{acc * 100:.2f}  ★")
        print("★"*50)
        
        print("\n--- DETAYLI RAPOR ---")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Saldırı']))
        
        print("\n--- KARMAŞIKLIK MATRİSİ ---")
        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=['Gerçek: Normal', 'Gerçek: Saldırı'], columns=['Tahmin: Normal', 'Tahmin: Saldırı'])
        print(df_cm)

    def save_model(self, path='model.pkl'):
        if self.is_trained:
            joblib.dump(self.model, path)
            print(f"Model kaydedildi: {path}")