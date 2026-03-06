import pandas as pd
import os
import sys

# ---------------------------------------------------------
# 1. AYARLAR: src klasörünü Python'a tanıtıyoruz
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

# Kendi modüllerimizi çağırıyoruz
try:
    from preprocessor import NslKddPreprocessor
    # DÜZELTME: Dosya ismin 'model.py' olduğu için buradan çağırıyoruz
    from model import NslKddClassifier 
except ImportError as e:
    print(f"HATA: Modüller 'src' klasöründe bulunamadı veya hatalı.\nDetay: {e}")
    print("Lütfen 'src' klasöründe 'preprocessor.py' ve 'model.py' olduğundan emin olun.")
    sys.exit(1)

# ---------------------------------------------------------
# 2. VERİ YÜKLEME FONKSİYONU
# ---------------------------------------------------------
def load_data():
    """
    Data klasöründeki txt dosyalarını okur ve sütun isimlerini ekler.
    """
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    # NSL-KDD Sütun İsimleri
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty_level'
    ]

    print(f"Veriler okunuyor... ({data_dir})")
    
    try:
        df_train = pd.read_csv(train_path, names=columns)
        df_test = pd.read_csv(test_path, names=columns)
        print("Veri yükleme başarılı.")
        return df_train, df_test

    except FileNotFoundError as e:
        print(f"\n[KRİTİK HATA] Dosya bulunamadı: {e}")
        return None, None

# ---------------------------------------------------------
# 3. ANA ÇALIŞTIRMA BLOĞU
# ---------------------------------------------------------
if __name__ == "__main__":
    # A) Veriyi Yükle
    train_data, test_data = load_data()

    if train_data is not None:
        # B) Ön İşleme (Preprocessing)
        print("\n--> Veri işleniyor (Sayısallaştırma & Etiketleme)...")
        
        # Preprocessor sınıfını başlat
        preprocessor = NslKddPreprocessor()
        
        # Eğitim setini işle
        train_processed = preprocessor.process_data(train_data, is_training=True)
        # Test setini işle
        test_processed = preprocessor.process_data(test_data, is_training=False)
        
        print("--> İşlem tamamlandı!")
        
        # C) MODEL HAZIRLIĞI
        ai_model = NslKddClassifier()

        # D) KONTROL PANELİ
        while True:
            print("\n" + "="*40)
            print("--- KONTROL PANELİ ---")
            print("1. Modeli EĞİT (Train)")
            print("2. Modeli TEST ET (Evaluate)")
            print("3. Veri İstatistiklerini Göster")
            print("4. ÇIKIŞ")
            print("="*40)
            
            secim = input("Yapmak istediğiniz işlem (1-4): ")
            
            if secim == '1':
                # --- EĞİTİM ---
                # 'label' sütunu hedefimiz (y), diğerleri özellik (X)
                y_train = train_processed['label']
                X_train = train_processed.drop('label', axis=1)
                
                ai_model.train(X_train, y_train)
            
            elif secim == '2':
                # --- TEST ---
                # DÜZELTME: 'if not' hatası giderildi
                if not ai_model.is_trained:
                    print("\n[UYARI] Lütfen önce 1'e basarak modeli eğitin!")
                    continue
                
                y_test = test_processed['label']
                X_test = test_processed.drop('label', axis=1)
                
                # Sütun sırasını garantiye al
                train_cols = train_processed.drop('label', axis=1).columns
                X_test = X_test.reindex(columns=train_cols, fill_value=0)
                
                ai_model.evaluate(X_test, y_test)
            
            elif secim == '3':
                # --- İSTATİSTİKLER ---
                print(f"\n[BİLGİ] Eğitim Seti Boyutu: {train_processed.shape}")
                print(f"[BİLGİ] Test Seti Boyutu:   {test_processed.shape}")
                if 'label' in train_processed.columns:
                    print("\n--- Sınıf Dağılımı (Eğitim) ---")
                    print(train_processed['label'].value_counts())
            
            elif secim == '4':
                print("Programdan çıkılıyor. İyi günler!")
                break
            
            else:
                print("Geçersiz seçim, lütfen tekrar deneyin.")