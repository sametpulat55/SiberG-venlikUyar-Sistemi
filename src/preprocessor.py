import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class NslKddPreprocessor:
    def __init__(self):
        # Neural Network ve Boosting ortaklığı için Standart Ölçekleme
        self.scaler = StandardScaler()
        self.train_columns = None
        self.encoders = {}
        
        # Kategorik Sütunlar
        self.categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
        # Gürültü Sütunları
        self.drop_cols = ['difficulty_level', 'num_outbound_cmds']

    def process_data(self, df, is_training=True):
        df = df.copy()

        # 1. TEMİZLİK: Gürültüyü At
        for col in self.drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # 2. KRİTİK ADIM: KOPYA VERİLERİ SİL (Sadece Eğitimde)
        # Bu işlem modelin "ezberlemesini" engeller!
        if is_training:
            initial_len = len(df)
            df.drop_duplicates(inplace=True)
            print(f"    [TEMİZLİK] {initial_len - len(df)} adet kopya (duplicate) satır silindi.")

        # 3. Label Ayırma
        y = None
        if 'class' in df.columns:
            df['label'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
            y = df['label']
            df.drop(['class', 'label'], axis=1, inplace=True)

        # 4. Logaritmik Dönüşüm (Sayısal Dengesizliği Düzelt)
        # 0 ile 500 milyon arasındaki farkı kapatır.
        numeric_cols = ['src_bytes', 'dst_bytes', 'duration', 'count', 'srv_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # 5. ORDINAL ENCODING (En Saf Hali)
        # One-Hot yerine Ordinal kullanıyoruz. 
        # Çünkü HistGradientBoosting sayısal kategorileri çok sever.
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str) # Garanti olsun
                if is_training:
                    # Bilinmeyenleri -1 olarak kodlayacak şekilde ayarla
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    df[col] = enc.fit_transform(df[[col]])
                    self.encoders[col] = enc
                else:
                    enc = self.encoders.get(col)
                    if enc:
                        df[col] = enc.transform(df[[col]])
                        # Bilinmeyen (-1) değerleri en sık görülen (0) ile değiştirme, bırak -1 kalsın.
                        # Model bunu "Bilinmeyen Saldırı" olarak algılayacaktır.

        # 6. Sütun Eşitleme
        if is_training:
            self.train_columns = df.columns.tolist()
        else:
            for col in self.train_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.train_columns]

        # 7. Ölçekleme
        # Kategorik sütunlara DOKUNMA! Sadece sayısalları ölçekle.
        # Bu sayede model "HTTP"nin "TCP"den büyük olmadığını anlar.
        num_cols_to_scale = [c for c in self.train_columns if c not in self.categorical_cols]
        
        if is_training:
            df[num_cols_to_scale] = self.scaler.fit_transform(df[num_cols_to_scale])
        else:
            df[num_cols_to_scale] = self.scaler.transform(df[num_cols_to_scale])

        # 8. Label Geri Ekle
        if y is not None:
            df['label'] = y.values

        return df