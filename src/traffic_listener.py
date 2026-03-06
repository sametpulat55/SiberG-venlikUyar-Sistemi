from scapy.all import sniff, IP, TCP, UDP, conf
from scapy.arch.windows import get_windows_if_list
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sys

# --- ADIM 3: AĞ TRAFİĞİ DİNLEME VE VERİ TOPLAMA ---
class NetworkSniffer:
    def __init__(self):
        print("--> [ADIM 3] Ağ Dinleyici Başlatılıyor...")
        
        # Encoder ve Scaler Yükleme
        try:
            self.encoders = joblib.load('models/encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("--> Model dosyaları yüklendi.")
        except:
            print("UYARI: Model dosyaları bulunamadı, ham veri gösterilecek.")
            print("(Önce 'python src/data_processor.py' çalıştırırsanız düzelir.)")
            self.encoders = {}
            self.scaler = None

    def packet_callback(self, packet):
        """Her pakette çalışan fonksiyon"""
        if IP in packet:
            try:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                length = len(packet)
                
                # Ekrana Bas (Gerçek Zamanlı İzleme)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {src_ip} -> {dst_ip} | Len: {length}")
                
            except Exception as e:
                pass

    def select_interface(self):
        """Kullanıcıya aktif ağ kartını seçtirir (Hata Korumalı)"""
        print("\n--- AĞ KARTLARI LİSTESİ ---")
        try:
            interfaces = get_windows_if_list()
        except Exception as e:
            print(f"HATA: Ağ kartları listelenemedi: {e}")
            return None

        valid_indices = []
        for i, iface in enumerate(interfaces):
            # HATA DÜZELTME: IP adresi listesi boşsa patlamaması için kontrol
            ips = iface.get('ips', [])
            if len(ips) > 0:
                ip_addr = ips[0]
            else:
                ip_addr = "Yok"
            
            print(f"[{i}] {iface['name']} (IP: {ip_addr}) - {iface['description']}")
            valid_indices.append(i)
            
        print("---------------------------")
        print("İPUCU: Yanında IP adresi yazan (özellikle Wi-Fi veya Ethernet) kartı seçmelisin.")
        
        while True:
            try:
                val = input("Dinlemek istediğin kartın numarasını gir: ")
                choice = int(val)
                if choice in valid_indices:
                    return interfaces[choice]['name']
                else:
                    print("Geçersiz numara, listedeki numaralardan birini gir.")
            except ValueError:
                print("Lütfen sadece sayı girin.")

    def start_sniffing(self):
        chosen_iface = self.select_interface()
        
        if chosen_iface:
            print(f"\n--> SEÇİLEN KART DİNLENİYOR: {chosen_iface}")
            print("--> Paketlerin akması için internette bir sayfa yenile (Google, YouTube vb.)...")
            print("--> Durdurmak için Ctrl+C bas.")
            try:
                sniff(iface=chosen_iface, filter="ip", prn=self.packet_callback, store=0)
            except Exception as e:
                print(f"Dinleme hatası: {e}")
                print("Lütfen Npcap'in kurulu olduğundan ve yönetici olarak çalıştırdığından emin ol.")

if __name__ == "__main__":
    sniffer = NetworkSniffer()
    sniffer.start_sniffing()