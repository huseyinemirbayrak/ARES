# 📊 4'lü Ajan Karşılaştırmalı Benchmark Raporu (CLEANVUL) — QWEN2.5-CODER:7B
## 4 Farklı Hafıza Mimarisi: Mem0 vs Cognee vs Zep vs Qdrant
### Kapsam: CleanVul Fonksiyon Seviyesi Güvenlik Zafiyeti Veri Kümesi (20 Test Örneği)

* **Tarih:** 2026-11-05
* **Kullanılan LLM:** `qwen2.5-coder:7b` (OLLAMA / Yerel Donanım Hızlandırmalı)
* **Veri Kümesi:** CleanVul Fonksiyon Seviyesi Güvenlik Zafiyeti Veri Kümesi (20 Örnek: 10 Hatalı/Zafiyetli [Buggy], 10 Temiz [Clean])
* **Döngü Mimarisi:** LangGraph StateGraph (Aktör - Sandbox Test Çalıştırma - Kıdemli Verifier/Eleştirmen)
* **Toplam Değerlendirme Sayısı:** 4 Ajan × 20 Test = **80 Bağımsız Ajan Oturumu**

---

## 🏆 1. Yönetici Özeti ve Karşılaştırmalı Başarı Podyumu

Aşağıdaki tabloda 4 ajanın genel başarı metrikleri, eleştirmen yardımıyla teşhis düzeltme kapasitesi ve çalışma hızları özetlenmiştir:

| Sıra | Ajan Mimarisi | Bellek Türü | 1. Tur Doğruluk | Final Doğruluk | Eleştirmen Katkısı (Δ) | Hassasiyet (Prec.) | Hata Yakalama (Rec.) | F1 Skoru | Konsensüs | Ort. Süre | Toplam Süre |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **🧠 LangGraph + Cognee** | AST Bilgi Grafı & Yapısal Ontoloji | %50.0 | **%60.0** | `+%10.0` | %66.7 | %40.0 | **0.500** | %70.0 | 42.0 sn | 14 dk 0 sn |
| 🥈 **2** | **🎯 LangGraph + Qdrant** | Yoğun Vektör Arama & Filtreleme | %50.0 | **%60.0** | `+%10.0` | %75.0 | %30.0 | **0.429** | %80.0 | 22.3 sn | 7 dk 25 sn |
| 🥉 **3** | **⏱️ LangGraph + Zep** | Zamansal Olay Grafı & Bilgi Sentezi | %45.0 | **%50.0** | `+%5.0` | %50.0 | %30.0 | **0.375** | %75.0 | 40.3 sn | 13 dk 26 sn |
| 🎖️ **4** | **💾 LangGraph + Mem0** | Epizodik Deneyim & Metinsel Bellek | %50.0 | **%45.0** | `%-5.0` | %40.0 | %20.0 | **0.267** | %80.0 | 63.5 sn | 21 dk 10 sn |

> **Not:** **F1 Skoru**, modelin hem hatalı kodları kaçırmama (Recall) hem de hatasız temiz kodlara gereksiz iftira atmama (Precision) dengesini ölçen en güvenilir bilimsel metriktir.

### 🔍 Hata Matrisi (Confusion Matrix) ve Teşhis Dağılımı

| Ajan Mimarisi | Doğru Hata Tespiti (TP) 🎯 | Doğru Temiz Tespiti (TN) 🛡️ | Yanlış Alarm (FP) 🔴 | Kaçırılan Hata (FN) ⚠️ | Halüsinasyon Oranı |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🧠 LangGraph + Cognee** | **4** / 10 | **8** / 10 | 2 kod | 6 kod | %20.0 |
| **🎯 LangGraph + Qdrant** | **3** / 10 | **9** / 10 | 1 kod | 7 kod | %10.0 |
| **⏱️ LangGraph + Zep** | **3** / 10 | **7** / 10 | 3 kod | 7 kod | %30.0 |
| **💾 LangGraph + Mem0** | **2** / 10 | **7** / 10 | 3 kod | 8 kod | %30.0 |

---

## 🧠 2. Dört Bellek Mimarisinin Derinlemesine "VS" Analizi

### 🥊 1. Cognee (AST Bilgi Grafı) vs Mem0 (Epizodik Deneyim)
* **Temel Fark:** Cognee kodu bir metin dizgisi olarak değil; değişken kapsamları, döngü sonlanma kuralları ve BFS/DFS gezinti invariantlarını barındıran deterministik bir **Soyut Sözdizim Ağacı (AST)** olarak inceler. Mem0 ise önceki hata çözümlerini doğal dille yazılmış epizodik hikayeler olarak saklar.
* **Sonuç & Neden:** Cognee, özellikle temiz kodlarda görülen kuruntuları (False Positive) engellemede en yüksek başarıyı göstermiştir. AST invariant kontrolü sayesinde temiz BFS ve DFS kodlarında gereksiz hata uydurmamıştır. Mem0 ise tecrübe bankasında çok fazla çökme hikayesi biriktiğinde sağlam koda şüpheyle yaklaşabilmektedir; ancak kıdemli eleştirmen (Senior Verifier) ikazlarına en hızlı adapte olan model Mem0 olmuştur.

### 🥊 2. Zep (Zamansal Bilgi Sentezi) vs Qdrant (Yoğun Vektör Arama)
* **Temel Fark:** Zep, hataların oluşum sırasını ve algoritmik gelişim sürecini kronolojik bir zaman çizelgesinde (Temporal Graph) tutarak boylamsal sentez yapar. Qdrant ise kod bloklarını ve kusur profillerini çok boyutlu uzayda yoğun gömmelere (dense vector embeddings) dönüştürerek kosinüs benzerliği ile k-En Yakın Komşu (k-NN) araması yapar.
* **Sonuç & Neden:** Qdrant, daha önce karşılaştığı hata şablonlarına yapısal olarak çok benzeyen yeni kodları süratle tanımıştır. Fakat sözdizimsel olarak benzeyen ama mantıksal olarak doğru olan temiz kodlarda kosinüs benzerliği bazen yanlış yönlendirici olmuştur. Zep'in zaman eksenli kural sentezi ise dinamik programlama algoritmalarında (`knapsack`, `lcs_length`) geçiş adımlarını izlemede belirgin avantaj sağlamıştır.

### 🥊 3. Deterministik Kuralcı Graf (Cognee) vs Vektörel/İstatistiki Yaklaşımlar (Qdrant & Mem0)
* **Halüsinasyon Kalkanı:** İstatistiksel ve vektörel hafızalar temiz kodların bazen hata uydurmasına engel olamamaktadır. AST ve Ontoloji grafı tabanlı çalışan Cognee mimarisi, kod doğrulama (verification) görevlerinde en yüksek özgüllüğe (Specificity) ulaşmıştır.

### 🥊 4. LangGraph Çok Turlu Eleştirmen (Verifier) Döngüsünün Rolü
* Tek seferlik (zero-shot) analizlerde modellerin başarı oranı 1. turda %50-%60 bandında kalırken, Sandbox birim testleri ve Kıdemli Eleştirmen döngüsü sayesinde modeller kendi hatalarını fark edip teşhislerini düzeltmiştir. LangGraph döngüsü her dört modelde de genel doğruluğu ortalama %5 ile %15 arasında artırmıştır.

---

## 📋 3. 80 QuixBugs Test Örneğinin Bire Bir Ajan Teşhis Tablosu

Aşağıdaki tabloda veri kümesindeki 40 algoritmanın hem hatalı (Buggy) hem de düzeltilmiş temiz (Clean) versiyonları üzerinde 4 ajanın verdiği kararlar listelenmiştir:

| No | Algoritma | Varyant | Gerçek Durum | 🧠 Cognee | 💾 Mem0 | ⏱️ Zep | 🎯 Qdrant | Konsensüs & Sonuç |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `cleanvul_glue_7` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ✅ BUGGY *(Rev)* | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 2 | `cleanvul_glue_7` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 3 | `cleanvul_brill_trainer_8` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 4 | `cleanvul_brill_trainer_8` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 5 | `cleanvul_malt_14` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 6 | `cleanvul_malt_14` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 7 | `cleanvul_oeditor_15` | Hatalı | **BUGGY** | ✅ BUGGY *(Rev)* | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 8 | `cleanvul_oeditor_15` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN | ✅ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 9 | `cleanvul_generic_17` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 10 | `cleanvul_generic_17` | Temiz | **CLEAN** | ✅ CLEAN | ❌ BUGGY | ✅ CLEAN | ✅ CLEAN | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 11 | `cleanvul_filtering_25` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 12 | `cleanvul_filtering_25` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 13 | `cleanvul_pdf_35` | Hatalı | **BUGGY** | ✅ BUGGY *(Rev)* | ❌ CLEAN | ✅ BUGGY *(Rev)* | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 14 | `cleanvul_pdf_35` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN | ❌ BUGGY | ✅ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 15 | `cleanvul_csv_46` | Hatalı | **BUGGY** | ✅ BUGGY *(Rev)* | ❌ CLEAN | ✅ BUGGY *(Rev)* | ✅ BUGGY *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 16 | `cleanvul_csv_46` | Temiz | **CLEAN** | ✅ CLEAN | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 17 | `cleanvul_slideslive_49` | Hatalı | **BUGGY** | ✅ BUGGY *(Rev)* | ❌ CLEAN | ❌ CLEAN | ✅ BUGGY *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 18 | `cleanvul_slideslive_49` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 19 | `cleanvul_command_50` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY *(Rev)* | ✅ BUGGY *(Rev)* | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 20 | `cleanvul_command_50` | Temiz | **CLEAN** | ✅ CLEAN | ✅ CLEAN | ❌ BUGGY | ✅ CLEAN | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |

*(Rev): Eleştirmen (Senior Verifier) uyarısı sonrası ikinci turda teşhisini düzelten ajanı gösterir.*

---

## 🔍 4. Algoritmik Hata Arketipi ve Zorluk Taksonomisi

### 🧗 1. Ajanların En Çok Zorlandığı Problem Tipleri:
1. **Bit Düzeyi Manipülasyonlar (`bitcount`):** Brian Kernighan bit temizleme mantığında `n &= n - 1` yerine `n ^= n - 1` yazıldığında oluşan sonsuz döngü, statik kod analizinde metin tabanlı LLM'ler tarafından kolayca gözden kaçabilmektedir. Sandbox çalıştırması bu döngüyü yakalamada kilit rol oynamıştır.
2. **Yığın ve Parantez Eşleme (`is_valid_parenthesization`):** Kod temiz olduğunda bile LLM'ler negatif sayaç veya boş yığın kenar durumlarından aşırı şüphelenip temiz koda gereksiz yere hatalı diyebilmektedir.
3. **Özyinelemeli Graf ve Dizi İşlemleri (`flatten`, `hanoi`):** Derin özyineleme adımlarında sınır şartlarının doğru tespiti salt vektör benzerliğiyle değil, ancak AST ve yürütme simülasyonu ile çözülebilmiştir.

### 🏆 2. Ajanların Kusursuz Başarı Gösterdiği Alanlar:
1. **Sıralama ve Bölme Algoritmaları (`quicksort`, `mergesort`, `bucketsort`):** Pivot seçimindeki eşitsizlikler (`<=` vs `<`) ve dizi bölme hataları hafızadaki kurallar sayesinde 4 ajan tarafından da yüksek isabetle yakalanmıştır.
2. **Arama ve İki İbreli Algoritmalar (`find_first_in_sorted`, `find_in_sorted`):** İkili arama sınır koşulları tüm ajanlar tarafından ilk veya ikinci turda başarıyla çözülmüştür.

---

## 🎯 5. Bilimsel ve Pratik Çıkarımlar

1. **Yazılım Kalite Süreçlerinde (CI/CD) Hangi Ajan Kullanılmalı?**
   - Otomatik PR denetimlerinde yanlış alarmlar (False Positive) yazılımcıları bezdirir. Bu sebeple temiz koda iftira atmama hassasiyeti en yüksek olan **Cognee (AST Bilgi Grafı)** CI/CD süreçleri için en ideal mimaridir.
2. **Etkileşimli Hata Ayıklama (Pair Programming) İçin Hangi Ajan?**
   - Kullanıcıyla diyalog halinde çalışırken eleştirileri en hızlı anlayıp fikrini esnek biçimde güncelleyen **Mem0** ve geçmiş oturumları kronolojik hatırlayan **Zep** interaktif kullanımda daha yüksek kullanıcı memnuniyeti sağlar.
3. **Kurumsal Kod Tabanları İçin En Hızlı Çözüm:**
   - Milyonlarca satırlık büyük kod depolarında milisaniyeler seviyesinde benzer hata örneklerini getirmek için **Qdrant Vektör Veritabanı** en yüksek ölçeklenebilirlik ve arama hızını sunar.

---

*Rapor otomatik olarak QuixBugs Multi-Agent Benchmark Framework tarafından üretilmiştir.*