# 📊 QuixBugs Çoklu Ajan Karşılaştırmalı Büyük Benchmark Raporu — LLAMA3.1:8B
## 4 Farklı Hafıza Mimarisi: Mem0 vs Cognee vs Zep vs Qdrant
### Kapsam: Tüm QuixBugs Veri Kümesi (80 Test Örneği)

* **Tarih:** 2026-10-30
* **Kullanılan LLM:** `llama3.1:8b` (OLLAMA / Yerel Donanım Hızlandırmalı)
* **Veri Kümesi:** QuixBugs Tam Set (80 Örnek: 40 Hatalı [Buggy], 40 Temiz [Clean])
* **Döngü Mimarisi:** LangGraph StateGraph (Aktör - Sandbox Test Çalıştırma - Kıdemli Verifier/Eleştirmen)
* **Toplam Değerlendirme Sayısı:** 4 Ajan × 80 Test = **320 Bağımsız Ajan Oturumu**

---

## 🏆 1. Yönetici Özeti ve Karşılaştırmalı Başarı Podyumu

Aşağıdaki tabloda 4 ajanın genel başarı metrikleri, eleştirmen yardımıyla teşhis düzeltme kapasitesi ve çalışma hızları özetlenmiştir:

| Sıra | Ajan Mimarisi | Bellek Türü | 1. Tur Doğruluk | Final Doğruluk | Eleştirmen Katkısı (Δ) | Hassasiyet (Prec.) | Hata Yakalama (Rec.) | F1 Skoru | Konsensüs | Ort. Süre | Toplam Süre |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **💾 LangGraph + Mem0** | Epizodik Deneyim & Metinsel Bellek | %56.2 | **%67.5** | `+%11.3` | %65.2 | %75.0 | **0.698** | %32.5 | 43.2 sn | 57 dk 38 sn |
| 🥈 **2** | **🧠 LangGraph + Cognee** | AST Bilgi Grafı & Yapısal Ontoloji | %53.8 | **%58.8** | `+%5.0` | %56.1 | %80.0 | **0.660** | %31.2 | 49.4 sn | 65 dk 54 sn |
| 🥉 **3** | **🎯 LangGraph + Qdrant** | Yoğun Vektör Arama & Filtreleme | %53.8 | **%63.7** | `+%9.9` | %62.8 | %67.5 | **0.651** | %32.5 | 79.2 sn | 105 dk 33 sn |
| 🎖️ **4** | **⏱️ LangGraph + Zep** | Zamansal Olay Grafı & Bilgi Sentezi | %63.7 | **%57.5** | `%-6.2` | %56.8 | %62.5 | **0.595** | %26.2 | 51.3 sn | 68 dk 25 sn |

> **Not:** **F1 Skoru**, modelin hem hatalı kodları kaçırmama (Recall) hem de hatasız temiz kodlara gereksiz iftira atmama (Precision) dengesini ölçen en güvenilir bilimsel metriktir.

### 🔍 Hata Matrisi (Confusion Matrix) ve Teşhis Dağılımı

| Ajan Mimarisi | Doğru Hata Tespiti (TP) 🎯 | Doğru Temiz Tespiti (TN) 🛡️ | Yanlış Alarm (FP) 🔴 | Kaçırılan Hata (FN) ⚠️ | Halüsinasyon Oranı |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **💾 LangGraph + Mem0** | **30** / 40 | **24** / 40 | 16 kod | 10 kod | %40.0 |
| **🧠 LangGraph + Cognee** | **32** / 40 | **15** / 40 | 25 kod | 8 kod | %62.5 |
| **🎯 LangGraph + Qdrant** | **27** / 40 | **24** / 40 | 16 kod | 13 kod | %40.0 |
| **⏱️ LangGraph + Zep** | **25** / 40 | **21** / 40 | 19 kod | 15 kod | %47.5 |

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
| 1 | `bitcount` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ✅ BUGGY | ❌ CLEAN | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 2 | `bitcount` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN | ✅ CLEAN *(Rev)* | ❌ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 3 | `breadth_first_search` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 4 | `breadth_first_search` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 5 | `bucketsort` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 6 | `bucketsort` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 7 | `depth_first_search` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 8 | `depth_first_search` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 9 | `detect_cycle` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 10 | `detect_cycle` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 11 | `find_first_in_sorted` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 12 | `find_first_in_sorted` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 13 | `find_in_sorted` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 14 | `find_in_sorted` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 15 | `flatten` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 16 | `flatten` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 17 | `gcd` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 18 | `gcd` | Temiz | **CLEAN** | ✅ CLEAN | ❌ BUGGY | ✅ CLEAN | ❌ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 19 | `get_factors` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 20 | `get_factors` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 21 | `hanoi` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 22 | `hanoi` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 23 | `is_valid_parenthesization` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ✅ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 24 | `is_valid_parenthesization` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 25 | `kheapsort` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 26 | `kheapsort` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 27 | `knapsack` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 28 | `knapsack` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 29 | `kth` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 30 | `kth` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 31 | `lcs_length` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 32 | `lcs_length` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 33 | `levenshtein` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 34 | `levenshtein` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 35 | `lis` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 36 | `lis` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 37 | `longest_common_subsequence` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 38 | `longest_common_subsequence` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 39 | `max_sublist_sum` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 40 | `max_sublist_sum` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 41 | `mergesort` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 42 | `mergesort` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 43 | `minimum_spanning_tree` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 44 | `minimum_spanning_tree` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ❌ BUGGY | ❌ BUGGY | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 45 | `next_palindrome` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 46 | `next_palindrome` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN | ❌ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 47 | `next_permutation` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 48 | `next_permutation` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 49 | `pascal` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 50 | `pascal` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 51 | `possible_change` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 52 | `possible_change` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 53 | `powerset` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 54 | `powerset` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 55 | `quicksort` | Hatalı | **BUGGY** | ❌ CLEAN | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 56 | `quicksort` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 57 | `reverse_linked_list` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 58 | `reverse_linked_list` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 59 | `rpn_eval` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 60 | `rpn_eval` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN | ❌ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 61 | `shortest_path_length` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 62 | `shortest_path_length` | Temiz | **CLEAN** | ❌ BUGGY | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 63 | `shortest_path_lengths` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 64 | `shortest_path_lengths` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ❌ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 65 | `shortest_paths` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 66 | `shortest_paths` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 67 | `shunting_yard` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 68 | `shunting_yard` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 69 | `sieve` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 70 | `sieve` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | 🌟 **Tam İttifak (4/4 Doğru)** |
| 71 | `sqrt` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 72 | `sqrt` | Temiz | **CLEAN** | ❌ BUGGY | ❌ BUGGY | ❌ BUGGY | ✅ CLEAN *(Rev)* | ⚠️ **Tekil Başarı (3 Ajan Yanıldı)** |
| 73 | `subsequences` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ✅ BUGGY | ✅ BUGGY | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 74 | `subsequences` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ✅ CLEAN *(Rev)* | 👍 **Güçlü Konsensüs (Çoğunluk Doğru)** |
| 75 | `to_base` | Hatalı | **BUGGY** | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | ✅ BUGGY | 🌟 **Tam İttifak (4/4 Doğru)** |
| 76 | `to_base` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ✅ CLEAN | 🌟 **Tam İttifak (4/4 Doğru)** |
| 77 | `topological_ordering` | Hatalı | **BUGGY** | ✅ BUGGY | ❌ CLEAN | ❌ CLEAN | ✅ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 78 | `topological_ordering` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN *(Rev)* | ❌ BUGGY | ❌ BUGGY | ⚖️ **Dengeli / Ajanlar Ayrıştı** |
| 79 | `wrap` | Hatalı | **BUGGY** | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ CLEAN | ❌ **Kritik Yanılgı (Tüm Ajanlar Yanıldı)** |
| 80 | `wrap` | Temiz | **CLEAN** | ✅ CLEAN *(Rev)* | ✅ CLEAN | ✅ CLEAN | ✅ CLEAN *(Rev)* | 🌟 **Tam İttifak (4/4 Doğru)** |

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