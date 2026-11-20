# 🧠 QuixBugs 4-Ajanlı Hafıza ve Hata Bulma Karşılaştırma Sistemi
## (LangGraph + Mem0 | LangGraph + Cognee | LangGraph + Zep | LangGraph + Qdrant)

Bu sistem, **QuixBugs** veri kümesindeki 40'tan fazla klasik algoritma hatasını (ve doğru hallerini) analiz etmek, hataları teşhis etmek ve 4 farklı hafıza (memory) mimarisinin hata bulma başarısını tur tur bilimsel olarak karşılaştırmak için geliştirilmiştir.

---

## 📌 1. Kurulan 4 Farklı Ajan Mimarisi

Her ajan temel akış orkestrasyonu için **LangGraph StateGraph** (Döngüsel Aktör-Eleştirmen / Actor-Critic) yapısını kullanır. Ancak her ajanın hafıza (memory) ve bağlam yönetim felsefesi birbirinden tamamen farklıdır:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                LANGGRAPH ORTAK OTONOM DÖNGÜSÜ                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
     [1. Hafıza / Ontoloji / Vektör Çekme]  (Mem0 / Cognee / Zep / Qdrant)
                                                │
                                                ▼
     [2. Analizci (Analyzer) Ajanı]       ---> Kod + Hafıza bilgisiyle teşhis koyar (JSON)
                                                │
                                                ▼
     [3. Test Çalıştırıcı (Sandbox)]      ---> Kodu QuixBugs test girdileriyle izole ortamda dener
                                                │
                                                ▼
     [4. Doğrulayıcı (Verifier) Eleştirmen]--> Test sonucuna ve mantığa göre onaylar veya reddeder
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
               [Reddedildi / Hata Var]                         [Onaylandı / Max Tur]
                        │                                               │
                        ▼                                               ▼
          (Döngü Başına: Düzeltme İsteği)               [5. Deneyimi Hafızaya Kaydet]
                                                                        │
                                                                        ▼
                                                              [6. Nihai Konsensüs]
```

---

### 1️⃣ Ajan 1: LangGraph + Mem0 (Epizodik & Deneyimsel Bellek)
* **Hafıza Türü:** Epizodik (Episodic) ve Anlamsal Uzun Vadeli Hafıza.
* **Nasıl Çalışır?** Ajan bir algoritmayı incelemeden önce geçmiş turlarda karşılaştığı hata deneyimlerini (`Mem0Manager`) sorgular.
  - *Örnek Hatırlama:* "Daha önce `bitcount` üzerinde `n ^= n - 1` görüldüğünde bitler sıfırlanmadığı için sonsuz döngü çıkmıştı. Brian Kernighan kuralı `n &= n - 1` olmalıdır."
  - Analizci bu geçmiş deneyimi okuyarak aynı tuzağa düşmez.
  - Kod doğrulandıktan sonra kazanılan yeni ders Mem0 veri tabanına (`.memory_data/mem0_episodes.json`) kalıcı olarak işlenir.

---

### 2️⃣ Ajan 2: LangGraph + Cognee (AST Bilgi Grafı & Kod Ontolojisi)
* **Hafıza Türü:** Kod Yapısal Ontolojisi & Abstract Syntax Tree (AST) Bilgi Grafı.
* **Nasıl Çalışır?** Kodu metin olarak değil, Python'un soyut sözdizim ağacı (AST) olarak parçalar (`NativeCogneeEngine`).
  - Fonksiyonlar, `While` döngüleri, `For` döngüleri, Bitwise işlemleri (`BitXor`, `BitAnd`) ve veri yapıları graf düğümü olarak haritalanır.
  - **Deterministik Değişmezlik Kuralları (Invariants):**
    - *Kural 1:* While döngüsü içinde sayaç üzerinde `^=` (XOR) varsa `VIOLATES_INVARIANT` uyarısı üretir.
    - *Kural 2:* BFS aramasında döngü içinde `visited.add(...)` çağrısı yoksa döngü patlaması uyarısı üretir.
  - Analizciye LLM halüsinasyonlarını sıfırlayan kesin yapısal graf bilgisi enjekte edilir.

---

### 3️⃣ Ajan 3: LangGraph + Zep (Zamansal Çizge & Kronolojik Gerçek Sentezi)
* **Hafıza Türü:** Zamansal Hafıza (Temporal Memory Graph) ve Kronolojik Olay Çizgesi.
* **Nasıl Çalışır?** Zep, test turları boyunca olayları bir zaman çizelgesi (timeline) olarak kaydeder (`NativeZepTemporalMemory`).
  - Hangi algoritmanın hangi girdide ne zaman çöktüğünü ve turlar içindeki hipotez değişimlerini zamansal gerçekler (synthesized facts) halinde sentezler.
  - *Örnek Sentez:* "(T1) Sıralama algoritmalarında pivot eşitliği ele alınmadığında derin özyineleme hatası meydana gelmektedir."
  - Analizciye bu kronolojik deneyim zinciri sunulur.

---

### 4️⃣ Ajan 4: LangGraph + Qdrant (Yoğun Vektör & Filtreli k-NN Arama)
* **Hafıza Türü:** Yoğun Vektör Uzayı (Dense Vector Embeddings) ve Yük Filtreleme (Payload Filtering).
* **Nasıl Çalışır?** Kod parçalarını 64 boyutlu normalize edilmiş anlamsal vektörlere dönüştürür (`NativeQdrantEngine`).
  - Yeni bir kod geldiğinde vektör uzayında Cosine Similarity ile en yakın `k` adet benzer hata şablonunu ve bunların doğrulanmış çözümlerini çeker.
  - `category` (bitwise, graph, sorting), `is_buggy` ve `language` metaverileriyle filtrelenmiş arama yapar.
  - Doğrulanan yeni analizler anında Qdrant koleksiyonuna (`.memory_data/qdrant_vectors.json`) eklenir.

---

## 🚀 2. Sistemi Çalıştırma ve Test Etme

Sistemi çalıştırmak için 3 kolay yol mevcuttur:

### Seçenek A: Tek Tıkla Windows Batch Dosyası (Önerilen)
Komut satırından veya çift tıklayarak:
```cmd
run_benchmark.bat
```
Özel parametrelerle çalıştırmak için:
```cmd
run_benchmark.bat --limit 10 --rounds 2 --provider ollama
```

### Seçenek B: PowerShell Scripti ile Çalıştırma
```powershell
.\run_benchmark.ps1 -Limit 10 -Provider ollama -Model "qwen2.5-coder:7b"
```

### Seçenek C: Doğrudan Python ile Çalıştırma
```powershell
& "..\..\..\.venv\Scripts\python.exe" run_agents_benchmark.py --provider ollama --limit 10
```

---

## ⚡ 3. Parametreler ve Çalışma Modları

| Parametre | Seçenekler | Varsayılan | Açıklama |
| :--- | :--- | :--- | :--- |
| `--dataset` | `quixbugs`, `cleanvul`, `agent_green`, `agent_green_td`, `all` | `quixbugs` | Test edilecek veri kümesi (QuixBugs, CleanVul, Agent-Green veya hepsi). |
| `--agent` | `all`, `mem0`, `cognee`, `zep`, `qdrant` | `all` | Hangi ajanın çalıştırılacağı (virgülle ayrılabilir). |
| `--provider` | `ollama`, `mock`, `gemini`, `openai` | `ollama` | Kullanılacak yapay zeka sağlayıcısı. |
| `--model` | örn: `qwen2.5-coder:7b` | Otomatik | Kullanılacak model adı. |
| `--limit` | Sayı (örn: `4`, `10`, `20`, `40`) | `10` | Test edilecek kod örneği sayısı (dengeli: %50 hatalı/zafiyetli, %50 temiz). |
| `--rounds` | Sayı (örn: `1`, `2`, `3`) | `2` | Doğrulayıcı reddederse izin verilen maksimum düzeltme turu. |
| `--out` | Dosya adı (örn: `sonuclar.json`) | `quixbugs_benchmark_summary.json` | Çıktı dosyasının adı. |

### Hızlı Örnekler:
1. **CleanVul ve Agent-Green Üzerinde 4'lü Ajan Testini Tek Tıkla Başlatma:**
   ```cmd
   4lu_ajan_testi_baslat.bat
   ```
2. **Sadece CleanVul Veri Kümesinde 4 Ajanı Yarıştırma:**
   ```cmd
   run_benchmark.bat --dataset cleanvul --provider ollama --model qwen2.5-coder:7b --limit 20
   ```
3. **Sadece Agent-Green (VulTrial) Veri Kümesinde 4 Ajanı Yarıştırma:**
   ```cmd
   run_benchmark.bat --dataset agent_green --provider ollama --model qwen2.5-coder:7b --limit 20
   ```
4. **Süper Hızlı Test (GPU veya API Key gerektirmez):**
   ```cmd
   run_benchmark.bat --dataset all --provider mock --limit 12
   ```

---

## 🧪 4. Otomatik Birim & Entegrasyon Testleri

Tüm hafıza motorlarının, sanal kum havuzunun (Sandbox) ve 4 ajanın graf derleme mekanizmalarının çalıştığını doğrulamak için:
```powershell
& "..\..\..\.venv\Scripts\python.exe" test_agents.py
```
*(Bu test; QuixBugs veri kümesi yükleyicisini, sonsuz döngü yakalayan sandbox'ı, 4 hafıza motorunu ve 4 LangGraph ajanını uçtan uca test eder - 13/13 Başarılı).*

---

## 📊 5. Çıktılar ve Raporlama

Benchmark tamamlandığında ekranda Rich kütüphanesiyle renkli bir performans matrisi basılır:
* **R1 Acc:** Ajanın kodu ilk gördüğü andaki doğruluk oranı.
* **Final Acc:** Eleştirmen (Verifier) ve Sandbox test geri bildiriminden sonraki nihai doğruluk.
* **Δ Imp:** Hafıza ve eleştirmen döngüsü sayesinde yanlışın doğruya çevrilme kazancı (%).
* **Precision:** Doğru çalışan koda yanlışlıkla "hata var" dememe başarısı (False Positive direnci).
* **Recall:** Gerçek hataları yakalama oranı (True Positive oranı).
* **F1:** Hassasiyet ve duyarlılığın harmonik ortalaması.
* **Approv:** Doğrulayıcının analizci teşhisine onay verme oranı (Konsensüs).
* **Latency:** Örnek başına ortalama işlem süresi (saniye).

Tüm detaylı loglar ve istatistikler otomatik olarak kaydedilir:
* **JSON Raporu:** `benchmark_results/quixbugs_benchmark_summary.json`
* **CSV Özeti:** `benchmark_results/quixbugs_benchmark_summary.csv`
* **Hafıza Kalıcı Verileri:** `.memory_data/` dizini altında saklanır.
