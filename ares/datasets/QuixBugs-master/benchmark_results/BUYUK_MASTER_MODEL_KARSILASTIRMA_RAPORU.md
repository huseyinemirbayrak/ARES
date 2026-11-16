# 🏆 QuixBugs Büyük Master Model Karşılaştırma Raporu
## 4 Farklı Yerel LLM × 4 Ajan Bellek Mimarisi (Toplam 1280 Test Oturumu)

* **Oluşturulma Tarihi:** 2026-10-30 02:11:16
* **Donanım:** NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM) + 16 GB RAM
* **Veri Kümesi:** QuixBugs 80 Dengeli Örnek (40 Hatalı + 40 Temiz Kod)
* **Test Edilen Modeller:** Qwen 2.5 Coder 7B, Meta Llama 3.1 8B

---

## 🥇 1. Genel Model Şampiyonlar Ligi Podyumu

Tüm bellek mimarilerinde (Mem0, Cognee, Zep, Qdrant) elde edilen ortalama başarılar:

| Sıra | Model | Geliştirici / Mimari | Ort. Doğruluk | Ort. F1 Skoru | Ort. Hata Yakalama (Recall) | Ort. Hassasiyet (Precision) | Ort. Gecikme | En Başarılı Olduğu Bellek |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | **Qwen 2.5 Coder 7B** | Yerel Ollama | **%67.2** | **0.695** | %75.0 | %64.9 | 38.3 sn | ⏱️ Zep (Zamansal) |
| 🥈 **2** | **Meta Llama 3.1 8B** | Yerel Ollama | **%61.9** | **0.651** | %71.2 | %60.2 | 55.8 sn | 💾 Mem0 (Epizodik) |

---

## 📊 2. Model × Bellek Mimarisi Çapraz Karşılaştırma Matrisi (F1 & Doğruluk)

| Model | 🧠 Cognee (AST Graf) | ⏱️ Zep (Zamansal Graf) | 🎯 Qdrant (Vektör k-NN) | 💾 Mem0 (Epizodik) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 2.5 Coder 7B** | %71.2 (F1: 0.723) | %70.0 (F1: 0.727) | %68.8 (F1: 0.719) | %58.8 (F1: 0.612) |
| **Meta Llama 3.1 8B** | %58.8 (F1: 0.660) | %57.5 (F1: 0.595) | %63.7 (F1: 0.651) | %67.5 (F1: 0.698) |

---

## 📂 3. Bireysel Model Benchmark Raporları

Her modelin 80 QuixBugs testindeki satır satır teşhis tablosu için aşağıdaki özel raporlara bakabilirsiniz:

* 📄 [Qwen 2.5 Coder 7B Özel Benchmark Raporu](BENCHMARK_RAPORU_QWEN2_5_CODER_7B.md)
* 📄 [Meta Llama 3.1 8B Özel Benchmark Raporu](BENCHMARK_RAPORU_LLAMA3_1_8B.md)
* 📄 [Alibaba Qwen 3 8B Özel Benchmark Raporu](BENCHMARK_RAPORU_QWEN3_8B.md)
* 📄 [Alibaba Qwen 3.5 9B Özel Benchmark Raporu](BENCHMARK_RAPORU_QWEN3_5_9B.md)

---

## 🎯 4. Nihai Sonuç ve Bilimsel Çıkarımlar

1. **Kod Hata Tespiti İçin En Güçlü Model ve Bellek Eşleşmesi:** Model ve mimari kombinasyonlarının verimliliği incelendiğinde, AST invariant denetimi ile model akıl yürütmesinin sinerjisi belirleyici olmuştur.

2. **VRAM Verimliliği vs Performans:** 6 GB VRAM sınırlarında tamamen GPU üzerinde çalışan 7B ve 8B modellerin gecikme avantajı, sistem RAM'ine taşan 9B modele kıyasla çok daha yüksek işlem hacmi sağlamıştır.


*Master rapor QuixBugs Çoklu Model Benchmark Orkestratörü tarafından üretilmiştir.*