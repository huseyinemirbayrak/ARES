# ARES: Adversarial Self-Play for Software Analysis (Tak-Çıkar / Plug-and-Play)

**ARES**, yazılım analizi, teknik borç (technical debt) tespiti ve otomatik kod onarımı için geliştirilmiş açık kaynaklı, **%100 Tak-Çıkar (Plug-and-Play)** bir **Adversarial Self-Play (Çekişmeli Öz-Oyun)** altyapısıdır.

---

## Temel Yetenekler (Core Capabilities)

1. **Tak-Çıkar (Plug-and-Play) Veriseti Adaptörü (`PlugAndPlayDatasetAdapter`)**:
   - Yarın herhangi bir dış verisetini (`.json`, `.jsonl`, `.yaml`, `.csv` formatında) sisteme doğrudan dahil edebilirsiniz (`--dataset` ve `--exam-dataset`).
   - Yeni ekleyeceğiniz verisetinde **önceden tanımlanmış saldırı mutasyonları (`candidate_mutations`), fonksiyon adı (`function_name`) veya beklenen çıktı değerleri bulunmak zorunda değildir!**
   - Sadece Python fonksiyon kodunu (`code` / `clean_code` / `solution`) ve `assert` ifadelerini (veya `input`/`output` çiftlerini) vermeniz yeterlidir:
     - [`PlugAndPlayDatasetAdapter`](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/ARES/ares/datasets/adapter.py) fonksiyon adını AST üzerinden otomatik bulur, `assert fn(x) == y` satırlarını deterministik test senaryolarına çevirir.
     - [`ASTSurgicalMutator`](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/ARES/ares/agents/ast_mutator.py) herhangi bir Python fonksiyonunun AST ağacını analiz ederek **1 veya 2 testi kıracak** sinsi mutasyonları otomatik üretir ve doğrular.
     - [`ASTSurgicalRepairer`](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/ARES/ares/agents/ast_mutator.py) değişken isimlerinden bağımsız genel AST dönüşümleriyle (ve yerel LLM ile) herhangi bir fonksiyondaki hatayı onarır.

2. **İsteğe Bağlı (On-Demand) Taşınabilir Cognee Hafıza Export / Import / Merge (`CogneeMemoryManager`)**:
   - Hafızayı istediğiniz anda **`.json`** (tek dosya Knowledge Graph) veya **`.zip`** (Knowledge Graph JSON + SQLite veritabanı + `episodes.jsonl` tam arşiv paketi) olarak dışa aktarabilir (`--export-only` veya `--export-memory`) ve içe aktarabilirsiniz (`--import-only` veya `--import-memory`).
   - **İki Farklı İçe Aktarma Modu (`--import-mode`)**:
     - `replace`: Dışarıdan verilen hafıza paketini birebir yükler.
     - `merge`: Farklı bir verisetinde veya farklı bir yerel LLM'de eğitilmiş hafıza paketini mevcut hafızanın üzerine **birleştirir (merge)**!
   - **Rol Bazlı Dışa Aktarma (`--export-role`)**: Sadece `attacker`, sadece `defender` veya `both` (her ikisi) hafızasını dışa aktarabilirsiniz.
   - **Hafıza İnceleme (`--inspect-memory`)**: Herhangi bir `.json` veya `.zip` hafıza paketinin içindeki öğrenilmiş hata sınıflarını, düğüm/kenar sayılarını ve doğrulanmış fonksiyon onarımlarını tablo halinde inceler.

3. **Ana Organizasyon Şefi — `LangGraph` (`ARESLangGraphOrchestrator`)**:
   - `select_problem_and_memory` $\rightarrow$ `attacker_turn` $\rightarrow$ `defender_turn` $\rightarrow$ `referee_evaluation` $\rightarrow$ `update_cognee_memory` $\rightarrow$ (`periodic_external_exam` veya sonraki tur).

4. **%100 Deterministik Hakem (`DeterministicReferee`)**:
   - Her problemdeki 4–5 deterministik testi çalıştırır. Saldırgan **sadece 1 veya 2 testi** bozarsa geçerli saldırı (`valid_subtle_attack`) sayılır; savunmacı kodu onarıp tüm testleri geçirirse turu kazanır.

---

## Yeni Bir Veriseti Nasıl Dahil Edilir? (Tak-Çıkar Veriseti Formatları)

`examples/sample_plug_and_play_dataset.jsonl` dosyasında görebileceğiniz gibi, en sade haliyle sadece `id`, `code` ve `tests` içeren bir `.jsonl`, `.json`, `.yaml` veya `.csv` dosyası hazırlamanız yeterlidir:

```json
{
  "id": "CUSTOM-001",
  "code": "def find_peak_altitude(altitudes: list[int]) -> int:\n    if not altitudes:\n        return 0\n    peak = altitudes[0]\n    for a in altitudes[1:]:\n        if a > peak:\n            peak = a\n    return peak\n",
  "tests": [
    "assert find_peak_altitude([10, 50, 30]) == 50",
    "assert find_peak_altitude([5, 12, 8, 20]) == 20",
    "assert find_peak_altitude([42]) == 42",
    "assert find_peak_altitude([-30, -10, -50, -20]) == -10",
    "assert find_peak_altitude([]) == 0"
  ]
}
```

---

## Komut Satırı (CLI) Kullanım Rehberi

### 1. Yeni Bir Verisetini Doğrulama ve Öz-Oyuna Sokma (Tak-Çıkar)
```powershell
# Yeni verisetini doğrula (temiz kod testleri + otomatik AST mutasyon sentezi kontrolü):
.\.venv\Scripts\python.exe run_ares.py --validate-dataset .\examples\sample_plug_and_play_dataset.jsonl

# Yeni verisetiyle öz-oyun ve dış sınav çalıştır:
.\.venv\Scripts\python.exe run_ares.py --dataset .\examples\sample_plug_and_play_dataset.jsonl --turns 15 --exam-interval 5
```

### 2. Hafızayı İsteğe Bağlı Dışa Aktarma (Export) ve İnceleme (Inspect)
```powershell
# Tur çalıştırmadan mevcut hafızayı anında .zip veya .json olarak dışa aktar:
.\.venv\Scripts\python.exe run_ares.py --export-only .\memory_store\ares_portable_brain.zip

# Sadece Defender hafızasını dışa aktar:
.\.venv\Scripts\python.exe run_ares.py --export-only .\memory_store\defender_only.json --export-role defender

# Dışa aktarılmış bir .zip veya .json hafıza paketinin içeriğini incele:
.\.venv\Scripts\python.exe run_ares.py --inspect-memory .\memory_store\ares_portable_brain.zip
```

### 3. Hafızayı İsteğe Bağlı İçe Aktarma (Import: Replace veya Merge)
```powershell
# Dışarıdan bir hafıza paketini mevcut hafızayla BİRLEŞTİR (merge):
.\.venv\Scripts\python.exe run_ares.py --import-only .\memory_store\ares_portable_brain.zip --import-mode merge

# Kaydedilmiş bir hafızayı yükleyip yeni bir veriseti üzerinde çalıştır ve güncellenmiş hafızayı kaydet:
.\.venv\Scripts\python.exe run_ares.py --dataset .\examples\sample_plug_and_play_dataset.jsonl --import-memory .\memory_store\ares_portable_brain.zip --import-mode merge --turns 10 --export-memory .\memory_store\updated_brain.zip
```

### 4. Python Kod İçinden (SDK) Tak-Çıkar Kullanım
```python
from ares.datasets import load_problems
from ares.memory import CogneeMemoryManager
from ares.orchestrator import ARESLangGraphOrchestrator

# 1. İstediğiniz verisetini takın (.json, .jsonl, .yaml, .csv)
custom_problems = load_problems("examples/sample_plug_and_play_dataset.jsonl")

# 2. Hafızayı yükleyin veya birleştirin (replace / merge)
memory = CogneeMemoryManager(store_dir="memory_store")
memory.import_portable_bundle("memory_store/ares_portable_brain.zip", mode="merge")

# 3. LangGraph öz-oyununu başlatın
orchestrator = ARESLangGraphOrchestrator(problems=custom_problems, memory=memory)
scoreboard = orchestrator.run_self_play(num_turns=10, exam_interval=5)

# 4. İstediğiniz anda hafızayı .zip veya .json olarak dışa aktarın
memory.export_portable_bundle("memory_store/yeni_hafiza_paketi.zip", role="both")
```
