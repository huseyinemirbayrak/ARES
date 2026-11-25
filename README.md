# ARES: Adversarial Self-Play for Software Analysis

**ARES**, yazılım analizi, teknik borç (technical debt) tespiti ve otomatik kod onarımı için geliştirilmiş açık kaynaklı, **Tak-Çıkar (Plug-and-Play)** bir **Adversarial Self-Play (Çekişmeli Öz-Oyun)** altyapısıdır.

---

## Temel Mimari (Core Architecture)

1. **5 Bölümlü Yapılandırılmış Prompt Sistemi (`StructuredPromptConfig`)**:
   - Attacker ve Defender ajanları YAML üzerinden parametre kontrollü 5 bölümlü promptlarla (`Task`, `Recipe`, `Output`, `Validation`, `Examples`) yönetilir ([attacker_prompt.yaml](configs/prompts/attacker_prompt.yaml), [defender_prompt.yaml](configs/prompts/defender_prompt.yaml)).
   - **Testleri Gör, Sonuçları Görme (`--hide-test-results`, varsayılan: AÇIK):** İki taraf da fonksiyon kodunu ve birim test tanımlarını (`call_expr == expected_output`) görür; ancak test çalıştırma sonuçlarını (kaç testin kaldığı, hangi testlerin patladığı, hata mesajları) hiçbir aşamada görmez.
   - `--blind-attacker` / `--blind-defender` (`--no-blind-*`) bayraklarıyla test tanımları da tamamen gizlenebilir; `--strict-llm` varsayılan olarak açıktır (LLM'siz AST heuristic baseline için `--no-strict-llm` kullanılır).

2. **Gerçek LangGraph Orkestratörü (`ARESLangGraphOrchestrator`)**:
   - Gerçek `langgraph.graph.StateGraph` + `MemorySaver` checkpoint yapısıyla çalışır:
     `select_problem_and_memory` -> `attacker_turn` -> `defender_turn` -> `referee_evaluation` -> `generate_reflexion` -> `update_memory` -> (`periodic_external_exam` | sonraki tur | `END`).

3. **Çift Arka Uçlu Hafıza Yönetimi (`CogneeMemoryManager` / `ARESMemoryManager`)**:
   - `--memory-backend cognee`: Gerçek `cognee` Python SDK (`cognee.add`, `cognee.cognify`, `cognee.search`, `cognee.prune`) kullanır; kurulu/erişilebilir değilse sahtelemeden açık hata verir.
   - `--memory-backend local_graph`: Dürüst yerel SQLite + JSON + AST/Unified-Diff episodik graf deposu (sıfır dış API çağrısı).
   - `--memory-backend auto` (varsayılan): Gerçek Cognee hazırsa onu kullanır, değilse `local_graph` moduna geçip aktif arka ucu açıkça raporlar.
   - On-demand `.json` / `.zip` Export, Import (`replace` / `merge`), rol bazlı export (`--export-role`) ve inceleme (`--inspect-memory`) destekler.

4. **%100 Deterministik, Subprocess-İzole Hakem (`DeterministicReferee`)**:
   - Kodları temiz ortam değişkenleriyle izole alt süreçte çalıştırır (timeout öldürme + AST güvenlik filtresi).
   - **1–2 Test Bozma Kuralı:** Attacker en az 1, en fazla 2 testi bozarak ince (subtle) bir bug üretmelidir (0 test bozan no-op, yorum değişikliği veya >2 test bozan yıkıcı mutasyonlar geçersizdir). Geçerli 1–2 test bozan bug'da Defender tüm testlerden (%100) geçerse **Defender kazanır**, en az 1 test kalırsa **Attacker kazanır**.

5. **Ana Test (`external_exam_benchmark.json`), Önce/Sonra Gelişim Raporu ve Doğrulanmış Bug Veriseti Üretimi**:
   - Oyun başlamadan önce (**Turn 0**) sabit ana test seti (`ares/datasets/external_exam_benchmark.json`) üzerinde Defender'ın bug tespit, sınıflandırma ve tam onarım başarısı ölçülür; self-play bittikten sonra (**Turn N**) aynı test tekrar koşulup `outputs/pre_post_comparison.md` ve `outputs/pre_post_comparison.json` dosyalarında problem bazlı gelişim (`GAINED`, `REGRESSED`, `ALWAYS_SOLVED`, `STILL_UNSOLVED`) raporlanır.
   - Oyun boyunca Attacker'ın ürettiği ve Hakem tarafından **gerçekten 1–2 test kırdığı doğrulanan** tüm benzersiz bug'lar (hem Defender'ı yenenler hem Defender'ın yakaladıkları) AST tekilleştirmesiyle `generated_datasets/` klasörüne (`generated_bugs.jsonl`, `generated_exam_benchmark.json`, `dataset_manifest.json`) kaydedilir.

6. **Tak-Çıkar Veriseti Adaptörü (`PlugAndPlayDatasetAdapter`)**:
   - `.json`, `.jsonl`, `.yaml`, `.csv` dosyalarının yanında yerleşik benchmark alias'larını destekler: `quixbugs` (40 metot), `agent_green` (12 metot), `cleanvul` (5 metot), `all_new` (57 metot).

---

## Kurulum (Installation)

```powershell
# Temel kurulum (LangGraph + Deterministic Referee + Local Episodic Graph)
uv sync

# Gerçek Cognee SDK desteği ile kurulum (opsiyonel extra)
uv sync --extra cognee
# veya: pip install -e ".[cognee]"
# (Not: `requirements.txt` dosyası Cognee dahil tam kurulum kilit listesini içerir.)
```

---

## Yeni Bir Veriseti Nasıl Dahil Edilir? (Tak-Çıkar Formatı)

`examples/sample_plug_and_play_dataset.jsonl` örneğindeki gibi sadece `id`, `code` ve `tests` içeren bir dosya yeterlidir:

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

### 1. Ana Deney Akışı: Turn 0 (Pre) -> Self-Play -> Turn N (Post) + Bug Veriseti Üretimi
```powershell
# Standart bilimsel deney: İki taraf sadece test tanımlarını görür (--hide-test-results varsayılan AÇIK),
# Turn 0 ve Turn 15'te external_exam_benchmark.json koşulur, doğrulanmış bug'lar generated_datasets/ altına yazılır:
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --turns 15 --exam-interval 15 --llm-mode ollama --attacker-model qwen2.5-coder:7b --defender-model qwen2.5-coder:7b

# Üretilen Önce/Sonra gelişim raporunu ve üretilen bug verisetini doğrula:
Get-Content .\outputs\pre_post_comparison.md
.\.venv\Scripts\python.exe run_ares.py --validate-dataset .\generated_datasets\generated_exam_benchmark.json

# Üretilen doğrulanmış bug'ları bir sonraki deneyde doğrudan sınav seti olarak kullan:
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --exam-dataset .\generated_datasets\generated_exam_benchmark.json --turns 10 --exam-interval 10

# (Opsiyonel) Turn N sınavı bittikten sonra yeni doğrulanmış bug'ları hedef sınav JSON'una da birleştir:
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --turns 15 --exam-interval 15 --append-to-exam-benchmark

# Gerçek Cognee SDK ile (Ollama + nomic-embed-text)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --memory-backend cognee --embedding-model nomic-embed-text --cognify-interval 3 --turns 15 --exam-interval 5

# Herhangi bir OpenAI-uyumlu yerel veya bulut LLM (LM Studio, vLLM, OpenAI, OpenRouter, Groq vb.)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --llm-mode openai_compatible --base-url http://localhost:1234/v1 --attacker-model qwen2.5-coder-7b --defender-model qwen2.5-coder-7b --strict-llm

# Bulut LLM + OpenAI embedding (boyut otomatik 1536 olarak çözümlenir)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --base-url https://api.openai.com/v1 --api-key $env:OPENAI_API_KEY --attacker-model gpt-4o-mini --defender-model gpt-4o-mini --memory-backend cognee --embedding-model text-embedding-3-small

# Tam kapalı-kutu (test tanımları da gizli) mod:
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --turns 15 --exam-interval 5 --blind-attacker --blind-defender

# Çevrimdışı (LLM'siz) AST heuristic baseline testi
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --turns 10 --exam-interval 5 --llm-mode adaptive_local
```

### 2. Veriseti Doğrulama ve Benchmark Seçimi
```powershell
# Yeni verisetini doğrula:
.\.venv\Scripts\python.exe run_ares.py --validate-dataset .\examples\sample_plug_and_play_dataset.jsonl

# Yerleşik benchmark alias'larıyla çalıştır (quixbugs, agent_green, cleanvul, all_new):
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --dataset quixbugs --exam-dataset cleanvul --turns 10 --exam-interval 5
```

### 3. Hafıza Export / Import / Merge / Inspect
```powershell
# Mevcut hafızayı .zip veya .json olarak dışa aktar:
.\.venv\Scripts\python.exe run_ares.py --export-only .\memory_store\ares_portable_brain.zip

# Sadece Defender hafızasını dışa aktar:
.\.venv\Scripts\python.exe run_ares.py --export-only .\memory_store\defender_only.json --export-role defender

# Hafıza paketini incele:
.\.venv\Scripts\python.exe run_ares.py --inspect-memory .\memory_store\ares_portable_brain.zip

# Mevcut hafızayla birleştir (merge):
.\.venv\Scripts\python.exe run_ares.py --import-only .\memory_store\ares_portable_brain.zip --import-mode merge
```

### 4. Önemli CLI Seçenekleri
- `--fresh-memory`: Turn-0 cold-start ölçümü için önceki hafıza deposunu temizler.
- `--hide-test-results` / `--no-hide-test-results`: Test tanımlarını iki tarafa da gösterip test çalıştırma sonuçlarını gizler (`game_config.yaml` varsayılanı: açık).
- `--generated-dataset-dir`, `--save-generated-bugs` / `--no-save-generated-bugs`, `--append-to-exam-benchmark`: Hakem onaylı 1–2 test bozan bug veriseti kaydı ve opsiyonel sınav birleştirme ayarları.
- `--memory-backend {auto,cognee,local_graph}`: Hafıza arka ucu seçimi.
- `--embedding-provider`, `--embedding-model`, `--embedding-dimensions`, `--cognee-search-type`, `--cognify-interval`: Gerçek Cognee ayarları.
- `--llm-mode {auto,ollama,openai_compatible,adaptive_local}`, `--base-url`, `--api-key`, `--llm-timeout`, `--num-ctx`: LLM sağlayıcı ve bağlam penceresi ayarları.
- `--strict-llm` / `--no-strict-llm`, `--blind-attacker` / `--no-blind-attacker`, `--blind-defender` / `--no-blind-defender`: Bilimsel değerlendirme ve kapalı-kutu test ayarları (`--strict-llm` varsayılan olarak açıktır).
