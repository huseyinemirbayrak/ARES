# ARES: Adversarial Self-Play for Software Analysis

**ARES**, yazılım analizi, teknik borç (technical debt) tespiti ve otomatik kod onarımı için geliştirilmiş açık kaynaklı, **Tak-Çıkar (Plug-and-Play)** bir **Adversarial Self-Play (Çekişmeli Öz-Oyun)** altyapısıdır.

---

## Temel Mimari (Core Architecture)

1. **5 Bölümlü Yapılandırılmış Prompt Sistemi (`StructuredPromptConfig`)**:
   - Attacker ve Defender ajanları YAML üzerinden parametre kontrollü 5 bölümlü promptlarla (`Task`, `Recipe`, `Output`, `Validation`, `Examples`) yönetilir ([attacker_prompt.yaml](file:///c:/Users/LENOVO/.gemini/antigravity/scratch/ARES/configs/attacker_prompt.yaml), [defender_prompt.yaml](file:///c:/Users/LENOVO/.gemini/antigravity/scratch/ARES/configs/defender_prompt.yaml)).
   - `--blind-attacker` / `--blind-defender` bayraklarıyla testlerin görünürlüğü kapatılıp açılabilir; `--strict-llm` ile saf LLM performansı ölçülür.

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
   - 1–2 test bozma (subtle attack) kuralını uygular, deterministik ödül + ELO günceller ve `ground_truth_source` (`curated` / `ast_diff` / `attacker_claim`) bilgisini kaydeder.

5. **Tak-Çıkar Veriseti Adaptörü (`PlugAndPlayDatasetAdapter`)**:
   - `.json`, `.jsonl`, `.yaml`, `.csv` dosyalarının yanında yerleşik benchmark alias'larını destekler: `quixbugs` (40 metot), `agent_green` (12 metot), `cleanvul` (5 metot), `all_new` (57 metot).

---

## Kurulum (Installation)

```powershell
# Temel kurulum (LangGraph + Deterministic Referee + Local Episodic Graph)
uv sync

# Gerçek Cognee SDK desteği ile kurulum (opsiyonel extra)
uv sync --extra cognee
# veya: pip install -e ".[cognee]"
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

### 1. Öz-Oyun Çalıştırma (Yerel Ollama, LM Studio/vLLM veya OpenAI-Uyumlu API)
```powershell
# Temiz başlangıç (--fresh-memory) + Yerel Ollama + Saf LLM + Blind Mode
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --turns 15 --exam-interval 5 --llm-mode ollama --attacker-model qwen2.5-coder:7b --defender-model qwen2.5-coder:7b --strict-llm --blind-attacker --blind-defender

# Gerçek Cognee SDK ile (Ollama + nomic-embed-text)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --memory-backend cognee --embedding-model nomic-embed-text --cognify-interval 3 --turns 15 --exam-interval 5

# Herhangi bir OpenAI-uyumlu yerel veya bulut LLM (LM Studio, vLLM, OpenAI, OpenRouter, Groq vb.)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --llm-mode openai_compatible --base-url http://localhost:1234/v1 --attacker-model qwen2.5-coder-7b --defender-model qwen2.5-coder-7b --strict-llm

# Bulut LLM + OpenAI embedding (boyut otomatik 1536 olarak çözümlenir)
.\.venv\Scripts\python.exe run_ares.py --fresh-memory --base-url https://api.openai.com/v1 --api-key $env:OPENAI_API_KEY --attacker-model gpt-4o-mini --defender-model gpt-4o-mini --memory-backend cognee --embedding-model text-embedding-3-small
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
- `--memory-backend {auto,cognee,local_graph}`: Hafıza arka ucu seçimi.
- `--embedding-provider`, `--embedding-model`, `--embedding-dimensions`, `--cognee-search-type`, `--cognify-interval`: Gerçek Cognee ayarları.
- `--llm-mode {auto,ollama,openai_compatible,adaptive_local}`, `--base-url`, `--api-key`, `--llm-timeout`, `--num-ctx`: LLM sağlayıcı ve bağlam penceresi ayarları.
- `--strict-llm`, `--blind-attacker`, `--blind-defender`: Bilimsel değerlendirme ve kapalı-kutu test ayarları.
