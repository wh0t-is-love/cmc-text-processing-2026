# ДЗ №1: маленькая LLM на чувашском

Decoder-only GPT (6,8 млн параметров), обученный с нуля — только претрейн — на
дампе чувашской Википедии (cvwiki). Чувашский: ~800 тыс. носителей, вне топ-50
по численности; из «малых» языков у него лучший по объёму и качеству дамп.

Итог: **val_loss 3.135, перплексия 23.0** — модель пишет связный чувашский текст
в жанре вики-статьи. Полный отчёт с экспериментами и историей багов —
`ОТЧЁТ.md` (и готовый `ОТЧЁТ.pdf`), график обучения — `reports/`.

Репозиторий самодостаточен: код всех шагов, токенизаторы, лучший чекпоинт
и исполненный демо-ноутбук уже внутри. Обучающий дамп скачивается одной
командой (шаг 2 ниже).

## Демо без обучения (5 минут)

Чекпоинт `checkpoints/base/best.pt` и токенизатор уже в репозитории — учить
ничего не нужно.

```bash
git clone <this-repo> && cd llm-on-chuvash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Консольная генерация:

```bash
python -I scripts/generate.py checkpoints/base/best.pt tokenizer/bpe-8192.json "Чăваш Ен" 200
```

Или откройте `notebooks/inference.ipynb` — он сам найдёт чекпоинт и токенизатор,
при наличии CUDA/MPS работает на ускорителе, в Colab сам предложит загрузить
файлы. Внутри: свободные продолжения, сравнение температур и бенчмарк фактов
с перефразировками. Ноутбук исполнен, выводы сохранены.

## Полный цикл с нуля

Все команды из каталога `llm-on-chuvash/`. Нужен Python 3.12+ и около 2 ГБ
на диске (дамп + корпус + токены).

```bash
# 1. Окружение (как в шаге «Демо»)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Дамп чувашской Википедии (~39 МиБ)
mkdir -p data/raw
curl -L -o data/raw/cvwiki-latest-pages-articles.xml.bz2 \
    https://dumps.wikimedia.org/cvwiki/latest/cvwiki-latest-pages-articles.xml.bz2

# 3. Корпус: потоковый экстрактор на iterparse, чистка викимаркапа (~15 с)
python -I scripts/extract_corpus.py data/raw/cvwiki-latest-pages-articles.xml.bz2 \
    data/corpus/cvwiki.txt

# 4. Токенизатор: byte-level BPE, словарь 8192 (~20 с)
python -I scripts/train_tokenizer.py data/corpus/cvwiki.txt tokenizer 8192

# 5. Токенизация корпуса в uint16 .npy под mmap (~10 с)
python -I scripts/tokenize_data.py data/corpus/cvwiki.txt tokenizer/bpe-8192.json \
    data/tokenized 0.01

# 6. Обучение: 5 эпох, 9295 шагов
#    Apple Silicon (MPS): ~40 минут  → --device mps
#    CPU (4 потока):      ~6.5 часов → --threads 4
python -I scripts/train.py --data-dir data/tokenized --tokenizer tokenizer/bpe-8192.json \
    --out-dir checkpoints --exp-name base --epochs 5 --device mps

# 7. Графики обучения
python -I scripts/plot_history.py .

# 8. Генерация
python -I scripts/generate.py checkpoints/base/best.pt tokenizer/bpe-8192.json "Чăваш Ен" 200
```

Прогон полноценно рескается (`--resume` поднимает модель, оптимизатор и шаг из
`last.pt`) и наблюдаем: статус каждые `--log-every` шагов с EMA-лоссом, lr,
скоростью и ETA, структурированный `log.jsonl` (loss/ppl/grad_norm/токены),
сэмплы генерации в `samples.log`, опциональный TensorBoard (`--tensorboard`).

Эксперименты из отчёта (словарь × качество, SWA) воспроизводятся так:

```bash
bash scripts/run_experiments.sh 1200
```

## Структура

```
scripts/          # полный пайплайн: extract_corpus, train_tokenizer,
                  # tokenize_data, train (модель TinyGPT + цикл), generate, plot
tokenizer/        # BPE 4096/8192/16384 + статистики компрессии
checkpoints/      # base/best.pt (в репозитории), логи всех прогонов (jsonl)
notebooks/        # inference.ipynb — отдельный демо-ноутбук (обучения в нём нет)
reports/          # кривые обучения
ОТЧЁТ.md / .pdf   # отчёт по всем критериям
requirements.txt
```

## Модель и решения

GPT-style на чистом PyTorch: pre-LN, RoPE, GELU-MLP 4×, tied embeddings,
AdamW + cosine LR с warmup, градиентный клип; `d_model=256`, 6 слоёв, 4 головы,
контекст 256, словарь 8192 → **6,82 млн параметров**. Опционально sliding
window attention (`--swa-window`).

Ключевые решения (подробно — в ОТЧЁТ.md):

- **словарь 8192** — по кривой «словарь ↔ компрессия» на полном корпусе;
  подтверждён короткой сеткой прогонов и полным обучением;
- **fp32, а не mixed precision** — замерено: bf16-autocast на CPU без
  аппаратной поддержки в ~10 раз медленнее; fp16 на MPS портит сходимость
  (нет GradScaler);
- **NFC без lower-case** — диакритики Ӑ/ӑ Ҫ/ҫ Ӳ/ӳ значимы в чувашском,
  их нельзя терять;
- **контекст 256 токенов ≈ 810 символов** при компрессии 3,16 симв/токен.
