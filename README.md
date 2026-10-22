# Persian MMS/VITS fine-tuning

Fine-tune Meta's Persian [`facebook/mms-tts-fas`](https://huggingface.co/facebook/mms-tts-fas)
checkpoint on the Persian [ManaTTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS)
dataset. The model is based on the VITS architecture; this project adapts the
training recipe from [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits)
for a single Apple Silicon Mac.

> The MMS checkpoint is licensed under CC BY-NC 4.0. Fine-tuned weights are
> therefore for non-commercial use unless Meta grants additional rights.

## اجرای سریع روی macOS

پیش‌نیازها: macOS روی Apple Silicon، پایتون 3.10 یا 3.11، حدود 15 گیگابایت فضای
خالی و اینترنت. آموزش روی MPS انجام می‌شود و در صورت نیاز عملیات پشتیبانی‌نشده به
CPU منتقل می‌شوند.

```bash
cd persian-mms-tts-finetuning
./scripts/setup_local.sh
./scripts/train_local.sh
```

اسکریپت اول محیط مجازی را می‌سازد، وابستگی‌ها را نصب می‌کند، افزونه Cython را
کامپایل می‌کند و checkpoint فارسی دارای discriminator را در
`checkpoints/mms-tts-fas-train` آماده می‌کند. اسکریپت دوم fine-tuning را با ۱۰۰۰
نمونه ManaTTS آغاز می‌کند و خروجی را در `outputs/persian-mms-tts` می‌نویسد.

برای تست صدای مدل:

```bash
./scripts/infer_local.sh "سلام، این یک آزمایش تبدیل متن فارسی به گفتار است."
```

فایل صوتی در `outputs/sample.wav` ذخیره می‌شود. اگر هنوز آموزش انجام نشده باشد،
اسکریپت به‌صورت خودکار مدل پایه `facebook/mms-tts-fas` را تست می‌کند.

## اجرای مرحله‌ای

```bash
source .venv/bin/activate

# فقط آماده‌سازی checkpoint قابل آموزش
python convert_original_discriminator_checkpoint.py \
  --language_code fas \
  --pytorch_dump_folder_path checkpoints/mms-tts-fas-train

# آموزش
PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --num_processes 1 --mixed_precision no \
  run_vits_finetuning.py training_config_examples/finetune_mms_fa_local.json

# inference
python scripts/infer.py \
  --model outputs/persian-mms-tts \
  --text "سلام دنیا" \
  --output outputs/sample.wav
```

برای یک تست سریع‌تر، مقدار `train_split_name`، تعداد epoch و batch size را در
[`training_config_examples/finetune_mms_fa_local.json`](training_config_examples/finetune_mms_fa_local.json)
کم کنید. برای ادامه آموزش قطع‌شده، همان فرمان train را دوباره اجرا کنید؛ آخرین
checkpoint به‌طور خودکار شناسایی می‌شود.

## ساختار پروژه

```text
.
├── checkpoints/                 # مدل پایه آماده آموزش (در Git ذخیره نمی‌شود)
├── outputs/                     # checkpointها و نمونه‌های صوتی
├── scripts/
│   ├── setup_local.sh
│   ├── train_local.sh
│   ├── infer_local.sh
│   └── infer.py
├── training_config_examples/
│   └── finetune_mms_fa_local.json
├── convert_original_discriminator_checkpoint.py
└── run_vits_finetuning.py
```

## مبنای علمی

این پروژه یک مقاله مستقل نیست؛ ترکیبی اجرایی از سه کار منتشرشده است:

- معماری VITS: *Conditional Variational Autoencoder with Adversarial Learning
  for End-to-End Text-to-Speech* (ICML 2021)
- مدل پایه MMS: *Scaling Speech Technology to 1,000+ Languages* (JMLR 2024)
- دیتاست فارسی: *ManaTTS Persian: a recipe for creating TTS datasets for lower
  resource languages* (NAACL 2025)

لینک و اطلاعات کامل استناد در [REFERENCES.md](REFERENCES.md) و
[`CITATION.cff`](CITATION.cff) آمده است.

## مجوز و اعتبار

کد این repository بر اساس پروژه MIT دارای مجوز
[`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) است؛
فایل [LICENSE](LICENSE) را ببینید. مجوز کد با مجوز وزن‌های Meta یکی نیست.
