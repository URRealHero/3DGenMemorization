# Scripts 

## 1\. caption\_label\_pair.py

Multi-view images captioning using Qwen3-VL models (supports 8B and MoE). It generates structured JSON outputs containing three levels of detail: a short phrase, a sentence, and a detailed paragraph.

**Usage:**

```bash
python text_granularity_captioning.py \
  --metadata_csv input_images.csv \
  --output_csv step3_captions.csv \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --num-views 12 \
  --views-per-sample 4 \
  --batch-size 32
```

**Arguments:**
  * `--metadata_csv`: Input CSV file (must contain `model_uid` and absolute `view_path`).
  * `--output_csv`: Path to save the resulting captions (JSON format).
  * `--model_id`: HuggingFace model ID to use (default: Qwen3-VL-8B-Instruct).
  * `--num-views`: Total number of views stacked vertically in the input image.
  * `--views-per-sample`: Number of views to crop and feed to the model per object.
  * `--batch-size`: Number of samples to process simultaneously (default: 32).
