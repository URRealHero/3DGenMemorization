# Scripts

We use two scripts to curate caption-label pairs and filter mis-aligned pairs. 
For Objaverse download, please refer to the [Objaverse documentation](https://objaverse.allenai.org/).

## 1\. caption\_label\_pair.py

Matches raw captions to the best object category.

**Usage:**

```bash
python caption_label_pair.py \
  --csv input_data.csv \
  --labels labels_list.txt \
  --out_csv step1_output.csv \
  --threshold 0.70
```

**Arguments:**

  * `--csv`: Input CSV file (must contain `uid` and `caption`).
  * `--labels`: Text file with the list of target object classes.
  * `--out_csv`: Path to save the results.
  * `--threshold`: Similarity score cutoff (default: 0.70).

## 2\. clean\_caption.py

Filters out bad captions and removes color/material descriptions using an LLM.

**Usage:**

```bash
python clean_caption.py \
  --in_csv step1_output.csv \
  --out_csv final_data.csv \
  --model_name "Qwen/Qwen3-4B-Instruct-2507"
```

**Arguments:**

  * `--in_csv`: The output file from script 1.
  * `--out_csv`: The final curated dataset.
  * `--batch`: Batch size (default: 64).