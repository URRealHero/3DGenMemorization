# Memorization in 3D Shape Generation
Official implementation of **"Memorization in 3D Shape Generation: An Empirical Study"**. This repository contains our evaluation framework for quantifying memorization ($Z_U$) and measuring generation quality (FPD), and the training code for our vector-set diffusion models (`hy3dshape`).

## 1\. Evaluation Framework

### Step 1: LFD Retrieval

Navigate to `retrieval/` to perform nearest-neighbor retrieval. This step generates the JSON files containing distances from generated samples ($Q$) and test samples ($P_{test}$) to the training set ($T$).

We recommend using **Light Field Distance (LFD)** as the retrieval metric, as it was found to be the most accurate in our benchmarks.

### Step 2: Memorization test ($Z_U$)

Navigate to `DataCopying/` to calculate the $Z_U$ score. This metric determines if the generative model is memorizing training data.

**Usage:**

```bash
cd DataCopying
python datacopying_test.py gen_retrieval_results.json test_retrieval_results.json
```

### Step 3: Quality evaluation (FPD)

You can get Uni3D embedding database from `retrieval/` [README](./retrieval/README.md).

```bash
cd fpd_eval
python npz_eval.py reference.npz generated.npz
```

> **Suggestion:**: You can normalize using the global training set statistics $\mu_R, \Sigma_R$ before saving them to `.npz`.

-----

## 2\. Model Training (`hy3dshape/`)

The `hy3dshape/` directory contains the implementation of our 3D generative models.

### Training
To be added soon.

### Inference
To be added soon.


-----

## 3\. Data Curation & Captioning

  * **`data_curation/`**: Contains our pipeline for filtering Objaverse-LVIS, removing mislabeled data using Qwen3, and preparing the dataset.
  * **`mvcaption/`**: Our Multi-View Captioning strategy to generate robust text descriptions.

Please refer to the README files inside those directories for detailed usage instructions.


## 4\. Acknowledgements

We deeply appreciate the open-source contributions that made this research possible. We would like to specifically acknowledge and thank the authors of the following projects for their excellent work:

* **[sdf_gen](https://github.com/1zb/sdf_gen?tab=readme-ov-file)**: Used for mesh normalization and SDF generation processing.
* **[LFD](https://github.com/kacperkan/light-field-distance) and [GET3D](https://github.com/nv-tlabs/GET3D)**: For Light Field Distance (LFD) implementation.
* **[data-copying](https://github.com/casey-meehan/data-copying/tree/master)**: The original implementation of the Data Copying Test ($Z_U$), which serves as the core of our memorization evaluation.
* **[3DShape2Vecset (VecSetX)](https://github.com/1zb/VecSetX)**: For the Vecset autoencoder architecture used in our experiments.
* **[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)**: For providing the robust diffusion backbone and training framework.