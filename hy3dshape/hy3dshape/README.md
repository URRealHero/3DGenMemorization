# Project Map
```
Hunyuan_Project/
├── main.py                     # entry point (builds config, datamodule, model, Trainer)
├── configs/                    # all YAMLs you pass with -c
│   └── *.yaml
├── scripts/                    # sbatch / launch helpers (yours)
├── utils/                      # top-level misc (CSV, etc.)  ← your dataset prints from here
│   ├── objaverse_train.csv
│   └── objaverse_val.csv
└── hy3dshape/                  # the actual package
    ├── __init__.py
    ├── pipelines.py            # sampling pipeline (CFG concat, batching, calls VAE decode)
    ├── data/
    │   ├── __init__.py
    │   ├── datamodule.py       # LightningDataModule (wraps dataset, DDP samplers)
    │   └── dataset.py          # scans *.pt latents + uses CSVs (paths, metadata)
    ├── models/
    │   ├── __init__.py
    │   ├── diffusion/
    │   │   ├── __init__.py
    │   │   ├── flow_matching_sit.py  # Diffuser (LightningModule)
    │   │   └── transport/            # flow-matching transport
    │   │       └── transport.py
    │   ├── autoencoders/
    │   │   ├── __init__.py
    │   │   ├── autoencoder.py        # CustomShapeVAE, VecSetAutoEncoder
    │   │   ├── bottleneck.py
    │   │   └── utils.py              # VanillaVolumeDecoder, MCSurfaceExtractor, Latent2MeshOutput
    │   ├── conditioner.py            # MultiConditioner
    │   └── embedders.py              # CLIP/class embedders
    └── utils/
        ├── __init__.py
        ├── misc.py                   # instantiate_* helpers, VAE ckpt routing, config merge
        ├── ema.py
        ├── schedulers.py
        ├── trainings/
        │   └── mesh_log_callback.py  # sampling during sanity/val
        |   └── ...
        ├── visualizers/
        │   └── pythreejs_viewer.py   # optional viewer
        |   └── ...
        └── utils.py                  # export_to_trimesh wrapper, timers, small helpers
```

# Who calls what (quick flow)

`python main.py -c configs/your.yaml --output_dir ...`
- loads config (hy3dshape.utils.misc.get_config_from_file)
- callbacks + logger (in main.py)
- datamodule = hy3dshape.data.datamodule.LatentDataModule
- model = hy3dshape.models.diffusion.flow_matching_sit.Diffuser
    - builds denoiser, conditioner
    - VAE via instantiate_non_trainable_model(...) (loads your VAE ckpt if given)
    - pipeline = hy3dshape.pipelines.Pipeline(...)
- Trainer.fit(model, datamodule=data)
    - train/val steps in Diffuser
    - sampling during sanity/val from utils/trainings/mesh_log_callback.py → pl_module.sample() → pipelines.py