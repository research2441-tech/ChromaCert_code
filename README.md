# ChromaCert

Code for *ChromaCert: Layout-Aware Masked Discrete Diffusion with Designer-Preference Alignment and Conformally Certified Text Legibility for Colour Harmonisation in Visual Communication Design*.

ChromaCert generates colours for layered graphic designs and then certifies text legibility:

1. **Element graph and encoder.** Each design becomes an element graph with on / under / adjacent / same-role relations. A relation-biased Transformer (d = 128, 4 blocks, 4 heads, about 1.03 M parameters) predicts tokens in a 256-entry OKLab colour codebook.
2. **Masked discrete diffusion.** Training uses an MDLM-style 1/t-weighted ELBO with random brand-colour locks. Sampling uses confidence-ordered unmasking (S = 8).
3. **Exact harmony guidance.** Each candidate token is reweighted by exp(−γ E_i(k)). The energy combines a hue-template term, a CIEDE2000 separation term and a lightness-hierarchy prior.
4. **Preference stage.** A DPO-style loss for absorbing-state diffusion contrasts designer colourings with samples from a frozen reference model, using shared mask rates.
5. **Accessibility certification.** A hue-preserving OKLCH repair is followed by evidence-score escalation (deferral or scrim). Conformal risk control calibrates the threshold against a pixel-level audit under 16 simulated observers (WCAG 2.2 and APCA).

All files sit in one folder. Run every command from inside it.

## Requirements

```
pip install -r requirements.txt
```

Everything runs on CPU. The experiments in the paper used two CPU cores and about 8 GB of RAM. Run the pixel audit (`evaluate.py --audit`) and `run_crc.py` one at a time, because each holds the layer thumbnails in memory.

## Data

Crello v5.1 (`cyberagent/crello` on the Hugging Face Hub) uses the official train, validation and test split.

```
mkdir -p raw prep data preds res runs figs logs
bash fetch_all.sh                 # download each shard, extract layers, colours and thumbnails into prep/, delete the parquet
python3 build_dataset.py          # element graphs, relations, proxy backgrounds and codebooks → data/
python3 embed.py                  # CLIP ViT-B/32 (LAION-2B) photo/text/title embeddings → data/emb_clip.pkl
```

Set `CC_DATA` if `data/` lives elsewhere.

## Training

```
python3 run_train.py --name mdm_s0   --objective mdm   --seed 0     # masked discrete diffusion
python3 run_train.py --name mlm_s0   --objective mlm   --seed 0     # one-shot masked colour model
python3 run_train.py --name gauss_s0 --objective gauss --seed 0     # Gaussian diffusion in OKLab
python3 run_dpo.py   --ref mdm_s0 --name dpo_s0 --seed 0            # preference stage
```

The paper uses seeds 0, 1 and 2 for the main models. For the ablations, add `--no-rel`, `--no-sem` or `--K 128/512`.

## Evaluation

```
python3 run_eval.py --method full --seed 0 --protocol P --tag full_P_s0    # brand-locked completion (half the layers locked)
python3 run_eval.py --method full --seed 0 --protocol G --tag full_G_s0    # free generation
python3 evaluate.py preds/full_P_s0.pkl
python3 evaluate.py --audit preds/full_G_s0.pkl                            # 16-observer pixel audit
python3 c2st_all.py preds/full_G_s0.pkl                                     # critic AUC on a fixed 1,000-design subset
python3 run_eval.py --method full --seed 0 --protocol G --split validation --tag full_G_s0_validation
python3 run_crc.py --cal preds/full_G_s0_validation.pkl --test preds/full_G_s0.pkl --metric wcag --out res/crc_full_s0_wcag.pkl
python3 obs_analysis.py preds/full_G_s0.pkl
python3 bench.py
python3 make_results.py && python3 -c "import make_results as m; m.group_de()"
python3 fig_arch.py && python3 fig_results.py
python3 fig_qual.py preds/full_G_s0.pkl 0.0 <id1> <id2> <id3> figs/fig7_qual.png
```

These are the `run_eval.py` methods:

| Method | Description |
|---|---|
| `freq` | Frequency prior |
| `knn` | Template retrieval |
| `img_resnet50`, `img_clip` | Image-level palette regression |
| `mcr` | One-shot masked colour model |
| `gauss` | Gaussian diffusion |
| `mdm` | Masked discrete diffusion |
| `mdm_guide` | Guided masked diffusion |
| `dpo` | Preference stage |
| `full` | ChromaCert |

`queue.sh <jobfile>` runs a list of commands one after another.

## Files

| File | Purpose |
|---|---|
| `colorlib.py` | sRGB/OKLab/OKLCH/CIELAB conversions, CIEDE2000, WCAG and APCA contrast, CVD simulation |
| `harmony.py` | hue templates, separation energy, lightness prior, Ou–Luo harmony index |
| `model.py` | relation-biased element-graph Transformer |
| `chroma.py` | data store, losses, training, preference fine-tuning, guided sampler |
| `baselines.py` | frequency, retrieval and image-level palette baselines |
| `access.py` | observers, proxy contrast, OKLCH repair, evidence score, pixel audit, CRC |
| `metrics.py`, `stats.py` | fidelity, harmony, critic AUC, paired tests, BH correction, bootstrap |
| `llm_baseline.py` | pilot script for an LLM recommender (not run at full scale; see the paper) |
| `summary.json`, `train_meta.json`, `bench.json`, `group_de.json`, `obs_full_G_s0.json` | aggregated numbers behind every table and figure (written to `res/` when the pipeline is rerun) |

## Licence

The code is released for research use. Crello is distributed under CDLA-Permissive-2.0, subject to the template licence of its original provider.
