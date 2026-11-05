# CVMJ revision TODO

This file is the current execution checklist. The SIGGRAPH 2025 paper and old
Pro checkpoints use legacy arbitrary-magnitude targets and are retained only
for historical comparison.

## Current status

- [x] Define the dimensionless target
  \(H(\mathbf{x}, f)=|p_{\mathrm{scene}}|/|p_{\mathrm{free}}|\).
- [x] Validate the analytic pulsating-sphere reference used by BEM.
- [x] Replace the empirical FDTD/BEM scale with a simulated FDTD free field.
- [x] Freeze a mesh-disjoint, approximately 9:1 train/validation manifest.
- [x] Diagnose the old BEM spikes as unchecked iterative-solver failures.
- [x] Validate an opt-in direct LU solve with iterative refinement.
  - The legacy `src/bem/BiCGSTAB.py` remains unchanged.
  - Full-Pro generation explicitly selects the checked direct solver.
  - Every BEM frequency must have relative residual at most \(10^{-6}\).
- [ ] Regenerate and publish the corrected full Pro dataset.
- [ ] Retrain the model and complete the paper experiments.
- [ ] Revise and submit the paper to Computational Visual Media.

## 1. Regenerate corrected BEM labels

Do **not** use `experiments/neuPAT_fix/generate.py` for this dataset: that
legacy path still selects the unchecked BiCGSTAB solver.

Choose up to four idle GPUs and generate into a new staging directory:

```bash
cd /home/houxinyun/NeuralSoundField/GeneralizableSound-Bem

conda run -n gensound python experiments/paper_next/generate_pro_full.py \
  --output /data5/houxinyun/GeneralizableSound/data/pro_relative_full_fixed.staging \
  --gpus 0 1 3 5 \
  --max-gpus 4 \
  --categories general enclosed \
  --splits train val
```

Replace `0 1 3 5` with idle GPU IDs. Re-running the same command resumes from
completed scene pairs. Expected BEM workload after excluding the invalid mesh:

- General BEM: 768 scenes.
- Enclosed BEM: 1334 scenes.
- Total BEM: 2102 scenes.

Each saved BEM sample must contain:

- `linear_solver = "direct_lu_refined"`;
- 63 per-frequency diagnostic records;
- finite transfer magnitudes;
- relative residual no greater than \(10^{-6}\).

## 2. Reuse the valid FDTD subset

The 435 FDTD scenes are not affected by the BEM solver bug. After BEM
generation finishes, copy their paired files from the old diagnostic staging
directory:

```bash
old=/data5/houxinyun/GeneralizableSound/data/pro_relative_full.staging
new=/data5/houxinyun/GeneralizableSound/data/pro_relative_full_fixed.staging

cp "$old"/train_mesh/fdtd_*.obj "$new"/train_mesh/
cp "$old"/train_data/fdtd_*.pt  "$new"/train_data/
cp "$old"/val_mesh/fdtd_*.obj   "$new"/val_mesh/
cp "$old"/val_data/fdtd_*.pt    "$new"/val_data/
```

Expected final composition:

- Train: 2280 scenes.
- Validation: 257 scenes.
- Total: 2537 scenes.
- Solver totals: 2102 BEM + 435 FDTD.

## 3. Audit and publish the dataset

Run the complete audit:

```bash
conda run -n gensound python experiments/paper_next/audit_pro_full.py \
  --dataset /data5/houxinyun/GeneralizableSound/data/pro_relative_full_fixed.staging \
  --output experiments/paper_next/validation/pro_relative_full_fixed_audit.json
```

The audit must report:

- 2537 paired scenes and no unexpected pairs;
- no train/validation source-mesh hash overlap;
- no NaN, Inf, negative transfer, or endpoint mismatch;
- complete solver/target/reference metadata;
- 63 diagnostics per BEM sample;
- maximum BEM residual at most \(10^{-6}\);
- no errors.

Only after this passes:

```bash
mv /data5/houxinyun/GeneralizableSound/data/pro_relative_full_fixed.staging \
   /data5/houxinyun/GeneralizableSound/data/pro_relative_full

ln -s /data5/houxinyun/GeneralizableSound/data/pro_relative_full \
  /home/houxinyun/NeuralSoundField/GeneralizableSound/data/pro_relative_full
```

Then remove the invalid old staging data and obsolete `pro_relative_v1/v2`
smoke datasets. Do not delete historical `NAT_data`.

## 4. Train the relative-transfer model

- [ ] Add a frozen experiment configuration for `data/pro_relative_full`.
- [ ] Train the final Pro model from scratch; old checkpoints are incompatible
  with the new target.
- [ ] Select checkpoints without tuning on the final mesh-disjoint validation
  results.
- [ ] Save the exact commit, configuration, random seed, GPU model, training
  duration, and checkpoint hash.
- [ ] Verify decoding uses \(H=10^a\), where \(a=\log_{10} H\).

## 5. Main quantitative evaluation

Report BEM-general, BEM-enclosed, and FDTD-teapot separately. Do not average
the solvers into one headline number.

- [ ] Log-amplitude MSE.
- [ ] Linear relative \(L_2\).
- [ ] Linear transfer-magnitude SNR in dB.
- [ ] FFAT PSNR and SSIM on a fixed spherical raster.
- [ ] Per-band results for 0–2, 2–4, 4–6, and 6–8 kHz.
- [ ] Scene-level mean, standard deviation, median, worst 10%, and worst case.
- [ ] GT / prediction / error visualizations for best, median, and worst cases.
- [ ] Fair NeuralSound baseline where its assumptions and data permit.

## 6. Ablations

- [ ] Cartesian coordinates versus the spherical representation.
- [ ] Spherical representation without seam handling.
- [ ] Current seam loss versus periodic graph topology.
- [ ] Optional periodic topology plus spherical Cartesian direction features.
- [ ] Identity radial coordinate \(r\) versus the current \(r^{1/3}\)
  near-source refinement.
- [ ] Report parameters, Log-MSE, linear SNR, and seam-local error.

Do not add an \(r^3\) ablation.

## 7. Timing

Measure with CUDA synchronization, warm-up, repeated trials, and fixed
hardware:

- [ ] Mesh preprocessing and octree construction.
- [ ] OGNN backbone initialization.
- [ ] Cached single-listener query.
- [ ] Cached batches of 32, 256, and 2048 listeners.
- [ ] Complete FFAT raster.
- [ ] Dynamic-geometry end-to-end update and FPS.
- [ ] Ten-second audio synthesis.
- [ ] BEM and FDTD offline ground-truth generation.

Describe dynamic geometry as an interactive update rate. Static geometry should
reuse the cached backbone and report the faster listener-query path separately.

## 8. Demos and figures

- [ ] Main zero-shot demo: one final Pro model on held-out geometry, including
  a held-out Utah teapot.
- [ ] Teapot opening animation with GT/predicted spectra and error.
- [ ] Main FFAT wall: GT | Ours | baseline with consistent columns and ranges.
- [ ] Worst-case FFAT and per-frequency error curves.
- [ ] Seam-ablation heatmap.
- [ ] Static cached-query and dynamic-update timing visualization.
- [ ] Keep cup, fans, trumpet, container, and 2-hole scene-specific demos only
  in supplementary material, explicitly disclosed as in-distribution
  per-scene training with mesh overlap.

Do not describe any scene-specific model as fine-tuned unless an actual
fine-tuning run is performed and documented.

## 9. Paper revision

- [ ] Replace arbitrary BEM amplitude with the dimensionless relative-transfer
  definition throughout the abstract, method, experiments, and limitations.
- [ ] Explain the analytic BEM free field and simulated FDTD free field.
- [ ] Document the checked direct solve and the residual acceptance criterion.
- [ ] Describe the mesh-level 9:1 split and invalid-mesh exclusion.
- [ ] Remove all unsupported fine-tuning claims.
- [ ] Separate zero-shot generalization from scene-specific applications.
- [ ] Add complete metrics, ablations, worst cases, and timing.
- [ ] State limitations: magnitude only, no phase cancellation, rigid
  boundaries, point-source/near-field assumptions, and solver discretization.
- [ ] Fix table headings, units, figure numbering, notation, captions, and
  inconsistent real-time claims identified by the SIGGRAPH reviews.

## 10. CVMJ submission

Computational Visual Media uses rolling submission; there is no fixed journal
deadline.

- [ ] Move the revised manuscript to the current CVM journal template.
- [ ] Prepare cover letter, graphical abstract, author biographies, and
  supplementary video.
- [ ] Include a concise statement explaining the substantial changes since the
  rejected SIGGRAPH submission.
- [ ] Freeze code/data commits and archive the final experiment manifests.
- [ ] Complete an internal technical and language review before submission.
