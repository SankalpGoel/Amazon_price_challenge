# ML Challenge 2025: Smart Product Pricing – Team Submission

### Team
*Project:* Smart Product Pricing Challenge 2025  
*Approach:* Multimodal Adaptive Learning (Text + Image + Anchors)  
*Team Name:* <Code Catalyst> 
*Team Lead:* <Sankalp Goel>  
*Date:* <13-10-2025>


---

## 1. Methodology Overview
We model product price as a *multimodal regression* problem using only the provided training data.  
Our pipeline fuses three signals:

| Modality | Core Model | Key Features |
|-----------|-------------|---------------|
| *Text* | TF-IDF (1-2 grams word, 3-5 char) → XGBoost Regressor | Product title + description, token & character n-grams, length/digit/case meta-features |
| *Image* | Fast statistical descriptors + XGBoost | width × height, aspect, brightness, entropy, RGB means/std, colorfulness |
| *Anchors (Adaptive)* | Unit-Anchor Decomposition (UAD) + Cluster Anchors + Stacker | Value × median price-per-unit(Unit) + cluster median prices + meta features fused via GBM stacker |

---

## 2. Model Architecture
1. *Text Head*  
   TF-IDF → XGBoost (1100 trees, depth 8, MAE objective).  
   Test-Time Text Augmentation (TTTA): 3 views (raw/lower/alnum) averaged; variance used as confidence.

2. *Image Head*  
   Lightweight image statistics → XGBoost (identical hyper-params).  
   Handles missing images gracefully (has_img = 0).

3. *Adaptive Layer (Ensemble)*  
   - *UAD – Unit-Anchor Decomposition:* extract Value & Unit from catalog_content; compute median price-per-unit per Unit.  
     → Anchor = Value × median PPU(Unit) ⇒ normalizes pack sizes.  
   - *CASS – Cluster Anchors + Stacker:* MiniBatch K-Means (2 K clusters) on TF-IDF space; cluster median price serves as family-level prior.  
   - *Stacker:* Gradient Boosting meta-learner combines Text pred, Image pred, Unit anchor, Cluster anchor, meta features, and TTTA variance.  
     Output is the final price.

---

## 3. Feature Engineering Highlights
- *Text Meta-Features:* len_char, len_word, num_digits, num_caps.  
- *Numeric Extraction:* Regex-based Value & Unit parsing (~98 % coverage).  
- *Image Descriptors:* entropy, brightness, colorfulness, RGB mean/std.  
- *Anchors:* per-Unit price-per-unit medians and per-Cluster median prices.  
- *Normalization:* All predictions clipped to positive floats.

---

## 4. Training & Validation
- *Data:* 75 k train / 75 k test samples.  
- *Cross-Validation:* 5-fold subject-stratified split.  
- *Metric:* Symmetric Mean Absolute Percentage Error (SMAPE).  
- *Objective:* reg:absoluteerror (MAE) ≈ SMAPE.  
- *Hardware:* Google Colab Pro (T4 GPU / 12 GB RAM).  

| Metric | Value |
|:--|--:|
| *CV SMAPE (final stack)* | *≈ 7 – 8 %* |
| Text head SMAPE | 10 – 11 % |
| Image head SMAPE | 13 – 15 % |
| Unit Anchor SMAPE | 9 – 10 % |
| Cluster Anchor SMAPE | 9 – 10 % |

---

## 5. Compliance & Licensing
- *No external price lookup*; uses only provided train/test text and images (downloaded via utils.download_images).  
- *Libraries:* scikit-learn (BSD-3), XGBoost (Apache-2.0), Pillow (HSP).  
- *Model size:* << 8 B parameters (~50 MB combined trees).  
- *Predictions:* all positive floats; output matches sample_test_out.csv exactly.

---

## 6. Innovation & Impact
- *Adaptive Anchoring:* UAD + CASS creates contextual priors that dramatically reduce price variance across units and brands.  
- *Test-Time Augmentation for Text:* improves robustness to token noise.  
- *Explainability:* anchors and cluster priors offer interpretable price drivers (“why is this item expensive?”).  
- *Efficiency:* runs end-to-end on CPU (< 30 min on Colab).

---

## 7. Results & Future Work
Our adaptive ensemble achieves *top-tier SMAPE (< 8 %)* on public leaderboard baselines.  
Future work: fine-grained CNN image embeddings and transformer-based text encoders with shared anchor priors.

---

*Final Deliverables:*  
- dataset/test_out.csv — submission file  
- Documentation.md — this document  
- outputs_adapt/cv_metrics.json — validation summary