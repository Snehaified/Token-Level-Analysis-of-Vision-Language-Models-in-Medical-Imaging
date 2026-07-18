# Token-Level-Analysis-of-Vision-Language-Models-in-Medical-Imaging
Token-level interpretability analysis of CLIP vs. BiomedCLIP on chest X-ray retrieval — quantifying the modality shortcut in general-purpose vision-language models

# Which Words Drive Medical Image Retrieval?

**A token-level interpretability analysis of CLIP and BiomedCLIP on chest X-rays.**

When a vision-language model retrieves chest X-rays from a query like *"chest X-ray showing pneumonia with lung opacity"*, which words are actually doing the work? Does the model respond to the diagnosis, the radiological finding, or just the imaging modality?

This project answers that question through systematic token ablation, and finds that general-purpose CLIP relies on a **modality shortcut**: the word "X-ray" contributes more to retrieval than the name of the disease.

---

## Key Findings

**1. CLIP retrieves via modality, not diagnosis.**
Removing "X-ray" from a Cardiomegaly query drops retrieval by **0.58 MAP@10**. Removing the word *"cardiomegaly"* drops it by only **0.28** — less than half the impact. Averaged across all prompts, modality tokens are CLIP's highest-importance category (mean Δ MAP@10 = +0.097), while clinical tokens are net-negative (−0.024).

**2. Domain adaptation inverts this.**
For BiomedCLIP, modality tokens become functionally inert (mean Δ MAP@10 ≈ 0.000) while clinical tokens become dominant (+0.071). Every BiomedCLIP training example is a medical image, so "X-ray" carries no discriminative signal — exactly what contrastive learning predicts.

**3. Neither model is compositional.**
Adding words to a prompt often *hurts*. The word **"normal"** reduces CLIP's retrieval of normal chest X-rays by **0.42 MAP@10** — the largest single effect measured. Function words like "showing" can matter more than diagnoses. Tokens destructively interfere rather than combining additively.

**4. Attention confirms the story from the vision side.**
CLIP's attention rollout is pathology-invariant: the same generic template appears for cardiomegaly, pneumonia, effusion, and normal cases. Attention concentrates on image corners, laterality markers, and equipment artifacts — never on the diagnostically relevant anatomy.

<p align="center">
  <img width="888" height="398" alt="Screenshot 2026-07-18 at 15 46 23" src="https://github.com/user-attachments/assets/710197a5-16a1-4c6a-8d28-985f37a76ff0" />

  <br>
  <em>Mean token importance by linguistic category. CLIP peaks on modality; BiomedCLIP peaks on clinical.</em>
</p>

---

## Method

**Dataset.** NIH ChestX-ray14, 800 images balanced across four classes (Cardiomegaly, Pneumonia, Effusion, No Finding), filtered to single-label cases.

**Models.** CLIP ViT-B/32 (LAION-2B) and BiomedCLIP (PubMedBERT + ViT-B/16, 15M biomedical image-text pairs).

**Prompt taxonomy.** For each pathology, prompts at four linguistic levels — clinical (`"pneumonia"`), radiological (`"lung opacity"`), modality (`"chest X-ray"`), and a composite containing one token of each.

**Token ablation.** For each composite prompt, remove one word at a time and measure the change in MAP@10. Token importance = baseline − ablated. Positive means the word helps; negative means it hurts.

**Supporting analyses.** Embedding-space displacement (cosine shift per ablated token), ViT attention rollout, and CodeCarbon emissions measurement linking token findings to inference efficiency.

---

## Repository Contents

```
├── token_level_analysis.ipynb   # Self-contained Colab notebook   <img width="888" height="398" alt="Screenshot 2026-07-18 at 15 46 23" src="https://github.com/user-attachments/assets/57a7fe78-d885-4a26-8a3c-aaa22efc684e" />

└── README.md
```

The notebook runs top-to-bottom on a free Colab T4. It handles dataset download, extraction, and caching; total runtime is roughly 25 minutes including setup.

**To reproduce:** open the notebook in Colab, enable GPU, and follow the dataset setup instructions in Section 3. You'll need the ChestX-ray14 image archives from the [NIH repository](https://nihcc.app.box.com/v/ChestXray-NIHCC) (three archives suffice for the 800-image subset).

---

## Limitations

Single-token ablation is a first-order probe — it doesn't capture interactions between tokens, which the compositional failures suggest are substantial. CLIP's tokenizer splits some medical terms into subwords, so word-level ablation removes only part of a concept. Attention rollout is not text-conditioned (CLIP's towers don't cross-attend), so the maps show global visual saliency rather than query-specific relevance; gradient-based attribution would be the stronger method. ChestX-ray14 labels are NLP-mined from reports with known noise, particularly for Pneumonia. Attention analysis is qualitative over a small sample.

---

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- Zhang et al. (2023). *BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.* arXiv:2303.00915.
- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.
- Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL.

---

Individual project for Deep Learning for Computer Vision II, EMJMD IPCVAI (UAM / PPCU), 2026.
