# Advanced AI — Food Image Segmentation & Multimodal Idiom Representation

**Module:** Advanced AI  
Two projects exploring modern deep learning architectures — one in computer vision, one in natural language processing.

---

## Task 1 — Food Image Segmentation with U-Net

Semantic segmentation of food images — the goal is pixel-level classification, labelling every pixel in an image as a specific food item rather than just classifying the whole image.

This matters in practice for things like calorie estimation apps or automated food logging, where you need to know not just "there's pizza in this image" but exactly where the pizza is and how much of the frame it occupies.

**Architecture:** U-Net — an encoder-decoder architecture originally developed for biomedical image segmentation. The encoder compresses the image into a rich feature representation; the decoder reconstructs it at full resolution with skip connections from the encoder to preserve spatial detail.

**Implementation:**
- segmentation-models-pytorch library for the U-Net backbone
- albumentations for augmentation (flips, brightness/contrast, grid distortion)
- Custom Dataset class handling image-mask pairs
- Mixed precision training with GradScaler for memory efficiency
- IoU (Intersection over Union) as the primary evaluation metric — more meaningful than pixel accuracy for segmentation

The skip connections are what make U-Net work well for segmentation — without them, spatial information gets lost in the bottleneck and the decoder struggles to reconstruct sharp boundaries.

---

## Task 2 — Multimodal Idiom Representation

Idioms are phrases where the meaning can't be derived from the individual words — "kick the bucket" has nothing to do with buckets. This project explored whether combining text and image embeddings produces better representations of idiomatic expressions than text alone.

**Approach:**
- CLIP (Contrastive Language-Image Pretraining) for joint text-image embeddings
- BERT / Sentence-BERT for text-only baseline embeddings
- XGBoost classifier on top of the embeddings for idiomaticity prediction
- UMAP for dimensionality reduction and visualisation of the embedding space
- WordCloud for qualitative analysis of idiomatic vs literal expression clusters

The interesting finding is that CLIP embeddings, trained to align images with text, sometimes capture idiomatic meaning better than pure text models — possibly because idioms often have visual metaphorical roots.

---

## Stack

Python · PyTorch · segmentation-models-pytorch · albumentations · Transformers (HuggingFace) · CLIP · XGBoost · UMAP · Matplotlib · Seaborn

