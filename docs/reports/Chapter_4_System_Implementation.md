# CHAPTER 4. SYSTEM IMPLEMENTATION

This chapter describes the implementation of the fashion product retrieval
system, from data preparation and CLIP fine-tuning to the web application and
Docker deployment. Experimental results are presented in Chapter 5.

## 4.1. Implementation Environment

- Briefly list the main technologies: Python, PyTorch, Transformers, FastAPI,
  React, Vite, Nginx, and Docker Compose.
- Summarize the source-code organization: `project/src/` for machine-learning
  and retrieval logic, and `project/web/` for the backend, frontend, and Nginx configuration.

**[Table 4.1. Main technologies and their roles]**

## 4.2. Data Preparation and Model Implementation

### 4.2.1. Data Preparation

- Describe metadata normalization, image validation, duplicate removal, and
  catalog sampling in `project/src/prepare_dataset.py`.
- Explain the prepared `products.csv` and category audit outputs.
- Describe prompt construction and reproducible train/validation/test splitting
  in `project/src/clip_data.py`.

### 4.2.2. CLIP Fine-Tuning

- Introduce the implementation in `project/src/finetune_clip.py`, using
  `openai/clip-vit-base-patch32` as the base model.
- Summarize the training loop, category-balanced sampling, contrastive loss,
  and validation-based selection of the best checkpoint.
- Identify the saved checkpoint and training metadata. Detailed experimental
  settings and training results belong in Chapter 5.

### 4.2.3. Embedding Generation

- Describe batch encoding and normalization in
  `project/src/generate_clip_embeddings.py`.
- Explain the saved image and text arrays and their alignment with catalog rows.
- Briefly describe compatibility checks between the catalog, embeddings, and
  selected checkpoint.

**[Figure 4.1. Data preparation, fine-tuning, and embedding generation]**

## 4.3. Recommendation and Backend Implementation

### 4.3.1. Image-Based Retrieval

- Describe image preprocessing, CLIP encoding, similarity calculation, and
  top-K ranking in the recommendation modules under `project/src/`.
- Explain how the automatic category policy is applied in code, referring to
  Section 3.6.1 for its scoring rule.

### 4.3.2. Text-Based Retrieval

- Describe Vietnamese-to-English translation through `project/src/translator.py`
  and query encoding with CLIP.
- Explain the implementation of image/text score blending and fallback when
  catalog text embeddings are unavailable.

### 4.3.3. FastAPI Integration

- Describe application initialization and recommendation routes under
  `project/web/backend/`.
- Explain how requests are validated, passed to the recommendation engine,
  and returned as product responses.
- Include one concise response example and summarize input-error handling.

## 4.4. User Interface Implementation

### 4.4.1. Search Interface

- Describe the React image and text search forms: image selection, query input,
  language, number of results, and text-search image weight.
- Explain API integration and loading/error feedback.

**[Figure 4.2. Main search interface]**

### 4.4.2. Product Results

- Describe product cards showing images, names, categories, and score information.
- Illustrate the image and text search flows with actual application screenshots.

**[Figure 4.3. Image search results]**

**[Figure 4.4. Text search results]**

## 4.5. Deployment and Functional Verification

### 4.5.1. Docker Deployment

- Describe `project/docker-compose.yml`, which starts the frontend, Nginx
  reverse proxy/load balancer, and two FastAPI backends.
- Explain mounting the prepared catalog, embeddings, and checkpoint, and briefly
  describe Nginx load distribution, bounded retries, and centralized rate limiting.
- Present `docker compose up -d --build` from `project/` after the required
  artifacts have been prepared.

### 4.5.2. Functional Verification

- Record checks for image search, English and Vietnamese text search, top-K
  selection, and invalid input handling.
- Summarize observed behavior in a compact table supported by responses or
  screenshots. Leave unexecuted checks explicitly marked as pending.

**[Table 4.2. Functional checks, expected behavior, and observed results]**

## 4.6. Chapter Summary

Summarize the implemented data pipeline, recommendation engine, web interface,
and deployment. Introduce the model evaluation and discussion in Chapter 5.
