# CHAPTER 3. SYSTEM ANALYSIS AND DESIGN

This chapter defines the requirements and design of the proposed fashion
product retrieval system. The system separates offline machine-learning tasks
from online recommendation. The offline pipeline prepares data, fine-tunes and
evaluates CLIP, and generates catalog embeddings. The online application uses
these artifacts to process image, text, or combined image-text queries.

## 3.1. Problem Analysis

Online shoppers may know the appearance of a desired product without knowing
its exact name, or they may only be able to describe it in words. A useful
retrieval system should therefore support three input situations:

1. **Image-only query:** retrieve products with similar visual characteristics.
2. **Text-only query:** retrieve products that match a written description.
3. **Combined query:** use an image as the visual reference and text to express
   additional or corrective information such as category, color, or pattern.

The output is a ranked Top-K list rather than a single class prediction. The
system is content-based: it ranks catalog products from their images and
descriptions and does not require user purchase history.

Category information must be used carefully. An incorrect category prediction
should not remove visually relevant products. For image-only search, category
is therefore an automatic soft preference controlled by a confidence
threshold. In a combined query, an explicitly written category may be treated
as a stronger constraint because it comes directly from the user.

The dataset also presents practical challenges. Categories are imbalanced,
descriptions may be short, and product images vary in background, viewpoint,
color, and style. The design consequently includes data auditing, balanced
training sampling, reproducible evaluation, and fallback behavior during
online retrieval.

## 3.2. System Requirements

### 3.2.1. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Prepare a configurable number of valid products and report their category distribution. |
| FR-02 | Create reproducible training, validation, and test assignments. |
| FR-03 | Fine-tune CLIP and retain the checkpoint with the best validation retrieval score. |
| FR-04 | Generate compatible image and text embeddings for the active catalog. |
| FR-05 | Accept image-only, text-only, or combined image-text queries. |
| FR-06 | Apply automatic soft-category preference to image search only when its confidence is sufficient. |
| FR-07 | Combine text-to-image and text-to-text similarity for text search when both catalog embeddings are available. |
| FR-08 | Use weighted raw cosine fusion for combined image-text retrieval. |
| FR-09 | Support English input and optional Vietnamese-to-English translation. |
| FR-10 | Return a configurable Top-K ranked list with product and score information. |
| FR-11 | Compare models and retrieval configurations under recorded experimental settings. |
| FR-12 | Expose service readiness and support requests through a load-balanced web deployment with centralized rate limiting. |

### 3.2.2. Non-Functional Requirements

- **Reproducibility:** dataset hashes, random seeds, split assignments,
  hyperparameters, checkpoints, and evaluation settings must be recorded.
- **Consistency:** catalog rows, embeddings, and model identity must belong to
  the same configuration.
- **Usability:** users are not required to understand or manually select an
  internal category strategy.
- **Performance:** catalog embeddings are generated offline so that an online
  request only encodes its query and ranks existing vectors.
- **Reliability:** invalid inputs and unavailable model instances must produce
  clear errors, while health checks and restart policies support recovery.
- **Maintainability:** data processing, model training, evaluation, serving,
  frontend, and deployment configuration remain separate modules.

## 3.3. Overall System Architecture

The architecture contains an offline pipeline and an online web application.
The website does not train a model when it starts; it loads a prepared catalog,
compatible embeddings, and a selected CLIP checkpoint.

**[Figure 3.1. Overall architecture of the proposed system]**

```text
Raw metadata and images
          |
          v
Preparation and category audit ---> Processed catalog
          |                               |
          v                               v
Train/validation/test split          Embedding generation
          |                               |
          v                               v
CLIP fine-tuning ---> Best checkpoint + image/text embeddings
          |
          +--------------------> Model comparisons

User ---> React frontend ---> Nginx reverse proxy/load balancer ---> FastAPI backend 1/2
                                                        |
                                                        v
                                      Catalog + model + embeddings
                                                        |
                                                        v
                                             Ranked Top-K products
```

| Component | Technology | Main responsibility |
|---|---|---|
| Frontend | React and Vite | Collect query inputs and display ranked products. |
| Reverse proxy and load balancer | Nginx | Provide one entry point, distribute requests, and enforce API rate limits. |
| Recommendation API | FastAPI | Validate requests, translate text when requested, and serialize results. |
| Recommendation engine | PyTorch, Transformers, NumPy | Encode queries, calculate scores, and rank products. |
| Artifact storage | CSV, JSON, NumPy arrays, safetensors | Store the catalog, embeddings, metadata, and checkpoint. |

The two backend containers use the same mounted artifacts but load separate
in-memory model instances. This provides basic load distribution and request
failover, although both instances still depend on the same Docker host.

## 3.4. Data and Artifact Design

The processed catalog contains one validated row per selected product. Its main
fields are `product_id`, `image_reference`, `product_name`, `description`,
`category`, `image_path`, and `image_sha256`. Exact duplicate image content is
removed before sampling.

| Artifact | Purpose |
|---|---|
| `data/processed/products.csv` | Active product catalog used for embedding generation and serving. |
| `products.category_audit.csv/.json` | Category counts, coverage, and warnings after preparation. |
| `runs/<run>/dataset.csv` | Exact training, validation, and test assignments. |
| `runs/<run>/training.json` | Training configuration, epoch history, and selected checkpoint metadata. |
| `runs/<run>/best/` | Best validation checkpoint and processor files. |
| `embeddings/image_embeddings.npy` | Precomputed product image vectors. |
| `embeddings/text_embeddings.npy` | Precomputed product text vectors. |
| Embedding metadata | Catalog hash, model identity, shape, and row-order information. |

The order of each embedding array must match the order of `products.csv`.
Metadata validation prevents an embedding array from being used with a changed
catalog or a different model checkpoint.

## 3.5. Offline Machine-Learning Pipeline

### 3.5.1. Data Preparation and Audit

The preparation stage validates image files, normalizes metadata fields,
calculates image hashes, removes exact duplicates, and selects the requested
catalog size. It also produces a category audit. A minimum per category is a
sampling target rather than a guarantee: when the source itself contains too
few products, the shortage is reported instead of creating artificial data.

### 3.5.2. CLIP Fine-Tuning

The prepared products are divided into training, validation, and test subsets.
Related records are grouped before splitting to reduce leakage. Fine-tuning
uses image-text pairs, a symmetric contrastive objective, AdamW, gradient
clipping, and category-balanced sampling for the training subset.

After each epoch, the model is evaluated on validation data. The checkpoint is
selected by the mean of text-to-image Recall@1 and image-to-text Recall@1. Test
data are reserved for final comparison and do not affect checkpoint selection.

### 3.5.3. Catalog Embedding Generation

The selected model encodes catalog images and constructed product text in
batches. The normalized vectors and their provenance metadata are saved for
online retrieval. Starting the website does not regenerate these embeddings.

## 3.6. Online Recommendation Design

### 3.6.1. Input Routing

The frontend includes a combined form that accepts optional image and text
fields. It automatically calls image search when only an image is present,
text search when only text is present, and combined search when both are
provided. Dedicated image and text tabs are also retained in the current
interface.

### 3.6.2. Image-Only Recommendation

An uploaded image is validated, converted to RGB, and encoded by the CLIP image
encoder. Its vector is compared with every catalog image vector using cosine
similarity, producing the image-to-image score \(S_{II}\).

The backend estimates a category from balanced groups of the strongest visual
neighbors. Category influence increases gradually between a confidence floor
of 0.50 and the full-strength threshold of 0.60:

\[
s_{image}(q,p_i)=S_{II}(q,p_i)
+\lambda g(h)\,\mathbf{1}[category_i=\hat{c}],
\]

where \(h\) is the heuristic confidence, \(g(h)\) maps confidence to a strength
between 0 and 1, and the default maximum bonus is \(\lambda=0.05\). Products
from other categories remain candidates. Confidence at or below 0.50 produces
no category bonus, while confidence at or above 0.60 produces the full bonus.

### 3.6.3. Text-Only Recommendation

English text is encoded directly. Vietnamese input can first be translated to
English. The query is compared with catalog image vectors and, when available,
catalog text vectors. The final text score is

\[
s_{text}(q,p_i)=\beta S_{TI}(q,p_i)
+(1-\beta)S_{TT}(q,p_i),
\]

where \(S_{TI}\) is text-to-image similarity, \(S_{TT}\) is text-to-text
similarity, and the default \(\beta\) is 0.50. If text embeddings are missing,
the system falls back to \(S_{TI}\). Text-only search does not apply automatic
category filtering.

### 3.6.4. Combined Image-Text Recommendation

For a combined query, CLIP independently encodes the image and the processed
text. The system calculates three cosine score families:

- \(S_{II}\): query image to product image;
- \(S_{TI}\): query text to product image;
- \(S_{TT}\): query text to product text.

The deployed system combines the raw cosine scores directly. The text branch
and final score are

\[
s_T=\gamma S_{TI}+(1-\gamma)S_{TT},
\]

\[
s_{combined}=\alpha S_{II}+(1-\alpha)s_T.
\]

With the default \(\alpha=\gamma=0.50\), the contributions are 0.50 for
image-to-image and 0.25 for each text component. The system uses these weighted
raw cosine scores directly for ranking.

Text attributes such as color or pattern can provide an additional metadata
match signal. Category handling follows the origin of the evidence: a category
inferred only from the image remains a soft preference, while a category stated
explicitly in the text may restrict the candidate set. This lets text correct
or refine information from the reference image.

## 3.7. API and User Interface Design

| Endpoint | Input | Main output |
|---|---|---|
| `POST /api/recommendations/image` | Multipart image, `top_k`, automatic category mode | Ranked products, visual score, applied category information, and timing. |
| `POST /api/recommendations/text` | JSON text, language, `top_k`, and image weight | Processed query, text component scores, final score, and timing. |
| `POST /api/recommendations/combined` | Multipart image and text, language, `top_k`, and image weight | Component scores, fusion score, applied category and attributes, and timing. |
| `GET /api/health/ready` | No query input | Backend model and artifact readiness. |

The frontend validates required fields and supported image types before sending
a request. Results from all modes use a common product-card presentation. The
combined form accepts one or both inputs and routes the request according to
what the user provides.

## 3.8. Deployment and Reliability Design

Docker Compose builds and starts the React frontend, Nginx reverse proxy/load balancer, and two
FastAPI backend instances. The processed catalog, embeddings, and selected
checkpoint must already exist or be downloaded before the web stack starts.

Nginx applies the `least_conn` strategy, centrally limits API traffic per client,
and retries eligible connection errors, timeouts, and selected upstream failures.
Requests that exceed the configured rate receive HTTP 429. Health checks distinguish a running
container from a backend that has successfully loaded its model and artifacts.
Docker restart policies provide basic service recovery. This design improves a
local demonstration deployment but does not provide automatic scaling or
protection from failure of the shared host.

## 3.9. Experimental Design and Reproducibility

The current unified comparison protocol uses the same 473 held-out test queries
and the same catalog of 5,000 products at \(K=1,5,10\). Category labels are
omitted from evaluation text to reduce target-label leakage. Embeddings are
cached only when their catalog and model identities match.

The experiment is divided into three groups:

1. **Controlled modality baselines:** SBERT and CLIP text representations,
   ResNet18 and CLIP image representations, and SBERT-ResNet18 rank fusion.
2. **CLIP fine-tuning comparison:** pretrained and fine-tuned CLIP on paired
   text-to-image and image-to-text retrieval.
3. **Serving ablation:** individual \(S_{II}\), \(S_{TI}\), and \(S_{TT}\)
   components, text fusion, and raw multimodal fusion.

A separate system-level category-aware ablation compares unrestricted image
ranking, the earlier binary-threshold strategy, and confidence-weighted soft
category ranking while keeping the same queries, gallery, checkpoint, and K
values.

The final system-level experiment follows the deployed query paths. It compares
image-only, text-only, and combined image-text retrieval, then evaluates the
combined ranking in four stages: raw score fusion, fusion with text attributes,
fusion with category handling, and the full pipeline with both mechanisms.
Attribute matching is reported separately because same-category relevance does
not by itself measure whether a requested color or pattern was satisfied.

For same-category retrieval, the query product is excluded and products sharing
its ground-truth category are relevant. For paired cross-modal retrieval, the
matching product remains in the gallery and exact `product_id` defines
relevance. Precision@K, Recall@K, and F1@K are reported together with evaluated
and skipped query counts. These category-based measures are retrieval proxies;
they do not fully represent detailed style similarity or user satisfaction.

## 3.10. Chapter Summary

This chapter presented the requirements and architecture of the fashion product
retrieval system. The offline pipeline prepares and audits data, fine-tunes
CLIP, generates compatible embeddings, and supports reproducible comparisons.
The online application supports image-only, text-only, and combined queries.
Image search uses confidence-controlled category preference, text search
combines text-to-image and text-to-text similarity, and combined search applies
weighted raw cosine fusion with optional category and attribute evidence. The next
chapter presents the corresponding implementation details.
