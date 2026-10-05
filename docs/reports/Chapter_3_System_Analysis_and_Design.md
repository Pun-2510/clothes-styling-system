# CHAPTER 3. SYSTEM ANALYSIS AND DESIGN

This chapter translates the theoretical concepts from Chapter 2 into the
requirements and architecture of the implemented system. The design separates
offline machine-learning tasks from online product retrieval. Offline tasks
prepare the catalog, fine-tune and evaluate CLIP, and generate reusable
embeddings. Online services receive user queries and rank the prepared catalog.

## 3.1. Problem Analysis

The system must help a user find existing catalog products when the user has
either a reference image or a textual description. A query does not necessarily
contain the exact product name or category stored in the dataset. The system
must therefore compare semantic or visual representations rather than depend
only on exact keyword matching.

The problem contains two principal retrieval cases:

1. **Image query:** the user uploads a product image, and the system returns
   visually related catalog items.
2. **Text query:** the user enters a description in English or Vietnamese, and
   the system returns products that match the meaning of the query.

The returned products form a ranked list rather than a single classification.
A category can improve image search when it is predicted reliably, but an
incorrect category must not remove all potentially relevant products. For this
reason, category information is designed as an automatic, confidence-controlled
preference instead of a mandatory option selected by the user.

Model training and model serving also have different requirements. Training
requires a reproducible data split, validation-based checkpoint selection, and
final test evaluation. Serving requires fast reuse of catalog embeddings,
validated inputs, stable API responses, and access to the model and artifacts.
The design separates these responsibilities so that starting the website does
not retrain the model or regenerate the catalog.

## 3.2. System Requirements

### 3.2.1. Functional Requirements

The main functional requirements are listed below.

| ID | Requirement |
|---|---|
| FR-01 | Prepare a configurable number of products from the raw metadata and image directory. |
| FR-02 | Remove records with missing or invalid images and remove exact duplicate image content. |
| FR-03 | Record the category distribution and warn about categories that do not meet the configured minimum. |
| FR-04 | Create reproducible training, validation, and test assignments without separating duplicate representations of the same product across subsets. |
| FR-05 | Fine-tune CLIP using fashion image–text pairs and save the checkpoint with the best validation score. |
| FR-06 | Generate image and text embeddings for the active product catalog. |
| FR-07 | Compare pretrained and fine-tuned CLIP on the same held-out queries, gallery, relevance rules, and values of K. |
| FR-08 | Accept JPEG, PNG, or WebP image queries and return ranked products. |
| FR-09 | Apply a soft category preference only when the category estimate reaches the configured confidence threshold; otherwise use unrestricted visual ranking. |
| FR-10 | Accept English or Vietnamese text queries and combine text-to-image and text-to-text similarity when both catalog representations are available. |
| FR-11 | Display the ranked products, category information, and similarity-related scores through a web interface. |
| FR-12 | Expose readiness information so that unavailable model instances are not treated as ready application services. |

### 3.2.2. Non-Functional Requirements

The system should satisfy the following non-functional requirements:

- **Reproducibility:** sampling seeds, dataset hashes, split assignments,
  hyperparameters, model identities, and evaluation settings must be saved.
- **Consistency:** the catalog row order, embeddings, and model checkpoint must
  belong to the same configuration. Incompatible artifacts must be rejected.
- **Usability:** users should search by image or text without choosing an
  internal category strategy.
- **Performance:** catalog embeddings should be calculated before serving so
  that an online request encodes only the query and performs vector ranking.
- **Availability:** requests should pass through a gateway that can distribute
  traffic between two backend instances and retry selected upstream failures.
- **Maintainability:** data preparation, training, evaluation, comparison,
  recommendation logic, API code, frontend code, and deployment configuration
  should remain in separate modules.
- **Input safety:** uploads must be checked for supported media type, size, and
  valid image content before model inference.

## 3.3. Overall System Architecture

The architecture contains an offline pipeline and an online application. The
offline pipeline produces versioned files used by the application. The online
application does not depend on the comparison output; it depends on the active
catalog, the corresponding embeddings, and the selected CLIP checkpoint.

**[Figure 3.1. Overall architecture of the fashion product retrieval system]**

The intended content of Figure 3.1 is summarized as follows:

```text
Raw CSV and images
        |
        v
Data preparation and audit --> processed products.csv
        |                              |
        v                              v
Train/validation/test split       embedding generation
        |                              |
        v                              v
CLIP fine-tuning --> best checkpoint  image/text arrays
        |                              |
        +------ model comparison       +-------------------+
                                                           |
User --> React frontend --> Nginx API gateway --> FastAPI backend 1/2
                                                   |
                                                   v
                                      model + catalog + embeddings
                                                   |
                                                   v
                                        ranked product response
```

The responsibilities of the online components are shown in Table 3.1.

| Component | Technology | Responsibility |
|---|---|---|
| Frontend | React and Vite | Collect an image or text query, send the request, and present ranked products. |
| API gateway | Nginx | Provide one API entry point, distribute requests by least connections, and retry eligible upstream failures. |
| Recommendation API | FastAPI | Validate inputs, translate Vietnamese text when requested, call the recommendation module, and serialize responses. |
| Recommendation engine | PyTorch, Transformers, NumPy | Load CLIP and the prepared artifacts, encode a query, calculate similarity, and rank products. |
| Artifact storage | CSV, JSON, NumPy arrays, safetensors | Store the catalog, metadata, embeddings, experimental records, and the selected checkpoint. |

Two backend containers use the same backend image and read the same mounted
catalog, embeddings, and checkpoint. Each process loads its own in-memory model
instance. This improves request availability and permits basic load distribution,
although it also approximately duplicates the model memory required by the
backend.

## 3.4. Data and Artifact Design

### 3.4.1. Input and Processed Catalog

The raw dataset contains product metadata and a directory of image files. The
preparation module recognizes common alternatives for image, product-name,
description, and category column names. It then constructs a normalized product
table. The main fields are shown below.

| Field | Purpose |
|---|---|
| `product_id` | Stable identifier assigned to the prepared product record. |
| `image_reference` | Original filename or image identifier from the source metadata. |
| `product_name` | Human-readable product name used in display and text construction. |
| `description` | Additional textual product information. |
| `category` | Ground-truth catalog category; blank values are represented as `Unknown`. |
| `image_path` | Resolved path to the validated product image. |
| `image_sha256` | Content hash used to detect duplicates and verify experiment data. |

The requested catalog size is a command-line parameter rather than a fixed
constant. This allows experiments or the web catalog to be prepared with, for
example, 4,000 or 5,000 products without changing source-code paths. Selection
first attempts to allocate the configured minimum to every eligible category.
Remaining positions are filled approximately in proportion to the available
source distribution while preserving the exact requested total.

The main output is `data/processed/products.csv`. The preparation summary records
the source path and hash, requested and selected sizes, random seed, number of
invalid or duplicate images, category counts, and output hash. Separate category
audit files report the counts and status of every category. These audit files
describe the prepared data; the web application still reads category values
from `products.csv`.

### 3.4.2. Training Run Artifacts

Each fine-tuning experiment has a separate directory under `runs/`. Its main
artifacts are listed below.

| Artifact | Content |
|---|---|
| `dataset.csv` | The exact records and saved train, validation, or test assignment used by the experiment. |
| `experiment.json` | Dataset identity, split information, and experiment configuration. |
| `training.json` | Training status, epoch history, model identities, validation scores, and selected epoch. |
| `dataset.category_audit.csv/.json` | Category distribution for the experiment and its subsets. |
| `best/` | The best validation checkpoint, processor configuration, and tokenizer files required to reload CLIP. |

The `best/` directory is updated only when the validation selection score
improves. It is therefore the checkpoint intended for final evaluation,
embedding generation, and application inference. Comparison reports are stored
under `comparisons/results/`, keeping evaluation output separate from the model
training directory.

### 3.4.3. Embedding Artifacts

The embedding stage produces `image_embeddings.npy` and
`text_embeddings.npy`. A metadata file associated with each array records the
catalog and model identity. At application startup, these records are checked
against the active catalog and checkpoint. The array row at position (i) must
refer to the product at position (i) in `products.csv`; otherwise, a ranked
index could be mapped to the wrong product.

## 3.5. Offline Machine-Learning Pipeline

### 3.5.1. Data Preparation and Audit

`src/prepare_dataset.py` reads the source metadata, resolves image paths,
normalizes required fields, verifies image files, calculates content hashes,
removes exact duplicates, and samples the requested catalog. The same execution
also creates the category audit through `src/audit_dataset.py`. A fixed random
seed makes the selection reproducible when the input and configuration remain
unchanged.

The category minimum is a target rather than a guarantee that unavailable data
can be created. If a source category contains fewer products than the target,
all eligible products can be selected and the audit reports the shortage. If
the total catalog budget is too small to reach the target for every category,
the sampler distributes the available positions as evenly as possible before
performing proportional filling.

### 3.5.2. Dataset Splitting

`src/clip_data.py` creates the training, validation, and test subsets. Records
that represent the same product or constructed prompt are grouped so that they
are not independently assigned to different subsets. The saved split is reused
by training and comparison rather than being regenerated for every command.
Category counts for the complete experiment and for each subset are recorded to
make missing or underrepresented categories visible.

### 3.5.3. CLIP Fine-Tuning

`src/finetune_clip.py` loads the image–text pairs through `ProductPairs` and
prepares model inputs through `PairCollator`. The collator also creates the
positive-pair matrix needed by the multi-positive symmetric contrastive loss.
Optimization uses AdamW, backpropagation, and gradient clipping. Depending on
the selected mode, either all CLIP parameters or only its projection layers and
logit scale are trainable.

A weighted random sampler reduces the dominance of large training categories.
The balancing power controls how strongly inverse category frequency affects
sampling, and a maximum sample weight limits the repetition of very small
categories. This process changes sampling probability only; it does not alter
the validation or test distribution.

After every epoch, retrieval is evaluated on the validation subset. The mean of
text-to-image Recall@1 and image-to-text Recall@1 is the checkpoint-selection
score. The test subset is not used to select an epoch. Once training is complete,
the selected model and its processor files are stored in `best/`.

### 3.5.4. Model Comparison

`comparisons/compare_clip.py` evaluates the original pretrained model and the
fine-tuned checkpoint. It first verifies that training completed and that the
dataset, base model, and checkpoint identities have not changed. Both variants
then use the same held-out test products, query set, gallery, values of (K),
and relevance definitions.

The evaluated tasks are text-to-image retrieval, image-to-text retrieval, and
category-based image-to-image retrieval. The output contains Precision@K,
Recall@K, and F1@K at the requested values, together with evaluated and skipped
query counts. Results are saved to `comparison.json`, `comparison.csv`, and
`comparison_by_category.csv` under a selected directory in
`comparisons/results/`.

The controlled modality experiment is stored under
`experiments/model_comparison/`. It evaluates SBERT as the text-only baseline,
ResNet18 as the image-only baseline, and a late-fusion system that combines the
two rankings using Reciprocal Rank Fusion. For the matched text-only comparison,
the same category-free text is also encoded by the pretrained and fine-tuned
CLIP text encoders. All systems use the exact test split saved by the CLIP
training run, the same gallery and values of (K), and the same category-based
relevance definition. The query product is excluded, and category labels are
omitted from all evaluation text to prevent target-label leakage.

ResNet18 and SBERT are task-specific baselines rather than direct replacements
for every CLIP experiment. Their embedding spaces are not aligned, so their
vectors are not compared directly. Rank fusion is used only when both image and
text are available. Direct cross-modal retrieval remains a separate CLIP task.
The `run_all.py` entry point executes the modality comparison and the existing
pretrained-versus-fine-tuned CLIP comparison while saving their tables in
separate subdirectories. It also produces `matched_comparison.csv`, which places
the compatible text-only and image-only category-retrieval results in one table.
The category-free fine-tuned CLIP text result is treated as an inference
ablation because the checkpoint was originally trained with category-bearing
prompts.

### 3.5.5. Catalog Embedding Generation

`src/generate_clip_embeddings.py` loads the selected catalog and CLIP model,
encodes product images and constructed product text in batches, normalizes the
vectors, and writes the embedding arrays and metadata. The website normally
uses embeddings generated by the same fine-tuned checkpoint mounted into its
backend containers. Regenerating comparison embeddings is not part of starting
the website.

## 3.6. Online Recommendation Design

### 3.6.1. Image-Based Recommendation

The image request follows these steps:

1. The API checks the media type, enforces the 5 MiB file limit, and verifies
   that the uploaded bytes form a valid image.
2. The image is converted to RGB and processed by the active CLIP image encoder.
3. Cosine similarity is calculated between the query vector and every catalog
   image vector.
4. The backend estimates a category from visual neighbors.
5. If the category confidence is sufficiently high, a small bonus is added to
   products in the predicted category. Otherwise, the original visual ranking
   is used without a category restriction.
6. The highest-scoring products and processing information are returned.

To reduce category-size bias, every category contributes only the same maximum
number of its strongest non-negative similarities to category estimation. The
implementation raises these similarities to the fourth power and averages the
strongest configured values. If the two highest category scores are (c_1) and
(c_2), the heuristic confidence is

\[
\operatorname{confidence} = \frac{c_1}{c_1+c_2}.
\]

This value expresses how clearly the first category exceeds the second under
the heuristic; it is not a calibrated classification probability. With the
current default threshold of 0.60, a confident result uses

\[
s_{image}(q,p_i) = \operatorname{cos}(q,v_i)
+ \lambda\,\mathbf{1}[category_i=\hat{c}],
\]

where (lambda=0.05). The category is a soft preference, so products from other
categories remain candidates. Below the threshold, (lambda) is effectively
zero and ranking depends only on visual similarity. The frontend exposes this
as one automatic behavior rather than presenting separate category buttons.

### 3.6.2. Text-Based Recommendation

For an English query, the text is sent directly to the CLIP text encoder. If
the user selects Vietnamese, the backend first translates the query into
English using `Helsinki-NLP/opus-mt-vi-en`, because the selected CLIP model is
primarily used with English text in this system.

The resulting query embedding is compared with catalog image vectors. When
catalog text embeddings are also available, the system calculates both
text-to-image and text-to-text scores. The final score is

\[
s_{text}(q,p_i)=
\beta\,\operatorname{cos}(q,v_i)
+(1-\beta)\,\operatorname{cos}(q,t_i),
\]

where (v_i) and (t_i) are the product image and text embeddings and
(0\leq\beta\leq1) is the selected image weight. If text embeddings are absent,
the system falls back to the cross-modal text-to-image score. The API returns
the original query, the translated or processed query, both component scores
when available, and the final ranking score.

## 3.7. API and User Interface Design

The recommendation endpoints are listed in Table 3.2.

| Endpoint | Input | Main output |
|---|---|---|
| `POST /api/recommendations/image` | Multipart image, `top_k`, and the automatic soft-category request mode | Applied category mode, optional prediction and confidence, timing, and ranked products. |
| `POST /api/recommendations/text` | JSON query, language, `top_k`, and image weight | Original and processed query, component scores, elapsed time, and ranked products. |
| `/api/health/ready` | No query content | Whether the backend model and required artifacts are ready. |

The text API accepts between 1 and 10 returned products, a text query of at most
512 characters, and an image weight from 0 to 1. The image API supports JPEG,
PNG, and WebP. Validation errors are returned with an HTTP status and a
machine-readable error code so that the frontend can display an appropriate
message.

The React interface separates image and text input while presenting results in
a common product-card format. Image users select a file and the number of
results; they are not required to choose a category mode. Text users provide a
query, language, number of results, and the desired balance between visual and
textual matching. During a request, the interface displays a loading state and
then presents either the returned products or a failure message.

## 3.8. Deployment and Reliability Design

Docker Compose defines one application configuration containing the frontend,
two backend services, and the Nginx API gateway. The complete web stack is
started with:

```powershell
docker compose up -d --build
```

The command builds the frontend and backend images and starts the services, but
it assumes that the processed catalog, embeddings, and selected checkpoint have
already been prepared or downloaded. The data, embeddings, and checkpoint are
mounted read-only into both backend containers. Logs are mounted separately,
and a named Hugging Face cache volume avoids downloading the same external model
files after every container recreation.

Nginx uses the `least_conn` strategy to route a new request to the backend with
fewer active connections. It retries connection errors, timeouts, and selected
502, 503, or 504 responses. Health checks distinguish gateway liveness from
backend readiness. Docker restarts failed services according to the configured
policy.

This arrangement provides basic load distribution and failure tolerance for a
local or demonstration deployment. It is not equivalent to an automatically
scaling production platform. Both backends run on the same Docker host, each
loads a copy of the model, and the system does not currently use a shared query
cache or external database. A host failure can still stop the complete system,
and model inference remains constrained by the host's CPU, GPU, and memory.

## 3.9. Experimental Validity and Reproducibility

The design uses the following controls to keep comparisons interpretable:

- The random seed and exact data assignments are saved.
- Source files, prepared data, and models are identified through metadata and
  hashes where applicable.
- Validation data select the best epoch; test data are reserved for the final
  report.
- Pretrained and fine-tuned CLIP use the same test queries, gallery, relevance
  definitions, and (K) values.
- Query-level and category-level summaries are both retained to expose the
  influence of category imbalance.
- The number of evaluated and skipped queries is included with the metrics.
- SBERT and ResNet18 are compared only on compatible modality-specific tasks;
  their combined result uses rank fusion rather than direct vector comparison.
- The web ranking policy is described separately from the controlled model
  comparison because category bonuses, translation, and score blending can
  affect application behavior.

These controls do not eliminate every limitation. Category labels are only a
proxy for image similarity, translation may change query meaning, and offline
retrieval metrics do not directly measure user satisfaction. These limitations
should be considered when interpreting the numerical results in Chapter 5.

## 3.10. Chapter Summary

This chapter defined the requirements, data artifacts, processing stages, and
deployment architecture of the fashion product retrieval system. The offline
pipeline prepares and audits the catalog, creates reproducible splits,
fine-tunes and compares CLIP, and generates compatible embeddings. The online
pipeline uses a React interface, an Nginx gateway, and two FastAPI instances to
serve image and text retrieval. Automatic category fallback reduces the risk of
an uncertain category decision, while precomputed embeddings make repeated
catalog ranking practical. The next chapter can present implementation details,
followed by the experimental results and discussion in Chapter 5.
