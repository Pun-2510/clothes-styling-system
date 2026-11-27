# CHAPTER 4. SYSTEM IMPLEMENTATION

## 4.1. Implementation environment and source organization

Python is used for data preparation, model training, embedding generation, evaluation, translation, and ranking. PyTorch and Transformers load and fine-tune CLIP. Pandas and NumPy process metadata and embedding arrays. FastAPI provides the backend, React and Vite provide the browser interface, and Docker Compose connects the services.

| Technology | Role |
|---|---|
| Python | Data preparation, training, encoding, retrieval, and evaluation. |
| PyTorch | CLIP optimization and inference. |
| Transformers | CLIP, BERT, processor, and tokenizer loading. |
| Torchvision | ResNet18, EfficientNet-B0, and ViT-B/16 baselines. |
| Sentence Transformers | SBERT text baseline. |
| Scikit-learn | TF-IDF baseline and supporting evaluation utilities. |
| FastAPI | Recommendation, catalog, health, and system-information APIs. |
| React and Vite | Unified search, catalog pages, explanations, Favorites, and Developer dashboard. |
| Nginx | Public reverse proxy and internal API request distribution. |
| Docker Compose | Local service construction and startup. |

The deployed application is self-contained under `project/`:

```text
project/
├── data/
│   ├── data.csv and data/            # source metadata and images
│   └── processed/                    # active processed catalog and audits
├── src/                              # preparation, training and retrieval logic
├── runs/                             # project-local checkpoints and training records
├── embeddings/                       # active image and text embeddings
├── retrieval_weights.json            # validation-selected serving configuration
├── web/
│   ├── backend/                      # FastAPI application
│   ├── frontend/                     # React application
│   ├── api-gateway/                  # internal Nginx load balancer
│   └── reverse-proxy/                # public Nginx entry point
├── logs/
├── docker-compose.yml
└── readme.txt
```

The repository-level `experiments/model_comparison/` directory contains controlled model comparisons and their result folders. The application never needs to load an experimental catalog or checkpoint from this directory. This separation prevents an experiment from changing the deployed website unintentionally.

## 4.2. Offline data and model preparation

### 4.2.1. Catalog preparation

`project/src/prepare_dataset.py` reads the source metadata and image directory. It normalizes text fields, verifies image files, calculates SHA-based image hashes, removes exact duplicate images, selects the requested catalog size, writes `products.csv`, and invokes the category audit.

The default `diverse` sampling strategy is designed to avoid selecting many nearly identical products from a large category. For each product, the script creates two lightweight descriptors:

- an image descriptor based on resized color and shape information;
- a deterministic text descriptor created from product metadata.

Within each category, max-min selection repeatedly chooses a product that is far from products already selected. This is a preprocessing method rather than a trained model. It keeps preparation reasonably fast and reproducible. The `random` strategy is retained for controlled experiments.

`project/src/audit_dataset.py` writes:

- `products.category_audit.csv`, containing per-category counts and status;
- `products.category_audit.json`, containing a summarized machine-readable report.

The audit reports the result but does not manufacture missing data. A warning can therefore indicate that the original source category does not contain enough valid products.

### 4.2.2. Dataset splitting

`project/src/clip_data.py` constructs the product text used by CLIP and creates grouped training, validation, and test subsets. Records representing the same product or duplicate constructed content are kept together to reduce information leakage. The dataset and experiment metadata are saved in the run directory so that training and evaluation can verify that the source data have not changed.

### 4.2.3. CLIP fine-tuning

`project/src/finetune_clip.py` loads `openai/clip-vit-base-patch32`, pairs each product image with its description, and optimizes the symmetric contrastive objective. The implementation supports multiple positive pairs when several batch records represent the same product.

A weighted sampler compensates moderately for category imbalance. The category-balance power determines how strongly rare categories are favored, and a maximum sample weight prevents extremely small categories from being repeated without limit.

After every epoch, validation text-to-image and image-to-text retrieval are calculated. The checkpoint with the best mean Recall@1 is saved under the run's `best/` directory. This directory contains the model weights, CLIP configuration, processor, and tokenizer files required for reloading the model.

### 4.2.4. Catalog embedding generation

`project/src/generate_clip_embeddings.py` loads the selected checkpoint and processes catalog products in batches. It creates normalized `image_embeddings.npy` and `text_embeddings.npy` arrays and writes metadata describing the source catalog and model.

Array row order is important: row \(i\) in each embedding array must represent row \(i\) in `products.csv`. Before serving, the backend checks metadata and dimensions so that a catalog cannot be silently combined with embeddings from another model or sample.

### 4.2.5. Validation-selected serving weights

The validation tuning step evaluates possible fusion and category settings without using test results. The selected values are written to `project/retrieval_weights.json`. This file stores values such as the text-branch weight, combined image weight, attribute weight, category contributions, prototype weight, graph-neighbor count, and related/unrelated category factors.

These values are deployment configuration, not user preferences. Docker mounts the file into both backend containers, and both instances use the same settings.

## 4.3. Retrieval implementation

### 4.3.1. Shared runtime initialization

When FastAPI starts, it loads the processed catalog, fine-tuned CLIP checkpoint, processor, image embeddings, optional text embeddings, translator, and retrieval weights. It also creates indexes for category names and product identifiers.

Category text prompts are encoded once during startup. Their embeddings are used both for image category evidence and for building the category-relation graph. As a result, the graph does not require manual category pairs in `.env` and does not require regenerating catalog embeddings whenever only the relation policy changes.

### 4.3.2. Image-only retrieval

The image endpoint accepts the original upload or a crop created in the browser. The file is validated, converted to RGB, and encoded by the fine-tuned CLIP vision encoder. Matrix multiplication with normalized catalog image embeddings produces image-to-image cosine similarity for all products.

`project/src/category_ranking.py` then calculates balanced visual-neighbor evidence. It groups similarities by category, keeps the same number of strongest scores per group, removes negative contributions, emphasizes strong matches, and averages the result. This prevents a large category from winning only because it contains more products.

The same query image is also compared with CLIP category prompts. `project/src/recommend.py` combines neighbor evidence and prompt evidence, identifies the strongest categories, and calculates confidence.

For intermediate confidence, the backend checks the mutual-neighbor category graph:

- related Top-1 and Top-2 categories receive asymmetric partial bonuses;
- unrelated categories do not both receive a bonus;
- the strongest unrelated category receives only a weak contribution;
- low confidence disables category support;
- high confidence applies full support to the strongest category.

The products are finally sorted by visual similarity plus the applicable category contribution. Category never removes every other product in image-only retrieval.

### 4.3.3. Text-only retrieval

`project/src/translator.py` detects whether translation is required. English text is passed directly to the CLIP text encoder, while supported Vietnamese text can be translated before encoding.

The normalized query vector is compared with catalog image and text embeddings. `project/src/recommend.py` fuses the two components using the validation-selected text weight. If text embeddings are missing, it falls back to text-to-image retrieval.

`project/src/query_signals.py` extracts category and attribute signals. One exact catalog category creates a category-specific candidate pool. Multiple exact categories, a likely misspelling, or a broad phrase keeps unrestricted ranking and produces guidance. This avoids silently choosing one category from an ambiguous request.

### 4.3.4. Combined retrieval

Combined retrieval calculates image and text scores separately and fuses them using the stored image weight. Supported colors and patterns are matched against catalog metadata and applied with a small validation-selected attribute weight.

An exact text category has priority because it directly expresses the user's requested product type. In that case, ranking occurs within the explicit category candidate pool. Otherwise, the route uses the same image category evidence, confidence function, mutual category graph, and dynamic support policy as image-only retrieval, with its own maximum combined-category contribution.

The implementation uses raw cosine components throughout. It does not convert them to percentiles. The final value is a relative ranking score for the current query, not a probability.

### 4.3.5. Result explanation

`project/web/backend/explanations.py` translates internal ranking signals into an explanation object. Recommendation response schemas include the selected route, score labels, category candidates, confidence, applied decision, reason codes, and a small query-specific graph.

The frontend components `RankingExplanation.jsx` and `CategoryRelationGraph.jsx` display this information in an expandable section above the result cards. The graph contains only categories relevant to the current query, making it easier to understand than the complete relation network.

## 4.4. FastAPI backend

The main routes are:

| Endpoint | Purpose |
|---|---|
| `POST /api/recommendations/image` | Image-only retrieval. |
| `POST /api/recommendations/text` | Text-only retrieval. |
| `POST /api/recommendations/combined` | Combined image–text retrieval. |
| `GET /api/catalog/categories` | List catalog categories. |
| `GET /api/catalog/products` | List or paginate products, optionally by category. |
| `GET /api/catalog/products/{product_id}` | Return one product's detail data. |
| `GET /api/health` | Basic backend health. |
| `GET /api/health/ready` | Model and artifact readiness. |
| `GET /api/health/system` | Read-only system information for the Developer page. |

Recommendation requests validate empty text, supported image types, image size, decodable content, and requested Top-*K*. A successful response contains product information, component scores when applicable, category evidence, query guidance, explanation data, processing time, and the backend instance identifier.

No database is required by the current application. Catalog data are read from prepared files, while user history and Favorites are browser-local.

## 4.5. Web interface

### 4.5.1. Unified search and crop

The React frontend provides one search bar. It calls the text, image, or combined endpoint according to the supplied inputs. Users do not choose a technical search mode.

Image preview and manual crop are implemented in the browser. The crop dialog creates a new image file from the selected rectangular region. This file is used consistently for preview, persistence, history, and API submission. The user can retain the original image when cropping is unnecessary.

After a successful request, the page scrolls smoothly to the result section. “New Search” clears the current query and results, while page-refresh persistence prevents accidental loss of an unfinished session.

### 4.5.2. Catalog and product pages

The home page presents featured categories and representative products. The category menu provides a compact selection and a link to the complete category page. Products can be paginated by category, and each product has a detail page containing its image, category, name, identifier, and available description.

Skeleton placeholders preserve layout while catalog data or images are loading. Product cards are used consistently across home, category, result, and Favorites pages.

### 4.5.3. History, Favorites, and language

Recent query history is stored in browser local storage. Users can reopen a search, delete one entry, or clear all history. Images stored with history remain on the same device.

Favorites use `web/frontend/src/services/favorites.js` and a separate local-storage key. A heart control is available on product cards and the detail page. The navigation bar shows the saved count, and `/favorites` lists the local collection. Favorites require no login and do not change ranking.

The bilingual interface changes navigation, instructions, errors, explanations, and status messages between Vietnamese and English. Translation used for model input remains a backend concern and is separate from interface language.

### 4.5.4. Developer dashboard

The read-only Developer page displays the serving architecture, backend instances, model and embedding readiness, current catalog information, offline and online pipelines, ranking formulas, and loaded runtime weights. It helps demonstrate how the application is operating but cannot retrain the model or edit system configuration.

## 4.6. Docker deployment

`project/docker-compose.yml` starts five services:

| Service | Responsibility |
|---|---|
| `frontend` | Runs the React/Vite interface inside the Docker network. |
| `backend` | First FastAPI model-serving instance. |
| `backend_2` | Second FastAPI instance using the same artifacts and configuration. |
| `api-gateway` | Internal Nginx layer for rate limiting, retry handling, and `least_conn` backend distribution. |
| `reverse-proxy` | Only public entry point on port 80; routes website, API, Swagger, and image requests. |

The frontend and backend ports are not published directly. The public reverse proxy routes page traffic to the frontend and API traffic to the internal Nginx layer. The internal layer selects the backend currently handling fewer active connections.

Both backends mount the same project-local catalog, image directory, embeddings, checkpoint, and `retrieval_weights.json`. Logs are mounted separately, and a shared Hugging Face cache avoids repeated downloads. Health checks report whether required artifacts have loaded successfully.

After offline artifacts have been prepared, the complete system is started from `project/` using:

```powershell
docker compose up -d --build
```

## 4.7. Functional verification

| Function | Expected behavior |
|---|---|
| Image-only query | Rank visually similar products and apply category support only according to confidence and relation evidence. |
| English/Vietnamese text | Process English directly and supported Vietnamese through translation. |
| Exact text category | Restrict candidates to the explicitly requested category. |
| Ambiguous text category | Keep unrestricted ranking and return correction or refinement suggestions. |
| Combined query | Use the image as reference, text as refinement, explicit text category as priority, and attributes when supported. |
| Manual crop | Submit the selected garment region through the same image pipeline. |
| Result explanation | Display score components, category candidates, confidence, decision, and a compact graph. |
| Top-*K* selection | Return the requested number within the supported range and available candidates. |
| Invalid image | Reject unsupported, oversized, or invalid files with a clear error. |
| Backend distribution | Allow both FastAPI instances to answer through the internal Nginx layer. |
| Search history | Save, reopen, delete, and clear local searches on the same device. |
| Favorites | Add and remove products locally without login and preserve them after refresh. |
| System health | Display checkpoint, embeddings, catalog, backend, and gateway readiness on the Developer page. |

The functional checks confirm that the implemented paths operate as designed. They do not replace the retrieval experiments in Chapter 5, which evaluate ranking quality quantitatively and analyze failure cases.
