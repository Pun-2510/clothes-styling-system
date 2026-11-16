# APPENDICES

The appendices provide supporting information for the system analysis, implementation, deployment, and experimental evaluation presented in Chapters 3, 4, and 5. They are intended to make the project easier to inspect and reproduce without interrupting the main discussion with long configuration tables or per-run results.

# APPENDIX A. DATASET PREPARATION AND AUDIT

## A.1. Input and processed data

The system uses product metadata together with the corresponding catalog images. The raw metadata and image directory are processed offline before model training or website deployment.

**Table A.1. Main input and processed data artifacts**

| Artifact | Repository location | Purpose |
|---|---|---|
| Raw product metadata | `project/data/data.csv` | Contains product identifiers, names, descriptions, categories, and image references |
| Raw product images | `project/data/data/` | Supplies the visual content associated with the metadata |
| Processed catalog | `project/data/processed/products.csv` | Provides the cleaned and sampled catalog used by the deployed system |
| Preparation summary | `project/data/processed/products.preparation.json` | Records the source hash, seed, sampling strategy, and preparation statistics |
| Category audit table | `project/data/processed/products.category_audit.csv` | Gives detailed category counts and audit status |
| Category audit summary | `project/data/processed/products.category_audit.json` | Gives a machine-readable summary of the audit |
| Audience audit table | `project/data/processed/products.audience_audit.csv` | Records the metadata-derived audience group assigned to each product |
| Audience audit summary | `project/data/processed/products.audience_audit.json` | Summarizes audience counts, evidence sources, and rows that require review |

The preparation stage normalizes metadata, checks image availability, removes duplicate images, and samples products within categories. The current serving catalog uses category-aware diverse sampling. This strategy considers color, foreground shape, a low-resolution visual descriptor, and product metadata so that the selected examples within a category are not unnecessarily repetitive.

## A.2. Current serving-catalog summary

**Table A.2. Data-preparation summary for the deployed catalog**

| Item | Value |
|---|---:|
| Eligible source products | 43,662 |
| Requested products | 10,000 |
| Selected products | 10,000 |
| Product categories | 142 |
| Duplicate images removed | 779 |
| Invalid images detected | 0 |
| Sampling seed | 42 |
| Minimum target per category | 25 |
| Categories with audit status `ok` | 87 |
| Categories with audit status `warning` | 55 |

An audit warning does not automatically mean that preprocessing failed. It often means that the source dataset itself contains fewer than 25 eligible products for a category. Therefore, the audit is used to document imbalance and support interpretation rather than to claim that all categories are equally represented.

## A.3. Metadata-derived audience summary

The audience browsing feature does not predict a person's gender or age from an image. It groups catalog products by terms already present in product names and descriptions. Products with insufficient or conflicting evidence remain in the `unknown` group instead of being forced into another audience.

**Table A.3. Audience groups in the deployed catalog**

| Audience group | Products |
|---|---:|
| Men | 4,042 |
| Women | 3,744 |
| Children | 448 |
| Unisex | 525 |
| Unknown or insufficient metadata | 1,241 |

The counts above cover all 10,000 serving products. The audience field is an optional browsing preference and is not used as ground truth in the retrieval experiments.

## A.4. Separation of serving and experimental outputs

The website uses the 10,000-product serving catalog and its matching model artifacts. The research experiments described in Appendix D are separate and use three controlled 5,000-product catalogs. Serving and experimental artifacts must not be mixed because their product order, embeddings, checkpoints, and hashes may differ.

# APPENDIX B. SYSTEM INTERFACES AND FUNCTIONAL VERIFICATION

## B.1. Main API endpoints

The React frontend communicates with the FastAPI backend through the same public origin. Nginx routes API requests to two backend instances, while catalog browsing endpoints remain read-only.

**Table B.1. Main system endpoints**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check whether the API process is responding |
| GET | `/api/health/ready` | Check model, embeddings, translator, and catalog readiness |
| GET | `/api/health/system` | Provide safe runtime information for the Developer dashboard |
| GET | `/api/catalog/categories` | Return catalog categories and representative images |
| GET | `/api/catalog/audiences` | Return metadata-derived audience groups |
| GET | `/api/catalog/products` | Browse products with category, audience, and pagination parameters |
| GET | `/api/catalog/products/{product_id}` | Return product details and related products |
| POST | `/api/recommendations/text` | Retrieve products using a textual query |
| POST | `/api/recommendations/image` | Retrieve products using an uploaded image |
| POST | `/api/recommendations/combined` | Retrieve products using an image and refining text together |

## B.2. Functional verification scenarios

**Table B.2. Main functional verification scenarios**

| Scenario | User action | Expected behavior |
|---|---|---|
| English text query | Enter an English product description | Return a ranked Top-K product list |
| Vietnamese text query | Enter a Vietnamese description | Translate the query and return related products |
| Image query | Upload a JPEG, PNG, or WebP image | Compare the image with catalog images and return Top-K results |
| Manual image crop | Select the garment region before searching | Use the selected region as the image query |
| Combined query | Upload an image and add a description | Use the image as the main reference and text as refinement |
| Exact text category | Enter an explicit catalog category | Restrict candidates to the clearly identified category |
| Ambiguous text | Enter a misspelled or overly broad expression | Avoid an uncertain hard category filter and provide a suggestion when possible |
| Audience preference | Select women, men, children, or unrestricted audience | Apply the selected metadata-derived preference without claiming automatic gender recognition |
| Result explanation | Open the explanation tab | Show the score components and category evidence used for ranking |
| Search history | Reopen a recent local search | Restore the query on the same browser device |
| Favorites | Select the heart icon on a product | Save or remove the product locally without requiring an account |
| System health | Open the Developer page | Display model, embedding, translator, and backend readiness |

Search history, favorites, and interface language preferences are stored locally in the browser. They are convenience features and are not used to retrain the model or personalize experimental results.

## B.3. Interface evidence

Insert the final screenshots at the marked positions. Use screenshots from the same deployed version described in Chapter 4.

**[Insert screenshot of the main image-text search interface here]**

*Figure B.1. Main interface for image, text, and combined product retrieval.*

**[Insert screenshot of Top-K results and the result-explanation tab here]**

*Figure B.2. Ranked results and the explanation of retrieval signals.*

**[Insert screenshot of the Developer dashboard here]**

*Figure B.3. Runtime health and retrieval-pipeline information shown on the Developer dashboard.*

# APPENDIX C. DEPLOYMENT AND REPRODUCTION GUIDE

## C.1. Docker services

**Table C.1. Services in the deployed web system**

| Service | Internal port | Responsibility |
|---|---:|---|
| Frontend | 5173 | Serve the React user interface |
| Backend | 8000 | First FastAPI model-serving instance |
| Backend 2 | 8000 | Second FastAPI model-serving instance |
| API gateway | 80 | Apply rate limiting and distribute API requests using `least_conn` |
| Reverse proxy | 80 | Expose the website, API, documentation, and catalog images through `http://localhost` |

Only the reverse proxy publishes a host port. The two backend instances use the same catalog, checkpoint, embeddings, and retrieval-weight file so that requests receive equivalent model behavior.

## C.2. Required serving artifacts

**Table C.2. Local artifacts required before website startup**

| Artifact | Default project location |
|---|---|
| Processed catalog | `project/data/processed/products.csv` |
| Catalog images | `project/data/data/` |
| Fine-tuned CLIP checkpoint | `project/runs/clip_finetune/best/` |
| Image embeddings | `project/embeddings/image_embeddings.npy` |
| Text embeddings | `project/embeddings/text_embeddings.npy` |
| Embedding metadata | `project/embeddings/*.json` |
| Selected retrieval weights | `project/retrieval_weights.json` |

The paths may be changed in `project/.env`. All artifacts must originate from the same processed catalog and model run.

## C.3. Startup and verification commands

From the cloned repository, the user first changes into the project directory. The repository folder name may differ and is therefore not hard-coded.

```powershell
cd .\project
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://localhost/api/health/ready
```

After changing `.env`, the checkpoint, embeddings, or retrieval weights, the containers can be recreated with:

```powershell
docker compose up -d --build --force-recreate
```

The website is then available at `http://localhost`, and the API documentation is available at `http://localhost/docs`.

# APPENDIX D. EXPERIMENTAL REPRODUCIBILITY

## D.1. Controlled experimental setting

The experimental catalogs are independent of the 10,000-product serving catalog. Three catalogs were prepared with seeds 42, 123, and 2026 to reduce dependence on one random sample. Each experiment compares methods on the same query set and full gallery within its seed.

**Table D.1. Common experimental configuration**

| Item | Setting |
|---|---|
| Catalog sampling strategy | Category-aware diverse sampling |
| Random seeds | 42, 123, and 2026 |
| Gallery size | 5,000 products per seed |
| Evaluated queries | 460, 464, and 462, respectively |
| Retrieval depths | K = 1, 5, and 10 |
| Main reported depth | K = 10 |
| Category-based relevance | For model baselines, category-aware ranking, and end-to-end serving evaluation, a returned product is relevant when it has the same ground-truth category as the query product; the query product itself is excluded from the gallery |
| Cross-modal CLIP alignment relevance | For text-to-image and image-to-text fine-tuning evaluation, the relevant item is the paired product with the same product identifier across modalities |
| Metrics | Precision@K, Recall@K, F1@K, NDCG@K, and MAP@K |
| Aggregation | Unweighted mean and sample standard deviation across the three seeds |
| Weight selection | Validation subset only |
| Final evaluation | Test queries not used to select checkpoints or retrieval weights |

In this report, *ground truth* means the category label already stored in the dataset. It is used as the reference for evaluation and is not a label generated by the retrieval system.

The compact evidence committed with the report is stored in `docs/reports/evidence/final_experiments/`. It contains aggregate tables and the detailed per-seed records used to produce the results in Chapter 5 and this appendix.

## D.2. Experiment groups

**Table D.2. Experimental groups and purposes**

| Experiment group | Purpose |
|---|---|
| Model comparison | Compare text-only, image-only, and multimodal representation methods under the same setting |
| CLIP fine-tuning comparison | Compare pretrained and fine-tuned CLIP on the same test data |
| Category-aware ranking | Compare category-free, hard-filter, threshold-based, confidence-weighted, and dynamic ranking |
| System-pipeline evaluation | Evaluate the complete image-only, text-only, and combined paths used by the application |
| Combined-query ablation | Measure the contribution of raw fusion, attribute evidence, and category evidence |

## D.3. Per-seed category-aware results

**Table D.3. Category-aware ranking results for each seed at K = 10**

| Seed | Ranking strategy | Precision | Recall | F1 | NDCG | MAP |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Category-free | 0.5676 | 0.1299 | 0.1755 | 0.5999 | 0.4920 |
| 42 | Hard category filter | 0.7130 | 0.1658 | 0.2364 | 0.7130 | 0.7130 |
| 42 | Threshold-based soft category | 0.5748 | 0.1328 | 0.1793 | 0.6056 | 0.5007 |
| 42 | Confidence-weighted category | 0.6600 | 0.1523 | 0.2087 | 0.6797 | 0.6256 |
| 42 | Dynamic relation-aware category | 0.6107 | 0.1401 | 0.1902 | 0.6393 | 0.5515 |
| 123 | Category-free | 0.5545 | 0.1295 | 0.1762 | 0.5804 | 0.4675 |
| 123 | Hard category filter | 0.6940 | 0.1532 | 0.2230 | 0.6940 | 0.6940 |
| 123 | Threshold-based soft category | 0.5625 | 0.1319 | 0.1797 | 0.5867 | 0.4765 |
| 123 | Confidence-weighted category | 0.6373 | 0.1472 | 0.2036 | 0.6539 | 0.5981 |
| 123 | Dynamic relation-aware category | 0.5933 | 0.1381 | 0.1891 | 0.6170 | 0.5234 |
| 2026 | Category-free | 0.5463 | 0.1244 | 0.1691 | 0.5716 | 0.4678 |
| 2026 | Hard category filter | 0.6753 | 0.1517 | 0.2202 | 0.6753 | 0.6753 |
| 2026 | Threshold-based soft category | 0.5506 | 0.1255 | 0.1707 | 0.5749 | 0.4730 |
| 2026 | Confidence-weighted category | 0.6288 | 0.1423 | 0.1991 | 0.6450 | 0.5984 |
| 2026 | Dynamic relation-aware category | 0.5911 | 0.1334 | 0.1833 | 0.6122 | 0.5280 |

The hard filter scores highly under category-based relevance because it limits retrieval to one predicted category. However, an incorrect category can remove useful products completely. The deployed dynamic method keeps the full gallery and uses category information as a soft ranking signal.

## D.4. Per-seed end-to-end query results

**Table D.4. End-to-end query-mode results for each seed at K = 10**

| Seed | Query mode | Precision | Recall | F1 | NDCG | MAP |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Image-only | 0.6107 | 0.1401 | 0.1902 | 0.6393 | 0.5515 |
| 42 | Text-only | 0.6996 | 0.1714 | 0.2357 | 0.7210 | 0.6646 |
| 42 | Image and text | 0.7211 | 0.1725 | 0.2416 | 0.7344 | 0.7039 |
| 123 | Image-only | 0.5933 | 0.1381 | 0.1891 | 0.6170 | 0.5234 |
| 123 | Text-only | 0.6974 | 0.1705 | 0.2367 | 0.7156 | 0.6585 |
| 123 | Image and text | 0.7209 | 0.1793 | 0.2485 | 0.7302 | 0.6969 |
| 2026 | Image-only | 0.5911 | 0.1334 | 0.1833 | 0.6122 | 0.5280 |
| 2026 | Text-only | 0.7160 | 0.1779 | 0.2461 | 0.7368 | 0.6815 |
| 2026 | Image and text | 0.7043 | 0.1720 | 0.2400 | 0.7159 | 0.6867 |

No query was skipped in these end-to-end evaluations. The variation among seeds supports reporting the mean and sample standard deviation in Chapter 5 instead of relying on one catalog sample.

# APPENDIX E. RETRIEVAL AND CATEGORY-EVIDENCE CONFIGURATION

## E.1. Selected retrieval weights

The deployed weights were selected on validation data. Test results were not used to choose these values.

**Table E.1. Selected retrieval configuration**

| Configuration item | Environment key | Selected value |
|---|---|---:|
| Text-to-image contribution in text search | `TEXT_RETRIEVAL_IMAGE_WEIGHT` | 0.50 |
| Image contribution in combined search | `COMBINED_IMAGE_WEIGHT` | 0.70 |
| Attribute contribution | `COMBINED_ATTRIBUTE_WEIGHT` | 0.00 |
| Combined-query category contribution | `COMBINED_CATEGORY_WEIGHT` | 0.15 |
| Maximum image-query category bonus | `CATEGORY_BONUS` | 0.20 |
| Category-prompt evidence weight | `CATEGORY_PROTOTYPE_WEIGHT` | 1.00 |
| Relation-graph neighbours per category | `CATEGORY_RELATION_NEIGHBORS` | 5 |
| Related Top-1 category factor | `RELATED_TOP1_BONUS_FACTOR` | 0.50 |
| Related Top-2 category factor | `RELATED_TOP2_BONUS_FACTOR` | 0.20 |
| Unrelated Top-1 category factor | `UNRELATED_TOP1_BONUS_FACTOR` | 0.10 |

## E.2. CLIP category prompt ensemble

For each catalog category, the system inserts its name into four fixed English templates:

1. `a catalog product photo of {category}`
2. `a product photo showing {category}`
3. `an ecommerce product image in the category {category}`
4. `a studio product image classified as {category}`

The CLIP text encoder converts the four prompts into vectors. Their normalized average represents the textual category prototype. Let \(\mathbf{v}_q\) denote the query-image embedding and \(\mathbf{t}_c\) the text prototype for category \(c\). Prompt evidence is:

\[
P_c = \operatorname{cosine}(\mathbf{v}_q,\mathbf{t}_c)
\]

The prompt evidence is combined with balanced visual-neighbour evidence \(V_c\):

\[
R_c = V_c + \alpha\max(P_c,0)
\]

Here, \(R_c\) is the combined category evidence and \(\alpha\) controls the contribution of the category prompt. Category evidence assists ranking; it is not treated as a probability and does not automatically prove that an image belongs to one category.

# APPENDIX F. ARTIFACT AND SOURCE TRACEABILITY

**Table F.1. Main implementation and evidence files**

| Purpose | Repository location |
|---|---|
| Data preparation and diverse sampling | `project/src/prepare_dataset.py` |
| Dataset and category audit | `project/src/audit_dataset.py` |
| CLIP fine-tuning | `project/src/finetune_clip.py` |
| Embedding generation | `project/src/generate_clip_embeddings.py` |
| Retrieval-weight selection | `project/src/tune_retrieval_weights.py` |
| Runtime retrieval and ranking | `project/src/recommend.py` |
| Category prompt templates | `project/src/category_prompts.py` |
| Category-ranking policy | `project/src/category_ranking.py` |
| Selected serving weights | `project/retrieval_weights.json` |
| Serving dataset audit | `project/data/processed/products.category_audit.csv` |
| Serving audience audit | `project/data/processed/products.audience_audit.csv` |
| Serving training history | `project/runs/clip_finetune/training.json` |
| Final evidence guide | `docs/reports/evidence/final_experiments/README.md` |
| Three-seed aggregate evidence | `docs/reports/evidence/final_experiments/` |
| Detailed per-seed evidence | `docs/reports/evidence/final_experiments/run_details/seed_<seed>/` |
| Local regenerable experiment outputs | `experiments/model_comparison/results/` |

The final lightweight CSV, JSON, Markdown summaries, and detailed run records are committed under `docs/reports/evidence/final_experiments/`. Large checkpoints, embeddings, model caches, and regenerable raw result directories remain local and are excluded from Git. Raw terminal logs were not produced by every completed experiment and are therefore not reconstructed. The tracked evidence above is sufficient to verify the reported aggregates without making the printed appendices unnecessarily long.
