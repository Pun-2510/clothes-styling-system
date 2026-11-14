# CHAPTER 3. SYSTEM ANALYSIS AND DESIGN

## 3.1. Problem analysis

The system must help a user find products when the available information is incomplete. The user may have a photograph but not know the product name, may have only a written description, or may use text to clarify an image. Therefore, the system supports three situations:

- **Image-only query:** find products with visual characteristics similar to an uploaded image.
- **Text-only query:** find products related to a written description.
- **Combined image–text query:** use an image as the visual reference and text to refine the same target product.

The output is a ranked Top-*K* list from an existing catalog. It is not only a category prediction. Category is supporting evidence that can improve ranking, but an uncertain prediction must not remove every product from the correct class. For this reason, image category evidence is applied softly and includes an unrestricted fallback. An exact category explicitly written by the user can be treated more strongly because it is a direct instruction.

The selected data contain imbalanced categories, short descriptions, repeated visual styles, and images with different backgrounds or viewpoints. The design must therefore include data auditing, diversity-aware sampling, category-balanced training, confidence-aware ranking, and clear fallback behavior.

## 3.2. System requirements

### 3.2.1. Functional requirements

| Requirement | Description |
|---|---|
| Data preparation | Validate metadata and images, remove exact duplicates, select a configurable and diverse catalog, and audit category distribution. |
| Model preparation | Fine-tune CLIP and retain the best validation checkpoint. |
| Embedding generation | Generate compatible image and text embeddings for the active serving catalog. |
| Unified query processing | Accept an image, text, or both through one search area and route the request automatically. |
| Category-aware ranking | Apply image category evidence according to confidence and category relationships; apply an exact textual category as a direct candidate constraint. |
| Product ranking | Calculate raw cosine-based scores and return a configurable Top-*K* list. |
| Query assistance | Suggest category corrections or refinements for misspelled, broad, or ambiguous text. |
| Result explanation | Explain the main score components, category candidates, confidence, and ranking decisions for the current query. |
| Catalog browsing | Browse categories and paginated products and open a product detail page. |
| Local user utilities | Store recent searches and Favorites in the browser without requiring an account. |
| System inspection | Display model, artifact, backend, pipeline, formula, and runtime status on a read-only Developer page. |
| Deployment | Serve the website through Nginx and distribute API requests across two FastAPI instances. |

### 3.2.2. Non-functional requirements

The system must be reproducible. Dataset selections, random seeds, split assignments, model settings, checkpoints, embedding metadata, validation-selected weights, and evaluation settings are recorded. The catalog, checkpoint, and embedding arrays must belong to the same configuration.

The online path must avoid repeating expensive catalog computation. Product embeddings are generated offline, while an online request only encodes the new query and compares it with stored vectors.

The interface must hide unnecessary model complexity. Users are not asked to choose a retrieval branch, category strategy, or fusion weight. The system determines these choices from the supplied inputs and validation-selected configuration.

The deployment must provide clear input errors and basic availability controls. Health checks, request-size limits, bounded retries, rate limits, two backend instances, and Docker restart policies provide reliability suitable for a local research prototype.

## 3.3. Overall system architecture

The design separates offline preparation from online retrieval. This distinction prevents model training or catalog encoding from running during a user request.

### 3.3.1. Offline project pipeline

The project-local offline flow is:

```text
Raw metadata and images
        ↓
Validation, duplicate removal and diversity-aware sampling
        ↓
Processed catalog and category audit
        ↓
Grouped train/validation/test split
        ↓
CLIP fine-tuning and best-checkpoint selection
        ↓
Catalog image/text embeddings
        ↓
Validation-based retrieval weights
        ↓
Serving artifacts under project/
```

The training subset updates CLIP. The validation subset selects the best checkpoint and ranking parameters. The test subset remains separate for evaluation. After selection, the checkpoint generates embeddings for the active serving catalog.

The repository-level `experiments/` directory is independent from the application artifacts in `project/`. Experimental seeds may create different catalogs, checkpoints, embeddings, and result files without changing the deployed website.

### 3.3.2. Online retrieval pipeline

The online flow is:

```text
Browser
   ↓
Public Nginx reverse proxy
   ├──→ React frontend
   └──→ Internal Nginx API load balancer
                     ├──→ FastAPI backend 1
                     └──→ FastAPI backend 2
                                  ↓
                Fine-tuned CLIP + catalog embeddings
                                  ↓
                  Ranking, explanation and Top-K
```

When a backend starts, it loads the processed catalog, checkpoint, embeddings, translator, category configuration, and retrieval weights. The request route is selected automatically from the available image and text inputs.

## 3.4. Data and model processing design

### 3.4.1. Data preparation

The preparation stage normalizes product fields, validates image files, calculates image hashes, and removes exact duplicate images. It then selects the requested number of products.

The default selection method is diversity-aware. Within each category, the method uses lightweight image color/shape information together with hashed text metadata. It starts with a deterministic product and repeatedly selects a product far from those already chosen. This encourages variation without running the full CLIP model during data preparation. A random strategy remains available for controlled comparison.

A category audit records source count, selected count, coverage, split distribution, and warning state for every category. The minimum-per-category value is a target, not a guarantee, because the source data may contain fewer products than requested.

Related records are grouped during splitting to reduce leakage. The saved split assignments allow training and evaluation to reuse the same data.

### 3.4.2. Fine-tuning and embedding generation

CLIP is fine-tuned using fashion image–text pairs and symmetric contrastive learning. A category-balanced sampler changes how often existing training products are observed so that common categories do not dominate every batch. It does not generate synthetic products.

After each epoch, validation retrieval is calculated. The checkpoint with the best average Recall@1 across text-to-image and image-to-text directions is retained. The selected model generates normalized catalog image and text embeddings in the same row order as the processed catalog.

Metadata checks verify that the checkpoint, catalog, and embeddings are compatible. Validation then selects serving weights, which are stored in a project-local JSON file and loaded automatically by both backend instances.

## 3.5. Recommendation design

### 3.5.1. Input routing and notation

At least one input must be provided:

- image without text → image-only retrieval;
- text without image → text-only retrieval;
- image and text → combined retrieval.

Let \(q\) be the current query and \(p_i\) the \(i\)-th catalog product. The main symbols are:

| Symbol | Meaning |
|---|---|
| \(S_{II}(q,p_i)\) | Similarity between the query image and product image. |
| \(S_{TI}(q,p_i)\) | Similarity between the query text and product image. |
| \(S_{TT}(q,p_i)\) | Similarity between the query text and product text. |
| \(S_T(q,p_i)\) | Fused text-branch score. |
| \(S_{base}(q,p_i)\) | Initial image–text fusion score. |
| \(A(p_i)\) | Metadata attribute match for supported colors or patterns. |
| \(V_c\) | Visual-neighbor evidence for category \(c\). |
| \(P_c\) | CLIP prompt evidence for category \(c\). |
| \(R_c\) | Combined category evidence. |
| \(h\) | Confidence of the strongest category. |
| \(g(h)\) | Confidence strength between 0 and 1. |
| \(W(c_i)\) | Dynamic support assigned to the category of product \(p_i\). |
| \(\beta\) | Text-to-image contribution in the text branch. |
| \(\omega_I\) | Image contribution in combined retrieval. |
| \(\omega_A\) | Attribute contribution. |
| \(\lambda_I,\lambda_C\) | Maximum image-only and combined category contributions. |

The numeric parameter values are selected from validation data and reported with the experimental settings rather than embedded permanently in the formulas.

### 3.5.2. Image-only retrieval

The submitted image is converted into a normalized CLIP image embedding. A manual crop, when used, simply becomes the submitted image; it does not create another score. The initial score is \(S_{II}(q,p_i)\).

Category evidence is calculated from two complementary sources.

**Visual-neighbor evidence.** Catalog products are grouped by category. The system takes the same maximum number of strongest non-negative similarities from each category and emphasizes strong matches before averaging them. The equal limit reduces the advantage of categories that contain many more products. The resulting evidence is \(V_c\).

**CLIP prompt evidence.** Each catalog category is represented by a short prompt such as “a catalog product photo of shirts.” The query image embedding is compared with the prompt embedding. Its non-negative score is \(P_c\).

The evidence is combined as:

\[
R_c=V_c+\alpha P_c
\]

where \(\alpha\) controls the prompt contribution. Let \(r_1\) and \(r_2\) be the two strongest supporting category scores. Confidence is:

\[
h=\frac{r_1}{r_1+r_2}
\]

The confidence-strength function is:

\[
g(h)=
\begin{cases}
0, & h\le h_0,\\
\dfrac{h-h_0}{h_1-h_0}, & h_0<h<h_1,\\
1, & h\ge h_1.
\end{cases}
\]

Here, \(h_0\) is the confidence floor and \(h_1\) is the full-confidence threshold.

The system does not manually declare related category pairs. It compares category prompt embeddings and builds a mutual-neighbor graph. Categories \(c_a\) and \(c_b\) are considered related only when each occurs among the other's nearest category neighbors.

The dynamic support \(W(c_i)\) follows three cases:

- At low confidence, \(W(c_i)=0\) for every product and ranking is unrestricted.
- At high confidence, products from the strongest category receive full support.
- At intermediate confidence, the two strongest categories receive different partial supports only when they are mutual neighbors. If they are unrelated, only the strongest category receives a small support.

The final image score is:

\[
S_{image}(q,p_i)=S_{II}(q,p_i)+\lambda_I g(h)W(c_i)
\]

This is a ranking bonus, not a probability and not a hard filter. Products from other categories remain eligible.

### 3.5.3. Text-only retrieval

English text is encoded directly. Vietnamese text can first be translated into English. The query is compared with both image and text catalog embeddings:

\[
S_T(q,p_i)=\beta S_{TI}(q,p_i)+(1-\beta)S_{TT}(q,p_i)
\]

If catalog text embeddings are unavailable, the backend falls back to \(S_{TI}\).

Text category handling is deliberately different from image prediction. If the query contains one exact catalog category, for example “red shirts,” the user has explicitly stated the product type, so products are ranked within that category. If the query contains multiple categories, a spelling error, or only a broad fashion phrase, the system keeps the full catalog available and returns suggestions instead of applying an uncertain filter.

### 3.5.4. Combined image–text retrieval

The image and text are encoded separately. The text branch first uses the same formula as text-only retrieval. The base combined score is:

\[
S_{base}(q,p_i)=\omega_I S_{II}(q,p_i)+(1-\omega_I)S_T(q,p_i)
\]

If a supported color or pattern is found in the text, the attribute-adjusted score is:

\[
S_{attr}(q,p_i)=(1-\omega_A)S_{base}(q,p_i)+\omega_A A(p_i)
\]

Otherwise, \(S_{attr}=S_{base}\).

When the text contains one exact category, that explicit instruction defines the candidate set. The system does not add a competing image category bonus. When no exact text category is available, the combined route reuses the image evidence, confidence function, and dynamic category-relation policy:

\[
S_{final}(q,p_i)=S_{attr}(q,p_i)+\lambda_C g(h)W(c_i)
\]

All similarity components are raw cosine similarities. Percentile normalization is not used. Combined retrieval assumes that the image and text describe the same desired product. It does not interpret two different product types as an outfit-matching request.

### 3.5.5. Ranking explanation

Every successful response can include a compact explanation. It identifies the selected query route, similarity components, category candidates, confidence state, applied bonuses or filters, and reason codes. A query-scoped graph displays only the relevant category nodes and relationships instead of exposing the complete internal graph.

The explanation describes why the current system returned a result; it does not claim that a product is objectively correct. This distinction is important because the final score is a relative ranking value, not a probability of user satisfaction.

## 3.6. Interface and deployment design

The React interface provides one search area, category browsing, product details, local history, local Favorites, Vietnamese–English switching, manual crop, result guidance, and a Developer dashboard. Search state may be retained after refresh, while “New Search” clears the current query and results.

The navigation and product cards are designed for retrieval rather than commerce. There are no payment or purchasing actions. Favorites are a browser-local collection and do not change model ranking.

The public Nginx service is the only service exposed on port 80. It sends page requests to the frontend and API/image requests to the internal API layer. The internal Nginx service distributes API traffic between two equivalent FastAPI instances using `least_conn`, applies endpoint-specific rate limits, forwards headers, and performs bounded retries. This separation makes the external reverse-proxy role and the internal API load-balancing role explicit.

## 3.7. Evaluation design

Validation is used for checkpoint and parameter selection; test data are used for final assessment. The controlled experiments use three independently prepared catalogs of 5,000 products with seeds 42, 123, and 2026. Every comparison within a seed uses the same catalog, queries, relevance rule, and values of \(K\). The final report presents the mean and standard deviation across the three seeds.

The experiment groups are:

1. **Controlled modality baselines:** TF-IDF, BERT, SBERT, and CLIP for text; ResNet18, EfficientNet-B0, ViT-B/16, and CLIP for images; and rank-level SBERT–ResNet18 fusion as an independent multimodal baseline.
2. **CLIP fine-tuning comparison:** pretrained CLIP versus fine-tuned CLIP for matched image, text, and cross-modal tasks.
3. **Serving-component comparison:** individual cosine components, text fusion, image–text fusion, and pretrained/fine-tuned serving variants.
4. **Category-ranking comparison:** unrestricted ranking, hard category filtering, confidence-weighted ranking, and dynamic relation-aware category ranking.
5. **Full-pipeline ablation:** base similarity, attribute adjustment, category handling, and the complete deployed pipeline.

The measurements are Precision@*K*, Recall@*K*, F1@*K*, NDCG@*K*, category-macro results, evaluated-query count, and skipped-query count at \(K\in\{1,5,10\}\). Chapter 5 presents the numeric results and error analysis.

## 3.8. Design limitations

The system retrieves only products contained in the prepared catalog. It assumes one principal target product per request. Manual cropping can reduce background influence, but the system does not automatically detect or segment all garments.

Category relationships are inferred from CLIP prompt representations and are therefore model-dependent. Closely related categories may still be visually confused, especially when source categories are scarce or inconsistently labeled. Text suggestions reduce uncertain filtering but do not fully understand every free-form sentence.

Category-based relevance is a practical evaluation proxy. Products in the same category may differ in style, material, price, or user preference. The system therefore requires both quantitative evaluation and qualitative error analysis.
