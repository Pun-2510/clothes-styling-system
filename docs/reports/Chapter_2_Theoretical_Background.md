# CHAPTER 2. THEORETICAL BACKGROUND AND RELATED WORK

## 2.1. Fashion product retrieval

A personalized recommendation system normally predicts products for a particular user from purchase history, ratings, or interaction logs. This project addresses a different problem: content-based retrieval. The current query is compared with product content already stored in a catalog, so the system can work without a user account or previous shopping behavior.

Let the catalog be \(C=\{p_1,p_2,\ldots,p_N\}\), where a product may contain an image, name, description, and category. Given a query \(q\), the system calculates a score \(s(q,p_i)\) for every candidate product, sorts the candidates from highest to lowest score, and returns the first \(K\) products.

An image query emphasizes appearance, including shape, color, texture, and pattern. A text query emphasizes the meaning expressed by the description. A combined query uses both sources: the image supplies the visual reference, while the text specifies how the desired product should be refined.

## 2.2. Feature representations and cosine similarity

Raw pixels and text tokens cannot be compared directly. An encoder converts each input into an embedding, which is a fixed-length numerical vector. Related inputs should be located near one another in the embedding space, while unrelated inputs should be farther apart.

The project uses cosine similarity:

\[
\operatorname{cos}(\mathbf{a},\mathbf{b})=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\lVert\mathbf{a}\rVert_2\lVert\mathbf{b}\rVert_2}
\]

where \(\mathbf{a}\) and \(\mathbf{b}\) are two embeddings, \(\mathbf{a}\cdot\mathbf{b}\) is their dot product, and \(\lVert\cdot\rVert_2\) is the vector length. A larger value means that the two representations point in more similar directions.

Catalog embeddings are generated offline and stored before the website starts. During a search, only the new query must be encoded. Its embedding can then be compared with all stored catalog embeddings efficiently.

## 2.3. Text retrieval approaches

### 2.3.1. TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) represents a document according to the words it contains. Term frequency increases the importance of words occurring in the document, while inverse document frequency reduces the importance of words that occur in many documents. The resulting sparse vectors can be compared using cosine similarity.

TF-IDF is simple, fast, and easy to interpret. However, it mainly depends on word overlap. It may not recognize that two different words express similar meanings. In this project, TF-IDF is used as a traditional text-only baseline rather than as part of the deployed system.

### 2.3.2. BERT

The Transformer architecture uses attention to model relationships between words. BERT learns contextual word representations by considering the surrounding text in both directions (Devlin et al., 2019). In the experiments, the frozen pretrained checkpoint `google-bert/bert-base-uncased` produces token representations. Mask-aware mean pooling converts the valid token representations into one vector for each query or product description.

Standard BERT was not originally optimized specifically for sentence-similarity retrieval. It is included to show the behavior of a general pretrained language model when used as a text encoder.

### 2.3.3. Sentence-BERT

Sentence-BERT (SBERT) adapts Transformer models to produce sentence embeddings that can be compared efficiently (Reimers & Gurevych, 2019). The project uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as a text-only baseline. Queries and catalog descriptions are encoded into the same textual space and ranked using cosine similarity.

SBERT supports semantic text matching better than a word-overlap method in many situations. However, its text space is not aligned with features produced by an independent image model such as ResNet18. Their vectors therefore cannot be directly added or compared.

When SBERT and ResNet18 are evaluated together, the project uses Reciprocal Rank Fusion (RRF). RRF combines their ranked lists rather than their incompatible embeddings. A product receives a larger contribution when it appears near the top of either list. This provides an independent-model multimodal baseline without claiming that the two encoders share one embedding space.

## 2.4. Image retrieval approaches

### 2.4.1. ResNet18

Convolutional Neural Networks learn visual patterns from images. Earlier layers commonly capture edges, colors, and textures, while later layers represent more complex shapes. ResNet introduced shortcut connections that improve the training of deep networks (He et al., 2015). ResNet18 contains 18 layers and is used as an image-only baseline through features from its penultimate layer.

### 2.4.2. EfficientNet-B0

EfficientNet scales network depth, width, and image resolution in a coordinated manner (Tan & Le, 2019). EfficientNet-B0 is the smallest standard variant and provides a resource-efficient image baseline. The experiment uses ImageNet-pretrained weights and extracts pooled visual features before the classifier.

### 2.4.3. Vision Transformer

Vision Transformer (ViT) divides an image into patches and processes them as a sequence using Transformer layers (Dosovitskiy et al., 2021). The project includes ImageNet-pretrained ViT-B/16 as another image-only baseline. Its class-token representation is used as the image embedding.

The standard ViT baseline and the visual encoder inside CLIP are not equivalent. Both use image patches, but standard ViT is pretrained for image classification, whereas CLIP's visual encoder is trained jointly with text descriptions. CLIP therefore learns visual features that are aligned with language.

ResNet18, EfficientNet-B0, and ViT-B/16 are compared only on matched image-to-image tasks. They do not contain compatible text encoders and are not direct text-to-image baselines.

## 2.5. Contrastive Language–Image Pre-training

CLIP contains an image encoder and a text encoder (Radford et al., 2021). Projection layers map both outputs into a shared embedding space. During pretraining, matching image–text pairs are encouraged to have high similarity and non-matching pairs in the same batch are encouraged to have lower similarity.

The project uses `openai/clip-vit-base-patch32`. Its visual encoder divides an image into 32 × 32 pixel patches. The model was selected because it supports both modalities, its pretrained checkpoint and processor are publicly available, and its size is practical for the available environment.

The shared space allows several comparisons:

- \(S_{II}\): query image to catalog image;
- \(S_{TI}\): query text to catalog image;
- \(S_{TT}\): query text to catalog text;
- \(S_{IT}\): query image to catalog text.

This is the main advantage of CLIP over independently pretrained text and image models. The deployed application can use one selected checkpoint for all three query modes.

## 2.6. Contrastive fine-tuning for fashion data

A general pretrained model may not fully represent the names, categories, colors, materials, and visual styles used by a particular fashion catalog. Fine-tuning adapts CLIP using the project's fashion image–text pairs. FashionCLIP provides related evidence that domain adaptation can improve fashion concept representations (Chia et al., 2022).

The project uses a symmetric contrastive objective:

\[
\mathcal{L}=\frac{1}{2}
(\mathcal{L}_{I\rightarrow T}+\mathcal{L}_{T\rightarrow I})
\]

where \(\mathcal{L}_{I\rightarrow T}\) retrieves the correct text for an image and \(\mathcal{L}_{T\rightarrow I}\) retrieves the correct image for a text. The implementation also supports multiple positive pairs so that repeated valid representations of the same product are not incorrectly treated as negatives.

The training subset updates the model. The validation subset is evaluated after every epoch and selects the checkpoint with the highest average Recall@1 across text-to-image and image-to-text retrieval. The test subset is not used for training or checkpoint selection.

## 2.7. Category-aware retrieval

Similarity alone can return visually related products from an unintended category. A strict category filter can solve this problem when the category is known, but it can also remove good results when the category prediction is uncertain. Category-aware retrieval therefore treats category evidence according to its source and reliability.

For image queries, the system estimates category evidence from two sources. The first source is the category distribution among strong visual neighbors. The second source compares the query image with CLIP text prompts created from catalog category names. Combining the two sources reduces dependence on either noisy neighbors or a single prompt score.

The system also builds a category-relation graph from CLIP category-prompt embeddings. Two categories are considered closely related only when each appears among the other's nearest category neighbors. This mutual rule is stricter than a one-way similarity and does not require manually listing pairs such as Jeans–Jeggings or Shirts–Kurtas.

Confidence determines how this evidence affects ranking. High confidence supports the strongest category. At intermediate confidence, two mutually related categories may receive different bonuses. If the two strongest categories are not related, only a small bonus is applied to the first category. Low confidence disables category support and returns to unrestricted similarity ranking.

For text queries, an explicitly written and unambiguous category is stronger evidence because it is supplied directly by the user. The system may restrict candidates to that category. Misspelled, broad, or multiple category expressions do not cause an arbitrary hard filter; suggestions are returned instead.

## 2.8. Category imbalance and data sampling

Fashion catalogs are often imbalanced: common categories contain many products, while rare categories contain only a few. Random selection and training may therefore be dominated by common categories.

The project addresses imbalance in three places. First, catalog preparation uses a minimum-per-category target when the source data and requested catalog capacity allow it. Second, diversity-aware sampling selects varied products within a category through deterministic max-min selection over lightweight image and text descriptors. Third, fine-tuning uses a weighted sampler so that smaller categories can be observed more often, while a maximum sample weight prevents extreme repetition.

These methods do not create new products and cannot repair a category that is already scarce in the source dataset. The category audit is therefore retained to distinguish a sampling problem from a source-data limitation.

## 2.9. Retrieval evaluation metrics

Retrieval evaluation requires a query, a gallery of candidate products, and a relevance rule. Same-category experiments consider another product relevant when it shares the query's ground-truth category. Exact paired image–text experiments use the matching product identifier. The query product itself is excluded where necessary to avoid a trivial self-match.

### 2.9.1. Precision@K

\[
\operatorname{Precision@K}=
\frac{\text{relevant products in Top-}K}{K}
\]

Precision@*K* answers: “Of the products returned near the top, how many are relevant?”

### 2.9.2. Recall@K

\[
\operatorname{Recall@K}=
\frac{\text{relevant products in Top-}K}
{\text{all relevant products in the gallery}}
\]

Recall@*K* answers: “How much of the available relevant catalog did the Top-*K* list recover?”

### 2.9.3. F1@K

\[
\operatorname{F1@K}=
2\frac{\operatorname{Precision@K}\operatorname{Recall@K}}
{\operatorname{Precision@K}+\operatorname{Recall@K}}
\]

F1@*K* is high only when both the quality of the returned list and the coverage of relevant products are high.

### 2.9.4. NDCG@K

Normalized Discounted Cumulative Gain gives more importance to relevant products appearing earlier in the list. For binary relevance:

\[
\operatorname{DCG@K}=\sum_{j=1}^{K}
\frac{rel_j}{\log_2(j+1)},
\qquad
\operatorname{NDCG@K}=\frac{\operatorname{DCG@K}}
{\operatorname{IDCG@K}}
\]

where \(rel_j\) is 1 when the product at position \(j\) is relevant and 0 otherwise. \(\operatorname{IDCG@K}\) is the best possible DCG for that query. NDCG@*K* is useful because two methods can retrieve the same number of relevant products but place them at different positions.

Metrics are first calculated for every eligible query and then averaged. Category-macro results give each eligible category equal importance. Queries without a relevant candidate are reported as skipped instead of being silently assigned zero. Because category agreement is only a practical relevance proxy, these metrics do not fully measure detailed style similarity, price suitability, or user satisfaction.

To reduce dependence on one selected catalog, the final experiment repeats the comparisons across three sampled catalogs and reports the mean and standard deviation. The mean describes average performance, while the standard deviation shows how much the result changes between samples.
