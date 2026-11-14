# CHAPTER 1. INTRODUCTION

## 1.1. Reason for choosing the topic

Online product catalogs provide many choices, but a large catalog can also make product discovery difficult. A user may see a fashion item in daily life, on social media, or on another website and want to find a similar item without knowing its exact name. In another situation, the user may only be able to describe the desired item, for example, “a red polo shirt with a simple pattern.” A keyword search may fail when the catalog uses different words, while an image-only search may be affected by the background, the person wearing the product, or other objects in the photograph.

This project addresses the problem as content-based fashion product retrieval. It compares the current query with the images and descriptions of products already available in a catalog, then returns a ranked Top-*K* list. The query may contain an image, text, or both. When both are provided, the image supplies the visual reference and the text clarifies the requested product type, color, or pattern.

Separate image and text models provide useful reference points. TF-IDF, BERT, and Sentence-BERT (SBERT) represent different approaches to text retrieval, while ResNet18, EfficientNet-B0, and Vision Transformer (ViT-B/16) represent image retrieval. However, these independently pretrained models do not naturally place images and text in the same representation space.

Contrastive Language–Image Pre-training (CLIP) solves this limitation by learning aligned image and text representations (Radford et al., 2021). This shared space makes image-to-image, text-to-image, text-to-text, and combined retrieval possible within one framework. The project uses pretrained CLIP as the starting point and fine-tunes it on fashion image–text pairs to study whether domain adaptation improves retrieval on the selected data. The idea is also motivated by FashionCLIP, which demonstrates the value of adapting CLIP to fashion concepts (Chia et al., 2022).

The project combines model research with a working web application. In addition to returning products, the application helps the user understand the result through category evidence and a compact explanation graph. This makes the model behavior easier to inspect without requiring the user to understand the underlying equations.

## 1.2. Target implementation

The main objective is to develop and evaluate a category-aware multimodal fashion retrieval system using CLIP. The specific objectives are:

- Prepare a configurable product catalog containing identifiers, images, product names, descriptions, and categories; validate images; remove exact duplicates; and report category distribution.
- Use diversity-aware sampling so that the selected catalog contains varied products within each category instead of repeatedly selecting highly similar records.
- Establish text-only baselines using TF-IDF, pretrained BERT, and SBERT, and image-only baselines using ResNet18, EfficientNet-B0, and ViT-B/16.
- Compare pretrained CLIP with fine-tuned CLIP under matched query, gallery, relevance, and Top-*K* conditions.
- Fine-tune CLIP on fashion image–text pairs using separate training, validation, and test subsets and category-balanced training sampling.
- Support image-only, text-only, and combined image–text retrieval through one search interface.
- Use category information as supporting evidence. Image categories influence ranking according to confidence and learned category relationships, while an exact category explicitly written by the user can restrict the text candidate set.
- Select fusion weights and category parameters using validation data rather than asking the website user to configure technical values.
- Evaluate retrieval using Precision@*K*, Recall@*K*, F1@*K*, NDCG@*K*, per-category results, and mean and standard deviation across multiple sampled catalogs.
- Provide catalog browsing, product details, local search history, local Favorites, manual image cropping, bilingual Vietnamese–English interface, result explanations, and a read-only Developer dashboard.
- Deploy the website with a public Nginx reverse proxy, an internal Nginx API load balancer, two FastAPI backend instances, a React frontend, and Docker Compose.

The project investigates three principal questions. First, how do text-only, image-only, and multimodal approaches differ on matched retrieval tasks? Second, does fashion-specific fine-tuning improve CLIP over its pretrained version? Third, how does category-aware ranking affect the complete serving pipeline compared with unrestricted similarity ranking?

## 1.3. Object and scope of the study

### 1.3.1. Objects of the study

The objects of the study are fashion product images, product names, descriptions, category labels, and the numerical representations used to retrieve products. The principal model is `openai/clip-vit-base-patch32`, evaluated in both pretrained and fashion-fine-tuned forms. Additional models are used as controlled baselines rather than as components of the deployed website.

The project uses the Mini Fashion Product Images and Text Dataset. Source records are converted into a configurable catalog containing product identifiers, image paths, names, descriptions, and categories. The deployed application and the controlled experiments use separately prepared artifacts so that experimental runs cannot silently replace the website catalog, checkpoint, embeddings, or retrieval weights.

### 1.3.2. Scope of the study

The study is limited to retrieving existing products from the prepared catalog.

**Data scope**

- Only products with usable metadata and valid image files are included.
- Exact duplicate images are removed before sampling.
- The catalog size is configurable. The deployed project may use a larger serving catalog, while controlled experiments use three independently sampled 5,000-product catalogs to measure result stability.
- Purchase history, ratings, payment information, and user-account data are not used.

**Functional scope**

- The system supports image-only, text-only, and combined image–text queries.
- The output is a ranked Top-*K* list of catalog products, not a generated product or a purchasing decision.
- For an image query, category evidence is estimated from visually similar catalog products and CLIP category prompts. Confidence and learned relationships between categories determine whether one category, two related categories, or no category receives ranking support.
- For a text query, an exact single category written by the user can restrict the candidate products. Misspelled, broad, or ambiguous category text does not trigger an uncertain hard filter; the interface instead provides suggestions.
- For a combined query, the image is the main visual reference and the text refines the same target product. An explicit text category has priority over an uncertain image category. Supported color and pattern words provide an additional metadata-based signal.
- Vietnamese text can be translated into English before CLIP encoding. English text is processed directly.
- A user may manually crop the intended garment before image retrieval. Cropping changes the submitted image but does not introduce a separate model or score.
- Search history and Favorites are stored only in the current browser. No login is required, and this local information is not used to personalize model ranking.

**Out-of-scope functions**

The project does not cover outfit compatibility, personalized recommendation from long-term behavior, product price prediction, size recommendation, virtual try-on, automatic product generation, payment, or inventory management. It also does not automatically detect and segment every garment in a complex scene. If an image contains several products, the user may crop the desired region; otherwise, CLIP encodes the complete image and unrelated content may affect the result.

The application is a functional research prototype. It demonstrates retrieval, category-aware ranking, explanation, and basic request distribution, but it is not intended to provide the security, automatic scaling, and operational guarantees of a commercial e-commerce platform.

## 1.4. Research method

The study combines literature review, system development, controlled experimentation, and error analysis.

First, source metadata and images are validated. Exact duplicate images are removed, text fields are normalized, and products are selected with a deterministic diversity-aware method. A category audit records how many products are available and selected in each category. Related records are kept together when the data are divided into training, validation, and test subsets to reduce information leakage.

Second, the project defines matched baselines. TF-IDF, BERT, SBERT, pretrained CLIP, and fine-tuned CLIP are compared for text retrieval. ResNet18, EfficientNet-B0, ViT-B/16, pretrained CLIP, and fine-tuned CLIP are compared for image retrieval. A rank-level combination of SBERT and ResNet18 provides an independent-model multimodal baseline. Comparisons use the same catalog, query set, relevance rule, and values of *K* whenever the task permits.

Third, pretrained CLIP is fine-tuned using image–text pairs and a symmetric contrastive objective. The training subset updates the model. The validation subset selects the best checkpoint and serving weights. The test subset is kept separate for final evaluation.

Fourth, experiments are repeated on three independently sampled catalogs using seeds 42, 123, and 2026. Each catalog contains 5,000 products. Results are summarized using the mean and standard deviation, reducing the risk of drawing a conclusion from one favorable or unfavorable sample.

Finally, the selected checkpoint generates the serving catalog embeddings. The deployed backend applies the query-specific ranking pipeline and records interpretable signals such as similarity components, category candidates, confidence, relation evidence, and reason codes. Functional verification covers the web interface, input validation, result explanation, local history and Favorites, service health, and request distribution.

## 1.5. Practical significance

The project provides a complete example of converting a multimodal model into an inspectable web retrieval system. Users can search with the information they already have: an image, a description, or both. They do not need to choose an internal model, category strategy, or fusion weight.

The controlled baselines show the strengths and limitations of different representation approaches. Repeated experiments help distinguish stable behavior from results caused by one sampled catalog. Category evidence and the explanation panel make the final ranking more understandable, while unrestricted fallback prevents uncertain category predictions from always controlling the result.

The resulting pipeline can serve as a foundation for later work involving larger catalogs, improved garment detection, user feedback, or personalized recommendation. These extensions remain separate from the objectives evaluated in the current project.
