# CHAPTER 1. INTRODUCTION

## 1.1. Reason for Choosing the Topic
e

## 1.2. Target Implementation

The overall objective is to develop a web-based fashion product recommendation system that retrieves relevant catalog items from an uploaded image or a textual description. Product images and descriptions are represented using CLIP, and candidate products are ranked using embedding similarity. The experimental objectives include comparing BERT and ResNet50 with CLIP on compatible tasks, and evaluating the effect of fine-tuning CLIP on fashion data.

The specific objectives are as follows:

1. Prepare a structured product catalog containing identifiers, image references, product names, descriptions, and categories, with checks for missing images and duplicate records.
2. Define a text baseline using BERT and an image baseline using ResNet50, with their exact configurations and evaluation tasks specified in Chapter 5, and establish pretrained CLIP as the shared image–text representation model.
3. Fine-tune CLIP on fashion image–text pairs using contrastive learning, with separate training, validation, and test subsets.
4. Support image-based retrieval with three category strategies: unrestricted ranking, hard category filtering, and a soft category preference.
5. Support text-based retrieval using similarity to catalog image embeddings and, when available, catalog text embeddings. Vietnamese queries are supported through a translation step before encoding.
6. Compare the baseline approaches with CLIP under matched evaluation conditions where applicable, and compare pretrained and fine-tuned CLIP using Precision@K, Recall@K, and F1@K for K values of 1, 5, and 10. Report classification results separately from retrieval results.
7. Provide a React web interface and a FastAPI backend, supported by Docker Compose deployment and instructions for preparing or loading the required artifacts.

The experiments address two questions: how CLIP compares with separate text and image approaches on compatible product search tasks, and whether fine-tuning improves CLIP retrieval on the selected fashion dataset. Chapter 5 will organize the numerical results by task and model configuration. The web application demonstrates how the selected model can be used for product search.

## 1.3. Object and Scope of the Study

### 1.3.1. Objects of the Study

The objects of the study are fashion product images, associated descriptions, category labels, and the representations used to retrieve products. The models considered are BERT for text processing, ResNet50 for visual feature extraction, pretrained CLIP, and fine-tuned CLIP. The main application model is `openai/clip-vit-base-patch32`, with its original weights and a checkpoint fine-tuned on the project's fashion data.

The project uses the Mini Fashion Product Images and Text Dataset available on Kaggle (nirmalsankalana, n.d.). The selected data are processed into a catalog containing product identifiers, image references, names, descriptions, and categories. Separate training, validation, and test subsets are prepared for the model experiment. Dataset statistics and the detailed preparation procedure are presented in the experimental chapter.

### 1.3.2. Scope of the Study

The system performs content-based recommendation: results depend on the current query and the information stored for each product. Image and text queries are supported separately. Image search includes options for unrestricted ranking, filtering by a predicted category, and giving that category a ranking preference. Text search can combine similarity to catalog image embeddings and catalog text embeddings. Vietnamese text input is supported through translation into English before encoding.

The planned comparisons are organized by input and task. For image retrieval, ResNet50 features can be compared with image embeddings from pretrained and fine-tuned CLIP. For text retrieval, a BERT-based representation can be compared with the CLIP text encoder using the same queries and product descriptions. The BERT checkpoint and method used to turn its outputs into sentence vectors must be specified. Any experiment using Sentence-BERT (SBERT) or another sentence embedding model must identify the actual model rather than label it simply as BERT. Direct image–text retrieval is evaluated separately for the two CLIP variants, because independently trained BERT and ResNet50 representations do not automatically share an aligned embedding space.

The scope is limited to retrieving existing products. User purchase histories, ratings, and long-term preference profiles are not included. The project also excludes size recommendation, virtual try-on, complete outfit generation, and queries that combine a reference image with a textual modification.

The existing CLIP comparison covers text-to-image retrieval, image-to-text retrieval, and category-based image-to-image retrieval. Matching product identifiers define relevance for the first two tasks. For the third task, relevant results share the query's category, with the query product excluded. Category agreement is used as a limited measure of relevance and does not establish whether products are suitable to wear together. The BERT and ResNet50 comparisons extend this evaluation plan; numerical results will be reported only for configurations actually evaluated.

## 1.4. Research Method

The study combines a focused review of BERT, ResNet50, CLIP, and CLIP adaptation to fashion data with system development and model evaluation. The methodology includes data preparation, baseline configuration, model fine-tuning, product retrieval, and web deployment.

First, the product metadata and images are checked and organized. Product names, categories, and descriptions are combined into the text associated with each image. Exact duplicate images are removed before the experiment. Products with the same identifier or identical constructed text are kept together when splitting the data, reducing overlap between the training, validation, and test subsets. The split assignments are saved for reuse.

Second, baseline comparisons are defined for the relevant modalities. ResNet50-based image retrieval uses visual features to rank candidate images. BERT-based text retrieval requires a specified sentence representation and uses text similarity to rank product descriptions. Within each comparison, the methods must use the same test queries, candidate products, relevance labels, and K values. Text queries must not reproduce the exact target descriptions in a way that makes retrieval a trivial self-match. Earlier category classification experiments are reported separately using classification metrics; their scores are not compared directly with retrieval metrics.

Third, pretrained CLIP provides the starting point for fine-tuning. The model is trained on image–text pairs from the training subset using contrastive learning, which encourages matching images and descriptions to have similar representations. The selected training settings and implementation details are presented in the later chapters.

The fine-tuned model is evaluated on the validation subset after each epoch. The checkpoint with the highest mean Recall@1 across text-to-image and image-to-text retrieval is retained. The test subset is used only for the final comparison. Both CLIP variants are evaluated using the same test queries, candidate products, relevance definitions, and K values.

The comparison reports Precision@K, Recall@K, and F1@K at K = 1, 5, and 10. Each metric is calculated per eligible query and then averaged. The definitions and treatment of queries without relevant candidates are explained in the evaluation chapter. These measurements assess retrieval on the selected test set; they do not directly measure user satisfaction or the effect of every web application feature.

Finally, image and text embeddings for the catalog are generated using the selected checkpoint. The backend converts a query into an embedding, calculates cosine similarity scores, and returns ranked products to the frontend. Saved metadata help verify that the model, catalog, and embeddings belong to the same configuration. Docker Compose starts the application services with access to the required model and data files.

## 1.5. Practical Significance

The project provides a working prototype for finding fashion products through an image or a description. Users can inspect ranked product results and, for image queries, choose how strongly the search depends on category. This offers a practical interface for exploring the catalog without requiring users to know an exact product name.

From an engineering perspective, the project connects data preparation, model training, embedding generation, backend processing, and frontend interaction. Computing catalog embeddings in advance avoids repeatedly encoding every product for each request. Saved experimental records and deployment instructions make the workflow easier to inspect and reproduce.

Comparing text and image baselines with CLIP is intended to clarify the practical role of each representation approach. The pretrained versus fine-tuned CLIP experiment further examines the effect of adaptation to the selected fashion data. Together, the application and the task-specific evaluations provide a basis for choosing a model for product search and for subsequent work on larger catalogs or user preferences.

## References

Chia, P. J., Attanasio, G., Bianchi, F., Terragni, S., Magalhães, A. R., Goncalves, D., Greco, C., & Tagliabue, J. (2022). Contrastive language and vision learning of general fashion concepts. *Scientific Reports, 12*, Article 18958. https://doi.org/10.1038/s41598-022-23052-9

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep residual learning for image recognition* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1512.03385

nirmalsankalana. (n.d.). *Mini fashion product images and text dataset* [Data set]. Kaggle. Retrieved September 28, 2026, from https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In M. Meila & T. Zhang (Eds.), *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 8748–8763). PMLR. https://proceedings.mlr.press/v139/radford21a.html
