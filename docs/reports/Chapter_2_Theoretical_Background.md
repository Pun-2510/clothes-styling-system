# CHAPTER 2. THEORETICAL BACKGROUND

This chapter presents the concepts used to design and evaluate the fashion
product retrieval system. The discussion focuses on content-based retrieval,
vector representations, ResNet18, BERT, SBERT, CLIP, contrastive fine-tuning, category
imbalance, and ranking metrics. These concepts also define the conditions under
which the models can be compared fairly in the experimental chapter.

## 2.1. Fashion Product Retrieval

A conventional personalized recommendation system predicts products for a
particular user from information such as purchase history, ratings, or
interaction logs. The present project addresses a different problem. It
performs content-based retrieval, in which the current image or text query is
compared with product content already stored in the catalog. Consequently, the
system can operate without a user account or a history of previous purchases.

Let the catalog be (C = \{p_1, p_2, \ldots, p_N\}), where every product can
contain an image, a name, a description, and a category. Given a query (q),
the system calculates a relevance score (s(q,p_i)) for each product and sorts
the products in descending order. The first (K) products form the returned
recommendation list. In this project, the input can be either an uploaded image
or a textual description.

Image-based retrieval emphasizes visual information such as shape, color,
pattern, and general appearance. Text-based retrieval emphasizes the concepts
expressed by the query. A multimodal model is valuable because it can represent
both inputs within one framework and can support comparisons between different
modalities.

## 2.2. Feature Representation and Embeddings

Raw pixels and words cannot be compared directly. A representation model
therefore converts an input into an embedding, which is a fixed-length numerical
vector. Inputs with related content should be located close to one another in
the embedding space, while unrelated inputs should be farther apart.

The project uses L2-normalized embeddings. For an embedding (z), its normalized
form is

\[
\hat{z} = \frac{z}{\lVert z \rVert_2}.
\]

The similarity between a query vector (q) and a product vector (p) is
measured using cosine similarity:

\[
\operatorname{cos}(q,p) =
\frac{q \cdot p}{\lVert q \rVert_2\lVert p \rVert_2}.
\]

When both vectors are normalized, cosine similarity is equal to their dot
product. This makes ranking efficient because catalog embeddings can be
generated once, saved as numerical arrays, and reused for later queries. Only
the new query must be encoded while the application is running.

## 2.3. Convolutional Neural Networks and ResNet18

Convolutional neural networks learn visual features through a sequence of
convolution, activation, and down-sampling operations. Earlier layers usually
respond to local patterns such as edges and textures, while deeper layers can
represent higher-level structures. A vector taken from a late network layer can
therefore be used as an image descriptor for similarity search.

ResNet introduced residual learning to make deep visual networks easier to
optimize (He et al., 2015). Instead of requiring a group of layers to learn a
complete mapping (H(x)), a residual block learns (F(x)=H(x)-x) and produces

\[
y = F(x) + x.
\]

The shortcut connection allows the original signal to pass through the block
and improves gradient propagation. ResNet18 is an 18-layer configuration that
uses basic residual blocks. It is smaller than deeper ResNet variants and is
practical for image classification and feature extraction on limited hardware.

In this study, ResNet18 is treated as a visual baseline. Features extracted
from product images can be compared using cosine similarity for image-to-image
retrieval. However, a standard ResNet18 has no text encoder and does not place
text and images into a shared representation space. It should therefore be
compared with the CLIP image encoder only on a matched image retrieval task,
not on direct text-to-image retrieval.

## 2.4. Transformers, BERT, and Sentence-BERT

The Transformer architecture uses attention to model relationships between
tokens without relying on recurrent processing (Vaswani et al., 2017). Through
self-attention, every token can incorporate information from other tokens in
the same sequence. Positional information is added because self-attention alone
does not express token order.

BERT is a pretrained Transformer encoder that learns bidirectional contextual
representations (Devlin et al., 2019). A word representation can depend on both
its left and right context, which helps the model distinguish meanings that
cannot be identified from an isolated word. Its original pretraining includes
masked language modeling, in which selected tokens are hidden and predicted
from their context.

BERT produces token-level contextual outputs. A retrieval experiment must
therefore state how these outputs are converted into one vector for the complete
description, for example by using a designated sequence token or a pooling
operation. A model explicitly trained for sentence similarity, such as a
Sentence-BERT variant, must be identified by its actual name rather than being
reported simply as BERT.

In this project, the text-only baseline is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a multilingual
Sentence-BERT model. It directly produces a sentence embedding for a product
name and description. It can rank catalog descriptions for a textual query,
but it cannot directly compare its vectors with ResNet18 image vectors. Its
fair role is consequently a text-to-text retrieval baseline under the same
query, gallery, relevance definition, and values of (K).

## 2.5. Contrastive Language–Image Pre-training

CLIP learns visual and textual representations from paired images and natural
language (Radford et al., 2021). Its dual-encoder architecture contains an image
encoder and a text encoder. Projection layers transform the two encoder outputs
to vectors of the same dimension, after which the vectors are normalized and
compared in a shared embedding space.

For a matching pair, the image and text vectors are trained to have a high
similarity. Non-matching combinations within the same batch act as negative
examples. Once trained, the encoders can be used independently: an uploaded
image is processed by the image encoder, while a text query is processed by the
text encoder. Because the two outputs occupy an aligned space, either query can
be compared with catalog embeddings.

The model selected for the main system is `openai/clip-vit-base-patch32`. Its
visual component is a Vision Transformer that divides an image into 32-by-32
pixel patches. The model was selected because its pretrained weights and
processor are publicly available, it supports both required input modalities,
and its size is practical for the available experimental environment.

The shared space gives CLIP an important advantage over independent SBERT and
ResNet18 models. CLIP supports text-to-image and image-to-text matching without
requiring an additional mapping between separately trained feature spaces. It
also supports image-to-image search by comparing the outputs of its image
encoder.

## 2.6. Contrastive Fine-Tuning for Fashion Data

A pretrained model provides general visual and linguistic knowledge, but the
distribution of a fashion catalog may differ from its original training data.
For example, product descriptions may repeatedly use specialized names,
categories, colors, and apparel attributes. Fine-tuning adapts the pretrained
parameters using image–text pairs from the selected catalog. Previous work on
FashionCLIP demonstrates the motivation for adapting CLIP representations to
fashion concepts (Chia et al., 2022).

For a mini-batch of (B) image–text pairs, let (v_i) be the normalized image
embedding and (t_j) the normalized text embedding. Their scaled similarity is

\[
s_{ij} = \frac{v_i^Tt_j}{\tau},
\]

where τ is a learned or configured temperature. The project implements a
symmetric contrastive objective. The image-to-text term encourages each image
to identify its matching text, while the text-to-image term performs the reverse
task:

\[
L = \frac{1}{2}\left(L_{I\rightarrow T}+L_{T\rightarrow I}\right).
\]

If more than one entry in a batch represents the same product, all corresponding
entries can be treated as positives. For an image (i) with a positive set
(P(i)), the multi-positive image-to-text loss is

\[
L_{I\rightarrow T} = -\frac{1}{B}\sum_{i=1}^{B}
\frac{1}{|P(i)|}\sum_{j\in P(i)}
\log\frac{\exp(s_{ij})}{\sum_{k=1}^{B}\exp(s_{ik})}.
\]

The reverse term is calculated in the same manner after transposing the
similarity matrix. This formulation avoids incorrectly treating repeated valid
pairs as negatives.

Fine-tuning can update the entire CLIP model or only selected projection layers
and the logit scale. Full fine-tuning offers greater adaptation capacity but
requires more memory and may overfit a small dataset. Projection-only training
is lighter but changes fewer parameters. The chosen mode and hyperparameters
must therefore be recorded with every experiment.

The training subset updates the model parameters. The validation subset is used
after each epoch to choose the best checkpoint. The test subset is reserved for
the final comparison and must not influence parameter updates or checkpoint
selection. In the implemented pipeline, the best checkpoint is selected by the
mean of text-to-image Recall@1 and image-to-text Recall@1 on the validation
subset.

## 2.7. Category Imbalance and Balanced Sampling

Fashion datasets frequently contain many examples for common categories and
only a few examples for rare categories. Random sampling can make this imbalance
worse, and training batches may then be dominated by common products. A model
may achieve an acceptable overall score while still performing poorly on rare
categories.

The project addresses this problem at two stages. During catalog preparation,
a configurable minimum number of products per category is selected when the
available data and requested catalog size permit it. During fine-tuning, a
weighted sampler gives products from smaller training categories a higher
probability of being selected. For a category (c), the sample weight follows
the capped inverse-frequency form

\[
w_c = \min\left(w_{max},
\left(\frac{n_{max}}{n_c}\right)^{\alpha}\right),
\]

where (n_c) is the category size, (n_{max}) is the largest category size,
(alpha) controls the strength of balancing, and (w_{max}) prevents extreme
weights. A value of (alpha=0) disables this balancing effect, whereas a larger
value increases the preference for rare categories.

Balanced sampling does not create new products and does not guarantee correct
classification. It changes how frequently existing training items are observed.
For this reason, overall retrieval metrics are accompanied by per-category and
category-macro results so that improvements are not inferred only from common
categories.

## 2.8. Retrieval Evaluation Metrics

Retrieval evaluation requires a query, a candidate gallery, and a rule defining
which gallery products are relevant. Let (R_q) be the set of relevant products
for query (q), and let (A_q^K) be the first (K) returned products. The core
metrics used by this project are Precision@K, Recall@K, and F1@K.

\[
\operatorname{Precision@K}(q) =
\frac{|A_q^K \cap R_q|}{|A_q^K|},
\]

\[
\operatorname{Recall@K}(q) =
\frac{|A_q^K \cap R_q|}{|R_q|},
\]

and

\[
\operatorname{F1@K}(q) =
\frac{2\,\operatorname{Precision@K}(q)\,
\operatorname{Recall@K}(q)}
{\operatorname{Precision@K}(q)+\operatorname{Recall@K}(q)}.
\]

Precision measures the proportion of returned items that are relevant. Recall
measures how much of the available relevant set is retrieved. F1 is their
harmonic mean and becomes high only when both values are high. The metrics are
calculated at (K=1), (5), and (10), allowing the evaluation to distinguish
the first result from a longer recommendation list.

For text-to-image and image-to-text evaluation, matching product identifiers
define positive pairs. For category-based image-to-image evaluation, another
product is relevant when it has the same ground-truth category as the query;
the query product itself is excluded. Category agreement is only a proxy for
similarity and is not a complete measurement of visual style or user preference.

Metrics are first calculated per eligible query and are then averaged. A
query-macro average gives every query equal importance. A category-macro average
first aggregates results within each category and then gives every eligible
category equal importance. Queries with no relevant item in the current gallery
are reported as skipped rather than being silently assigned an arbitrary score.

## 2.9. Chapter Summary

This chapter established the theoretical basis for the project. ResNet18 and
SBERT provide separate visual and textual reference approaches, whereas CLIP
aligns both modalities in one embedding space. Cosine similarity supports
content-based ranking, contrastive fine-tuning adapts CLIP to fashion data, and
balanced sampling reduces the dominance of large categories during training.
Precision@K, Recall@K, and F1@K provide the common retrieval measures used in
the later experiments. Chapter 3 applies these concepts to the analysis and
design of the implemented system.

## References

Chia, P. J., Attanasio, G., Bianchi, F., Terragni, S., Magalhães, A. R., Goncalves, D., Greco, C., & Tagliabue, J. (2022). Contrastive language and vision learning of general fashion concepts. *Scientific Reports, 12*, Article 18958. https://doi.org/10.1038/s41598-022-23052-9

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep residual learning for image recognition* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1512.03385

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In M. Meila & T. Zhang (Eds.), *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 8748–8763). PMLR. https://proceedings.mlr.press/v139/radford21a.html

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In I. Guyon et al. (Eds.), *Advances in Neural Information Processing Systems 30* (pp. 5998–6008). Curran Associates, Inc.
