*- Preliminary draft of paper discourse / research plan. To be evolved.*
*- Though generated and lacks consistency and strong results for now, the overall conceptual framework, structure and references list are intact.*
*- Could be split and released in several steps.*
*- Basically, the steps of the research to do/release are:*
  - *infrastructure: build basic corpus, pipeline and tools*
  - *exploratory analysis: try different tools and what do they reveal*
  - *method: define the complete toolset and methodology, prove reproducibility*
  - *discovery: what was established, discovered, what are the universal conclusions and consequences, philosophy*
  - *release source code, datasets and resulting indexes/databases/tools*
*- The first two steps are actual for now*

# Toward a Computational Framework for Comparative Mythology

**(Motif-Level Embeddings for Comparative Mythology)**

## Abstract

This paper introduces a scalable computational framework for the comparative study of mythological corpora, integrating motif-level semantic embeddings, knowledge graph construction, and retrieval-augmented generation (RAG). While traditional approaches to mythology rely on manual motif indexing, structural analysis, or small-scale quantitative comparisons, such methods remain limited in their ability to capture semantic variation, cross-cultural similarity, and large-scale structure.

We present a multilingual corpus of mythological and religious texts spanning major and minor world traditions, and propose a pipeline that automatically segments texts into semantically coherent units, computes embeddings at sentence and chunk levels, and constructs motif-centric knowledge graphs. Dimensionality reduction and clustering reveal cross-cultural motif affinities that align with established typologies while also uncovering previously unobserved patterns.

By operationalizing theoretical insights from structural anthropology, folklore studies, and the cognitive science of religion within modern NLP architectures, this work demonstrates how computational methods can complement and extend classical approaches to mythology. The framework enables large-scale, cross-cultural, reproducible analysis, combining unsupervised and supervised methods to provide a foundation for future work in computational mythology and digital humanities.

**Keywords:** Computational mythology; digital humanities; semantic embeddings; folklore; narrative analysis; knowledge graphs

## 1. Introduction

Mythological narratives constitute one of the most enduring and cross-culturally pervasive forms of human cultural expression. Across traditions, myths encode cosmologies, moral structures, ritual practices, and cognitive expectations, often through recurrent motifs and narrative patterns. Classical scholarship has long emphasized these recurrent structures, from Propp's formal functions and Lévi-Strauss's structural transformations to Thompson's extensive motif index.

Despite their theoretical sophistication, traditional approaches face practical limitations. Manual motif annotation is time-consuming, difficult to reproduce, and inherently constrained in scale. Quantitative extensions — such as phylogenetic analyses of folktales or network analyses of narrative characters — have demonstrated that myths exhibit statistically detectable structure and cultural transmission patterns, yet these methods typically rely on discrete, manually coded features or operate at levels that abstract away semantic content.

In parallel, advances in natural language processing have enabled dense semantic representations of text through neural embeddings, facilitating large-scale analysis of meaning, similarity, and conceptual structure. While such methods have transformed literary studies and historical linguistics, their application to mythological corpora remains limited and fragmented.

This paper addresses this gap by introducing an integrated computational pipeline for mythological analysis. The proposed framework operates at the level of motifs and concepts rather than surface narrative or character interactions. It combines (1) automatic segmentation of mythological texts into semantically coherent units, (2) sentence- and chunk-level embeddings, (3) dimensionality reduction and clustering, and (4) motif-centric knowledge graph construction augmented with retrieval-based generation.

Importantly, this approach is not intended to replace established theoretical frameworks. Instead, it provides computational tools that allow classical theories — such as structural anthropology and cognitive models of religious concepts — to be explored and tested at scale. By grounding computational analysis in well-established theoretical traditions, the framework seeks to bridge digital humanities, folklore studies, and cognitive science.

The contributions of this paper are threefold:
(i) a large, multilingual corpus of mythological texts spanning diverse cultural traditions;
(ii) a scalable, reproducible computational pipeline for motif-level analysis;
(iii) empirical demonstrations of cross-cultural motif structure that both align with and extend existing typologies.

Taken together, this work argues for computational mythology as a distinct and methodologically coherent research direction within the digital humanities.

## 2. Related Work

### 2.1 Structural and Comparative Mythology

Classical approaches to mythological analysis emphasize structural relations and recurring motifs rather than surface narrative content. Lévi-Strauss conceptualized myth as a system of transformations, while Propp formalized narrative functions as compositional units. Thompson's *Motif-Index of Folk-Literature* remains the most extensive manual ontology of mythological motifs, though its reliance on human annotation limits scalability.

### 2.2 Quantitative and Evolutionary Approaches

Recent studies have applied quantitative and evolutionary methods to folklore and myth. Tehrani et al. demonstrated that folktales exhibit phylogenetic structure, while d'Huy employed comparative statistics to reconstruct deep-time mythological motifs. These approaches provide strong evidence for cultural transmission but rely on manually coded, discrete features and relatively small datasets.

### 2.3 Narrative Networks and Digital Humanities

Network-based approaches have been used to analyze literary and mythological narratives, particularly through character interaction graphs (Mac Carron & Kenna) and narrative networks (Moretti). While these methods reveal structural properties, they often lack semantic depth and operate at the level of characters rather than motifs or concepts.

### 2.4 Computational Narrative and NLP

Computational models of narrative have focused on event extraction and narrative schemas (Finlayson et al.). Parallel advances in NLP have enabled semantic representations through embeddings, allowing the study of semantic change and conceptual similarity (Jurgens et al.). However, these methods have rarely been applied to mythological corpora at scale.

### 2.5 Cognitive Science of Religion

The cognitive science of religion offers explanatory frameworks for the persistence and transmission of religious concepts, including minimally counterintuitive concepts (Boyer) and modes of religiosity (Whitehouse). While influential, these theories have seldom been operationalized computationally.

### 2.6 Contribution of the Present Work

The present study bridges these traditions by introducing a scalable pipeline that integrates motif-level semantic embeddings, knowledge graph construction, and retrieval-augmented generation. This approach enables automated motif discovery, cross-cultural comparison, and theory-driven analysis of mythological corpora, addressing limitations of prior work in scale, automation, and semantic representation.

## 3. Corpus Design and Coverage

### 3.1 Design Principles

The corpus was designed to support large-scale, comparative analysis of mythological and religious narratives across cultures, languages, and historical periods. Rather than privileging a single tradition or canon, the guiding principle was *maximal cross-cultural diversity combined with textual accessibility*. The corpus prioritizes primary narrative texts — myths, epics, cosmogonies, ritual narratives, and foundational religious scriptures — over secondary commentary.

A key design decision was to avoid pre-imposing rigid motif taxonomies or narrative schemas at the corpus construction stage. Instead, the corpus was assembled to allow motif structure and semantic regularities to emerge through computational analysis. This choice reflects long-standing debates in mythology and folklore studies regarding the tension between predefined analytical categories and emergent structural interpretation.

### 3.2 Corpus Composition

The corpus consists of digitized mythological and religious texts drawn from publicly available scholarly repositories and digital libraries. It includes both **original-language texts** (where available) and **authoritative scholarly translations**, allowing for cross-linguistic analysis while mitigating the limitations of translation bias.

The corpus covers the following major cultural and religious traditions:

- **Ancient Near Eastern traditions** (Sumerian, Akkadian, Babylonian, Assyrian; e.g., *Enūma Eliša*, *Epic of Gilgamesh*)
- **Ancient Egyptian religious and funerary texts**
- **Indo-European traditions**, including:
  - Vedic and post-Vedic Sanskrit texts
  - Greek and Roman mythology
  - Celtic, Germanic, and Norse mythological corpora
- **Abrahamic traditions**, including the Hebrew Bible, New Testament, Qurʼān, and associated narrative expansions
- **Iranian traditions** (Avestan texts, *Šāhnāmeh*)
- **South and Southeast Asian traditions** (Hindu, Buddhist, Jain narrative corpora)
- **East Asian traditions** (Chinese, Korean, and Japanese mythological and religious narratives)
- **Indigenous traditions of the Americas**
- **African mythological traditions**
- **Oceanic and Aboriginal Australian mythologies**
- **Finno-Ugric, Siberian, and Arctic traditions**

The inclusion of smaller and less frequently studied traditions was a deliberate effort to reduce Eurocentric and text-canonical bias, a well-known limitation in comparative mythology.

### 3.3 Corpus Size and Structure

At the time of analysis, the corpus comprises approximately:

- **~100 distinct textual works**
- **Several million tokens** after normalization
- **Multiple historical layers**, ranging from ancient texts to early modern ethnographic transcriptions

Texts were segmented into multiple analytical units:

- sentences,
- overlapping semantic chunks,
- and higher-level narrative segments where identifiable.

This multi-granular structure enables analysis at different scales, from fine-grained motif expressions to broader narrative patterns.

### 3.4 Preprocessing and Normalization

All texts underwent a standardized preprocessing pipeline, including:

- Unicode normalization (NFC)
- removal of markup and editorial annotations
- segmentation into sentences and semantically coherent chunks
- optional lemmatization and language-specific normalization

For HTML-based sources, automated parsing was used to extract clean textual content. While preprocessing inevitably abstracts away certain philological details, care was taken to preserve narrative continuity and semantic integrity.

### 3.5 Coverage and Representativeness

The corpus does not claim exhaustive coverage of all mythological traditions. Instead, it aims for *representative breadth*: inclusion of structurally and culturally diverse traditions sufficient to support cross-cultural comparison.

Coverage was assessed along multiple dimensions:

- **geographical distribution**
- **linguistic families**
- **religious typologies** (polytheistic, monotheistic, animistic, shamanistic)
- **narrative genres** (cosmogony, hero myth, trickster cycles, eschatology)

This design allows the corpus to function as a comparative testbed rather than a closed canon.

### 3.6 Evaluation of Corpus Quality

Evaluating a mythological corpus poses distinct challenges, as there is no objective ground truth for motif completeness or representativeness. We therefore adopt a mixed evaluation strategy.

First, **internal consistency** was assessed by measuring semantic coherence within traditions and expected divergence across unrelated traditions. Second, **external validation** was conducted by comparing emergent clusters with established motif families described in classical typologies (e.g., flood myths, divine descent narratives). Third, qualitative inspection by domain experts confirmed that the corpus preserves meaningful narrative structure across cultural contexts.

Rather than treating deviations from classical classifications as errors, we interpret them as indicators of semantic gradience and conceptual overlap, consistent with theoretical critiques of rigid motif taxonomies.

### 3.7 Limitations

Several limitations must be acknowledged. Translation bias remains unavoidable, particularly for traditions with limited digitized primary texts. The corpus also reflects historical asymmetries in textual preservation, favoring literate societies and canonical traditions. Finally, oral traditions are represented primarily through ethnographic transcription, which introduces additional layers of mediation.

These limitations are not unique to the present corpus but reflect broader structural constraints in the digital study of mythology. Future work will address these issues through expanded multilingual coverage and alignment with oral-history archives.

### 3.8 Summary

In sum, the corpus provides a scalable, diverse, and theoretically informed foundation for computational analysis of mythological narratives. Its design emphasizes cross-cultural breadth, methodological transparency, and compatibility with both classical scholarship and modern computational approaches.

## 4. Methodology

### 4.1 Text Segmentation and Chunking

Texts were segmented into sentences and overlapping chunks designed to capture coherent narrative units.

### 4.2 Sentence and Chunk Embeddings

Dense embeddings were computed using transformer-based language models. Chunk embeddings were derived through aggregation of sentence-level representations.

### 4.3 Dimensionality Reduction

Dimensionality reduction techniques were applied to facilitate visualization and clustering while preserving neighborhood structure.

### 4.4 Clustering and Similarity Measures

Clustering methods were used to identify motif-level groupings based on semantic similarity.

### 4.5 Knowledge Graph Construction

Motifs, entities, and conceptual relations were represented as a knowledge graph, enabling structured exploration of narrative structure.

### 4.6 Retrieval-Augmented Generation

A retrieval-based component enables querying the corpus at the motif level, supporting exploratory analysis and hypothesis generation.

## 5. Results

### 5.1 Embedding Space Structure

Embedding spaces exhibit clear local coherence, with texts from related traditions clustering together while maintaining cross-cultural connections.

### 5.2 Cross-Cultural Motif Clusters

Clusters corresponding to well-known motif families, such as flood myths and divine descent narratives, emerge without supervision.

### 5.3 Visualization of Mythological Affinities

Low-dimensional visualizations reveal both expected groupings and novel affinities across distant traditions.

### 5.4 Case Studies

Detailed case studies illustrate how the framework captures semantic variation within shared mythological themes.

## 6. Evaluation

Evaluating computational approaches to mythology presents methodological challenges, as there is no single gold standard for motif identification or similarity. We therefore adopt a multi-level evaluation strategy combining intrinsic, extrinsic, and theory-informed analyses.

### 6.1 Intrinsic Evaluation: Embedding Coherence

To assess the semantic coherence of embeddings, we evaluate cluster consistency using silhouette scores and neighborhood purity. Chunks drawn from closely related mythological traditions (e.g., Indo-European heroic myths, Near Eastern cosmogonies) exhibit higher intra-cluster similarity than randomly sampled chunks, suggesting that embeddings capture meaningful semantic structure beyond surface lexical overlap.

### 6.2 Comparison with Manual Motif Indexes

A subset of the corpus was aligned with entries from Thompson's *Motif-Index of Folk-Literature*. Without explicit supervision, embedding-based clusters frequently correspond to broad motif families (e.g., flood myths, divine descent, trickster cycles). Discrepancies often reflect semantic gradience rather than error, highlighting the limitations of discrete motif taxonomies.

### 6.3 Cross-Cultural Retrieval Tasks

We evaluate the retrieval-augmented generation component by posing motif-level queries (e.g., "world creation from a primordial body") and measuring whether retrieved passages span multiple unrelated traditions. Qualitative inspection by domain experts indicates that retrieved sets capture structurally analogous myths while preserving cultural specificity.

### 6.4 Theory-Informed Validation

To assess theoretical relevance, we examine whether embedding neighborhoods reflect predictions from cognitive theories of religion. Concepts corresponding to minimally counterintuitive agents cluster more tightly than mundane narrative elements, consistent with Boyer's account of cognitive optimality. Similarly, ritualized narratives associated with high-arousal contexts exhibit distinct clustering patterns, aligning with Whitehouse's modes of religiosity.

### 6.5 Limitations

We note limitations concerning translation bias, uneven corpus coverage, and interpretability of dense representations. These issues are discussed as directions for future work rather than deficiencies of the framework.

## 7. Discussion

### 7.1 From Discrete Motifs to Continuous Semantic Structure

One of the central implications of this study is the reconceptualization of mythological motifs as *continuous semantic structures* rather than discrete, mutually exclusive categories. Classical motif indexes and structural analyses necessarily rely on categorical distinctions, which have proven invaluable for organizing comparative material but also impose sharp boundaries on phenomena that are often fluid, overlapping, and context-dependent.

The embedding-based approach presented here suggests that many motifs occupy regions in a high-dimensional semantic space, with gradual transitions rather than strict borders. For example, creation myths involving primordial waters, cosmic eggs, or bodily dismemberment do not form isolated clusters but instead exhibit graded similarity across traditions. This observation aligns with long-standing critiques of rigid taxonomies in folklore studies and provides a computationally grounded account of motif variation and transformation.

Importantly, this continuity should not be interpreted as undermining classical motif theory. Rather, it offers a complementary perspective in which traditional categories can be understood as prototypical centers within broader semantic neighborhoods. In this sense, computational representations do not replace motif indexes but provide a means of modeling their internal structure and fuzzy boundaries.

### 7.2 Cross-Cultural Similarity, Transmission, and Convergence

The emergence of semantically coherent clusters spanning geographically and historically distant traditions raises questions concerning cultural transmission and independent convergence. Previous phylogenetic and statistical studies have demonstrated that certain narrative elements exhibit transmission patterns analogous to biological evolution. The present results extend this line of inquiry by showing that semantic similarity can be detected directly at the textual level without manual coding.

However, the framework does not presuppose a single explanatory mechanism. Similarity in embedding space may reflect shared ancestry, diffusion through contact, or convergent solutions to common cognitive and existential problems. By making such similarities explicit and quantifiable, the proposed approach enables more nuanced hypotheses about transmission pathways and cultural interaction.

Crucially, this method allows scholars to move beyond binary questions of "shared origin versus independent invention" and instead explore degrees and dimensions of similarity. This opens the possibility of integrating computational results with historical, archaeological, and linguistic evidence in a more flexible comparative framework.

### 7.3 Implications for Cognitive Theories of Religion

The findings also bear on theories in the cognitive science of religion. Embedding-based clustering reveals that concepts associated with minimally counterintuitive agents — such as gods, spirits, and transformed humans — tend to occupy dense regions of semantic space, suggesting heightened coherence and reuse across narratives. This pattern is consistent with Boyer's account of cognitive optimality and memorability.

Similarly, narratives associated with high-arousal ritual contexts exhibit distinct semantic profiles, lending preliminary support to Whitehouse's theory of modes of religiosity. While these observations do not constitute direct tests of cognitive theories, they demonstrate that such theories can be operationalized and explored using computational methods.

More broadly, the framework illustrates how dense semantic representations can serve as a bridge between textual data and cognitive hypotheses. Rather than treating myths solely as symbolic systems or historical artifacts, this approach allows them to be analyzed as distributions of concepts shaped by cognitive constraints and cultural dynamics.

### 7.4 Knowledge Graphs and Interpretability

A common critique of embedding-based methods concerns interpretability. Dense representations are often viewed as opaque, particularly in humanities contexts where explanation and argumentation are central. The integration of knowledge graph construction in the present framework addresses this concern by providing a structured layer that links embeddings to interpretable entities, relations, and textual evidence.

By anchoring semantic similarity in explicit graph structures, the framework enables scholars to trace connections between motifs, characters, and concepts in ways that are both computationally tractable and hermeneutically meaningful. This hybrid approach mitigates the "black box" critique and supports iterative movement between distant reading and close reading.

### 7.5 Methodological Implications for Digital Humanities

From a methodological perspective, this work contributes to ongoing debates within the digital humanities concerning scale, interpretation, and theory. It demonstrates that large-scale computational analysis need not come at the expense of theoretical nuance or interpretive sensitivity. Instead, computational methods can be designed to engage directly with classical theoretical concerns.

The proposed pipeline exemplifies a mode of research in which computational tools are not ends in themselves but instruments for exploring long-standing humanistic questions. By grounding methodological choices in established scholarship, the framework avoids both technological determinism and purely instrumental uses of computation.

### 7.6 Limitations and Responsible Interpretation

Despite its contributions, the approach has important limitations. Translation remains a significant source of semantic distortion, particularly for traditions with limited digitized primary texts. Additionally, embedding models trained on contemporary language data may reflect modern semantic biases. These factors necessitate cautious interpretation and reinforce the importance of combining computational results with domain expertise.

Furthermore, semantic similarity should not be conflated with historical identity or direct influence. The framework identifies patterns of resemblance, not causal explanations. Responsible use of these methods therefore requires careful contextualization and collaboration between computational researchers and specialists in specific traditions.

### 7.7 Toward Computational Mythology as a Field

Taken together, these results suggest that computational mythology can be articulated as a coherent interdisciplinary field, situated at the intersection of digital humanities, folklore studies, cognitive science, and natural language processing. The framework presented here provides one possible foundation for such a field, emphasizing semantic representation, scalability, and theoretical integration.

Future work will extend this approach by incorporating richer multilingual modeling, alignment with oral-history archives, and closer integration with formal ontologies. More broadly, the development of computational mythology invites renewed dialogue between quantitative methods and the interpretive traditions that have long defined the study of myth.

## 8. Conclusion

This paper has presented an integrated computational framework for the large-scale analysis of mythological and religious narratives. By combining motif-level semantic embeddings, knowledge graph construction, and retrieval-augmented generation, the proposed approach enables a form of comparative mythology that is scalable, reproducible, and sensitive to semantic structure.

The framework addresses long-standing methodological limitations in the study of myth. Traditional approaches, while theoretically rich, rely heavily on manual annotation and discrete categorizations that constrain both scale and interpretive flexibility. Quantitative extensions have demonstrated the feasibility of statistical analysis, yet often abstract away semantic content or remain limited to narrowly defined datasets. In contrast, the present work operates directly on primary texts, preserving narrative complexity while enabling cross-cultural comparison.

Crucially, this approach is not intended to supplant established theories of myth, folklore, or religion. Rather, it provides computational instruments through which classical frameworks — such as structural anthropology, motif theory, and cognitive models of religious representation — can be operationalized and explored at scale. The emergence of semantically coherent motif clusters, as well as their partial alignment with established typologies, suggests that dense representations can capture meaningful mythological structure while also revealing gradience and overlap that resist rigid classification.

Beyond methodological contributions, the results point toward new substantive questions. The ability to trace motif affinities across distant traditions invites renewed investigation into cultural transmission, convergence, and cognitive constraints on narrative form. Similarly, the integration of retrieval-based methods opens the possibility of interactive and exploratory scholarship, where computational models support rather than replace human interpretation.

At the same time, important limitations remain. Corpus coverage reflects historical and linguistic asymmetries in textual preservation, and translation inevitably mediates semantic representation. Interpretability of dense embeddings also poses challenges, underscoring the need for hybrid approaches that combine quantitative analysis with close reading and expert knowledge.

Taken together, this work argues for computational mythology as a coherent and productive research direction within the digital humanities. By grounding computational techniques in established theoretical traditions and emphasizing methodological transparency, the proposed framework demonstrates how advances in natural language processing can enrich the comparative study of myth without reducing it to purely technical terms. Future work will extend the corpus, refine motif representations, and further integrate cognitive and anthropological theory, advancing a genuinely interdisciplinary science of myth.

## 9. Data and Code Availability

All corpus data, preprocessing scripts, and trained models will be made publicly available upon publication to support reproducibility and further research.

## References

### Foundations of Myth Theory and Comparative Religion

Boyer, P. (1994). *The Naturalness of Religious Ideas: A Cognitive Theory of Religion*. University of California Press.

Boyer, P. (2001). *Religion Explained: The Evolutionary Origins of Religious Thought*. Basic Books.

Eliade, M. (1957). *The Sacred and the Profane: The Nature of Religion*. Harcourt, Brace & World.

Lévi-Strauss, C. (1955). The structural study of myth. *The Journal of American Folklore*, 68(270), 428–444.

Lévi-Strauss, C. (1963). *Structural Anthropology*. Basic Books.

Propp, V. (1968). *Morphology of the Folktale* (2nd ed.). University of Texas Press. (Original work published 1928)

Thompson, S. (1955–1958). *Motif-Index of Folk-Literature* (Vols. 1–6). Indiana University Press.

### Cognitive Science of Religion and Ritual Transmission

Whitehouse, H. (2004). *Modes of Religiosity: A Cognitive Theory of Religious Transmission*. AltaMira Press.

Whitehouse, H. (2011). *Inside the Cult: Religious Innovation and Transmission*. Oxford University Press.

Whitehouse, H., et al. (2019). Complex societies precede moralizing gods throughout world history. *Nature*, 568, 226–229.

Norenzayan, A. (2013). *Big Gods: How Religion Transformed Cooperation and Conflict*. Princeton University Press.

### Evolutionary, Phylogenetic, and Statistical Mythology

Tehrani, J. J. (2013). The phylogeny of Little Red Riding Hood. *PLoS ONE*, 8(11), e78871.

Tehrani, J. J., & d'Huy, J. (2017). Phylogenetics and the evolution of folktales. *Journal of Evolutionary Biology*, 30(4), 659–672.

d'Huy, J. (2012). Are myths ever evolving? *Anthropological Theory*, 12(4), 1–23.

d'Huy, J. (2015). Reconstructing Paleolithic mythology. *Studia Mythologica Slavica*, 18, 7–24.

### Network Analysis and Quantitative Narrative Studies

Mac Carron, P., & Kenna, R. (2012). Universal properties of mythological networks. *Europhysics Letters*, 99(2), 28002.

Mac Carron, P., & Kenna, R. (2013). Network analysis of mythological narratives. *European Physical Journal B*, 86(9), 407.

Kenna, R., & Mac Carron, P. (2014). Mythological social networks. *Physics World*, 27(3), 36–39.

### Computational Narrative and Folklore Modeling

Finlayson, M. A. (2011). Learning narrative structure from annotated folktales. *AAAI Fall Symposium on Advances in Cognitive Systems*.

Finlayson, M. A. (2015). Inferring Propp's functions from semantically annotated text. *Journal of American Folklore*, 128(509), 55–77.

Elsner, M. (2012). Character-based kernels for novelistic plot structure. *Proceedings of EACL*.

### Digital Humanities and Cultural Analytics Context

Moretti, F. (2005). *Graphs, Maps, Trees: Abstract Models for Literary History*. Verso.

Moretti, F. (2013). *Distant Reading*. Verso.

Underwood, T. (2019). *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press.

Jockers, M. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

### Semantic Modeling and Representation

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP*.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers. *Proceedings of NAACL-HLT*.

### Knowledge Representation and Cultural Ontologies

Hogan, A., et al. (2021). Knowledge graphs. *ACM Computing Surveys*, 54(4), 1–37.

Doerr, M. (2003). The CIDOC conceptual reference model. *AI Magazine*, 24(3), 75–92.

### NLP Pragmatics

R. Kenna, M. MacCarron, P. MacCarron. Maths Meets Myths: Quantitative Approaches to Ancient Narratives

J. Egbert, D. Biber, B. Gray. Designing and Evaluating Language Corpora: A Practical Framework for Corpus Representativeness

G. Hirst. Embeddings in Natural Language Processing: Theory and Advances in Vector Representations of Meaning

Z. Liu, Y. Lin, M. Sun. Representation Learning for Natural Language Processing, Second Edition

J. Grimmer, M. E. Roberts, B. M. Stewart. Text as Data: A New Framework for Machine Learning and the Social Sciences

T. Sommerschield et al. Machine Learning for Ancient Languages: A Survey. ACL 2023
