# DAPT
**Predicting Transcriptional Outcomes of Multigene Perturbations under Multi-Omics**
## 📍 Overview
<img width="1408" height="736" alt="Gemini_Generated_Image_klw283klw283klw2" src="https://github.com/user-attachments/assets/592e1985-5e86-47b9-9f57-b5b1093fa6d2" />


We present **DAPT**, a novel deep learning framework designed to predict the transcriptional outcomes of single and multigene perturbations. Developed to overcome the structural limits of existing models. DAPT introduces a descriptor-driven perturbation adapter integrated with Graph Neural Networks (GNNs) and prior biological knowledge.

Understanding how gene perturbations alter cellular transcriptional responses is central to modern biomedicine, enabling rational therapeutic design and the identification of genetic interactions. While recent single-cell CRISPR screening platforms like Perturb-seq provide large-scale datasets, existing computational methods exhibit instability when predicting multigene perturbations, particularly for genes unseen during training or entirely missing from biological knowledge graphs.

DAPT replaces standard ID-based perturbation embeddings with a descriptor-based representation. This allows our model to learn the intrinsic biological properties of genes, significantly improving zero-shot prediction for both unseen and out-of-vocabulary (OOV) perturbations.

## 💡 Key Innovations
- Descriptor-Based Representation: Replaces discrete, ID-based embeddings with semantic biological gene descriptors, enabling parameter sharing across genes.
- Perturbation Regularized Autoencoder (RAE): Transforms high-dimensional descriptors into dense latent embeddings through non-linear transformations.
- Biological Knowledge Integration: Utilizes Gene Ontology (GO) and Gene Co-expression graphs as dense substrates for message passing, ensuring predictions are grounded in functional pathways.
- OOV Generalization : Generates perturbation representations directly from input descriptors, enabling out-of-distribution generalization, particularly for unseen or novel gene perturbations.
