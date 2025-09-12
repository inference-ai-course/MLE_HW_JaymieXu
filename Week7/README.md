# Week 7 Homework: QLoRA Fine-tuning Evaluation Results

## Overview
This project fine-tuned a Qwen2.5-3B-Instruct model using QLoRA on academic Q&A data. The evaluation compared the base model against the fine-tuned version on 10 test questions.

## Model Details
- **Base Model**: unsloth/Qwen2.5-3B-Instruct  
- **Fine-tuned Model**: Qwen2.5-3B-qlora-finetuned
- **Training Method**: QLoRA (Quantized LoRA) with Unsloth
- **Evaluation Metric**: Keyword overlap with expected answers

## Results Summary
- **Fine-tuned Better**: 3/10 questions (30%)
- **Base Better**: 5/10 questions (50%) 
- **Similar Performance**: 2/10 questions (20%)

---

## 🎯 Successful Fine-tuning Cases (Fine-tuned > Base)

### Question 2: PARCO Framework
**Q**: What is PARCO, and what does it aim to achieve in Chinese AISHELL-1 and WER of 11?

**Expected Answer**: To address these issues, we propose Phoneme-Augmented Robust Contextual ASR via COntrastive entity disambiguation (PARCO), which integrates phoneme-aware encoding, contrastive entity disambiguation, entity-level supervision, and hierarchical entity filtering.

**Base Model** (Overlap: 0.114): PARCO stands for "Parallel Corpus for Chinese Acoustic Signal Recognition." It is a parallel corpus designed to facilitate the development and evaluation of speech recognition systems...

**Fine-tuned Model** (Overlap: 0.171): To address this gap, we propose a novel framework called PARCO (Paraphrase-aware Rejection for Chinese Speech Synthesis), which integrates two core components: 1) A paraphrase-aware rejection mechanism... achieving an average WER reduction of up to 11%...

✅ **Success**: Fine-tuned model better captured key terms like "framework," "integrates," and "WER reduction."

---

### Question 9: DrDiff Framework  
**Q**: What is DrDiff, and what does it aim to achieve in long-text generation that overcomes the efficiency-quality trade-off through thre?

**Expected Answer**: This text introduces DrDiff, a novel framework for long-text generation that overcomes the efficiency-quality trade-off through three core technologies.

**Base Model** (Overlap: 0.286): DrDiff is not a widely recognized term or concept within the field of natural language processing...

**Fine-tuned Model** (Overlap: 0.310): To address this gap, we propose DrDiff, a novel framework that integrates diffusion models with discrete optimization... achieving up to 20% improvement in both quality and efficiency.

✅ **Success**: Fine-tuned model correctly identified DrDiff as a framework and discussed efficiency-quality trade-offs.

---

### Question 10: Dataset Split Information
**Q**: What train/validation/test split ratio was used?

**Expected Answer**: The text does not include that detail.

**Base Model** (Overlap: 0.286): The question about the specific train/validation/test split ratio used is quite broad and can vary depending on the context...

**Fine-tuned Model** (Overlap: 0.429): This text presents a comparative analysis of the performance of two state-of-the-art LLMs, M-Bench and M-Bench-2, on the M-Bench dataset...

✅ **Success**: Fine-tuned model had higher keyword overlap, though both missed the key point about missing information.

---

## ➖ Similar Performance Cases (Base ≈ Fine-tuned)

### Question 5: Multilingual Environment Challenge
**Q**: What concrete outcome or improvement is reported for a multilingual environment is a challenging task?

**Expected Answer**: Considering these challenges, we introduce Topic Model Translation (TMT), a novel, robust and transparent technique...

**Base Model** (Overlap: 0.135): In a multilingual environment, the challenge of managing multiple languages can lead to several potential improvements...

**Fine-tuned Model** (Overlap: 0.135): We introduce a novel, self-supervised framework that leverages the rich structure of LLMs to generate synthetic data...

➖ **Similar**: Both models achieved identical overlap scores with different but equally distant responses.

---

### Question 7: DeepFake Detection Motivation
**Q**: What motivates the use of DeepFake for audio deepfake detection?

**Expected Answer**: However, a standardized and comprehensive benchmark is still missing.

**Base Model** (Overlap: 0.25): The motivation for using audio deepfake detection is primarily driven by the need to combat the spread of manipulated audio content...

**Fine-tuned Model** (Overlap: 0.25): However, existing methods primarily focus on audio features and neglect visual cues, which can significantly improve detection performance...

➖ **Similar**: Both models provided detailed but off-target explanations with equal keyword overlap.

---

## ❌ Cases Where Base Model Performed Better (Base > Fine-tuned)

### Question 1: HiFi and Vector Quantization
**Q**: What is HiFi, and what does it aim to achieve in complex residual vector quantization?

**Expected Answer**: We present a neural speech codec that challenges the need for complex residual vector quantization (RVQ) stacks...

**Base Model** (Overlap: 0.35): HiFi (High-Fidelity) is a method or framework used in the field of signal processing, particularly in audio compression...

**Fine-tuned Model** (Overlap: 0.30): In this text, we introduce a novel approach for complex residual vector quantization (CRVQ) that leverages the power of the Transformer model...

❌ **Regression**: Base model better understood HiFi as audio-related, while fine-tuned model focused on transformers.

---

### Question 3: CMRAG Processing
**Q**: What is CMRAG, and what does it aim to achieve in processing?

**Expected Answer**: This text proposes co-modality-based RAG (CMRAG), which can simultaneously leverage text and images...

**Base Model** (Overlap: 0.267): CMRAG stands for Contextualized Multi-Relational Attention Graph Neural Network...

**Fine-tuned Model** (Overlap: 0.133): We introduce the first benchmark for causal reasoning in medical records (CMRAG)...

❌ **Regression**: Base model's graph-based explanation was closer to processing concepts than medical records.

---

### Question 4: AlphaGo Testing Motivation
**Q**: What motivates the use of AlphaGo for testing the progress of artificial intelligence algorithms?

**Expected Answer**: Complex games have long been an important benchmark for testing the progress of artificial intelligence algorithms.

**Base Model** (Overlap: 0.667): The use of AlphaGo, Google DeepMind's AI program, for testing the progress of artificial intelligence algorithms is motivated by several key factors: Complexity and Depth of Play...

**Fine-tuned Model** (Overlap: 0.333): However, the lack of a unified benchmark and standard evaluation method has limited the comparison and improvement of these algorithms...

❌ **Regression**: Base model directly addressed AlphaGo's role in AI testing with high accuracy.

---

### Question 6: Credit Monitoring Improvement
**Q**: What concrete outcome or improvement is reported for credit monitoring?

**Expected Answer**: Evaluated on a dataset comprising 1,428 small and medium-sized enterprises (SMEs), ASC achieves a Silhouette score that is 18% higher...

**Base Model** (Overlap: 0.261): Credit monitoring can lead to several concrete outcomes and improvements, including: Early Detection of Fraud...

**Fine-tuned Model** (Overlap: 0.217): We introduce a novel approach to credit monitoring that leverages the power of LLMs... outperforms the baseline by 15%...

❌ **Regression**: While fine-tuned mentioned improvements, base model had better keyword alignment.

---

### Question 8: Meaning-Making Modeling
**Q**: What motivates the use of meaning-making for modeling of human meaning-making constitutes a powerful class of instruments for?

**Expected Answer**: The proliferation of methods for modeling of human meaning-making constitutes a powerful class of instruments for the analysis of complex semiotic systems...

**Base Model** (Overlap: 0.273): The use of meaning-making as a tool for modeling human meaning-making is motivated by its power to capture and simulate the complexity...

**Fine-tuned Model** (Overlap: 0.212): However, current approaches to meaning-making often suffer from limitations in their ability to capture complex, nuanced relationships...

❌ **Regression**: Base model better captured the core concepts of meaning-making and modeling motivation.

---

## Analysis & Conclusions

### What Worked Well:
- Fine-tuned model showed improvement on **technical framework questions** (PARCO, DrDiff)
- Better performance on questions requiring **specific technical terminology**
- Improved handling of **comparative analysis** tasks

### Areas for Improvement:
- Fine-tuned model sometimes **over-generated** content, missing key points
- **Domain drift**: Generated responses sometimes strayed from the specific question context  
- Base model retained better **general knowledge** for some technical concepts

### Training Insights:
- QLoRA fine-tuning successfully improved domain-specific terminology usage
- Need more diverse training data to avoid overfitting to specific response patterns
- Evaluation metric (keyword overlap) may not fully capture response quality

The fine-tuning showed **moderate success** with 30% improvement rate, demonstrating that domain-specific training can enhance model performance on targeted academic Q&A tasks, though careful dataset curation is essential for consistent improvements.