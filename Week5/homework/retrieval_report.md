# Retrieval Report

- generated: 2025-08-11T19:20:33.332002
- model: `sentence-transformers/all-MiniLM-L6-v2`
- index size: 4411 vectors
- parameters: k=3, min_score=0.25


## Query: `transformer attention mechanism`

- **score=0.556 · p3 · Crisp Attention: Regularizing Transformers via Structured Sparsity**

  ```
  ##strains encoder and decoder outputs to mitigate overfitting in sequence tasks. our work frames structured attention sparsity as a data - dependent regularizer, analogous to a structured form of dropout. by retaining only the highest - scoring attention weights via a top - k approach, we prevent the model from relying on low - value connections, promoting more robust feature learning. this aligns with findings from [ 18 ], who observe natural sparsity in transformer activations ( e. g., 3. 0 % nonzero entries in t5 - base ) and demonstrate that enforcing sparsity via top - k thresholding enha…
  ```
- **score=0.532 · p4 · Crisp Attention: Regularizing Transformers via Structured Sparsity**

  ```
  2. 5 sparsity as regularization the connection between sparsity and regularization is well - established ; l1 regularization, for example, induces sparsity in weight vectors to enhance generalization [ 15 ]. modern techniques like feature flow regularization also leverage this principle to improve structured sparsity [ 19 ]. in the context of transformers, enforcing sparsity in mlp activations has been shown to improve model robustness and calibration [ 18 ]. our work builds on this principle but applies it directly to the attention mechanism itself. we provide the first empirical proof that s…
  ```
- **score=0.531 · p2 · Crisp Attention: Regularizing Transformers via Structured Sparsity**

  ```
  ] introduce modernbert, an encoder - only transformer optimized for long - context tasks, achieving state - of - the - art results in retrieval and classification with improved efficiency. similarly, [ 9 ] propose tkwinformer, which uses top - k window attention for vision tasks, reporting improved matching accuracy. these works primarily aim to approximate dense attention to preserve performance while enhancing efficiency. in contrast, our research demonstrates that structured sparsity in the attention mechanism can not only reduce computational costs but also act as a regularizer, leading to…
  ```

## Query: `beam search decoding`

- **score=0.459 · p14 · AURA: Affordance-Understanding and Risk-aware Alignment Technique for Large Language Models**

  ```
  ##b ). this design choice ensures that aura is evalu - ated in a model - agnostic fashion across architectures with different inductive biases and decoding styles. these mod - els also represent state - of - the - art performance in lightweight instruction - following and reasoning tasks ( qi et al. 2024 ; team 2024 ), making them strong base policies for safety - sensitive generation. reward - guided search depth ( k ) we use k ∈2, 4, 8 response samples in reward - guided search to investigate the impact of search depth on safety alignment. this range strikes a balance between computa - tiona…
  ```
- **score=0.395 · p4 · Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation**

  ```
  ##ds like greedy or beam search, to produce k diverse sum - mary candidates for each query. each candidate summary is then evaluated by computing its sps using the reader ’ s em - bedding parameters. finally, the summary with the lowest sps, indicating optimal alignment with the reader ’ s internal representation, is selected as input to the reader llm for downstream answer generation. test - time sampling in text - to - embedding compression. embedding - level compression maps retrieved passages ( and the query ) directly to a summary embedding via a trained projector ( cheng et al. 2024a ). …
  ```
- **score=0.377 · p4 · Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation**

  ```
  the sum - mary – query embedding before passing the fusion represen - tation through the reader llm. for each probe, we extract the reader ’ s penultimate - layer hidden state at the probe po - sition, denoted hr, and compute a simple diversity score fol - lowing ( hu et al. 2025 ) : sprobe = p x i = 1 ∆ ( i ) 2, ( 3 ) where ∆ ( i ) is the gap between the i - th and ( i + 1 ) - th largest elements of hr. smaller sprobe indicates stronger seman - tic deviation from the existing summary – query signal. we retain the m probes with the smallest scores and form m + 1 candidate embedding summaries (…
  ```

## Query: `contrastive learning for embeddings`

- **score=0.676 · p13 · Efficient Knowledge Probing of Large Language Models by Adapting Pre-trained Embeddings**

  ```
  , xin zhang, yanzhao zhang, dingkun long, pengjun xie, and meishan zhang. towards general text embeddings with multi - stage contrastive learning. arxiv : 2308. 03281, 2023. [ 46 ] aivin v. solatorio. gistembed : guided in - sample selection of training negatives for text embedding fine - tuning. arxiv : 2402. 16829, 2024. 13
  ```
- **score=0.517 · p8 · Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation**

  ```
  . beyond prompting : an efficient embedding frame - work for open - domain question answering. arxiv preprint arxiv : 2503. 01606. izacard, g. ; caron, m. ; hosseini, l. ; riedel, s. ; bojanowski, p. ; joulin, a. ; and grave, e. 2022a. unsupervised dense in - formation retrieval with contrastive learning. transactions on machine learning research. izacard, g. ; caron, m. ; hosseini, l. ; riedel, s. ; bojanowski, p. ; joulin, a. ; and grave, e. 2022b. unsupervised dense in - formation retrieval with contrastive learning. transactions on machine learning research. izacard, g. ; lewis, p. ; lomel…
  ```
- **score=0.498 · p4 · Efficient Knowledge Probing of Large Language Models by Adapting Pre-trained Embeddings**

  ```
  ( 1 ) sentence embedding models. facts are sentences that combine entities with specific relations. thus, one can find meaningful representations of them by leveraging pre - trained sentence embedding models [ 24, 22 ]. these models are trained to represent “ similar ” sentences in a database close to each other. two sentences can be deemed “ similar ” based on relevance for retrieval [ 35 ] and clustering tasks [ 36 ]. while these models were traditionally trained from scratch on contrastive loss, newer models have adapted the progress in generative llms by leveraging their learned parameters…
  ```

## Query: `reinforcement learning for language models`

- **score=0.625 · p9 · Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation**

  ```
  restructure retrieved content efficiently to advance question - answering capabilities. in al - onaizan, y. ; bansal, m. ; and chen, y. - n., eds., findings of the asso - ciation for computational linguistics : emnlp 2024, 8548 – 8572. miami, florida, usa : association for computational linguistics. liu, w. ; qi, s. ; wang, x. ; qian, c. ; du, y. ; and he, y. 2025. nover : incentive training for language models via verifier - free reinforcement learning. arxiv preprint arxiv : 2505. 16022. mialon, g. ; dessi, r. ; lomeli, m. ; nalmpantis, c. ; pa - sunuru, r. ; raileanu, r. ; roziere, b. ; sch…
  ```
- **score=0.602 · p11 · Sample-efficient LLM Optimization with Reset Replay**

  ```
  , christopher hesse, and john schulman. training verifiers to solve math word problems. arxiv preprint arxiv : 2110. 14168, 2021. [ 7 ] ganqu cui, lifan yuan, ning ding, guanming yao, wei zhu, yuan ni, guotong xie, zhiyuan liu, and maosong sun. ultrafeedback : boosting language models with high - quality feedback, 2023. [ 8 ] ganqu cui, lifan yuan, zefan wang, hanbin wang, wendi li, bingxiang he, yuchen fan, tianyu yu, qixin xu, weize chen, et al. process reinforcement through implicit rewards. arxiv preprint arxiv : 2502. 01456, 2025. 11
  ```
- **score=0.598 · p10 · Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models**

  ```
  zhou, y. ; stanczyk, p. ; garea, s. r. ; geist, m. ; and bachem, o. 2024. on - policy distillation of language models : learning from self - generated mistakes. in the twelfth international conference on learning representations. anthropic. 2024. claude 3. 5 sonnet. https : / / www. anthropic. com / news / claude - 3 - 5 - sonnet. published : 21 jun 2024. aryan, a. ; nain, a. k. ; mcmahon, a. ; meyer, l. a. ; and sahota, h. s. 2023. the costly dilemma : generalization, evaluation and cost - optimal deployment of large language models. arxiv preprint arxiv : 2308. 08061. austin, j. ; odena, a. …
  ```

## Query: `CTC loss`

- **score=0.377 · p10 · InfoCausalQA:Can Models Perform Non-explicit Causal Reasoning Based on Infographic?**

  ```
  ##f - ference between the number of predicted and actual cor - rect answers. a positive value indicates over - selection ; a negative value indicates under - selection. ctdif = 1 n n x i = 1 | | − | yi | ( 4 ) • mean absolute count difference ( ctdifabs ) : the aver - age absolute error between the predicted and true number of correct answers. ctdifabs = 1 n n x i = 1 | | − | yi | ( 5 ) • count accuracy ( ctacc ) : the proportion of questions for which the model predicted the exact number of cor - rect answers, regardless of which specific options were selected. ctacc = 1 n n x i = 1 1 h | | =…
  ```
- **score=0.349 · p7 · Prosocial Behavior Detection in Player Game Chat: From Aligning Human-AI Definitions to Efficient Annotation at Scale**

  ```
  697 0. 714 0. 723 0. 721 0. 730 0. 746 0. 776 0. 838 0. 870 recall 0. 785 0. 861 0. 886 0. 886 0. 892 0. 867 0. 873 0. 873 0. 854 0. 817 0. 804 0. 7 ( ↑precision ) precision 0. 670 0. 718 0. 792 0. 856 0. 892 0. 884 0. 887 0. 875 0. 869 0. 870 0. 870 recall 0. 785 0. 741 0. 722 0. 677 0. 734 0. 772 0. 791 0. 798 0. 798 0. 804 0. 804 aucs of 0. 76 with inference times of 0. 45 – 2. 13s per 2, 000 instances. the svm outperformed both ( auc = 0. 77, precision = 0. 67, recall = 0. 78 ), with moderate inference latency ( 2. 93s ). neural models — lstm, setfit, and bert — offered improved pre - dict…
  ```
- **score=0.311 · p19 · Efficient Knowledge Probing of Large Language Models by Adapting Pre-trained Embeddings**

  ```
  0 10 20 30 40 epoch 0. 2 0. 3 0. 4 0. 5 0. 6 0. 7 loss training loss on gpt - 4o model gist linq gte mpnet mxbai nve2 ultra ( a ) gpt - 4o 0 10 20 30 40 epoch 0. 3 0. 4 0. 5 0. 6 0. 7 0. 8 loss training loss on gpt - 4o cot model gist linq gte mpnet mxbai nve2 ultra ( b ) gpt - 4o cot 0 10 20 30 40 epoch 0. 2 0. 3 0. 4 0. 5 0. 6 0. 7 loss training loss on gpt - 4o - mini model gist linq gte mpnet mxbai nve2 ultra ( c ) gpt - 4o - mini 0 10 20 30 40 epoch 0. 2 0. 3 0. 4 0. 5 0. 6 0. 7 loss training loss on gpt - 4o - mini cot model gist linq gte mpnet mxbai nve2 ultra ( d ) gpt - 4o - mini cot …
  ```