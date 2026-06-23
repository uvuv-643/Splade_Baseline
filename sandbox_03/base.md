# Reading priority for agents

Read compressed corpus first: `assets/compressed_papers/ALL_COMPRESSED_PAPERS.md` or `assets/compressed_papers/INDEX.md` + per-paper Markdown files.
Do not read raw PDFs by default. PDFs in `assets/pdfs/` are only archival/source restore data.
Ignore raw extraction/source/cache files if any appear.

---

1. SPLADE v1 (SIGIR 2021) — оригинал

    Formal, Piwowarski, Clinchant. "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
    🔗 https://dl.acm.org/doi/10.1145/3404835.3463098
    Первый ранкер на основе явной регуляризации разреженности и log-saturation эффекта для весов терминов, что приводит к высоко-разреженным представлениям, конкурирующим с dense-методами.

2. SPLADE v2 (arXiv 2021)

    Formal, Lassance, Piwowarski, Clinchant — arXiv:2109.10086
    🔗 https://arxiv.org/abs/2109.10086
    Введён max-pooling механизм, SPLADE-doc (document-only encoder с предвычисленными весами), и distillation с hard negatives.

3. From Distillation to Hard Negative Sampling (SIGIR 2022)

    Formal, Lassance, Piwowarski, Clinchant — SPLADE++
    Состояние SOTA на тот момент с дистилляцией от cross-encoder.

4. Efficiency Study for SPLADE (SIGIR 2022)

    Lassance, Clinchant — Efficient-SPLADE
    🔗 https://dl.acm.org/doi/10.1145/3477495.3531833
    L1-регуляризация для запросов, разделение query/document энкодеров, FLOPS-регуляризация на стадии middle-training, более быстрые query-энкодеры.

5. Towards Effective and Efficient Sparse Neural IR (TOIS 2024)

    Formal et al. — журнальная версия всего «SPLADE-family»
    🔗 https://dl.acm.org/doi/10.1145/3634912
    Предлагаются улучшения SPLADE для эффективности и качества — SPLADE++ и Efficient-SPLADE.

6. SPLADE-v3 (arXiv 2024)

    Lassance, Déjean, Formal, Clinchant — arXiv:2403.06789
    🔗 https://arxiv.org/abs/2403.06789
    Статистически значимо эффективнее BM25 и SPLADE++, сравним с cross-encoder re-ranker'ами; >40 MRR@10 на MS MARCO dev и +2% out-of-domain на BEIR.

7. Static Pruning Study on Sparse Neural Retrievers (SIGIR 2023)

    Lassance, Lupart, Déjean, Clinchant, Tonellotto
    Базовая работа по static pruning для SPLADE-индексов.

🚀 НОВЫЕ РАБОТЫ 2024–2026 (perspective directions для твоей статьи)
A) LLM-based SPLADE / decoder-only архитектуры
8. Mistral-SPLADE (arXiv 2024)

    Doshi, Kumar, Murthy, P, Sen — arXiv:2408.11119
    🔗 https://arxiv.org/abs/2408.11119
    Decoder-only модели видели больше данных и лучше выучивают keyword expansions; используется Mistral как backbone. Echo-Mistral-SPLADE — SOTA на BEIR среди LSR.

9. CSPLADE (IJCNLP-AACL 2025)

    Xu, Feng, Tian, Ding, Cheong (Amazon) — arXiv:2504.10816
    🔗 https://arxiv.org/abs/2504.10816
    Возможно обучение LSR на 8B LLM с конкурентным качеством и уменьшенным index size; одна из первых работ по анализу performance-efficiency trade-off через model quantization.

10. LACONIC (arXiv 2026)

    Xu, Zhuang, C. Zhang, X. Ma, Tian, Mehta, Lin, Srikumar — arXiv:2601.01684
    🔗 https://arxiv.org/abs/2601.01684
    Семейство LSR на Llama-3 (1B/3B/8B); two-phase curriculum: (1) weakly-supervised pre-finetuning для адаптации causal LLM к bidirectional контексту, (2) finetuning на hard negatives. 8B-вариант: 60.2 nDCG на MTEB Retrieval.

11. Leveraging Decoder Architectures for LSR (KEIR 2025)

    Qiao, Nguyen, Kanoulas, Yates — Springer LNCS 16086, pp.19–35

12. Scaling Sparse and Dense Retrieval in Decoder-Only LLMs (SIGIR 2025)

    Zeng, Killingback, Zamani — SIGIR'25

B) Inference-free LSR (горячее направление!)
13. Towards Competitive Search Relevance for Inference-Free LSR (arXiv 2024, v2 2025)

    Geng, Wang, Ru, Yang (OpenSearch) — arXiv:2411.04403
    🔗 https://arxiv.org/abs/2411.04403
    Предложены IDF-aware penalty для подавления вклада low-IDF токенов; модель превосходит SOTA inference-free на +3.3 NDCG@10 на BEIR.

14. Effective Inference-Free Retrieval for LSR (SIGIR 2025) — Li-LSR

    Nardini, Nguyen, Rulli, Venturini, Yates — arXiv:2505.01452
    🔗 https://arxiv.org/abs/2505.01452

15. Exploring ℓ₀ Sparsification for Inference-Free Sparse Retrievers (SIGIR 2025)

    Shen, Geng et al. — arXiv:2504.14839
    OpenSearch разработали два ℓ₀-вдохновлённых подхода: ℓ₀ mask loss (исключает уже разреженные документы из регуляризации) и ℓ₀ Approximation Activation (использует лог-преобразования для штрафа на токены с низкими активациями).

16. An Alternative to FLOPS Regularization to Productionize SPLADE-Doc (SIGIR 2025)

    Porco, Mehra, Malioutov, Radhakrishnan, Keymanesh, Preoțiuc-Pietro, MacAvaney, Cheng — arXiv:2505.15070
    DF-FLOPS — снижает использование высоко-DF токенов, ~10× ускорение в production-engine.

C) Multilingual / Cross-lingual SPLADE
17. MILCO (ICLR 2026)

    Nguyen, Lei, Ju, Yang, Yates — arXiv:2510.00671
    🔗 https://arxiv.org/abs/2510.00671
    SOTA multilingual/cross-lingual LSR, обходит BGE-M3 и Qwen3-Embed; при mass-based pruning до 30 активных размерностей MILCO 560M обгоняет Qwen3-Embed 0.6B с 3× более низкой latency и 10× меньшим индексом.

18. SPLARE — Learning Retrieval Models with Sparse Autoencoders (ICLR 2026)

    Formal, Louis, Déjean, Clinchant — arXiv:2603.13277
    SAE-based LSR превосходит vocabulary-based аналоги, особенно в multilingual и out-of-domain; SPLARE — конкурентоспособная 7B multilingual модель с обобщаемыми sparse latent embeddings.

19. SPLADE-X (Nair et al., 2022) и BLADE (Nair et al., 2023) — ранние cross-lingual подходы.
D) Эффективные индексы и pruning для SPLADE
20. Faster Learned Sparse Retrieval with Block-Max Pruning (BMP, SIGIR 2024)

    Mallia, Suel, Tonellotto

21. Seismic — Efficient Inverted Indexes for Approximate Retrieval (SIGIR 2024)

    Bruch, Nardini, Rulli, Venturini

22. Dynamic Superblock Pruning (SP, SIGIR 2025)

    Carlson, Xie, He, Yang — arXiv:2504.17045
    🔗 https://arxiv.org/abs/2504.17045
    Для rank-safe поиска на SPLADE SP на 32% быстрее BMP при k=10; до 2.9× быстрее BMP, 3.3× Seismic, 9.1× ASC при recall 99%.

23. Efficiency Optimizations for Superblock-based Sparse Retrieval (arXiv 2026)

    arXiv:2602.20986 / arXiv:2602.02883

24. Investigating Scalability of Approximate Sparse Retrieval to Massive Datasets (ECIR 2025)

    Bruch, Nardini, Rulli, Venturini, Venuta

25. Threshold-driven Pruning with Segmented Maximum Term Weights (EMNLP 2024)

    Qiao, Carlson, He, Yang, Yang

26. Representation Sparsification with Hybrid Thresholding (arXiv 2306.11293)

    Yang group — hybrid threshold-based sparsification.

E) Расширение словаря / новые архитектуры
27. DyVo: Dynamic Vocabularies with Entities (EMNLP 2024)

    Nguyen, Chatterjee, MacAvaney, Mackie, Dalton, Yates — arXiv:2410.07722
    🔗 https://arxiv.org/abs/2410.07722
    DyVo head использует существующие entity embeddings и entity retrieval компонент; entity-веса объединяются с word-piece весами в joint-представления для inverted index.

28. PromptReps (arXiv 2024)

    Zhuang, Ma, Koopman, Lin, Zuccon — arXiv:2404.18424
    Промптинг LLM для генерации dense+sparse представлений.

29. To Case or Not to Case: Empirical Study in LSR (arXiv 2601.17500, 2026)

    Эффект casing на LSR.

F) Мультимодальный и conversational SPLADE
30. Multimodal LSR with Probabilistic Expansion Control (arXiv 2024)

    Nguyen, Hendriksen, Yates, de Rijke — arXiv:2402.17535

31. Multimodal LSR for Image Suggestion (2024)
32. STAIR: Sparse Text and Image Representation in Grounded Tokens (EMNLP 2023)

    Chen et al.

33. Sparse and Dense Retrievers Learn Better Together (CIKM 2025)

    arXiv:2508.16707 — joint sparse-dense оптимизация для text-image.

34. DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search (2024)

    Lupart et al. — дистилляция score-based для conversational search.

35. SERVAL: Zero-shot Visual Document Retrieval (arXiv 2509.15432, 2025)
G) Адаптация к длинным документам и tutorials/surveys
36. Adapting Learned Sparse Retrieval for Long Documents (2023)
37. On the Reproducibility of LSR Adaptations for Long Documents (2025)
38. Neural Lexical Search with Learned Sparse Retrieval (SIGIR 2025 Tutorial)

    Yates, Lassance, Rulli, Lei et al. — отличная отправная точка для review
    🔗 https://lsr-tutorial.github.io/

39. A Survey of Model Architectures in Information Retrieval (arXiv 2502.14822, 2025)
40. Semantic Search for Information Retrieval (arXiv 2508.17694, 2025)