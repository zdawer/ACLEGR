# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Paper
- **Title:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
- **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
- **Venue/Date:** NAACL-HLT 2019 (arXiv:1810.04805)  
- **Citation:** Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.  
- **BibTeX:**
```bibtex
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={4171--4186},
  year={2019}
}
```

## Abstract-style summary
BERT is a Transformer encoder pre-trained with masked language modeling (MLM) and next sentence prediction (NSP) objectives on large-scale unlabeled corpora (BookCorpus and English Wikipedia). The resulting contextual representations are fine-tuned with minimal task-specific changes for downstream NLP tasks. The model yields strong gains across natural language understanding benchmarks, notably GLUE and SQuAD, by leveraging bidirectional attention, WordPiece tokenization, and task-specific classification or span prediction heads during fine-tuning.

## Datasets and resources
- **Pre-training corpora:**  
  - BookCorpus (800M words). Public redistribution is restricted; many reproductions substitute [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) alongside Wikipedia.  
  - English Wikipedia (~2.5B words). Latest dumps: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2  
- **Fine-tuning benchmarks:**  
  - GLUE (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI): https://gluebenchmark.com/tasks  
  - SQuAD v1.1 and v2.0: https://rajpurkar.github.io/SQuAD-explorer/  
- **Pretrained weights:** Official checkpoints are mirrored on Hugging Face (e.g., [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), [bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased)).  
- **Reference logs:** GLUE and SQuAD fine-tuning logs are available in the Hugging Face Transformers examples repository (see `examples/pytorch/text-classification` and `examples/pytorch/question-answering` runs at https://github.com/huggingface/transformers).

## Expected benchmarks
Original paper results (single-task fine-tuning, no ensembling):

| Model | Benchmark | Metric | Score |
| --- | --- | --- | --- |
| BERT-Large (cased) | GLUE Test | Average | 80.5 |
| BERT-Large (cased) | MNLI Test (m/mm) | Accuracy | 86.7 / 85.9 |
| BERT-Large (cased) | SQuAD v1.1 Test | F1 / EM | 93.2 / 86.9 |
| BERT-Large (cased) | SQuAD v2.0 Test | F1 | 83.1 |

Reproductions with BERT-Base (uncased) typically reach ~79–80 average on GLUE dev and ~88.5 F1 on SQuAD v1.1 dev when trained with the standard hyperparameters below.

## Reproducibility tips
- **Seeds:** Use fixed seeds across libraries (e.g., `PYTHONHASHSEED=0`, `torch.manual_seed(42)`, `numpy.random.seed(42)`). For TPU runs, also set `XLA_FLAGS=--xla_gpu_deterministic_ops`.  
- **Hardware:** The original models were trained on 16 TPU chips (BERT-Base ~4 days, BERT-Large ~4 days). Comparable GPU training can be done with 8× V100 32GB; single-GPU fine-tuning fits in 12–16GB.  
- **Batching & sequence length:** Pre-train with a mix of 128 and 512 token sequences; fine-tune with `max_seq_length` 128–512 depending on the task (512 for SQuAD, 128/256 for GLUE).  
- **Optimization:** AdamW with β1=0.9, β2=0.999, weight decay 0.01, learning rate warmup over the first 10% of steps, then linear decay. Typical LR: 1e-4 (pre-training), 2e-5/3e-5/5e-5 grid for fine-tuning.  
- **Runtime expectations:** GLUE task fine-tuning on BERT-Base typically completes within 2–6 hours on a single 16GB GPU; SQuAD v1.1 fine-tuning runs in ~8–10 hours under the same constraints.  
- **Logging:** Track evaluation metrics after each epoch and save the best checkpoint on dev performance to mirror the reporting in the paper.
