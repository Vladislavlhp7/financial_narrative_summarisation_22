# Extractive Summarisation of UK Annual Reports
## Introduction
Although there has been considerable progress in Natural Language Processing (NLP) over the years, it has not fully reached the Accounting and Finance (AF) industry.
In the meantime, companies worldwide produce vast amounts of textual data as part of their reporting packages to comply with regulations and inform shareholders of their financial performance.
The glossy annual report is such an example, widely read by investors, which also tends to be quite long.
Inspired by the Financial Narrative Summarisation (FNS) workshops (Zmandar et al., 2021, El-Haj et al., 2022) we design an Automatic Text Summarisation (ATS) system for the narrative parts of UK financial annual reports.

## Methodology
With this goal in mind, we implement and explore the following models for Extractive Text Summarisation (ETS): 
* attention-based Financial Recurrent Neural Network (FinRNN) with data augmentation, and 
* fine-tuned Financial BERT (FinBERT) (Yang et al., 2020).

## Results
Our evaluations based on the ROUGE-2 metric show both models to be outperforming the standard ATS baselines: (TextRank Mihalcea et al., 2004), and LexRank (Erkan et al., 2004).
Furthermore, our proposed FinBERT-base demonstrates competitive performance against official FNS 2022 models on the validation set - achieving an average ROUGE-2 F1 score of 0.382 and beating with 8% the best performing model overall in the FNS22 competition - the mT5 (Foroutan et al., 2022).

## References
```
@article{Erkan2004LexRankGC,
  title={LexRank: Graph-based Centrality as Salience in Text Summarization},
  author={Gunes Erkan and Dragomir R. Radev},
  journal={Journal of Artificial Intelligence Research},
  year={2004}
}
@inproceedings{mihalcea-tarau-2004-textrank,
    title = "TextRank: Bringing Order into Text",
    author = "Mihalcea, Rada and Tarau, Paul",
    booktitle = "Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W04-3252",
    pages = "404--411",
}
@inproceedings{zmandar-etal-2021-financial,
    title = "The Financial Narrative Summarisation Shared Task {FNS} 2021",
    author = "Zmandar, Nadhem  and
      El-Haj, Mahmoud  and
      Rayson, Paul  and
      Abura{'}Ed, Ahmed  and
      Litvak, Marina  and
      Giannakopoulos, Geroge  and
      Pittaras, Nikiforos",
    booktitle = "Proceedings of the 3rd Financial Narrative Processing Workshop",
    month = "15-16 " # sep,
    year = "2021",
    address = "Lancaster, United Kingdom",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.fnp-1.22",
    pages = "120--125",
}
@proceedings{fnp-2022-financial,
    title = "Proceedings of the 4th Financial Narrative Processing Workshop @LREC2022",
    editor = "El-Haj, Mahmoud  and
      Rayson, Paul  and
      Zmandar, Nadhem",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.fnp-1.0",
}
@misc{yang2020finbert,
      title={FinBERT: A Pretrained Language Model for Financial Communications},
      author={Yi Yang and Mark Christopher Siy UY and Allen Huang},
      year={2020},
      eprint={2006.08097},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

