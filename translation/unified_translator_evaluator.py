#!/usr/bin/env python3
"""
Unified Translation Evaluation Framework
Provides a consistent, reusable class for evaluating translation quality.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint
from typing import List, Dict, Union

class TranslationEvaluator:
    """
    A unified class to calculate multiple standard translation metrics.
    This class ensures that evaluation standards are consistent and reusable.
    """
    def __init__(self, comet_model_name: str = "Unbabel/wmt22-comet-da"):
        """
        Initializes the evaluator and loads the COMET model.
        Loading the model can take time and resources, so it's done once at initialization.
        """
        print(f"Loading COMET model: {comet_model_name}. This may take a while...")
        try:
            model_path = download_model(comet_model_name)
            self.comet_model = load_from_checkpoint(model_path)
            print("✅ COMET model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load COMET model: {e}")
            print("COMET scores will not be available.")
            self.comet_model = None

    def calculate_bleu_chrf_ter(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculates BLEU, chrF++, and TER metrics.
        Useful when COMET is calculated separately.

        Args:
            hypotheses (List[str]): A list of translated sentences (machine translations).
            references (List[str]): A list of reference sentences.

        Returns:
            Dict[str, float]: A dictionary containing the scores for each metric.
        """
        # Sacrebleu expects references to be a list of lists, with one reference list per hypothesis
        # e.g., [[ref1_for_hyp1, ref2_for_hyp1], [ref1_for_hyp2]]
        # For our case, it's one reference per hypothesis.
        references_for_sacrebleu = [references]

        # 1. BLEU (via SacreBLEU for consistency)
        bleu = sacrebleu.corpus_bleu(hypotheses, references_for_sacrebleu, tokenize="ja-mecab")

        # 2. chrF++ (character n-gram F-score)
        chrf = sacrebleu.corpus_chrf(hypotheses, references_for_sacrebleu, word_order=2)

        # 3. TER (Translation Edit Rate)
        ter = sacrebleu.corpus_ter(hypotheses, references_for_sacrebleu, asian_support=True)

        metrics = {
            "BLEU": bleu.score,
            "SacreBLEU": bleu.score,  # sacrebleu.corpus_bleu is the official SacreBLEU score
            "chrF++": chrf.score,
            "TER": ter.score
        }

        return metrics
