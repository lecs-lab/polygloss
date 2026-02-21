"""Used to wrap some nice functions when a user loads the model from HF.
This file, and the appropriate lang_perplexities.csv, should be added to the model repo.
Not used directly during training or anywhere in this repo."""

from huggingface_hub import hf_hub_download
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class PolyGlossSeq2Seq(T5ForConditionalGeneration):
    def estimate_perf(self, glottocode: str):
        perplexities_path = hf_hub_download(
            repo_id=self.config.name_or_path, filename="lang_perplexities.csv"
        )
        perplexity = None
        with open(perplexities_path) as f:
            for line in f:
                next_glottocode, _, ppl, num_tokens, num_morphemes, num_words = (
                    line.split(",")
                )
                if next_glottocode.strip('"') == glottocode:
                    perplexity = {
                        "ppl": float(ppl.strip('"')),
                        "num_tokens": int(num_tokens.strip('"')),
                        "num_morphemes": int(num_morphemes.strip('"')),
                        "num_words": int(num_words.strip().strip('"')),
                    }
                    break
            else:
                raise ValueError(
                    "Unknown glottocode -- performance is probably very poor."
                )
        print("PolyGloss has seen this language during training.")
        print("-------------------------")
        print(f"Perplexity: {perplexity['ppl']}")
        print(f"# Tokens: {perplexity['num_tokens']}")
        print(f"# Morphemes: {perplexity['num_morphemes']}")
        print("-------------------------")
        print("PREDICTIONS (warning: high variance)")
        regressions_path = hf_hub_download(
            repo_id=self.config.name_or_path, filename="regressions.csv"
        )
        with open(regressions_path) as f:
            skip = False
            for line in f:
                if not skip:
                    skip = True
                    continue
                metric, slope, intercept = line.split(",")
                slope = float(slope)
                intercept = float(intercept.strip())
                predicted = intercept + slope * perplexity["ppl"]
                print(f"{metric} = {predicted}")

    def predict_igt(
        self,
        transcription: str,
        translation: str | None,
        language: str | None,
        metalanguage: str | None,
    ):
        if translation is None:
            translation = "None"
        if language is None:
            language = "an unknown language"
        if metalanguage is None:
            metalanguage = "an unknown language"
        prompt = f"""Predict the glosses and morphological segmentation (in parentheses) for the following text in {language}.

        Text in {language}: {transcription}
        Translation in {metalanguage}: {translation}

        Output:
        """

        if getattr(self, "tokenizer", None) is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name_or_path, use_fast=False
            )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.tokenizer.batch_decode(
            self.generate(**inputs, max_length=1024, num_beams=2),
            skip_special_tokens=True,
        )
        return outputs[0]
