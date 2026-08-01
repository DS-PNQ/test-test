# Standardized input/output for NLLB-200

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Simple language keys used across the pipeline, mapped to NLLB's FLORES-200 codes.
# Scope per pipeline_overview.md: VN<->EN and VN<->CN only.
LANG_CODES = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "zh_hans": "zho_Hans",
    "zh_hant": "zho_Hant",
}


class NLLBTranslator:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        inputs = self.tokenizer(text, return_tensors="pt")
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang])
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
