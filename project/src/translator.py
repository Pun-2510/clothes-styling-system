"""Offline Vietnamese-to-English translation for CLIP text queries."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import TRANSLATION_MODEL


class VietnameseEnglishTranslator:
    """Load MarianMT once and translate Vietnamese queries to English."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
            .to(self.device)
        )
        self.model.eval()

    def translate(self, text):
        text = str(text).strip()
        if not text:
            raise ValueError("Vui lòng nhập nội dung cần dịch.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        translation = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()
        if not translation:
            raise RuntimeError("Model dịch không trả về nội dung.")
        return translation
