# services/summarize.py

import re
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pathlib import Path

# 🔹 Load summarization model from local directory
MODEL_PATH = Path(r"D:/models/summarization_model")
tokenizer = BartTokenizer.from_pretrained(str(MODEL_PATH))
model = BartForConditionalGeneration.from_pretrained(str(MODEL_PATH))

# 🔹 Use GPU if available and has enough memory, else CPU
device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6 * 1024 ** 3 else "cpu")
model.to(device)
model.eval()


def mask_sensitive_info(text: str) -> str:
    """
    Masks sensitive financial information like:
    - Currency values ($, €, £, etc.)
    - Numbers with units (million, billion, etc.)
    - Percentages
    - Years and IDs
    """
    patterns = [
        r'\$?\d{1,3}(,\d{3})+(\.\d+)?\s?(million|billion|M|B)?',  # USD amounts with commas
        r'[\u20AC\u00A3\u00A5]?\d+(\.\d+)?\s?(million|billion|M|B)?',  # Other currencies
        r'\d+(\.\d+)?\%',  # Percentages
        r'\b(19|20)\d{2}\b',  # Years (1900-2099)
        r'\b[A-Z]{2,5}-\d{2,5}\b',  # Document or form IDs (e.g., SEC-1234)
        r'\b\d{4,}\b'  # Any long numeric IDs (e.g., tax IDs)
    ]

    for pattern in patterns:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)

    return text


def generate_masked_summary(text: str) -> str:
    """
    Generates a summary of the text and applies masking on sensitive info.
    """
    try:
        # 🔹 Limit input to avoid overloading BART
        inputs = tokenizer(text[:10000], return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=50,
                length_penalty=1.0,
                num_beams=6,
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=tokenizer.eos_token_id,
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        clean_summary = summary.strip()
        masked_summary = mask_sensitive_info(clean_summary)

        return masked_summary

    except Exception as e:
        return f"Summary failed: {str(e)}"
