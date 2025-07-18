from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ✅ Path to your saved model
classification_model_dir = r"D:\models\classification_model"

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(classification_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(classification_model_dir)

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Your label list
label_list = [
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "10k_filing",
    "financial_news_article",
    "contract_agreement",
    "audit_report",
    "prospectus",
    "invoice"
]

def classify_text(text: str):
    """
    Classify the input text into one of the financial document categories.
    """
    if not text or not isinstance(text, str):
        return {
            "label": "unknown",
            "confidence": 0.0
        }
    
    # ✅ Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # ✅ Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ✅ Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    # ✅ Return result
    return {
        "label": label_list[pred_idx],
        "confidence": round(confidence, 3)
    }
