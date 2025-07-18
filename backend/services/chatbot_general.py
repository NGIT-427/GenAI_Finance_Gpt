from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import re

# Load model and tokenizer
MODEL_DIR = Path(r"D:\models\qa_model")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Expanded keywords covering finance, economics, business, etc.
FINANCE_KEYWORDS = [
    # General finance
    "finance",
    "financial",
    "money",
    "investment",
    "invest",
    "investing",
    "savings",
    "saving",
    "budget",
    "fund",
    "capital",
    "wealth",
    "income",
    "expense",
    "expenditure",
    "profit",
    "loss",
    "revenue",
    "earnings",
    "cash",
    "liquidity",
    "credit",
    "debit",
    "loan",
    "debt",
    "interest",
    "mortgage",
    "asset",
    "liability",
    "valuation",
    "dividend",
    "return",
    "roi",
    "yield",
    "equity",
    "shares",
    "stock",
    "bond",
    "securities",
    "portfolio",
    "broker",
    "account",
    "bank",
    "banking",
    "ledger",
    "balance sheet",
    "audit",
    "accounting",
    "tax",
    "taxation",
    "invoice",
    "payment",
    "payroll",
    "currency",
    "exchange rate",
    "inflation",
    "deflation",
    "risk",
    "hedge",
    "hedging",
    "derivative",
    "commodity",
    "futures",
    "options",
    "trading",
    "market",
    "stock market",
    "forex",
    "cryptocurrency",
    "bitcoin",
    "ethereum",
    "blockchain",
    "fintech",
    "gdp",
    "gross domestic product",
    "unemployment",
    "recession",
    "demand",
    "supply",
    "macroeconomics",
    "microeconomics",
    "economics",
    "economy",
    "trade",
    "export",
    "import",
    "tariff",
    "budget deficit",
    "monetary policy",
    "fiscal policy",
    "central bank",
    "interest rate",
    "quantitative easing",
    "inflation rate",
    "consumer price index",
    "cpi",
    "producer price index",
    "ppi",
    "foreign direct investment",
    "fdi",
    # Business & management
    "business",
    "entrepreneur",
    "entrepreneurship",
    "startup",
    "corporation",
    "company",
    "enterprise",
    "merger",
    "acquisition",
    "strategy",
    "management",
    "operations",
    "supply chain",
    "logistics",
    "marketing",
    "sales",
    "revenue model",
    "kpi",
    "benchmark",
    "franchise",
    "partnership",
    "proprietorship",
    "valuation",
    "assets under management",
    "aum",
    "esg",
    "sustainability",
    "corporate governance",
    # Insurance
    "insurance",
    "premium",
    "claim",
    "underwriting",
    "actuary",
    "policy",
]

def answer_general_query(question: str) -> str:
    q_lower = question.lower()

    # Compile regex to detect any keyword with word boundaries
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in FINANCE_KEYWORDS) + r")\b")
    if not pattern.search(q_lower):
        return "I'm sorry, I can only answer finance-related questions."

    prompt = (
        "You are an expert financial assistant. "
        "Answer clearly in one sentence.\n\n"
        f"Question: {question}\nAnswer:"
    )

    result = generator(
        prompt,
        max_length=120,
        do_sample=False,
        temperature=0.3,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw_text = result[0]["generated_text"].strip()
    answer_only = raw_text.split("Answer:")[-1].strip()
    clean_answer = re.sub(r"<.*?>", "", answer_only).strip()

    return clean_answer
