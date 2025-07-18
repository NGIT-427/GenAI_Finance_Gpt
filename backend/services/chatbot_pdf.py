from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# ✅ Load your local fine-tuned model
MODEL_PATH = r"D:\models\qa_model_roberta"

print("🔄 Loading QA model from local directory...")

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create pipeline using GPU (device=0) or CPU (device=-1)
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def answer_pdf_query(question, context):
    """
    Splits the context into chunks, runs QA pipeline, and returns best answer.
    """
    max_chunk_size = 350  # smaller chunk size to avoid truncation
    stride = 100
    answers = []

    # Split context into overlapping chunks
    for start in range(0, len(context), stride):
        chunk = context[start:start + max_chunk_size]
        if not chunk.strip():
            continue

        try:
            result = qa_pipeline(
                question=question,
                context=chunk,
                handle_impossible_answer=True
            )

            print("====== CHUNK RESULT ======")
            print(result)
            print("==========================")

            # Skip empty or NaN answers
            if not result.get("answer") or not result["answer"].strip():
                continue
            if "score" not in result or result["score"] != result["score"]:
                continue

            answers.append((result["score"], result["answer"]))

            # Early stop if confident
            if result["score"] > 0.85:
                break

        except Exception as e:
            print(f"Chunk error: {e}")

    if answers:
        # Return best scoring answer
        best = max(answers, key=lambda x: x[0])
        return best[1]

    return "Sorry, no answer found in the PDF."
