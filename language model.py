from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

# Load the pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load the SQuAD dataset
squad = load_dataset("squad")

# Use the first question and context from the dataset
question = squad["train"][0]["question"]
context = squad["train"][0]["context"]

# Tokenize the input and make predictions
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Extract start and end positions for the answer
start_logits = outputs.start_logits
end_logits = outputs.end_logits
answer_start = start_logits.argmax()
answer_end = end_logits.argmax() + 1

# Decode the answer
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)

# Print the results
print(f"Question: {question}")
print(f"Answer: {answer}")
