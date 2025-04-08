from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained("./bert-base-uncased")
model.save_pretrained("./bert-base-uncased")
