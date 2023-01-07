import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "C:\Users\furqa\OneDrive\Desktop\imdb\input\IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)