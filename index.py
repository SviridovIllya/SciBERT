import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import torch
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score, recall_score
import transformers

print(f"Transformers version: {transformers.__version__}")

dataset = load_dataset("conll2003")

# Обмеження розміру датасету для швидшого виконання
train_data = dataset['train'].shuffle(seed=42).select(range(1000))  # 1000 прикладів для тренування
test_data = dataset['test'].shuffle(seed=42).select(range(200))  # 200 прикладів для тестування

# Завантаження токенізатора та моделі SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased",
                                                        num_labels=9)  # 9 класів у CoNLL-2003

label_map = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC"
}


# Токенізація та вирівнювання міток
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=512)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ігноруємо субтокени
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_data_tokenized = train_data.map(tokenize_and_align_labels, batched=True)
test_data_tokenized = test_data.map(tokenize_and_align_labels, batched=True)

train_data_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_data_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./scibert_ner_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# Функція для обчислення метрик
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_map[l] for l in label if l != -100] for label in labels]
    pred_labels = [[label_map[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]

    return {
        "f1": seqeval_f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels)
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_tokenized,
    eval_dataset=test_data_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Тренування моделі
trainer.train()

# Оцінка SciBERT
scibert_results = trainer.evaluate()
scibert_f1 = scibert_results['eval_f1']
scibert_precision = scibert_results['eval_precision']
scibert_recall = scibert_results['eval_recall']
print(f"SciBERT F1-score: {scibert_f1:.4f}")
print(f"SciBERT Precision: {scibert_precision:.4f}")
print(f"SciBERT Recall: {scibert_recall:.4f}")


# --- Логістична регресія
def prepare_token_level_data(dataset):
    X, y = [], []
    for example in dataset:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        for token, tag in zip(tokens, ner_tags):
            X.append(token)
            y.append(tag)
    return X, y


X_train, y_train = prepare_token_level_data(train_data)
X_test, y_test = prepare_token_level_data(test_data)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Тренування логістичної регресії
logreg = LogisticRegression(max_iter=1000, multi_class='multinomial')
logreg.fit(X_train_tfidf, y_train)

# Оцінка логістичної регресії
y_pred = logreg.predict(X_test_tfidf)
# Конвертація міток логістичної регресії у строковий формат для seqeval
y_test_str = [label_map[y] for y in y_test]
y_pred_str = [label_map[y] for y in y_pred]
logreg_f1 = seqeval_f1_score([y_test_str], [y_pred_str])
logreg_precision = precision_score([y_test_str], [y_pred_str])
logreg_recall = recall_score([y_test_str], [y_pred_str])
print(f"Logistic Regression F1-score: {logreg_f1:.4f}")
print(f"Logistic Regression Precision: {logreg_precision:.4f}")
print(f"Logistic Regression Recall: {logreg_recall:.4f}")

# Порівняння результатів
print("\nПорівняння результатів:")
print(f"SciBERT F1-score: {scibert_f1:.4f}, Precision: {scibert_precision:.4f}, Recall: {scibert_recall:.4f}")
print(f"Logistic Regression F1-score: {logreg_f1:.4f}, Precision: {logreg_precision:.4f}, Recall: {logreg_recall:.4f}")
print(f"Різниця (SciBERT - LogReg) F1-score: {(scibert_f1 - logreg_f1):.4f}")
print(f"Різниця (SciBERT - LogReg) Precision: {(scibert_precision - logreg_precision):.4f}")
print(f"Різниця (SciBERT - LogReg) Recall: {(scibert_recall - logreg_recall):.4f}")