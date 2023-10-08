from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoTokenizer
import numpy as np
from datasets import load_dataset, load_metric
import nltk
import os


dataset = load_dataset("quora")
model_id="google/flan-t5-small"
max_input_length = 1024
max_target_length = 512
batch_size = 92
num_epochs = 12
accumulation = 1
lr = 5e-5

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


tokenizer = AutoTokenizer.from_pretrained(model_id)


metric = load_metric("rouge")
data_collator = DataCollatorForSeq2Seq(tokenizer)
def compute_metrics(eval_pred):
    predictions, labels= eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = [p[0] + '\n' for p in decoded_preds]
    decoded_labels = [l[0] + '\n' for l in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def preprocess(dataset):
  inputs = []
  for data in dataset['questions']:
    inputs.append("Are question 1 and question 2 paraphrase?: question 1: " + data['text'][0].strip() + ' question 2: ' + data['text'][1].strip())
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
  labels = tokenizer([str(l) for l in dataset['is_duplicate']], max_length=max_target_length, truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


tokenizedDataset = dataset['train'].map(preprocess,batched=True)
datasets_train_test = tokenizedDataset.train_test_split(test_size=0.2)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
training_args = Seq2SeqTrainingArguments(
    output_dir=model_id+'out',
    evaluation_strategy="epoch",
    eval_steps=1,
    logging_strategy="steps",
    logging_steps=2000,
    save_strategy="epoch",
    #save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4*batch_size,
    #weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=False,
    metric_for_best_model="rouge1",
    report_to="tensorboard",
    gradient_accumulation_steps=accumulation,
    #push_to_hub=True,
    #hub_token="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"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets_train_test["train"],
    eval_dataset=datasets_train_test["test"],
    compute_metrics=compute_metrics,
)

trainer.train()