from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import json
import torch
from utils.generators import augment_sentence

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# 학습 데이터 준비
base_data = [
    {"annotation": {"err_sentence": "안녕하세요", "cor_sentence": "안녕하세요"}},
    {"annotation": {"err_sentence": "반갑습니다", "cor_sentence": "반갑습니다"}},
    {"annotation": {"err_sentence": "좋아요", "cor_sentence": "좋아요"}},
]
augmented_data = []
for item in base_data:
    cor_sentence = item["annotation"]["cor_sentence"]
    augmented_data.append({"annotation": {"err_sentence": cor_sentence, "cor_sentence": cor_sentence}})
    for _ in range(10):
        err_sentence = augment_sentence(cor_sentence, prob=0.3)
        augmented_data.append({"annotation": {"err_sentence": err_sentence, "cor_sentence": cor_sentence}})

data = {"data": augmented_data}
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 데이터 로드 및 Dataset 변환
with open("data/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
dataset = Dataset.from_list([item["annotation"] for item in train_data["data"]])


# 전처리 함수
def preprocess_function(examples):
    inputs = tokenizer(examples["err_sentence"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples["cor_sentence"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1,
    learning_rate=5e-5,
)

# 트레이너 설정
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 학습 시작
trainer.train()

# 학습된 모델과 토크나이저 저장
save_path = "models/final_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"모델과 토크나이저가 {save_path}에 저장되었습니다.")

# 학습된 모델로 테스트
input_text = "안뇽하세요"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(
    inputs["input_ids"],
    max_length=128,
    num_beams=4
)
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"입력: {input_text} -> 교정: {corrected_text}")
