from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")

# 오타 문장 입력
input_text = "안뇽하세요"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"])
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"입력: {input_text} -> 교정: {corrected_text}")