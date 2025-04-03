from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")


def correct_typo(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 테스트
test_sentences = [
    "안녕하셰요, 반갑숩니다.",
    "나는 어제 학꾜에 갓습니다.",
    "내일 친구를 만낟서 영화를 볼거예요."
]

for sentence in test_sentences:
    corrected = correct_typo(sentence)
    print(f"원본: {sentence}")
    print(f"교정: {corrected}")
    print("-" * 50)
