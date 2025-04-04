from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

# FastAPI 앱 생성
app = FastAPI(title="한국어 오타 교정 API", description="KoBART 모델을 활용한 한국어 오타 교정 서비스")

# 모델과 토크나이저 변수 선언
tokenizer = None
model = None
device = None


# 모델 로드 함수
def load_model():
    global tokenizer, model, device

    model_base_path = "./models/final/final_model_10"  # 실제 경로로 수정 필요

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    try:
        # 학습된 모델 로드
        print("학습된 모델 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_base_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_base_path)
        model.to(device)
        print(f"모델이 {device}에 로드되었습니다.")
        return True

    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        # 오류 발생 시 기본 모델로 대체
        print("기본 모델로 대체 시도 중...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
            model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
            model.to(device)
            print(f"기본 모델이 {device}에 로드되었습니다.")
            return True
        except Exception as fallback_e:
            print(f"기본 모델 로드 실패: {fallback_e}")
            return False


# 앱 시작 시 모델 로드
if not load_model():
    raise RuntimeError("모델을 로드할 수 없습니다. 서비스를 시작할 수 없습니다.")


# 입력 데이터 모델 정의
class TextInput(BaseModel):
    text: str


class CorrectionResult(BaseModel):
    original: str
    corrected: str


@app.post("/correct", response_model=CorrectionResult)
async def correct_text(input_data: TextInput):
    try:
        text = input_data.text
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")

        # 입력 데이터 토큰화
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # 오타 교정 생성
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
        )
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"original": text, "corrected": corrected_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오타 교정 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
