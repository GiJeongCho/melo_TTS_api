from fastapi import FastAPI, HTTPException, Header, Depends,Body
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
from pathlib import Path
from typing import Dict
import torch
from melo.api import TTS  

app = FastAPI()

models: Dict[str, TTS] = {}  # 화자 이름을 키로 하고 TTS 인스턴스를 값으로 하는 딕셔너리

def load_models():
    base_path = Path("/home/edutem/joge/TTS/melo_api/model/en")
    if torch.cuda.is_available():
        print("CUDA is available. Models will be loaded on GPU.")
    else:
        print("CUDA is not available. Models will be loaded on CPU.")
    for gender_dir in base_path.iterdir():
        for speaker_dir in gender_dir.iterdir():
            if speaker_dir.is_dir():
                ckpt_path = speaker_dir / "G_0.pth"
                config_path = speaker_dir / "config.json"
                if ckpt_path.exists() and config_path.exists():
                    speaker_name = speaker_dir.stem  # 디렉토리 이름을 화자 이름으로 사용
                    models[speaker_name] = TTS(
                        language="EN",
                        device="auto",
                        use_hf=False,
                        config_path=str(config_path),
                        ckpt_path=str(ckpt_path)
                    )
                    print(f"Model for {speaker_name} loaded successfully.")

@app.on_event("startup")
async def startup_event():
    load_models()  # 서버 시작 시 모든 모델 로드

@app.post("/synthesize/")
async def synthesize(text: str = Body(...), speaker_name: str = Body(...)):
    try:
        if speaker_name in models:
            model = models[speaker_name]
            audio_buffer = model.tts_to_bytes(text, speaker_id=0)  # speaker_id를 사용자 정의에 맞게 조정
            return StreamingResponse(audio_buffer, media_type="audio/wav")
        else:
            raise HTTPException(status_code=404, detail="Speaker not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
