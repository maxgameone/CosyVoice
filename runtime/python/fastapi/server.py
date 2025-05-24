import asyncio
import os
import sys
import argparse
import logging
import multiprocessing
import io
from fastapi import FastAPI, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tts_worker_process import run_inference

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

pool = None
args = None

def stream_audio(result):
    for audio in result:
        if isinstance(audio, bytes):
            yield audio

@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pool.apply(run_inference, (args.model_dir, "inference_sft", (tts_text, spk_id)))
    )
    if result and isinstance(result[0], str) and result[0].startswith("error:"):
        return {"error": result[0]}
    return StreamingResponse(stream_audio(result))

@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_wav_bytes = await prompt_wav.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pool.apply(run_inference, (args.model_dir, "inference_zero_shot", (tts_text, prompt_text, prompt_wav_bytes)))
    )
    if result and isinstance(result[0], str) and result[0].startswith("error:"):
        return {"error": result[0]}
    return StreamingResponse(stream_audio(result))

@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_wav_bytes = await prompt_wav.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pool.apply(run_inference, (args.model_dir, "inference_cross_lingual", (tts_text, prompt_wav_bytes)))
    )
    if result and isinstance(result[0], str) and result[0].startswith("error:"):
        return {"error": result[0]}
    return StreamingResponse(stream_audio(result))

@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pool.apply(run_inference, (args.model_dir, "inference_instruct", (tts_text, spk_id, instruct_text)))
    )
    if result and isinstance(result[0], str) and result[0].startswith("error:"):
        return {"error": result[0]}
    return StreamingResponse(stream_audio(result))

@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_wav_bytes = await prompt_wav.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pool.apply(run_inference, (args.model_dir, "inference_instruct2", (tts_text, instruct_text, prompt_wav_bytes)))
    )
    if result and isinstance(result[0], str) and result[0].startswith("error:"):
        return {"error": result[0]}
    return StreamingResponse(stream_audio(result))

@app.websocket("/ws_tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        prompt_wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_shot_prompt.wav")
        with open(prompt_wav_path, "rb") as f:
            prompt_wav_bytes = f.read()
        texts = []
        async def receive_texts():
            while True:
                data = await websocket.receive_json()
                tts_text = data.get("tts_text")
                if tts_text is None or tts_text == "__end__":
                    break
                texts.append(tts_text)
        await receive_texts()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pool.apply(run_inference, (args.model_dir, "ws_tts", (texts, prompt_wav_bytes)))
        )
        if result and isinstance(result[0], str) and result[0].startswith("error:"):
            await websocket.send_json({"error": result[0]})
        else:
            for tts_audio in result:
                await websocket.send_bytes(tts_audio)
    except WebSocketDisconnect:
        await websocket.close()
    except Exception as e:
        await websocket.send_json({"error": str(e)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='iic/CosyVoice-300M')
    parser.add_argument('--workers', type=int, default=5, help='进程池大小')
    args = parser.parse_args()
    pool = multiprocessing.Pool(processes=args.workers)
    uvicorn.run(app, host="0.0.0.0", port=args.port)