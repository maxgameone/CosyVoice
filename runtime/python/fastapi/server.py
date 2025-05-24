# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import queue
import sys
import argparse
import logging
import pyogg
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import uuid
import logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
user_sessions = {}

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))

@app.websocket("/ws_tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    logging.info("ws_tts: 连接已建立")
    try:
        # 每次连接都新建模型实例
        try:
            local_cosyvoice = CosyVoice(args.model_dir)
            logging.info("ws_tts: CosyVoice实例加载成功")
        except Exception:
            local_cosyvoice = CosyVoice2(args.model_dir)
            logging.info("ws_tts: CosyVoice2实例加载成功")
        logging.info("ws_tts: 本连接模型实例已加载")

        prompt_wav_path = os.path.join(ROOT_DIR, "zero_shot_prompt.wav")
        with open(prompt_wav_path, "rb") as f:
            prompt_speech_16k = load_wav(f, 16000)
            logging.info("ws_tts: 音色载入成功")
       
        # 初始化 Opus 编码器（16kHz, 单声道）
        opus_encoder = pyogg.OpusEncoder()
        opus_encoder.set_application("audio")
        opus_encoder.set_sampling_frequency(16000)
        opus_encoder.set_channels(1)

        text_queue = queue.Queue()

        async def receive_texts():
            while True:
                data = await websocket.receive_json()
                tts_text = data.get("tts_text")
                logging.info(f"ws_tts: 收到文本: {tts_text}")
                if tts_text is None or tts_text == "__end__":
                    text_queue.put(None)
                    logging.info("ws_tts: 收到结束信号，退出文本接收")
                    break
                text_queue.put(tts_text)
                logging.info("ws_tts: 文本已放入队列")

        import asyncio
        asyncio.create_task(receive_texts())

        def text_generator():
            while True:
                t = text_queue.get()
                logging.info(f"ws_tts: 生成器取到文本: {t}")
                if t is None:
                    logging.info("ws_tts: 生成器收到结束信号，退出")
                    break
                yield t

        logging.info("ws_tts: 开始推理循环")
        for i, j in enumerate(local_cosyvoice.inference_zero_shot(
                text_generator(),
                "希望你以后能够做的比我还好呦。",
                prompt_speech_16k,
                stream=False)):
            pcm_data = (j['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            opus_data = opus_encoder.encode(pcm_data)
            await websocket.send_bytes(opus_data)
            logging.info(f"ws_tts: 已发送音频数据 {i}")
        logging.info("ws_tts: 推理循环结束")
    except WebSocketDisconnect:
        logging.info("ws_tts: 连接断开")
        await websocket.close()
    except Exception as e:
        logging.error(f"ws_tts: 异常: {e}")
        await websocket.send_json({"error": str(e)})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
