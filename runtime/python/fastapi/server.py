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
import asyncio
import os
import queue
import sys
import argparse
import logging
import threading
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
cosyvoice_lock = threading.Lock()  # 全局锁，保护cosyvoice调用
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
    receive_task = None
    tts_thread = None
    text_queue = None
    stop_event = None

    try:
        # 1. 加载音色提示（每个连接独立加载）
        prompt_wav_path = os.path.join(ROOT_DIR, "zero_shot_prompt.wav")
        with open(prompt_wav_path, "rb") as f:
            prompt_speech_16k = load_wav(f, 16000)
            logging.info("音色载入成功")

        # 2. 初始化连接专用资源
        text_queue = queue.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()

        # 3. 异步接收前端文本
        async def receive_texts():
            try:
                while not stop_event.is_set():
                    data = await websocket.receive_json()
                    tts_text = data.get("tts_text", "")
                    if tts_text == "__end__":
                        logging.info("收到终止信号，结束生成")
                        text_queue.put(None)  # 发送终止信号
                        break
                    text_queue.put(tts_text)
                    logging.info(f"文本已入队: {tts_text}")
            except asyncio.CancelledError:
                logging.info("接收任务被取消")
            except WebSocketDisconnect:
                logging.info("客户端主动断开连接")
            except Exception as e:
                logging.error(f"接收异常: {e}")
                text_queue.put(None)  # 异常时终止生成器

        receive_task = asyncio.create_task(receive_texts())

        # 4. 文本生成器（支持超时检查）
        def text_generator():
            while not stop_event.is_set():
                try:
                    t = text_queue.get(timeout=0.1)  # 避免永久阻塞
                    logging.info(f"生成器取出文本: {t}")
                    if t is None:
                        break
                    yield t
                except queue.Empty:
                    continue

        # 5. TTS工作线程（加锁保护cosyvoice）
        def tts_worker():
            logging.info("TTS线程启动")
            try:
                with cosyvoice_lock:  # 确保同一时间只有一个线程调用cosyvoice
                    for i, j in enumerate(cosyvoice.inference_zero_shot(
                        text_generator(),
                        "希望你以后能够做的比我还好呦。",
                        prompt_speech_16k,
                        stream=True
                    )):
                        if stop_event.is_set():
                            break
                        audio = (j['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                        fut = asyncio.run_coroutine_threadsafe(
                            websocket.send_bytes(audio),
                            loop
                        )
                        fut.result()  # 等待发送完成
                        logging.info(f"已发送第{i}段音频")
            except Exception as e:
                logging.error(f"TTS线程异常: {e}")
            finally:
                logging.info("TTS线程退出")

        tts_thread = threading.Thread(target=tts_worker)
        tts_thread.start()

        # 6. 监控连接状态
        while True:
            await asyncio.sleep(0.1)
            if not websocket.client_state == "CONNECTED":
                break

    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    except Exception as e:
        logging.error(f"WebSocket异常: {e}")
    finally:
        # 7. 清理资源
        logging.info("开始清理资源...")
        if stop_event:
            stop_event.set()  # 通知所有线程退出
        if text_queue:
            text_queue.put(None)  # 确保生成器退出
        if receive_task:
            receive_task.cancel()  # 取消接收任务
        if tts_thread:
            tts_thread.join(timeout=2)  # 等待线程退出
            if tts_thread.is_alive():
                logging.warning("TTS线程未正常退出，强制终止")
        await websocket.close()
        logging.info("资源清理完成")
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
