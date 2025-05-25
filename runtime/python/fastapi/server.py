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
import multiprocessing
import os
import queue
import sys
import argparse
import logging
import opuslib
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


# TTS worker 进程函数，每个进程独立加载模型和 Opus 编码器
def tts_worker(model_dir, task_queue, result_queue):
    logging.info(f"[Worker {os.getpid()}] 启动，加载模型中...")
    model = CosyVoice(model_dir)
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    frame_size = 320  # 20ms帧
    logging.info(f"[Worker {os.getpid()}] 模型加载完成，等待任务")
    while True:
        task = task_queue.get()
        if task is None:  # 收到退出信号
            logging.info(f"[Worker {os.getpid()}] 收到退出信号，退出")
            break
        tts_text, prompt_text, prompt_speech_16k = task
        logging.info(f"[Worker {os.getpid()}] 收到任务: {tts_text}")
        # 逐句推理并分帧编码
        for j in model.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=True):
            pcm = (j['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            for start in range(0, len(pcm), frame_size * 2):
                frame = pcm[start:start + frame_size * 2]
                if len(frame) < frame_size * 2:
                    frame += b'\x00' * (frame_size * 2 - len(frame))
                opus_data = encoder.encode(frame, frame_size)
                result_queue.put(opus_data)
        result_queue.put(None)  # 一段文本结束信号
        logging.info(f"[Worker {os.getpid()}] 任务完成")

# 启动进程池
NUM_WORKERS = 2
task_queues, result_queues, workers = [], [], []
def start_workers(model_dir):
    """启动多个 TTS worker 进程，每个进程有独立的任务队列和结果队列"""
    for idx in range(NUM_WORKERS):
        tq, rq = multiprocessing.Queue(), multiprocessing.Queue()
        p = multiprocessing.Process(target=tts_worker, args=(model_dir, tq, rq))
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        workers.append(p)
        logging.info(f"[Main] 启动worker进程 {p.pid}")

def get_worker():
    """简单轮询分配空闲 worker"""
    for tq, rq in zip(task_queues, result_queues):
        if rq.empty():
            return tq, rq
    return task_queues[0], result_queues[0]

@app.websocket("/ws_tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket TTS 推理接口，接收文本，返回 Opus 音频流"""
    await websocket.accept()
    logging.info("[Main] 新WebSocket连接")
    try:
        # 载入提示音色
        prompt_wav_path = os.path.join(ROOT_DIR, "zero_shot_prompt.wav")
        with open(prompt_wav_path, "rb") as f:
            prompt_speech_16k = load_wav(f, 16000)
        while True:
            data = await websocket.receive_json()
            tts_text = data.get("tts_text")
            logging.info(f"[Main] 收到文本: {tts_text}")
            if not tts_text or tts_text == "__end__":
                break
            tq, rq = get_worker()
            # 发送任务到 worker
            tq.put((tts_text, "希望你以后能够做的比我还好呦。", prompt_speech_16k))
            # 实时读取 worker 返回的音频帧并发送给前端
            frame_count = 0
            while True:
                opus_data = await asyncio.get_event_loop().run_in_executor(None, rq.get)
                if opus_data is None:
                    break
                await websocket.send_bytes(opus_data)
                frame_count += 1
            logging.info(f"[Main] 文本推理完成，发送帧数: {frame_count}")
    except WebSocketDisconnect:
        logging.info("[Main] WebSocket断开")
        await websocket.close()
    except Exception as e:
        logging.error(f"[Main] 异常: {e}")
        await websocket.send_json({"error": str(e)})

import atexit
def cleanup():
    """服务退出时关闭所有 worker 进程"""
    logging.info("[Main] 服务退出，清理worker进程")
    for tq in task_queues:
        tq.put(None)
    for p in workers:
        p.join()
atexit.register(cleanup)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='iic/CosyVoice-300M')
    args = parser.parse_args()
    start_workers(args.model_dir)
    logging.info(f"[Main] 服务启动，监听端口 {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)