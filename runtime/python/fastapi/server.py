import os
import sys
import argparse
import logging
import multiprocessing
import asyncio
import time
import opuslib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 用 Manager 解决 Queue 跨进程传递问题
manager = multiprocessing.Manager()

# -------- Worker进程代码 --------
def tts_worker(model_dir, task_queue, result_queue):
    logging.info(f"[Worker {os.getpid()}] 启动，加载模型中...")
    try:
        local_cosyvoice = CosyVoice(model_dir)
        logging.info(f"[Worker {os.getpid()}] CosyVoice实例加载成功")
    except Exception:
        local_cosyvoice = CosyVoice2(model_dir)
        logging.info(f"[Worker {os.getpid()}] CosyVoice2实例加载成功")
    logging.info(f"[Worker {os.getpid()}] 模型加载完成，等待任务")
    while True:
        task = task_queue.get()
        if task is None:
            logging.info(f"[Worker {os.getpid()}] 收到退出信号，退出")
            break
        text_queue, prompt_speech_16k = task

        # 保留你的流式生成器推理代码
        def text_generator():
            while True:
                t = text_queue.get()
                logging.info(f"ws_tts: 生成器取到文本: {t}")
                if t is None:
                    logging.info("ws_tts: 生成器收到结束信号，退出")
                    break
                yield t

        logging.info("ws_tts: 开始推理循环")
        encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
        frame_size = 320  # 20ms帧，16kHz采样率，单声道
        for i, j in enumerate(local_cosyvoice.inference_zero_shot(
                text_generator(),
                "希望你以后能够做的比我还好呦。",
                prompt_speech_16k,
                stream=True)):
            pcm_data = (j['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            # 分帧编码
            for start in range(0, len(pcm_data), frame_size * 2):  # 2字节每采样点
                frame = pcm_data[start:start + frame_size * 2]
                if len(frame) < frame_size * 2:
                    # 填充最后一帧
                    frame += b'\x00' * (frame_size * 2 - len(frame))
                opus_data = encoder.encode(frame, frame_size)
                result_queue.put(opus_data)
        logging.info("ws_tts: 推理循环结束")
        result_queue.put(None)

# -------- 进程池管理 --------
task_queues, result_queues, workers = [], [], []
def start_workers(model_dir, num_workers=5):
    for idx in range(num_workers):
        tq, rq = manager.Queue(), manager.Queue()  # 用 manager.Queue()
        p = multiprocessing.Process(target=tts_worker, args=(model_dir, tq, rq))
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        workers.append(p)
        logging.info(f"[Main] 启动worker进程 {p.pid}")

def get_worker(timeout=10):
    """等待timeout秒，超时抛异常"""
    start = time.time()
    while True:
        for tq, rq in zip(task_queues, result_queues):
            if rq.empty():
                return tq, rq
        if time.time() - start > timeout:
            raise RuntimeError("没有空闲worker，请稍后重试")
        time.sleep(0.01)

# -------- WebSocket 路由 --------
@app.websocket("/ws_tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    logging.info("[Main] 新WebSocket连接")
    try:
        prompt_wav_path = os.path.join(ROOT_DIR, "zero_shot_prompt.wav")
        with open(prompt_wav_path, "rb") as f:
            prompt_speech_16k = load_wav(f, 16000)
        tq, rq = get_worker()
        text_queue = manager.Queue()  # 用 manager.Queue()
        tq.put((text_queue, prompt_speech_16k))
        # 启动异步文本接收任务
        async def receive_texts():
            while True:
                data = await websocket.receive_json()
                tts_text = data.get("tts_text")
                logging.info(f"[Main] 收到文本: {tts_text}")
                if tts_text is None or tts_text == "__end__":
                    text_queue.put(None)
                    logging.info("[Main] 收到结束信号，退出文本接收")
                    break
                text_queue.put(tts_text)
                logging.info("[Main] 文本已放入队列")
        receive_task = asyncio.create_task(receive_texts())
        # 实时读取 worker 返回的音频帧并发送给前端
        while True:
            opus_data = await asyncio.get_event_loop().run_in_executor(None, rq.get)
            if opus_data is None:
                break
            await websocket.send_bytes(opus_data)
    except WebSocketDisconnect:
        logging.info("[Main] WebSocket断开")
        await websocket.close()
    except Exception as e:
        logging.error(f"[Main] 异常: {e}")
        await websocket.send_json({"error": str(e)})

import atexit
def cleanup():
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
    parser.add_argument('--worker_num', type=int, default=5)
    args = parser.parse_args()
    start_workers(args.model_dir, args.worker_num)
    logging.info(f"[Main] 启动服务，模型目录: {args.model_dir}, worker数: {args.worker_num}")
    logging.info(f"[Main] 服务启动，监听端口 {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)