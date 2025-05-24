import os
import io
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def run_inference(model_dir, api, params):
    try:
        try:
            model = CosyVoice(model_dir)
        except Exception:
            model = CosyVoice2(model_dir)
        if api == "inference_sft":
            tts_text, spk_id = params
            result = []
            for i in model.inference_sft(tts_text, spk_id):
                tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        elif api == "inference_zero_shot":
            tts_text, prompt_text, prompt_wav_bytes = params
            prompt_speech_16k = load_wav(io.BytesIO(prompt_wav_bytes), 16000)
            result = []
            for i in model.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k):
                tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        elif api == "inference_cross_lingual":
            tts_text, prompt_wav_bytes = params
            prompt_speech_16k = load_wav(io.BytesIO(prompt_wav_bytes), 16000)
            result = []
            for i in model.inference_cross_lingual(tts_text, prompt_speech_16k):
                tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        elif api == "inference_instruct":
            tts_text, spk_id, instruct_text = params
            result = []
            for i in model.inference_instruct(tts_text, spk_id, instruct_text):
                tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        elif api == "inference_instruct2":
            tts_text, instruct_text, prompt_wav_bytes = params
            prompt_speech_16k = load_wav(io.BytesIO(prompt_wav_bytes), 16000)
            result = []
            for i in model.inference_instruct2(tts_text, instruct_text, prompt_speech_16k):
                tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        elif api == "ws_tts":
            texts, prompt_wav_bytes = params
            prompt_speech_16k = load_wav(io.BytesIO(prompt_wav_bytes), 16000)
            result = []
            for i, j in enumerate(model.inference_zero_shot(
                    iter(texts),
                    "希望你以后能够做的比我还好呦。",
                    prompt_speech_16k,
                    stream=True)):
                tts_audio = (j['tts_speech'].numpy() * (2 ** 15)).astype('int16').tobytes()
                result.append(tts_audio)
            return result
        else:
            return ["Unknown API"]
    except Exception as e:
        return [f"error:{str(e)}"]