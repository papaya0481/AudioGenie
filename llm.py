import os, json, mimetypes
from typing import Optional, List, Dict, Any
from pathlib import Path
import requests, base64

_MAX_INLINE_BYTES = int(os.environ.get("GEMINI_INLINE_LIMIT", str(15 * 1024 * 1024)))

def _mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

class LLM:
    def chat(self, system: str, user: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        raise NotImplementedError

class GeminiLLM(LLM):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self._client = None
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError("google-genai not installed. pip install google-genai") from e
        from google import genai
        self._genai = genai
        self._types = __import__("google.genai", fromlist=["types"]).types
        self._client = genai.Client(api_key=self.api_key)


    def _tolist(self, x):
        if not x:
            return []
        return x if isinstance(x, (list, tuple)) else [x]

    def _parts_for_media(self, images, videos, texts):
        parts = []

        # image
        for img in self._tolist(images):
            p = str(img)
            mt = _mime(p)
            try:
                size = Path(p).stat().st_size
            except Exception:
                size = 0
            if size and size <= _MAX_INLINE_BYTES:
                parts.append(self._types.Part(
                    inline_data=self._types.Blob(data=_read_bytes(p), mime_type=mt or "image/jpeg")
                ))
            else:
                try:
                    uploaded = self._client.files.upload(file=p) 
                    parts.append(self._types.Part(
                        file_data=self._types.FileData(file_uri=getattr(uploaded, "uri", None) or getattr(uploaded, "path", None), mime_type=mt or "image/jpeg")
                    ))
                except Exception:
                    parts.append(self._types.Part(
                        inline_data=self._types.Blob(data=_read_bytes(p), mime_type=mt or "image/jpeg")
                    ))

        # video
        for vid in self._tolist(videos):
            p = str(vid)
            mt = _mime(p) or "video/mp4"
            try:
                size = Path(p).stat().st_size
            except Exception:
                size = 0

            if size and size <= _MAX_INLINE_BYTES:
                parts.append(self._types.Part(
                    inline_data=self._types.Blob(data=_read_bytes(p), mime_type=mt)
                ))
            else:
                # files API
                try:
                    uploaded = self._client.files.upload(file=p)
                    # file_uri, file.name
                    file_uri = getattr(uploaded, "uri", None) or getattr(uploaded, "path", None) or getattr(uploaded, "name", None)
                    parts.append(self._types.Part(
                        file_data=self._types.FileData(file_uri=file_uri, mime_type=mt)
                    ))
                except Exception as e:
                    raise RuntimeError(f"[GeminiLLM] The video is too large and the file upload failed: {p}; please verify network connectivity and Files API permissions. Original error: {e}")
                                            
        # text
        for text in self._tolist(texts):
            if isinstance(text, str):
                parts.append(self._types.Part(
                    inline_data=self._types.Blob(data=text.encode("utf-8"), mime_type="text/plain")
                ))
            else:
                raise TypeError("The text must be a string.")

        return parts

    def _read_bytes(self, path: str) -> bytes:
        """audio file"""
        with open(path, "rb") as f:
            return f.read()

    def chat(self, system: str, user: str, stop=None, media: Optional[Dict[str, Any]] = None) -> str:
        if self._client is None:
            self._init_client()

        media = media or {}
        images = media.get("images")
        videos = media.get("videos")
        texts = media.get("texts")
        audio = media.get("audio")

        parts = []
        if images or videos or texts:
            parts.extend(self._parts_for_media(images, videos, texts))
        
        if audio:
            data = self._read_bytes(audio) if isinstance(audio, str) else audio
            audio_part = self._types.Part(
                inline_data=self._types.Blob(data=data, mime_type="audio/wav")
            )
            parts.append(audio_part)


        parts.append(self._types.Part(text=f"[SYSTEM]\n{system}"))
        parts.append(self._types.Part(text=f"[USER]\n{user}"))

        content = self._types.Content(parts=parts)
        resp = self._client.models.generate_content(model=self.model, contents=content)
        text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        return text or ""


class OpenaiLLM(LLM):
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai not installed. pip install openai") from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, stop=None, media: Optional[Dict[str, Any]] = None) -> str:
        if self._client is None:
            self._init_client()

        media = media or {}
        content = []

        for img in media.get("images", []) if isinstance(media.get("images"), list) else ([media.get("images")] if media.get("images") else []):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(img)}"}})

        content.append({"type": "text", "text": user})

        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]

        resp = self._client.chat.completions.create(model=self.model, messages=messages, stop=stop)
        return resp.choices[0].message.content or ""

    def _encode_image(self, path: str) -> str:
        import base64
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
class NvidiaLLM(LLM):
    def __init__(self, model: str = "microsoft/phi-4-multimodal-instruct", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.invoke_url = base_url or "https://integrate.api.nvidia.com/v1/chat/completions"
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set.")

    def chat(self, system: str, user: str, stop=None, media: Optional[Dict[str, Any]] = None) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        content = [{"type": "text", "text": user}]
        media = media or {}

        if media.get("texts"):
            texts = media["texts"] if isinstance(media["texts"], list) else [media["texts"]]
            for text in texts:
                content.append({"type": "text", "text": text})

        if media.get("images"):
            imgs = media["images"] if isinstance(media["images"], list) else [media["images"]]
            for img in imgs:
                with open(img, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        if media.get("videos"):
            vids = media["videos"] if isinstance(media["videos"], list) else [media["videos"]]
            for vid in vids:
                content.append({"type": "video_url", "video_url": {"url": vid}})

        if media.get("audio"):
            audios = media["audio"] if isinstance(media["audio"], list) else [media["audio"]]
            for audio in audios:
                with open(audio, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.3,
            "repetition_penalty": 1.2,
            "reasoning": "false"
        }

        resp = requests.post(self.invoke_url, headers=headers, json=payload)
        return resp.json()["choices"][0]["message"]["content"]
    

class HuggingfaceLLM(LLM):
    def __init__(self, model: str = "Qwen/Qwen3-VL-4B-Instruct", **parameters):
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            import torch
        except Exception as e:
            raise RuntimeError("transformers not installed. pip install transformers") from e
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # 如果你想进一步压榨显存，可以改成 4-bit：
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16
        )

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(model,
                                                                      quantization_config=quantization_config,
                                                                      dtype=torch.bfloat16,
                                                                      **parameters)
        self.processor = AutoProcessor.from_pretrained(model)

    def chat(self, system: str, user: str, stop=None, media: Optional[Dict[str, Any]] = None) -> str:
        media = media or {}

        content = []
        messages = []
        # if system:
        #     content.append({"type": "text", "text": f"{system}"})
        #     messages.append({"role": "system", "content": content})
        #     content = []

        for img in (media.get("images") or []) if isinstance(media.get("images"), list) else ([media["images"]] if media.get("images") else []):
            content.append({"type": "image", "image": str(img)})

        for vid in (media.get("videos") or []) if isinstance(media.get("videos"), list) else ([media["videos"]] if media.get("videos") else []):
            content.append({"type": "video", "video": str(vid)})
            
        for text in (media.get("texts") or []) if isinstance(media.get("texts"), list) else ([media["texts"]] if media.get("texts") else []):
            content.append({"type": "text", "text": str(text)})

        content.append({"type": "text", "text": f"{user}"})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self._model.device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""