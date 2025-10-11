import os, json, mimetypes
from typing import Optional, List, Dict, Any
from pathlib import Path

class LLM:
    def chat(self, system: str, user: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        raise NotImplementedError


_MAX_INLINE_BYTES = int(os.environ.get("GEMINI_INLINE_LIMIT", str(15 * 1024 * 1024)))

def _mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

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
