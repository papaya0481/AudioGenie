import argparse, os, sys
from pathlib import Path

def add_repo(home: str):
    home = os.path.expanduser(home)
    if home not in sys.path:
        sys.path.insert(0, home)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.environ.get("COSYVOICE_HOME", "./bin/cosyvoice"))
    ap.add_argument("--model", default="FunAudioLLM/CosyVoice2-0.5B")
    ap.add_argument("--target_text", default="大家好啊，我是你的专属数字人朵拉，今天你的心情怎么样呀")
    ap.add_argument("--prompt_transcript", default="希望你以后能够做的比我还好呦。")
    ap.add_argument("--prompt_wav", default=os.environ.get("COSYVOICE_PROMPT_WAV", "asset/zero_shot_prompt.wav"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    home = Path(args.home).expanduser().resolve()
    os.chdir(home)
    os.environ["PYTHONPATH"] = str(home / "third_party/Matcha-TTS") + os.pathsep + os.environ.get("PYTHONPATH", "")

    add_repo(str(home))

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio

    cosyvoice = CosyVoice2(args.model, load_jit=False, load_trt=False)
    prompt_wav = Path(args.prompt_wav).expanduser()
    if not prompt_wav.is_absolute():
        prompt_wav = home / prompt_wav
    prompt_speech_16k = load_wav(str(prompt_wav), 16000)
    gen = None
    for item in cosyvoice.inference_zero_shot(args.target_text, args.prompt_transcript, prompt_speech_16k, stream=False):
        gen = item
    wav = gen['tts_speech']
    sr = cosyvoice.sample_rate

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out, wav, sr)

if __name__ == "__main__":
    main()
