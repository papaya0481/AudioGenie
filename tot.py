from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os, uuid, json, copy, shlex, subprocess, re

from AudioGenie.tools import ToolLibrary, run_tool
from AudioGenie.critiquers import AudioEvalCritic
from AudioGenie.llm import LLM
from AudioGenie.tools import _SafeDict
from AudioGenie.critiquers import AudioEvalCritic


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _sec_from_event(ev: Dict[str, Any]) -> float:
    st = _safe_float(ev.get("start_time"), 0.0)
    et = _safe_float(ev.get("end_time"), st)
    dur = _safe_float(ev.get("duration"), et - st)
    return max(dur, 0.2)

def _norm_type(t: str) -> str:
    return (t or "").strip().lower()

def _best_threshold_met(scores: Dict[str, float]) -> bool:
    return (
        scores.get("quality", 0.0) >= 0.7 and
        scores.get("alignment", 0.0) >= 0.7 and
        scores.get("aesthetics", 0.0) >= 0.7
    )

def _pick_text_key(args: Dict[str, Any]) -> Optional[str]:
    for k in ("text", "prompt", "ref_prompt", "tts_text"):
        if k in args and isinstance(args[k], str):
            return k
    return None

def _as_flag(k: str, v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, bool):
        return [f"--{k}"] if v else []
    return [f"--{k}", str(v)]


# ============== ToT ==============
@dataclass
class ToTNode:
    node_id: str
    node_type: str  # "initial" | "generation" | "refinement"
    model: Optional[str] = None
    output_wav: Optional[str] = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

class ToTExecutor:
    def __init__(
        self,
        tool_lib: ToolLibrary,
        llm: LLM,
        critic: AudioEvalCritic,
        prompt_max_retries: int = 1,
        max_depth: int = 3,
        max_siblings: int = 2,
        prefer_bin: bool = True, 
    ):
        self.tool_lib = tool_lib
        self.llm = llm
        self.critic = critic
        self.prompt_max_retries = max(0, int(prompt_max_retries))
        self.max_depth = max_depth
        self.max_siblings = max_siblings
        self.prefer_bin = prefer_bin
        self.nodes: Dict[str, ToTNode] = {}

    def _new_node(self, node_type: str, parent: Optional[str] = None, **meta) -> ToTNode:
        nid = str(uuid.uuid4())[:8]
        n = ToTNode(node_id=nid, node_type=node_type, parent=parent, meta=meta)
        self.nodes[nid] = n
        if parent:
            self.nodes[parent].children.append(nid)
        return n

    def _revise_text_prompt(self, model: str, prev_args: Dict[str, Any], event: Dict[str, Any],
                            scores: Dict[str, float], suggestions: List[str]) -> Dict[str, Any]:
        if model not in ("MMAudio", "InspireMusic", "CosyVoice2", "DiffRhythm"):
            return prev_args

        text_key = _pick_text_key(prev_args)
        if not text_key:
            return prev_args

        system = (
            "You are an audio prompt refining assistant. "
            "Given the previous generation arguments and evaluation feedback, "
            "rewrite ONLY the text prompt to better match the described scene and timing. "
            "Return JSON with one field: {\"text\": \"...\"}."
        )
        user = json.dumps({
            "model": model,
            "previous_args": prev_args,
            "event": {
                "audio_type": event.get("audio_type"),
                "Object": event.get("Object"),
                "description": event.get("description"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "duration": event.get("duration"),
            },
            "eval_scores": scores,
            "eval_suggestions": suggestions,
            "requirements": [
                "Keep structure unchanged; revise only the text field.",
                "Respect timing/duration and scene realism.",
                "Avoid generic terms; add concrete acoustic details."
            ]
        }, ensure_ascii=False)

        try:
            raw = self.llm.chat(system, user)
            raw = (raw or "").strip()
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
                new_args = copy.deepcopy(prev_args)
                new_args[text_key] = obj["text"].strip()
                return new_args
        except Exception:
            pass
        return prev_args

    def _call_model(self, model_name: str, args: Dict[str, Any], out_wav: str, workdir: str) -> Tuple[str, Dict[str, Any]]:
        a = copy.deepcopy(args)
        a.setdefault("out", out_wav)

        try:
            tool = self.tool_lib.get(model_name)
            wav = run_tool(tool, a, out_wav)
            return wav or "", {"runner": "tool_lib"}
        except Exception as e:
            return "", {"runner": "tool_lib", "error": str(e)}


    def run(self, event: Dict[str, Any], workdir: str) -> Tuple[str, Dict[str, Any], Dict[str, dict]]:
        if event.get("keep") and isinstance(event.get("keep_wav"), str) and os.path.exists(event["keep_wav"]):
            nodes_snapshot = {
                "kept": {
                    "node_id": "kept",
                    "node_type": "kept",
                    "model": "PROBE_KEEP",
                    "output_wav": event["keep_wav"],
                    "parent": None,
                    "children": [],
                    "meta": {"note": "Kept from Stage-2 video probe"}
                }
            }
            scores = {"quality": 1.0, "alignment": 1.0, "aesthetics": 1.0}
            return event["keep_wav"], scores, nodes_snapshot

        root = self._new_node("initial", meta={"event": event})
        candidates: List[str] = list(event.get("model_candidates") or [])[: self.max_siblings]
        refined_inputs: Dict[str, Dict[str, Any]] = dict(event.get("refined_inputs") or {})
        if not candidates:
            self.nodes[root.node_id].meta["note"] = "no candidates in event; skipped"
            nodes_snapshot = {nid: self.nodes[nid].__dict__ for nid in self.nodes}
            return "", {"quality":0,"alignment":0,"aesthetics":0}, nodes_snapshot

        best_wav: Optional[str] = None
        best_scores: Dict[str, float] = {"quality": 0.0, "alignment": 0.0, "aesthetics": 0.0}

        for model_name in candidates:
            base_args = copy.deepcopy(refined_inputs)
            tries = 1 + self.prompt_max_retries
            prev_scores: Dict[str, float] = {}
            prev_suggestions: List[str] = []

            for attempt in range(tries):
                node = self._new_node(
                    "generation" if attempt == 0 else "refinement",
                    parent=root.node_id,
                    model=model_name,
                    attempt=attempt
                )

                args = base_args if attempt == 0 else self._revise_text_prompt(
                    model_name, base_args, event, prev_scores, prev_suggestions
                )

                out_wav = os.path.join(workdir, f"{node.node_id}_{model_name}.wav")
                wav_path, meta_extras = self._call_model(model_name, args, out_wav, workdir)
                node.output_wav = wav_path
                node.meta["argv"] = args
                node.meta["runner"] = meta_extras.get("runner")
                if "cmd" in meta_extras: node.meta["cmd"] = meta_extras["cmd"]
                if "stdout" in meta_extras: node.meta["stdout"] = meta_extras["stdout"]
                if "stderr" in meta_extras: node.meta["stderr"] = meta_extras["stderr"]
                if "mp4" in meta_extras: node.meta["mp4"] = meta_extras["mp4"]

                scores, suggestions = self.critic.evaluate(event, wav_path, self.llm)
                node.meta["scores"] = scores
                node.meta["suggestions"] = suggestions

                if sum(scores.values()) > sum(best_scores.values()):
                    best_scores = scores
                    best_wav = wav_path

                if _best_threshold_met(scores):
                    nodes_snapshot = {nid: self.nodes[nid].__dict__ for nid in self.nodes}
                    return wav_path or "", scores, nodes_snapshot

                base_args = args
                prev_scores = scores
                prev_suggestions = suggestions

        nodes_snapshot = {nid: self.nodes[nid].__dict__ for nid in self.nodes}
        return best_wav or "", best_scores, nodes_snapshot
