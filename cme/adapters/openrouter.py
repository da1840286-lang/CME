"""OpenRouter adapters: Brain (structured atoms) and a batch vision judge. Model id is explicit config."""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from ..atoms import ATOM_SCHEMA, KINDS
from ..providers import InfraError, ProviderCallError
from .http import check_status, data_uri, urllib_transport

URL = "https://openrouter.ai/api/v1/chat/completions"


def _chat(params, api_key, send, messages, response_format=None):
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ProviderCallError("OPENROUTER_API_KEY is not set")
    body = {"model": params["model_id"], "messages": messages,
            "temperature": params.get("temperature", 0.2), "max_tokens": params.get("max_tokens", 4000)}
    if response_format:
        body["response_format"] = response_format
    status, raw = send("POST", URL, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
    check_status("openrouter", status, raw)
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ProviderCallError(f"non-JSON response: {raw[:200]!r}") from e
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message", err) if isinstance(err, dict) else err
        if code == 429 or (isinstance(code, int) and code >= 500):
            raise InfraError(f"openrouter error {code}: {msg}")
        raise ProviderCallError(f"openrouter error {code}: {msg}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderCallError(f"unexpected response shape: {raw[:200]!r}") from e
    usage = data.get("usage") or {}
    return content or "", {"model": data.get("model"), "cost": usage.get("cost")}


def extract_json(text: str):
    """Models wrap JSON in fences or prose; take the first object. None when there is none."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return None


def _format(params):
    mode = params.get("response_format_mode", "json_object")
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "atoms", "strict": True, "schema": ATOM_SCHEMA}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def make_openrouter_brain(params: dict, api_key=None, transport=None):
    send = transport or urllib_transport

    def call(request: dict):
        system = ("You are the Brain of an ad production system. Reply with ONE JSON object {\"atoms\": [...]} "
                  "and nothing else. Each atom: id, kind, text (max 120 chars), priority (MUST/SHOULD/MAY), "
                  f"channel (image/motion/qa), shot. Allowed kinds: {', '.join(KINDS)}. "
                  + " ".join(request.get("rules", [])) + " JSON schema: " + json.dumps(request["schema"]))
        user = {k: request.get(k) for k in ("offer", "brief", "shots", "only_fix") if request.get(k) is not None}
        content, meta = _chat(params, api_key, send,
                              [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                              _format(params))
        data = extract_json(content)
        if not isinstance(data, dict) or "atoms" not in data:
            return {"atoms": [], "_error": "response was not a JSON object with an 'atoms' list"}, meta
        return data, meta

    return call


def make_openrouter_judge(params: dict, api_key=None, transport=None):
    """judge(image_path, atoms) -> ({atom_id: {'pass': bool, 'evidence': str}}, meta). Evidence only."""
    send = transport or urllib_transport

    def judge(image_path, atoms):
        system = ("You verify one image against numbered requirements. Be strict: if you cannot clearly see "
                  "the requirement satisfied, answer false. Reply ONLY with JSON "
                  "{\"results\": [{\"id\": \"...\", \"pass\": true|false, \"evidence\": \"<12 words\"}]}.")
        reqs = [{"id": a.id, "requirement": a.text} for a in atoms]
        content, meta = _chat(params, api_key, send, [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(reqs)},
                                         {"type": "image_url", "image_url": {"url": data_uri(image_path)}}]}],
            {"type": "json_object"} if params.get("response_format_mode", "json_object") != "prompt" else None)
        data = extract_json(content)
        out = {}
        requested_ids = {a.id for a in atoms}
        rows = data.get("results") if isinstance(data, dict) else None
        if isinstance(rows, list):
            ids = [r.get("id") for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)
                   and r.get("id") in requested_ids]
            counts = Counter(ids)
            for r in rows:
                if not isinstance(r, dict) or not isinstance(r.get("id"), str):
                    continue
                rid, passed, evidence = r["id"], r.get("pass"), r.get("evidence")
                if rid not in requested_ids:
                    continue
                # A duplicated ID is ambiguous: omit every occurrence rather than keeping the last one.
                if counts[rid] != 1 or type(passed) is not bool or not isinstance(evidence, str) or not evidence.strip():
                    continue
                out[rid] = {"pass": passed, "evidence": evidence.strip()[:120]}
        return out, meta

    return judge


ANALYST_SYSTEM = (
    "You analyse ONE short product ad from a few still frames taken at known timestamps. You did NOT see the full video. "
    "Reply ONLY with JSON: {\"hook\":S,\"buyer_problem\":S,\"product_intervention\":S,\"result_demo\":S,"
    "\"evidence\":[{\"t_sec\":number,\"frame\":int,\"observation\":str}],\"cta\":S,\"mechanism_hypothesis\":S,"
    "\"transferability\":S,\"limits\":{\"text\":str,\"missing\":[str]},"
    "\"ratings\":{\"problem_clarity\":1-5|null,\"product_demo\":1-5|null,\"proof_visibility\":1-5|null,\"sequence_quality\":1-5|null}} "
    "where S = {\"text\": str, \"kind\": \"OBSERVED\"|\"INFERRED\"}. OBSERVED only for what is visible in the frames; "
    "interpretations are INFERRED. mechanism_hypothesis and transferability are always INFERRED and are hypotheses, not proof "
    "of profitability. evidence t_sec must be one of the given frame timestamps. If something cannot be known from the frames, "
    "say so in limits.missing and use null ratings. Write all text values in Russian."
)


def make_openrouter_analyst(params: dict, api_key=None, transport=None):
    """analyst(frames, context) -> ({'data': dict|None, 'raw_excerpt': str}, meta). frames: [{'t_sec', 'path'}]."""
    send = transport or urllib_transport

    def analyze(frames, context):
        header = {"ad": context, "frames": [{"index": i, "t_sec": f["t_sec"]} for i, f in enumerate(frames)]}
        content = [{"type": "text", "text": json.dumps(header, ensure_ascii=False)}]
        content += [{"type": "image_url", "image_url": {"url": data_uri(f["path"])}} for f in frames]
        text, meta = _chat(params, api_key, send,
                           [{"role": "system", "content": ANALYST_SYSTEM}, {"role": "user", "content": content}],
                           {"type": "json_object"} if params.get("response_format_mode", "json_object") != "prompt" else None)
        return {"data": extract_json(text), "raw_excerpt": text[:1500]}, meta

    return analyze
