"""Adapter catalog + run settings: choose which adapter to test, explicitly.

Free and paid adapters sit side by side. Nothing falls back silently, and an adapter whose
parameters were never confirmed by a live call runs only when the settings opt in.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from .adapters.fal_http import make_fal_image
from .adapters.hf_space import make_hf_space_image
from .adapters.openrouter import make_openrouter_analyst, make_openrouter_brain, make_openrouter_judge
from .providers import CallLog, InfraError, ModelRef, ProviderPolicyError, call_with_policy, check_model

class SettingsError(ValueError):
    pass


FACTORIES = {
    ("hf_space", "image"): lambda params, **ctx: make_hf_space_image(
        {**params, **({"seed": ctx["image_seed"]} if ctx.get("image_seed") is not None else {})},
        token=ctx.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        client_factory=ctx.get("client_factory")),
    ("fal", "image"): lambda params, **ctx: make_fal_image(params, api_key=ctx.get("fal_key"), transport=ctx.get("transport")),
    ("openrouter", "brain"): lambda params, **ctx: make_openrouter_brain(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
    ("openrouter", "judge"): lambda params, **ctx: make_openrouter_judge(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
    ("openrouter", "analyst"): lambda params, **ctx: make_openrouter_analyst(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
}


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    kind: str          # image / brain / judge / analyst
    provider: str
    model_id: str
    tier: str          # paid / free
    params: dict = field(default_factory=dict)
    verified: bool = False
    price_hint: str = ""
    notes: str = ""

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.provider, self.model_id, self.tier)


@dataclass(frozen=True)
class Settings:
    image_adapter: str = ""
    fallback_adapter: str = ""        # image fallback, must be declared to be used
    allow_free: bool = False
    brain_adapter: str = ""
    judge_adapter: str = ""
    allow_unverified: bool = False    # first live test of a freshly configured adapter
    max_attempts_per_shot: int = 3
    analyst_adapter: str = ""          # Creative Radar: manual ad analysis


def load_catalog(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["name"]: AdapterSpec(e["name"], e["kind"], e["provider"], e["model_id"], e["tier"],
                                   e.get("params", {}), bool(e.get("verified", False)),
                                   e.get("price_hint", ""), e.get("notes", "")) for e in data["adapters"]}


def load_settings(path) -> Settings:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SettingsError(f"cannot read settings JSON: {e}") from e
    if not isinstance(d, dict):
        raise SettingsError("settings.json root must be a JSON object")
    for key in ("allow_free", "allow_unverified"):
        value = d.get(key, False)
        if type(value) is not bool:
            raise SettingsError(f"{key} must be a JSON boolean (true/false), not {value!r}")
    attempts = d.get("max_attempts_per_shot", 3)
    if type(attempts) is not int or attempts < 1:
        raise SettingsError("max_attempts_per_shot must be a positive integer")
    names = ("image_adapter", "fallback_adapter", "brain_adapter", "judge_adapter", "analyst_adapter")
    for key in names:
        if key in d and not isinstance(d[key], str):
            raise SettingsError(f"{key} must be a string")
    return Settings(image_adapter=d.get("image_adapter", ""), fallback_adapter=d.get("fallback_adapter", ""),
                    allow_free=d.get("allow_free", False), brain_adapter=d.get("brain_adapter", ""),
                    judge_adapter=d.get("judge_adapter", ""), allow_unverified=d.get("allow_unverified", False),
                    max_attempts_per_shot=attempts, analyst_adapter=d.get("analyst_adapter", ""))


def describe(catalog: dict) -> str:
    rows = ["name | kind | tier | verified | price hint"]
    for s in catalog.values():
        rows.append(f"{s.name} | {s.kind} | {s.tier} | {'yes' if s.verified else 'NO (confirm on first live run)'} | {s.price_hint}")
    return "\n".join(rows)


def build_adapter(spec: AdapterSpec, allow_free: bool = False, allow_unverified: bool = False, **ctx):
    check_model(spec.ref, allow_free)
    if "CONFIRM_ME" in json.dumps(spec.params):
        raise ProviderPolicyError(f"adapter {spec.name!r} is not configured: replace CONFIRM_ME in adapters.json")
    if not spec.verified and not allow_unverified:
        raise ProviderPolicyError(
            f"adapter {spec.name!r} was never confirmed by a live call: set allow_unverified for the first test only")
    factory = FACTORIES.get((spec.provider, spec.kind))
    if factory is None:
        raise ProviderPolicyError(f"no factory for {spec.provider!r}/{spec.kind!r}")
    return factory({**spec.params, "model_id": spec.model_id}, **ctx)


def _info(spec, primary, image_seed=None):
    info = {"adapter": spec.name, "tier": spec.tier, "model_id": spec.model_id, "provider": spec.provider,
            "verified": spec.verified, "fallback_used": spec.name != primary.name}
    if spec.kind == "image" and spec.provider == "hf_space" and image_seed is not None:
        info["seed"] = image_seed
    return info


def make_image_provider(settings: Settings, catalog: dict, log: CallLog, run_id: str, **ctx):
    """provider(prompt, refs) -> bytes. `provider.info` describes the last call for the receipt."""
    if settings.image_adapter not in catalog:
        raise ProviderPolicyError(f"unknown image_adapter {settings.image_adapter!r} in settings.json")
    if settings.fallback_adapter and settings.fallback_adapter not in catalog:
        raise ProviderPolicyError(f"unknown fallback_adapter {settings.fallback_adapter!r} in settings.json")
    primary = catalog[settings.image_adapter]
    fb = catalog[settings.fallback_adapter] if settings.fallback_adapter else None
    image_seed = ctx.get("image_seed")
    specs = {primary.ref: primary, **({fb.ref: fb} if fb else {})}
    fns = {s.name: build_adapter(s, settings.allow_free, settings.allow_unverified, **ctx) for s in specs.values()}

    def provider(prompt: str, refs) -> bytes:
        def go(ref):
            spec = specs[ref]
            provider.info = _info(spec, primary, image_seed) | {"call_status": "STARTED"}
            try:
                data = fns[spec.name](prompt, refs)
            except Exception as e:
                provider.info = _info(spec, primary, image_seed) | {"call_status": "ERROR",
                    "error_type": type(e).__name__, "error": str(e)[:300]}
                raise
            provider.info = _info(spec, primary, image_seed) | {"call_status": "OK"}
            return data, {}  # image services do not report which model ran: logged as 'unreported'
        return call_with_policy(primary.ref, go, log, run_id, fb.ref if fb else None, settings.allow_free)

    provider.info = {}
    return provider


def _make_llm(adapter_name, kind, settings, catalog, log, run_id, **ctx):
    if not adapter_name:
        raise ProviderPolicyError(f"{kind}_adapter is not configured in settings.json")
    if adapter_name not in catalog:
        raise ProviderPolicyError(f"unknown {kind}_adapter {adapter_name!r} in settings.json")
    spec = catalog[adapter_name]
    if spec.kind != kind:
        raise ProviderPolicyError(f"adapter {adapter_name!r} is a {spec.kind!r} adapter, not {kind!r}")
    fn = build_adapter(spec, settings.allow_free, settings.allow_unverified, **ctx)

    def run(*args):
        def go(ref):
            out, meta = fn(*args)
            run.info = _info(spec, spec) | {"responded_model": meta.get("model")}
            return out, meta
        # QA Judge only: retry one transient infrastructure failure on the same explicit model.
        # Never retry malformed/negative verdicts, policy errors, or non-transient provider errors.
        attempts = 2 if kind == "judge" else 1
        for attempt in range(1, attempts + 1):
            try:
                return call_with_policy(spec.ref, go, log, run_id, allow_free=settings.allow_free)
            except InfraError as e:
                log.write(run_id=run_id, component="qa_judge_resilience", attempt=attempt,
                          max_attempts=attempts, status="retry" if attempt < attempts else "exhausted",
                          error=str(e)[:300])
                if attempt == attempts:
                    raise

    run.info = {}
    return run


def make_brain_call(settings, catalog, log, run_id, **ctx):
    """call(request) -> dict, for brain.request_atoms."""
    return _make_llm(settings.brain_adapter, "brain", settings, catalog, log, run_id, **ctx)


def make_judge(settings, catalog, log, run_id, **ctx):
    """judge(image_path, atoms) -> {atom_id: {pass, evidence}}, for qa.judge_batch_checks."""
    return _make_llm(settings.judge_adapter, "judge", settings, catalog, log, run_id, **ctx)


def make_analyst(settings, catalog, log, run_id, **ctx):
    """analyst(frames, context) -> ({'data', 'raw_excerpt'}), for the Creative Radar manual analysis."""
    return _make_llm(settings.analyst_adapter, "analyst", settings, catalog, log, run_id, **ctx)
