"""Offer card: the only place where product facts enter the system.

The engine never knows a concrete product. Everything about the offer is data.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

MAX_CLAIM_CHARS = 120


class OfferError(ValueError):
    pass


@dataclass(frozen=True)
class OfferCard:
    offer_id: str
    name: str
    product_type: str
    verified_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    photos: tuple[str, ...]
    platform: str = ""
    geo: str = "US"
    product_category: str = ""
    confirmations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "offer_id": self.offer_id, "name": self.name, "product_type": self.product_type,
            "verified_claims": list(self.verified_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "photos": [Path(x).name for x in self.photos], "platform": self.platform, "geo": self.geo,
            "product_category": self.product_category,
            "confirmations": list(self.confirmations),
        }


def load_offer(path: str | Path, check_files: bool = True) -> OfferCard:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    errors = []
    for key in ("offer_id", "name", "product_type"):
        if not str(data.get(key, "")).strip():
            errors.append(f"missing {key}")
    claims = [str(c).strip() for c in data.get("verified_claims", []) if str(c).strip()]
    if not claims:
        errors.append("verified_claims must not be empty")
    for c in claims:
        if len(c) > MAX_CLAIM_CHARS:
            errors.append(f"claim longer than {MAX_CLAIM_CHARS} chars: {c[:30]}...")
    photos = [str(x) for x in data.get("photos", [])]
    if not photos:
        errors.append("at least one product photo is required")
    resolved = []
    for ph in photos:
        cand = Path(ph) if Path(ph).is_absolute() else (p.parent / ph)
        if check_files and not cand.exists():
            errors.append(f"photo not found: {ph}")
        resolved.append(str(cand.resolve()))
    photos = resolved
    if errors:
        raise OfferError("; ".join(errors))
    return OfferCard(
        offer_id=str(data["offer_id"]).strip(), name=str(data["name"]).strip(),
        product_type=str(data["product_type"]).strip(),
        verified_claims=tuple(claims),
        forbidden_claims=tuple(str(c).strip() for c in data.get("forbidden_claims", []) if str(c).strip()),
        photos=tuple(photos), platform=str(data.get("platform", "")).strip().lower(),
        geo=str(data.get("geo", "US")).strip().upper() or "US",
        product_category=str(data.get("product_category", "")).strip().lower(),
        confirmations=tuple(str(c).strip() for c in data.get("confirmations", []) if str(c).strip()),
    )
