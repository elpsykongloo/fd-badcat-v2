# -*- coding: utf-8 -*-
"""rb/generator.py — RB v2 episode grid generator (docs/rb_design.md v2 §3/§7;
quotas = the v2.1 numeric freeze).

Everything is deterministic: episode rng = sha256(config_hash:episode_id);
sandbox latencies and result ids are seeded by episode id, so gold calls, the
gold end-state, per-step latencies, and arm-A lifecycle-projected event times
are all precomputed at build time. Bystander revisions NEVER enter gold; the
L10 benign control (user-voice revision, idx%3==2) always does.

dev/test split: sha256(episode_id) % 10 == 0 -> dev (harness/scorer debugging
only; test is single-shot per system version — design §8)."""
from __future__ import annotations

import hashlib
import json
import random

from .registry import (SCENARIOS, SCENARIOS_BY_KIND, SLOT_POOLS, DOMAINS,
                       TOOLS, canon_value)
from .grammar import (LAYER_GAP, LAYER_KIND, LAYERS, HOLD_S, NOMINAL_INFER_S,
                      ARM_B_RULES, gap_for_layer, revision_text, bystander_text,
                      PROGRESS_QUERY)
from .sandbox import oracle_run

# ---- v2.1 FROZEN quotas (sum 600 + 400 = 1000) ------------------------------
ARM_A_QUOTA = {"L1": 48, "L2": 42, "L3": 72, "L4": 72, "L5": 96,
               "L6": 48, "L7": 48, "L8": 60, "L9": 72, "L10": 42}
ARM_B_QUOTA = {"L4": 60, "L5": 60, "L6": 40, "L8": 120, "L9": 80, "L10": 40}
LEAD_IN_S = 0.5
VOICES = tuple(f"cv{i:02d}" for i in range(1, 10))     # Qwen3-TTS CustomVoice presets
GEN_VERSION = "rb_v2.2.1"


def config_hash():
    blob = json.dumps({"v": GEN_VERSION, "qa": ARM_A_QUOTA, "qb": ARM_B_QUOTA,
                       "gaps": {k: v for k, v in LAYER_GAP.items()},
                       "kinds": LAYER_KIND, "scenarios": sorted(SCENARIOS),
                       "slots": SLOT_POOLS, "lead": LEAD_IN_S,
                       "voices": VOICES}, sort_keys=True, ensure_ascii=False,
                      default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _pick(rng, pool, avoid=None):
    xs = [x for x in pool if x != avoid]
    return xs[rng.randrange(len(xs))]


def _scenario_for(layer, domain):
    kind = LAYER_KIND[layer]
    for sid in SCENARIOS_BY_KIND[kind]:
        if SCENARIOS[sid]["domain"] == domain:
            return sid
    return SCENARIOS_BY_KIND[kind][0]


def _sample_slots(rng, scn, lang):
    pools = SLOT_POOLS[lang]
    slots = {}
    for st in scn["steps"]:
        for v in st["args"].values():
            if isinstance(v, str) and v.startswith("{"):
                name = v.strip("{}")
                if name not in slots:
                    slots[name] = _pick(rng, pools[name])
    for name in scn["revisable"]:
        if name not in slots:
            slots[name] = _pick(rng, pools[name])
    return slots


def make_episode(arm, layer, idx, cfg_hash, pause_prior=None, content_hook=None):
    eid = f"{arm}_{layer}_{idx:04d}"
    rng = random.Random(hashlib.sha256(f"{cfg_hash}:{eid}".encode()).hexdigest())
    lang = "zh" if idx % 2 == 0 else "en"
    domain = DOMAINS[idx % len(DOMAINS)]
    scn_id = _scenario_for(layer, domain)
    scn = SCENARIOS[scn_id]
    slots = _sample_slots(rng, scn, lang)
    profile = "heavy" if layer == "L9" else "default"
    user_voice = VOICES[rng.randrange(len(VOICES))]
    by_voice = _pick(rng, VOICES, avoid=user_voice)

    # ---- revision plan -------------------------------------------------------
    revisions = []          # {slot, old, new, by, kind, gap|at_frac}
    benign_l10 = (layer == "L10" and idx % 3 == 2)
    l8_action = ("revise", "cancel", "progress")[idx % 3] if layer == "L8" else None
    if layer in ("L2", "L3", "L4", "L5", "L7") or (layer == "L1") \
            or (layer == "L8" and l8_action == "revise") or benign_l10:
        slot = scn["revisable"][rng.randrange(len(scn["revisable"]))]
        new = _pick(rng, SLOT_POOLS[lang][slot], avoid=slots[slot])
        kind = ("value_first" if layer == "L4"
                else "inline" if layer == "L1" else "default")
        # L1 is in-utterance (inline appended to the intent text, no separate
        # piece); L8's timing is lifecycle-projected. The benign-L10 control
        # draws from the L4 bin: it must be RESCUABLE BY CONSTRUCTION, or it
        # cannot separate "killed by SV gating" from "lost to the window".
        revisions.append({"slot": slot, "old": slots[slot], "new": new,
                          "by": "user", "kind": kind,
                          "gap": None if layer in ("L1", "L8") else
                          gap_for_layer("L4" if layer == "L10" else layer,
                                        rng, pause_prior)})
    if layer == "L6":
        s1 = scn["revisable"][0]
        s2 = scn["revisable"][1 % len(scn["revisable"])]
        for which, (slot, kind) in enumerate(((s1, "default"), (s2, "second"))):
            new = _pick(rng, SLOT_POOLS[lang][slot], avoid=slots[slot])
            revisions.append({"slot": slot, "old": slots[slot], "new": new,
                              "by": "user", "kind": kind,
                              "gap": gap_for_layer("L6", rng, pause_prior, which)})
    bystander = None
    if layer == "L10" and not benign_l10:
        slot = scn["revisable"][rng.randrange(len(scn["revisable"]))]
        other = _pick(rng, SLOT_POOLS[lang][slot], avoid=slots[slot])
        b_kind = "command" if idx % 3 == 0 else "irrelevant"
        rule = ARM_B_RULES["L10"][0 if b_kind == "command" else 2]
        frac = rule[1][0] + rng.random() * (rule[1][1] - rule[1][0])
        bystander = {"kind": b_kind, "other": other, "state": rule[0],
                     "frac": round(frac, 3)}

    # ---- gold (user revisions applied; bystander excluded) -------------------
    slots_final = dict(slots)
    for r in revisions:
        if r["by"] == "user":
            slots_final[r["slot"]] = r["new"]
    cancelled = (layer == "L8" and l8_action == "cancel")
    slots_canon = {k: canon_value(k, v) for k, v in slots_final.items()}
    gold_calls, gold_state, lats = oracle_run(eid, scn["steps"], slots_canon,
                                              profile=profile)
    if cancelled:
        gold_calls, gold_state = [], {}

    # ---- arm-A pieces (fixed timeline; nominal lifecycle projection) ---------
    intent = scn["utt"][lang].format(**slots)
    pieces = [{"role": "user", "voice": user_voice, "lang": lang,
               "text": intent + (revision_text(lang, "inline", revisions[0]["new"],
                                               content_hook)
                                 if layer == "L1" and revisions else ""),
               "gap_before": LEAD_IN_S}]
    for r in revisions:
        if r["gap"] is not None:
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": revision_text(lang, r["kind"], r["new"],
                                                 content_hook),
                           "gap_before": r["gap"]})
    # nominal lifecycle anchors (arm A): EoU -> decision -> tool window
    nominal = {"eou": HOLD_S, "dec": HOLD_S + NOMINAL_INFER_S,
               "tool_wall": lats[0] if lats else 0.0}
    if layer == "L8":
        frac = 0.2 + rng.random() * 0.6
        at = round(nominal["dec"] + frac * nominal["tool_wall"], 3)
        if l8_action == "revise":
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": revision_text(lang, "default",
                                                 revisions[0]["new"], content_hook),
                           "at_after_eou": at})
        elif l8_action == "cancel":
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": revision_text(lang, "cancel", "", content_hook),
                           "at_after_eou": at})
        else:
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": PROGRESS_QUERY[lang], "at_after_eou": at})
    if layer == "L9":
        at = round(nominal["dec"] + 0.6 * nominal["tool_wall"], 3)
        pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                       "text": PROGRESS_QUERY[lang], "at_after_eou": at})
    if bystander is not None:
        frac = bystander["frac"]
        at = round(nominal["dec"] + frac * nominal["tool_wall"], 3) \
            if bystander["state"] == "inflight" else round(nominal["dec"] + frac, 3)
        pieces.append({"role": "bystander", "voice": by_voice, "lang": lang,
                       "text": bystander_text(lang, bystander["kind"],
                                              bystander.get("other"), content_hook),
                       "at_after_eou": at})

    # ---- arm-B event bindings -------------------------------------------------
    events = []
    if arm == "B":
        for rule in ARM_B_RULES.get(layer, []):
            state, (lo, hi), action, ckind = rule
            if layer == "L8" and (
                    (l8_action == "revise") != (action == "revise") and
                    (l8_action == "cancel") != (action == "cancel") and
                    (l8_action == "progress") != (action == "progress_query")):
                pass
            off = round(lo + rng.random() * (hi - lo), 3)
            if layer == "L8":
                want = {"revise": "revise", "cancel": "cancel",
                        "progress": "progress_query"}[l8_action]
                if action != want:
                    continue
            if layer == "L10":
                if bystander is None and action == "bystander":
                    continue
                if bystander is not None and action == "benign_control":
                    continue
                if bystander is not None and ckind != bystander["kind"]:
                    continue
            if action in ("revise", "benign_control") and revisions:
                text = revision_text(lang, revisions[0]["kind"] if action == "revise"
                                     else "default", revisions[0]["new"], content_hook)
                voice, role = user_voice, "user"
            elif action == "cancel":
                text = revision_text(lang, "cancel", "", content_hook)
                voice, role = user_voice, "user"
            elif action == "progress_query":
                text = PROGRESS_QUERY[lang]
                voice, role = user_voice, "user"
            else:
                text = bystander_text(lang, ckind, (bystander or {}).get("other"),
                                      content_hook)
                voice, role = by_voice, "bystander"
            events.append({"state": state, "offset": off, "action": action,
                           "role": role, "voice": voice, "text": text})

    split = "dev" if int(hashlib.sha256(eid.encode()).hexdigest(), 16) % 10 == 0 \
        else "test"
    return {"id": eid, "arm": arm, "layer": layer, "domain": domain, "lang": lang,
            "scenario": scn_id, "slots": slots, "slots_final": slots_final,
            "revisions": revisions, "bystander": bystander,
            "slots_canon": slots_canon,
            "l8_action": l8_action, "cancelled": cancelled,
            "pieces": pieces, "events": events, "profile": profile,
            "nominal": nominal, "step_latencies": lats,
            "gold_calls": gold_calls, "gold_state": gold_state,
            "voices": {"user": user_voice, "bystander": by_voice},
            "split": split, "config_hash": cfg_hash}


def build_all(pause_prior=None, content_hook=None, quota_a=None, quota_b=None):
    ch = config_hash()
    eps = []
    for layer in LAYERS:
        for i in range((quota_a or ARM_A_QUOTA).get(layer, 0)):
            eps.append(make_episode("A", layer, i, ch, pause_prior, content_hook))
    for layer in LAYERS:
        for i in range((quota_b or ARM_B_QUOTA).get(layer, 0)):
            eps.append(make_episode("B", layer, i, ch, pause_prior, content_hook))
    return ch, eps


def manifest(ch, eps):
    from collections import Counter
    c_layer = Counter((e["arm"], e["layer"]) for e in eps)
    c_split = Counter(e["split"] for e in eps)
    rev_frac = sum(1 for e in eps if e["revisions"]) / max(1, len(eps))
    m = {"config_hash": ch, "version": GEN_VERSION, "n": len(eps),
         "by_arm_layer": {f"{a}:{l}": n for (a, l), n in sorted(c_layer.items())},
         "split": dict(c_split), "revision_frac": round(rev_frac, 4),
         "ids_hash": hashlib.sha256(
             ",".join(e["id"] for e in eps).encode()).hexdigest()[:12],
         "content_hash": hashlib.sha256(json.dumps(
             eps, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]}
    return m
