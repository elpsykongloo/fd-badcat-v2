# -*- coding: utf-8 -*-
"""rb/generator.py — RB episode grid generator (docs/rb_design.md v2 §3/§7;
v2.3 = review-driven revision, quotas rebalanced, layers L11/L12 added).

Everything is deterministic: episode rng = sha256(config_hash:episode_id);
sandbox latencies and result ids are keyed by (episode, fn, occurrence), so
gold calls, the gold end-state, per-step latencies, and arm-A lifecycle-
projected event times are all precomputed at build time. Bystander revisions
NEVER enter gold; the L10 benign control (user-voice revision, idx%3==2)
always does.

v2.3 changes (rb_design §15):
  * language is drawn from the episode hash, DECOUPLED from the domain cycle
    (v2.2's lang=idx%2 x domain=idx%4 made ecommerce/housing 100% zh and
    finance/travel 100% en);
  * arm-B revision/cancel content for ARM_B_EVENT_ONLY layers is delivered
    ONLY through reactive events (v2.2 shipped a script piece AND an event
    echo — double delivery);
  * arm-B event text is selected by the rule's content kind (v2.2 always used
    revisions[0] — L6's committed-anchored second event carried the wrong
    revision);
  * `at_after_eou` is now truly EoU-relative (v2.2 baked hold+infer into the
    value and the assembler added HOLD again — every arm-A lifecycle
    projection landed 0.64 s late, 0/57 L8 events inside the tool window);
  * L7 = compensation arena (revision timed past the reference commit
    horizon; gold = forward(new) net — reachable by reverse+relaunch, or by
    patch for longer-window systems; fee/time difference lands in the
    transactional track);
  * L11 = TTS barge-in revision (arm B); L12 = attribution arena (step-2-only
    slot revision while step 1's window is open — the v2.2.1 L7 anti-window
    mechanism, now a designed layer);
  * per-episode seeded audio perturbation params (rate/gain/SNR);
  * utterance text may draw from the FROZEN content bank
    (exp/rb/content_bank.json, hash in config_hash) with seeded disfluencies.

dev/test split: sha256(episode_id) % 10 == 0 -> dev (harness/scorer debugging
only; test is single-shot per system version — design §8)."""
from __future__ import annotations

import hashlib
import json
import random

from .registry import (SCENARIOS, SCENARIOS_BY_KIND, SLOT_POOLS, DOMAINS,
                       TOOLS, canon_value)
from .grammar import (LAYER_GAP, LAYER_KIND, LAYERS, HOLD_S, NOMINAL_INFER_S,
                      ARM_B_RULES, ARM_B_EVENT_ONLY, DELTA_REF_S, L7_MARGIN_S,
                      gap_for_layer, l7_gap, revision_text, bystander_text,
                      progress_text, intent_text, bank_hash)
from .sandbox import Sandbox, oracle_run

# ---- v2.3 FROZEN quotas (sum 600 + 400 = 1000) ------------------------------
ARM_A_QUOTA = {"L1": 48, "L2": 42, "L3": 60, "L4": 72, "L5": 84,
               "L6": 48, "L7": 48, "L8": 60, "L9": 60, "L10": 42, "L12": 36}
ARM_B_QUOTA = {"L4": 50, "L5": 50, "L6": 40, "L7": 40, "L8": 100,
               "L9": 50, "L10": 40, "L11": 30}
LEAD_IN_S = 0.5
VOICES = tuple(f"cv{i:02d}" for i in range(1, 10))     # Qwen3-TTS CustomVoice presets
GEN_VERSION = "rb_v2.3.0"


def config_hash():
    blob = json.dumps({"v": GEN_VERSION, "qa": ARM_A_QUOTA, "qb": ARM_B_QUOTA,
                       "gaps": {k: v for k, v in LAYER_GAP.items()},
                       "kinds": LAYER_KIND, "scenarios": sorted(SCENARIOS),
                       "slots": SLOT_POOLS, "lead": LEAD_IN_S,
                       "voices": VOICES, "bank": bank_hash(),
                       "event_only": ARM_B_EVENT_ONLY,
                       "l7": [DELTA_REF_S, L7_MARGIN_S]},
                      sort_keys=True, ensure_ascii=False, default=list)
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


def _step_lats(eid, scn, profile):
    """Per-step wall latencies, (episode, fn, occurrence)-keyed — computable
    before gold execution (needed for the L7 commit-horizon gap)."""
    sb = Sandbox(eid, profile=profile)
    seen = {}
    lats = []
    for st in scn["steps"]:
        k = seen.get(st["fn"], 0)
        seen[st["fn"]] = k + 1
        lats.append(sb.latency_of_at(st["fn"], k))
    return lats


def _step2_only_slots(scn):
    """Revisable slots that appear ONLY in steps[1:] (the L12 attribution
    construction: the revision names a field of a NOT-YET-LAUNCHED action)."""
    step0 = {v.strip("{}") for v in scn["steps"][0]["args"].values()
             if isinstance(v, str) and v.startswith("{")}
    return [s for s in scn["revisable"] if s not in step0]


def make_episode(arm, layer, idx, cfg_hash, pause_prior=None, content_hook=None):
    eid = f"{arm}_{layer}_{idx:04d}"
    rng = random.Random(hashlib.sha256(f"{cfg_hash}:{eid}".encode()).hexdigest())
    lang = "zh" if rng.random() < 0.5 else "en"     # v2.3: decoupled from domain
    domain = DOMAINS[idx % len(DOMAINS)]
    scn_id = _scenario_for(layer, domain)
    scn = SCENARIOS[scn_id]
    slots = _sample_slots(rng, scn, lang)
    profile = "heavy" if layer == "L9" else "default"
    user_voice = VOICES[rng.randrange(len(VOICES))]
    by_voice = _pick(rng, VOICES, avoid=user_voice)
    lats = _step_lats(eid, scn, profile)

    # ---- revision plan -------------------------------------------------------
    revisions = []          # {slot, old, new, by, kind, gap}
    benign_l10 = (layer == "L10" and idx % 3 == 2)
    l8_action = ("revise", "cancel", "progress")[idx % 3] if layer == "L8" else None
    if layer in ("L1", "L2", "L3", "L4", "L5", "L7", "L11", "L12") \
            or (layer == "L8" and l8_action == "revise") or benign_l10:
        if layer == "L12":
            cand = _step2_only_slots(scn) or scn["revisable"]
        else:
            cand = scn["revisable"]
        slot = cand[rng.randrange(len(cand))]
        new = _pick(rng, SLOT_POOLS[lang][slot], avoid=slots[slot])
        kind = ("value_first" if layer == "L4"
                else "inline" if layer == "L1" else "default")
        # L1 is in-utterance; L8/L11 timing is lifecycle-driven; L7's gap is
        # past the reference commit horizon BY CONSTRUCTION (hold + infer +
        # delta* + forward wall + margin) — under the reference window the
        # only route to gold is reverse + relaunch. The benign-L10 control
        # draws from the L4 bin: rescuable by construction.
        gap = (None if layer in ("L1", "L8", "L11") else
               l7_gap(rng, lats[0]) if layer == "L7" else
               gap_for_layer("L4" if layer == "L10" else layer,
                             rng, pause_prior))
        revisions.append({"slot": slot, "old": slots[slot], "new": new,
                          "by": "user", "kind": kind, "gap": gap})
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
    gold_calls, gold_state, _ = oracle_run(eid, scn["steps"], slots_canon,
                                           profile=profile)
    if cancelled:
        gold_calls, gold_state = [], {}

    # ---- arm-A pieces (fixed timeline; nominal lifecycle projection) ---------
    utt_tpl = intent_text(lang, scn_id, scn["utt"][lang], rng)
    try:
        intent = utt_tpl.format(**slots)
    except (KeyError, IndexError):
        intent = scn["utt"][lang].format(**slots)
    pieces = [{"role": "user", "voice": user_voice, "lang": lang,
               "text": intent + (revision_text(lang, "inline",
                                               revisions[0]["new"],
                                               content_hook, rng)
                                 if layer == "L1" and revisions else ""),
               "gap_before": LEAD_IN_S}]
    arm_b_event_only = (arm == "B" and layer in ARM_B_EVENT_ONLY)
    if not arm_b_event_only:
        for r in revisions:
            if r["gap"] is not None:
                pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                               "text": revision_text(lang, r["kind"], r["new"],
                                                     content_hook, rng),
                               "gap_before": r["gap"]})
    # nominal lifecycle anchors (arm A), all relative to the UTTERANCE END:
    # eou = +HOLD, dec = +HOLD+INFER. Pieces carry at_after_eou = offset from
    # the EOU (the assembler adds exactly one HOLD; v2.2 double-counted it).
    nominal = {"eou": HOLD_S, "dec": HOLD_S + NOMINAL_INFER_S,
               "tool_wall": lats[0] if lats else 0.0}
    if layer == "L8":
        frac = 0.2 + rng.random() * 0.6
        at = round(NOMINAL_INFER_S + frac * nominal["tool_wall"], 3)
        if l8_action == "revise":
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": revision_text(lang, "default",
                                                 revisions[0]["new"],
                                                 content_hook, rng),
                           "at_after_eou": at})
        elif l8_action == "cancel":
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": revision_text(lang, "cancel", "",
                                                 content_hook, rng),
                           "at_after_eou": at})
        else:
            pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                           "text": progress_text(lang, rng),
                           "at_after_eou": at})
    if layer == "L9":
        at = round(NOMINAL_INFER_S + 0.6 * nominal["tool_wall"], 3)
        pieces.append({"role": "user", "voice": user_voice, "lang": lang,
                       "text": progress_text(lang, rng), "at_after_eou": at})
    if bystander is not None:
        frac = bystander["frac"]
        at = round(NOMINAL_INFER_S + frac * nominal["tool_wall"], 3) \
            if bystander["state"] == "inflight" else \
            round(NOMINAL_INFER_S + frac, 3)
        pieces.append({"role": "bystander", "voice": by_voice, "lang": lang,
                       "text": bystander_text(lang, bystander["kind"],
                                              bystander.get("other"),
                                              content_hook, rng),
                       "at_after_eou": at})

    # ---- arm-B event bindings -------------------------------------------------
    events = []
    if arm == "B":
        for rule in ARM_B_RULES.get(layer, []):
            state, (lo, hi), action, ckind = rule
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
                # v2.3: pick the revision MATCHING the rule's content kind
                # (v2.2 always used revisions[0] — L6's second event carried
                # the wrong revision).
                rev = next((r for r in revisions if r["kind"] == ckind),
                           revisions[0])
                text = revision_text(lang,
                                     rev["kind"] if action == "revise"
                                     else "default",
                                     rev["new"], content_hook, rng)
                voice, role = user_voice, "user"
            elif action == "cancel":
                text = revision_text(lang, "cancel", "", content_hook, rng)
                voice, role = user_voice, "user"
            elif action == "progress_query":
                text = progress_text(lang, rng)
                voice, role = user_voice, "user"
            else:
                text = bystander_text(lang, ckind, (bystander or {}).get("other"),
                                      content_hook, rng)
                voice, role = by_voice, "bystander"
            events.append({"state": state, "offset": off, "action": action,
                           "role": role, "voice": voice, "text": text})

    # ---- seeded audio perturbation family (v2.3; applied by the assembler) ---
    perturb = {"rate": round(0.94 + rng.random() * 0.12, 3),
               "gain_db": round(-6.0 + rng.random() * 8.0, 1),
               "snr_db": None if rng.random() < 0.5
               else round(15.0 + rng.random() * 10.0, 1)}

    split = "dev" if int(hashlib.sha256(eid.encode()).hexdigest(), 16) % 10 == 0 \
        else "test"
    return {"id": eid, "arm": arm, "layer": layer, "domain": domain, "lang": lang,
            "scenario": scn_id, "slots": slots, "slots_final": slots_final,
            "revisions": revisions, "bystander": bystander,
            "slots_canon": slots_canon,
            "l8_action": l8_action, "cancelled": cancelled,
            "pieces": pieces, "events": events, "profile": profile,
            "nominal": nominal, "step_latencies": lats,
            "scenario_steps": [st["fn"] for st in scn["steps"]],
            "perturb": perturb,
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
    c_dl = Counter((e["domain"], e["lang"]) for e in eps)
    rev_frac = sum(1 for e in eps if e["revisions"]) / max(1, len(eps))
    m = {"config_hash": ch, "version": GEN_VERSION, "n": len(eps),
         "by_arm_layer": {f"{a}:{l}": n for (a, l), n in sorted(c_layer.items())},
         "split": dict(c_split), "revision_frac": round(rev_frac, 4),
         "domain_lang": {f"{d}:{l}": n for (d, l), n in sorted(c_dl.items())},
         "content_bank": bank_hash(),
         "ids_hash": hashlib.sha256(
             ",".join(e["id"] for e in eps).encode()).hexdigest()[:12],
         "content_hash": hashlib.sha256(json.dumps(
             eps, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]}
    return m
