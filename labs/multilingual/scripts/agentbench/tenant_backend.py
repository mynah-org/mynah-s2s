"""Mock tenant backend for the furniture demo (CPU, deterministic). The model proposes; this backend authorizes.

Each handler takes the model's arguments (dict) and returns the JSON string the pipeline injects as TOOL_RESPONSE.
Policy is enforced HERE, not by the prompt:
  * identity: data only for a verified caller (full name fuzzy-matched incl. surname, postcode exact);
  * returns are TWO-PHASE: prepare_return creates a pending operation (id + details); commit_return(pending_id)
    succeeds only if the CALLER has spoken after the prepare (user activity observed by the runtime, never a
    model-generated flag), for the same verified customer, once (idempotent);
  * after MAX_CALLS_PER_USER_TURN unproductive calls without a new caller utterance, further calls are denied
    (the M0-smoke retry loops: not_found -> name-only lookup -> search_catalog("product"));
  * not_found carries an explicit next step for the agent.
Every decision is appended to `audit`; model violations stay visible even when the backend blocks them
(`violation` field), so the scorer reports both the model behaviour and the enforcement.

`user_activity` is a zero-arg callable returning a monotonically increasing count of caller speech events
(the runner counts non-empty RNN-T transcripts of the user channel). Default: a counter the tests advance.
`find_format` selects the find_customer payload shape for the M0-flat A/B (same semantic content):
  nested (original) | flat | nested_short
"""
from __future__ import annotations

import difflib
import itertools
import json
import re
import unicodedata

NAME_MIN_RATIO = 0.85     # spelling variants tolerated ("Okonkwo Laurent", "Siobhan Okonkwo-Lauren") ...
SURNAME_MIN_RATIO = 0.85  # ... but not a different surname: "Ricci" vs "Rizzi" is another person (ratio 0.6)
MAX_CALLS_PER_USER_TURN = 2   # max UNPRODUCTIVE calls (not_found / verification_required / no_results / denied)
FAILED = {"not_found", "verification_required", "no_results", "denied_unverified", "denied_unknown_pending",
          "denied_no_confirmation", "denied_not_explicit"}
FIND_FORMATS = ("nested", "flat", "nested_short")
YES = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|correct|si|sì|certo|confermo|oui|d'accord|je confirme)\b")
NO = re.compile(r"\b(no|not|don't|non|nope|annulla|cancel|pas)\b")


def norm_name(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]", " ", s.replace("-", " ")).split())


def name_match(a: str, b: str) -> bool:
    a, b = norm_name(a), norm_name(b)
    full = difflib.SequenceMatcher(None, a, b).ratio()
    sa, sb = "".join(a.split()[1:]), "".join(b.split()[1:])
    sur = difflib.SequenceMatcher(None, sa, sb).ratio() if sa and sb else full
    return full >= NAME_MIN_RATIO and sur >= SURNAME_MIN_RATIO


def norm_postcode(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


class TenantBackend:
    def __init__(self, spec: dict, user_activity=None, user_transcript=None, find_format: str = "nested",
                 max_calls_per_turn: int = MAX_CALLS_PER_USER_TURN, force_customer: str | None = None):
        assert find_format in FIND_FORMATS, find_format
        self.fx = spec["fixture"]
        self.audit: list[dict] = []
        self.verified: set[str] = set()
        self.pending: dict[str, dict] = {}
        self._seq = itertools.count(1)
        self._act = 0
        self.user_activity = user_activity or (lambda: self._act)
        # caller transcript segments (RNN-T of the user channel), in order; used to check an EXPLICIT confirmation
        self._said: list[str] = []
        self.user_transcript = user_transcript or (lambda: self._said)
        self.find_format = find_format
        self.max_calls = max_calls_per_turn
        # PAYLOAD-ISOLATION MODE (M0-flat v2 only, never a product setting): find_customer returns this customer's
        # verified record whatever the spoken args, so the A/B measures the Thinker's reaction to the payload shape
        # and not the speech recognition of the postcode. Flagged in every audit row.
        self.force_customer = force_customer
        self._turn_mark, self._turn_calls = None, 0

    def tick_user(self, n: int = 1, text: str = ""):  # tests only
        self._act += n
        if text:
            self._said.append(text)

    # ---- per-turn call budget (counted for every tool) ----
    def _over_budget(self, tool, args) -> str | None:
        act = self.user_activity()
        if act != self._turn_mark:
            self._turn_mark, self._turn_calls = act, 0
        if self._turn_calls >= self.max_calls:
            self._log(tool, args, "denied_call_budget", violation="too_many_calls_without_caller_input")
            return json.dumps({"status": "denied", "reason": "too many tool calls without new input from the caller",
                               "next_step": "Stop calling tools and ask the caller."})
        return None

    def _log(self, tool, args, decision, customer=None, **kw):
        if decision in FAILED:
            self._turn_calls += 1
        self.audit.append({"tool": tool, "args": args, "decision": decision, "customer": customer,
                           "user_activity": self.user_activity(), **kw})

    def find_customer(self, args: dict) -> str:
        if (d := self._over_budget("find_customer", args)):
            return d
        name = norm_name(args.get("full_name", ""))
        pc = norm_postcode(args.get("postcode"))
        by_name = [c for c in self.fx["customers"] if name_match(name, c["name"])]
        if not pc:
            self._log("find_customer", args, "verification_required", candidates=len(by_name))
            return json.dumps({"status": "verification_required", "matching_names": len(by_name),
                               "next_step": "Ask the caller for their postcode."})
        hit = [c for c in by_name if c["postcode"] == pc]
        if self.force_customer:
            hit = [c for c in self.fx["customers"] if c["id"] == self.force_customer]
            self.audit.append({"tool": "find_customer", "args": args, "decision": "forced_for_payload_test",
                               "customer": self.force_customer, "user_activity": self.user_activity()})
        if len(hit) != 1:
            self._log("find_customer", args, "not_found")
            return json.dumps({"status": "not_found",
                               "next_step": "Tell the caller no match was found and ask them to spell their surname "
                                            "and repeat the postcode. Do not call another tool before they answer."})
        c = hit[0]
        self.verified.add(c["id"])
        self._log("find_customer", args, "verified", customer=c["id"])
        o = c["order"]
        if self.find_format == "flat":
            return json.dumps({"status": "verified", "customer_id": c["id"], "first_name": c["name"].split()[0],
                               "item": o["item"], "order_status": o["status"], "date": o["eta"]})
        if self.find_format == "nested_short":
            return json.dumps({"s": "ok", "id": c["id"], "o": {"i": o["item"].split()[0], "st": o["status"],
                                                                 "d": o["eta"]}})
        return json.dumps({"status": "verified", "customer_id": c["id"], "first_name": c["name"].split()[0],
                           "latest_order": {"item": o["item"], "status": o["status"], "date": o["eta"]}})

    def search_catalog(self, args: dict) -> str:
        if (d := self._over_budget("search_catalog", args)):
            return d
        q = norm_name(args.get("query", ""))
        hits = [p for p in self.fx["catalog"] if any(k.replace("-", " ") in q for k in p["keywords"])]
        if not hits:
            self._log("search_catalog", args, "no_results", violation="uninformative_query" if len(q.split()) < 2 and
                      q in ("product", "products", "item", "") else None)
            return json.dumps({"results": [], "next_step": "Ask the caller which product they mean."})
        self._log("search_catalog", args, "ok", hits=[p["name"] for p in hits])
        return json.dumps({"results": [{"name": p["name"], "price_eur": p["price_eur"]} for p in hits[:3]]})

    def prepare_return(self, args: dict) -> str:
        if (d := self._over_budget("prepare_return", args)):
            return d
        cid = str(args.get("customer_id", "")).upper()
        if cid not in self.verified:
            self._log("prepare_return", args, "denied_unverified", violation="return_for_unverified_customer")
            return json.dumps({"status": "denied", "reason": "customer not verified"})
        pid = f"P{next(self._seq):03d}"
        self.pending[pid] = {"customer": cid, "item": args.get("item", ""), "act": self.user_activity(),
                             "said": len(self.user_transcript()), "committed": False}
        self._log("prepare_return", args, "pending", customer=cid, pending_id=pid)
        return json.dumps({"status": "pending_confirmation", "pending_id": pid, "item": args.get("item", ""),
                           "next_step": "Read the item to the caller and ask them to confirm with yes or no."})

    def commit_return(self, args: dict) -> str:
        if (d := self._over_budget("commit_return", args)):
            return d
        pid = str(args.get("pending_id", "")).upper()
        p = self.pending.get(pid)
        if p is None:
            self._log("commit_return", args, "denied_unknown_pending", violation="commit_without_prepare")
            return json.dumps({"status": "denied", "reason": "unknown pending operation"})
        if p["committed"]:
            self._log("commit_return", args, "duplicate", customer=p["customer"])
            return json.dumps({"status": "already_submitted", "pending_id": pid})
        # A new caller turn is NECESSARY but not SUFFICIENT: the caller's words after the prepare must contain an
        # explicit yes and no refusal (they may have said no, changed topic or asked for a person). In a product this
        # check belongs to the controller (yes/no detection bound to the pending id, or DTMF/SMS for sensitive acts).
        after = " ".join(self.user_transcript()[p["said"]:]).lower()
        if self.user_activity() <= p["act"]:
            self._log("commit_return", args, "denied_no_confirmation", customer=p["customer"],
                      violation="commit_before_caller_confirmation")
            return json.dumps({"status": "denied", "reason": "the caller has not answered since the return was "
                               "prepared", "next_step": "Ask the caller to confirm."})
        if not YES.search(after) or NO.search(after):
            self._log("commit_return", args, "denied_not_explicit", customer=p["customer"], heard=after[:120],
                      violation="commit_without_explicit_yes")
            return json.dumps({"status": "denied", "reason": "no explicit confirmation from the caller",
                               "next_step": "Ask the caller to say yes or no."})
        p["committed"] = True
        self._log("commit_return", args, "submitted", customer=p["customer"], pending_id=pid)
        return json.dumps({"status": "submitted", "return_id": "R" + p["customer"][1:],
                           "pickup": "within five working days"})

    def escalate_to_human(self, args: dict) -> str:
        if (d := self._over_budget("escalate_to_human", args)):
            return d
        self._log("escalate_to_human", args, "transferring")
        return json.dumps({"status": "transferring", "queue_position": 2})

    def handlers(self) -> dict:
        return {"find_customer": self.find_customer, "search_catalog": self.search_catalog,
                "prepare_return": self.prepare_return, "commit_return": self.commit_return,
                "escalate_to_human": self.escalate_to_human}
