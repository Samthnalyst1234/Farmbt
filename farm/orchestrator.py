"""Runs the farm: plan -> parallel research -> analyse -> hunt gaps -> red-team -> report.

Every stage's output is saved before the next one starts, so a run stopped by the budget limit or an
error can be resumed without paying for the finished stages again.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import config, prompts
from .llm import LLM, BudgetExceeded, QuotaExhausted
from .pages import fetch_text, relevant_excerpt
from .store import Store

VERDICT_RANK = {"strong": 0, "plausible": 1, "weak": 2, "already_solved": 3, "unreviewed": 4}
LEVEL_RANK = {"high": 0, "medium": 1, "low": 2}


def norm_url(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (url or "").strip().lower()).rstrip("/")


def ground(findings: list[dict], sources: list[dict]) -> None:
    """Mark how well each finding's source is backed by what the agent actually saw.

    read        the agent opened or cited that page
    seen        the page appeared in its search results
    unverified  the URL never showed up in the agent's tool results (possibly invented)
    """
    index = {norm_url(s["url"]): s for s in sources}
    for f in findings:
        s = index.get(norm_url(f["source_url"]))
        f["grounding"] = ("read" if s and (s.get("opened") or s.get("cited")) else
                          "seen" if s else "unverified")


THIN_PAGE_CHARS = 4000  # pages with less readable text than this can't show a claim is missing

# The most confidence a finding may claim, given how well its source is backed.
CONFIDENCE_CAP = {"verified": "high", "read": "high", "seen": "medium", "unverified": "low", "disputed": "low"}
CONFIDENCE_LEVELS = ["low", "medium", "high"]


def cap_confidence(f: dict) -> None:
    """Models often rate claims 'high' from a search snippet alone; cap the rating at what the source supports."""
    f.setdefault("model_confidence", f["confidence"])
    cap = CONFIDENCE_CAP.get(f.get("grounding"), "low")
    f["confidence"] = min(f["model_confidence"], cap, key=CONFIDENCE_LEVELS.index)


def dump(data) -> str:
    if config.COMPACT:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(data, indent=1, ensure_ascii=False)


class Farm:
    def __init__(self, store: Store, llm: LLM, *, questions: int, workers: int, solve: int = 1, log=print):
        self.store = store
        self.llm = llm
        self.questions = questions
        self.workers = workers
        self.solve_count = solve
        self.log = log

    # --- run control -----------------------------------------------------------------------------

    def run(self, run_id: int) -> str | None:
        """Run (or resume) a run to completion. Returns the report path, or None if it paused."""
        return self._guarded(run_id, self._pipeline)

    def solve(self, run_id: int, gap_id: str | None = None) -> str | None:
        """Run the solver stage on one gap of a finished run (default: the best-ranked open gap)."""
        return self._guarded(run_id, lambda rid, project: self._solve_existing(rid, project, gap_id))

    def _pipeline(self, run_id: int, project) -> str:
        prior = self._prior(project["id"], run_id)
        plan = self._stage(run_id, "plan", lambda: self._plan(project, prior))
        research = self._research(run_id, plan)
        findings = self._stage(run_id, "findings", lambda: self._findings(run_id, plan, research))
        findings = self._stage(run_id, "verify", lambda: self._verify(run_id, findings))
        landscape = self._stage(run_id, "landscape", lambda: self._analyse(project, findings))
        gaps = self._stage(run_id, "gaps", lambda: self._hunt(run_id, project, landscape, findings, prior))
        ranked = self._stage(run_id, "critique", lambda: self._critique(run_id, project, gaps))
        for gap in self._open_gaps(ranked)[: self.solve_count]:
            self._solve(run_id, project, gap, landscape, findings)
        report = self._stage(run_id, "report",
                             lambda: self._report(project, plan, landscape, ranked, findings, prior))
        return self._write_report(run_id, project, report["markdown"], findings)

    def _solve_existing(self, run_id: int, project, gap_id: str | None) -> str:
        ranked = self.store.stage(run_id, "critique")
        report = self.store.stage(run_id, "report")
        if not ranked or not report:
            raise RuntimeError(f"run {run_id} hasn't finished its research yet; resume it first")
        findings = self.store.stage(run_id, "verify") or self.store.stage(run_id, "findings")
        landscape = self.store.stage(run_id, "landscape")
        if gap_id:
            gap = next((g for g in ranked["gaps"] if g["id"].lower() == gap_id.lower()), None)
            if not gap:
                raise RuntimeError(f"run {run_id} has no gap {gap_id}")
        else:
            open_gaps = self._open_gaps(ranked)
            if not open_gaps:
                raise RuntimeError(f"run {run_id} has no open gaps to solve")
            gap = open_gaps[0]
        self._solve(run_id, project, gap, landscape, findings)
        return self._write_report(run_id, project, report["markdown"], findings)

    def _guarded(self, run_id: int, body) -> str | None:
        run = self.store.run(run_id)
        project = self.store.project_by_id(run["project_id"])
        self._base_cost = run["cost_usd"]
        self.store.set_status(run_id, "running")
        self.log(f"run {run_id} for '{project['slug']}'")
        try:
            path = body(run_id, project)
            self.store.set_report(run_id, path)
            self.store.set_status(run_id, "done")
            self.log(f"done. report: {path}")
            return path
        except BudgetExceeded as exc:
            self.store.set_status(run_id, "paused", str(exc))
            hint = "" if isinstance(exc, QuotaExhausted) else " --max-cost <usd>"
            self.log(f"paused: {exc}. Continue later with: python -m farm resume {run_id}{hint}")
            return None
        except Exception as exc:
            self.store.set_status(run_id, "failed", repr(exc))
            self.log(f"failed: {exc!r}. Retry with: python -m farm resume {run_id}")
            raise
        finally:
            self.store.set_cost(run_id, self._base_cost + self.llm.meter.cost)
            m = self.llm.meter
            self.log(f"this session: {m.calls} calls, {m.searches} web searches, ~${m.cost:.2f}")

    def _stage(self, run_id: int, name: str, fn):
        cached = self.store.stage(run_id, name)
        if cached is not None:
            self.log(f"{name}: reusing saved output")
            return cached
        self.log(f"{name}: working")
        data = fn()
        self.store.save_stage(run_id, name, data)
        self.store.set_cost(run_id, self._base_cost + self.llm.meter.cost)
        return data

    def _prior(self, project_id: int, run_id: int) -> dict | None:
        """What the last finished run of this project concluded, for monitoring runs."""
        last = self.store.latest_run(project_id)
        if not last or last["id"] >= run_id:
            return None
        plan = self.store.stage(last["id"], "plan") or {}
        gaps = self.store.stage(last["id"], "critique") or {"gaps": []}
        return {
            "date": last["finished_at"],
            "questions": [q["question"] for q in plan.get("research_questions", [])],
            "gaps": [{"id": g["id"], "title": g["title"], "description": g["description"],
                      "verdict": g["review"]["verdict"]} for g in gaps["gaps"]],
        }

    # --- stages ----------------------------------------------------------------------------------

    def _plan(self, project, prior) -> dict:
        prompt = f"Mission:\n{project['mission']}\n\nWrite exactly {self.questions} research questions."
        if prior:
            prompt += f"\n\nPrevious run ({prior['date']}) asked:\n{dump(prior['questions'])}"
            prompt += f"\n\nGaps it found:\n{dump(prior['gaps'])}"
        plan = self.llm.ask("planner", prompts.planner(), prompt, schema=prompts.PLAN).data
        plan["research_questions"] = plan["research_questions"][: self.questions]
        for i, q in enumerate(plan["research_questions"], 1):
            q["id"] = f"Q{i}"
            self.log(f"  {q['id']} [{q['angle']}] {q['question']}")
        return plan

    def _research(self, run_id: int, plan: dict) -> dict:
        results, todo = {}, []
        for q in plan["research_questions"]:
            saved = self.store.stage(run_id, f"research:{q['id']}")
            if saved is not None:
                results[q["id"]] = saved
            else:
                todo.append(q)
        if not todo:
            return results
        self.log(f"research: {len(todo)} researchers working, {self.workers} at a time")
        budget_error = None
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self._research_one, run_id, q): q for q in todo}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    results[q["id"]] = fut.result()
                    self.log(f"  {q['id']} done: {len(results[q['id']]['findings'])} findings")
                except BudgetExceeded as exc:
                    budget_error = exc
                except Exception as exc:  # one failed researcher shouldn't sink the run
                    self.log(f"  {q['id']} failed: {exc!r}")
        if budget_error:
            raise budget_error
        if not results:
            raise RuntimeError("every researcher failed")
        return results

    def _research_one(self, run_id: int, q: dict) -> dict:
        notes = self.llm.ask("researcher", prompts.researcher(),
                             f"Research question {q['id']}: {q['question']}\n\nWhy it matters: {q['why']}",
                             web=config.RESEARCHER_WEB)
        extracted = self.llm.ask(
            "extractor", prompts.extractor(),
            f"Research notes:\n{notes.text}\n\nSources the researcher saw:\n{dump(notes.sources)}\n\n"
            "Extract every distinct finding.",
            schema=prompts.FINDINGS).data
        result = {"question": q, "notes": notes.text, "sources": notes.sources,
                  "findings": extracted["findings"]}
        self.store.save_stage(run_id, f"research:{q['id']}", result)
        return result

    def _findings(self, run_id: int, plan: dict, research: dict) -> list[dict]:
        findings = []
        for q in plan["research_questions"]:
            if q["id"] not in research:
                self.log(f"  {q['id']} has no research; continuing without it")
                continue
            ground(research[q["id"]]["findings"], research[q["id"]]["sources"])
            # Keep the best-grounded findings when there are more than the profile allows.
            ranked_findings = sorted(research[q["id"]]["findings"],
                                     key=lambda f: ("read", "seen", "unverified").index(f["grounding"]))
            for f in ranked_findings[: config.FINDINGS_PER_QUESTION]:
                finding = {**f, "evidence": f["evidence"][: config.EVIDENCE_CHARS],
                           "id": f"F{len(findings) + 1}", "question_id": q["id"]}
                cap_confidence(finding)
                findings.append(finding)
        self.store.save_findings(run_id, findings)
        unverified = sum(f["grounding"] == "unverified" for f in findings)
        capped = sum(f["confidence"] != f["model_confidence"] for f in findings)
        self.log(f"  {len(findings)} findings in the knowledge base ({unverified} with unverified sources, "
                 f"{capped} confidence ratings lowered to match their sources)")
        return findings

    # --- page checking: re-read the source and confirm it says what the finding claims -------------

    def _verify(self, run_id: int, findings: list[dict]) -> list[dict]:
        if any("grounding" not in f for f in findings):  # saved before source checking existed
            for research in self.store.stages_like(run_id, "research:%"):
                ground([f for f in findings if f["question_id"] == research["question"]["id"]], research["sources"])
            for f in findings:
                f.setdefault("grounding", "unverified")
                cap_confidence(f)
        if config.VERIFY_PAGES <= 0:
            return findings
        by_url: dict[str, list[dict]] = {}
        for f in findings:
            if f["source_url"].startswith("http"):
                by_url.setdefault(f["source_url"], []).append(f)
        # Check the riskiest sources first: only seen in search results, then never seen, then already read;
        # within each, the pages behind the most high-confidence claims.
        risk = {"seen": 0, "unverified": 1, "read": 2}
        pages = sorted(by_url.items(), key=lambda kv: (min(risk.get(f["grounding"], 3) for f in kv[1]),
                                                       -sum(f["model_confidence"] == "high" for f in kv[1])))
        pages = pages[: config.VERIFY_PAGES]
        self.log(f"  re-reading {len(pages)} of {len(by_url)} source pages to check their claims")

        def check(item):
            url, page_findings = item
            try:
                return url, page_findings, *self._check_page(url, page_findings)
            except BudgetExceeded:
                raise
            except Exception as exc:  # unreachable page, blocked, PDF, bad JSON...
                return url, page_findings, None, f"{type(exc).__name__}: {exc}"[:200]

        with ThreadPoolExecutor(self.workers) as pool:
            results = list(pool.map(check, pages))

        for url, page_findings, checks, detail in results:
            if checks is None:
                for f in page_findings:
                    f["verification"] = {"status": "unreachable", "quote": "", "detail": detail}
                self.log(f"  could not read {url}: {detail}")
                continue
            # "Not found" only counts against a claim when the checker saw the whole of a substantial page;
            # on an excerpt or a thin page (paywall, abstract only, JavaScript app) it proves nothing.
            complete = detail == "complete"
            by_id = {c["finding_id"].strip(): c for c in checks}
            for f in page_findings:
                c = by_id.get(f["id"])
                if not c:
                    continue
                f["verification"] = {"status": c["status"], "quote": c["quote"][:300], "page_complete": complete}
                if detail == "thin":
                    f["verification"]["detail"] = "the page shows little text (abstract, paywall or app), so this is inconclusive"
                if c["status"] == "supported":
                    f["grounding"] = "verified"
                elif c["status"] == "partial" and f["grounding"] != "read":
                    f["grounding"] = "read"
                elif c["status"] == "contradicted" or (c["status"] == "not_found" and complete):
                    f["grounding"] = "disputed"
        for f in findings:
            cap_confidence(f)
        counts = {k: sum(f["grounding"] == k for f in findings) for k in ("verified", "disputed")}
        self.log(f"  page check: {counts['verified']} findings confirmed by their source, "
                 f"{counts['disputed']} not supported by it")
        self.store.save_findings(run_id, findings)
        return findings

    def _check_page(self, url: str, page_findings: list[dict]) -> tuple[list[dict], str]:
        claims = page_findings[:8]
        text = "Mock page text." if self.llm.mock else fetch_text(url)
        if len(text) < 200 and not self.llm.mock:
            raise ValueError("page has almost no readable text (it may need JavaScript)")
        excerpt = relevant_excerpt(text, [f["claim"] for f in claims], config.PAGE_CHARS)
        prompt = (f"Page: {url}\n\nPage text:\n{excerpt}\n\nClaims to check:\n"
                  + "\n".join(f"{f['id']}: {f['claim']}" for f in claims))
        checks = self.llm.ask("verifier", prompts.verifier(), prompt, schema=prompts.VERIFY).data["checks"]
        # Second opinion before condemning: small models' single verdicts are noisy, so a claim is only
        # judged unsupported if a fresh check agrees; any support found the second time wins.
        doubtful = {c["finding_id"].strip() for c in checks if c["status"] in ("not_found", "contradicted")}
        if doubtful:
            recheck = [f for f in claims if f["id"] in doubtful]
            retry = self.llm.ask("verifier", prompts.verifier(),
                                 f"Page: {url}\n\nPage text:\n{excerpt}\n\nClaims to check:\n"
                                 + "\n".join(f"{f['id']}: {f['claim']}" for f in recheck),
                                 schema=prompts.VERIFY).data["checks"]
            second = {c["finding_id"].strip(): c for c in retry}
            for c in checks:
                again = second.get(c["finding_id"].strip())
                if again and again["status"] in ("supported", "partial"):
                    c.update(status=again["status"], quote=again["quote"])
                elif again and again["status"] != c["status"]:
                    c["status"] = "not_found"  # the checks disagree on how it fails: weakest verdict
        if len(text) < THIN_PAGE_CHARS and not self.llm.mock:
            return checks, "thin"
        return checks, "complete" if len(text) <= config.PAGE_CHARS else "excerpt"

    def _analyse(self, project, findings) -> dict:
        return self.llm.ask("analyst", prompts.analyst(),
                            f"Mission:\n{project['mission']}\n\nFindings:\n{self._findings_text(findings)}",
                            schema=prompts.LANDSCAPE).data

    def _hunt(self, run_id: int, project, landscape, findings, prior) -> dict:
        prompt = (f"Mission:\n{project['mission']}\n\nLandscape:\n{dump(landscape)}\n\n"
                  f"Findings:\n{self._findings_text(findings, brief=config.COMPACT)}")
        if prior:
            prompt += f"\n\nGaps from the previous run ({prior['date']}):\n{dump(prior['gaps'])}"
        else:
            prompt += "\n\nThis is the first run, so every gap's status is 'new'."
        passes = max(1, config.GAP_VOTES)
        results = []
        for i in range(1, passes + 1):
            data = self._stage(run_id, f"gaps:pass{i}", lambda: self.llm.ask(
                "gap_hunter", prompts.gap_hunter(), prompt, schema=prompts.GAPS).data)
            results.append(data["gaps"])
        gaps = results[0] if passes == 1 else self._vote(results)
        for i, g in enumerate(gaps, 1):
            g["id"] = f"G{i}"
            g.setdefault("votes", 1)
            g.setdefault("passes", 1)
            self.log(f"  {g['id']} [{g['impact']} impact, found in {g['votes']}/{g['passes']} passes] {g['title']}")
        return {"gaps": gaps}

    def _vote(self, results: list[list[dict]]) -> list[dict]:
        """Majority vote across independent gap-hunting passes: keep the gaps most passes agree on."""
        passes = len(results)
        candidates, listing = {}, []
        for p, gaps in enumerate(results, 1):
            for j, g in enumerate(gaps, 1):
                cid = f"P{p}.{j}"
                candidates[cid] = (p, g)
                listing.append({"id": cid, "title": g["title"], "description": g["description"][:300]})
        groups = self.llm.ask("gap_merger", prompts.gap_merger(), f"Gaps:\n{dump(listing)}",
                              schema=prompts.MERGE).data["groups"]
        grouped, merged = set(), []
        for group in groups + [{"members": [cid], "representative": cid} for cid in candidates]:
            members = [m.strip() for m in group["members"] if m.strip() in candidates and m.strip() not in grouped]
            if not members:
                continue  # every leftover gap the merger forgot ends up as its own group here
            grouped.update(members)
            rep = group["representative"].strip()
            gap = dict(candidates[rep if rep in members else members[0]][1])
            ideas, evidence = [], []
            for m in members:
                ideas += [i for i in candidates[m][1]["attack_ideas"] if i not in ideas]
                evidence += [e for e in candidates[m][1]["evidence"] if e not in evidence]
            gap.update(attack_ideas=ideas[:6], evidence=evidence,
                       votes=len({candidates[m][0] for m in members}), passes=passes)
            merged.append(gap)
        merged.sort(key=lambda g: (-g["votes"], LEVEL_RANK.get(g["impact"], 1)))
        need = passes // 2 + 1
        kept = [g for g in merged if g["votes"] >= need]
        if len(kept) < 2:  # little agreement: keep the best-supported few, their vote counts show how weak
            kept = merged[:3]
        self.log(f"  majority vote: {len(kept)} of {len(merged)} distinct gaps kept "
                 f"(needed {need} of {passes} passes)")
        return kept[:8]

    def _critique(self, run_id: int, project, gaps: dict) -> dict:
        items = [{k: g[k] for k in ("id", "title", "description", "why_unsolved", "opportunity")}
                 for g in gaps["gaps"]]
        notes = self.llm.ask("critic", prompts.critic(),
                             f"Mission:\n{project['mission']}\n\nProposed gaps:\n{dump(items)}",
                             web=config.CRITIC_WEB)
        reviews = self.llm.ask(
            "extractor", prompts.extractor(),
            f"Critic's notes:\n{notes.text}\n\nSources the critic saw:\n{dump(notes.sources)}\n\n"
            f"Gap IDs under review: {', '.join(g['id'] for g in gaps['gaps'])}. Write one review per gap.",
            schema=prompts.CRITIQUE).data
        # A counter-argument only counts if its URLs came from pages the critic actually saw.
        seen = {norm_url(s["url"]) for s in notes.sources}
        for r in reviews["reviews"]:
            urls = r["sources"] + re.findall(r"https?://[^\s)\]]+", r["counter_evidence"])
            r["grounded"] = any(norm_url(u) in seen for u in urls)
        by_id = {r["gap_id"].strip(): r for r in reviews["reviews"]}
        missing = {"verdict": "unreviewed", "confidence": 0.0, "counter_evidence": "",
                   "what_to_verify": "", "sources": []}
        ranked = [{**g, "review": by_id.get(g["id"], missing)} for g in gaps["gaps"]]
        ranked.sort(key=lambda g: (VERDICT_RANK.get(g["review"]["verdict"], 4), -g.get("votes", 1),
                                   LEVEL_RANK.get(g["impact"], 1), -LEVEL_RANK.get(g["difficulty"], 1)))
        for g in ranked:
            self.log(f"  {g['id']} -> {g['review']['verdict']}")
        self.store.save_gaps(run_id, ranked)
        return {"gaps": ranked, "critic_notes": notes.text}

    # --- solver: from a gap to a tested plan --------------------------------------------------------

    @staticmethod
    def _open_gaps(ranked: dict) -> list[dict]:
        return [g for g in ranked["gaps"] if g["review"]["verdict"] not in ("already_solved", "weak")]

    def _solve(self, run_id: int, project, gap: dict, landscape: dict, findings: list[dict]) -> dict:
        """Design solutions for one gap, pick the best, and plan an experiment to test it."""
        gid = gap["id"]
        self.log(f"solver: working on {gid} {gap['title']}")
        context = self._gap_context(project, gap, landscape, findings)
        designs = self._stage(run_id, f"solve:{gid}:designs", lambda: self.llm.ask(
            "solution_designer", prompts.solution_designer(), f"{context}\n\nPropose 3 to 5 solutions.",
            schema=prompts.SOLUTIONS).data)
        for s in designs["solutions"]:
            self.log(f"  idea: {s['name']}")
        prior = self._stage(run_id, f"solve:{gid}:prior_art",
                            lambda: self._prior_art(project, gap, designs["solutions"]))
        prior_text = self._prior_art_text(prior)
        review = self._stage(run_id, f"solve:{gid}:review", lambda: self.llm.ask(
            "solution_critic", prompts.solution_critic(),
            f"{context}\n\nProposed solutions:\n{dump(designs['solutions'])}\n\nPrior-art report:\n{prior_text}",
            schema=prompts.SOLUTION_REVIEW).data)
        best = self._pick(designs["solutions"], review)
        self.log(f"  chosen: {best['name']}")
        plan = self._stage(run_id, f"solve:{gid}:plan", lambda: self.llm.ask(
            "experiment_planner", prompts.experiment_planner(),
            f"{context}\n\nChosen solution:\n{dump(best)}\n\nWhy it was chosen: {review['rationale']}\n\n"
            f"What already exists (the baseline should include the closest existing product or method):\n"
            f"{prior_text}",
            schema=prompts.EXPERIMENT).data)
        result = {"gap": {k: gap[k] for k in ("id", "title", "description")}, "solutions": designs["solutions"],
                  "prior_art": prior["checks"], "review": review, "best": best["name"], "plan": plan}
        self.store.save_stage(run_id, f"solution:{gid}", result)
        return result

    def _prior_art(self, project, gap: dict, solutions: list[dict]) -> dict:
        """Search for existing products, projects and papers that already implement each solution."""
        items = [{k: s[k] for k in ("name", "core_idea", "how_it_works")} for s in solutions]
        notes = self.llm.ask("prior_art", prompts.prior_art(),
                             f"Mission:\n{project['mission']}\n\nGap: {gap['title']}\n{gap['description']}\n\n"
                             f"Proposed solutions:\n{dump(items)}", web=config.PRIOR_ART_WEB)
        data = self.llm.ask(
            "extractor", prompts.extractor(),
            f"Prior-art notes:\n{notes.text}\n\nSources the scout saw:\n{dump(notes.sources)}\n\n"
            f"Solutions checked: {dump([s['name'] for s in solutions])}. Write one check per solution, using "
            "these names exactly.", schema=prompts.PRIOR_ART).data
        # Same rule as elsewhere: a match only counts if the scout actually came across its page.
        seen = {norm_url(s["url"]) for s in notes.sources}
        for check in data["checks"]:
            for match in check["closest"]:
                match["grounded"] = norm_url(match["url"]) in seen
            if check["status"] != "not_found" and not any(m["grounded"] for m in check["closest"]):
                check["status_note"] = "the scout's matches weren't among pages it actually saw"
            self.log(f"  prior art: {check['name']} -> {check['status'].replace('_', ' ')}")
        return data

    @staticmethod
    def _prior_art_text(prior: dict) -> str:
        lines = []
        for c in prior["checks"]:
            matches = "; ".join(f"{m['title']} ({m['kind']}, {m['url']}): {m['how_close']}"
                                + ("" if m.get("grounded", True) else " [UNVERIFIED]") for m in c["closest"])
            lines.append(f"- {c['name']}: {c['status'].replace('_', ' ')}. Closest: {matches or 'none found'}. "
                         f"Still new: {c['still_new'] or 'nothing stated'}")
        return "\n".join(lines)

    @staticmethod
    def _same_name(a: str, b: str) -> bool:
        """Models don't always repeat a name exactly (hyphen characters, dropped suffixes)."""
        a, b = (" ".join(re.findall(r"[a-z0-9]+", x.lower())) for x in (a, b))
        return bool(a and b) and (a == b or a in b or b in a)

    @classmethod
    def _pick(cls, solutions: list[dict], review: dict) -> dict:
        for s in solutions:
            if cls._same_name(s["name"], review["best"]):
                return s
        scored = sorted(review["reviews"], key=lambda r: r["feasibility"] + r["impact"], reverse=True)
        for r in scored:
            for s in solutions:
                if cls._same_name(s["name"], r["name"]):
                    return s
        return solutions[0]

    def _gap_context(self, project, gap: dict, landscape: dict, findings: list[dict]) -> str:
        r = gap["review"]
        cited = set(re.findall(r"F\d+", " ".join(gap["evidence"])))
        approaches = "\n".join(f"- {a['name']}: {a['how_it_works']} Weaknesses: {'; '.join(a['weaknesses'])}"
                               for a in landscape["approaches"])
        return (f"Mission:\n{project['mission']}\n\n"
                f"Gap {gap['id']}: {gap['title']}\n{gap['description']}\n"
                f"Why it is unsolved: {gap['why_unsolved']}\nOpportunity: {gap['opportunity']}\n"
                f"Ideas so far: {'; '.join(gap['attack_ideas'])}\n\n"
                f"Critic's review ({r['verdict']}, confidence {r['confidence']}): {r['counter_evidence']}\n"
                f"Verify first: {r['what_to_verify']}\n\n"
                f"Existing approaches:\n{approaches}\n\n"
                f"Supporting findings:\n{self._findings_text([f for f in findings if f['id'] in cited], brief=True)}")

    def _report(self, project, plan, landscape, ranked, findings, prior) -> dict:
        if config.COMPACT:
            refs = [{"id": f["id"], "claim": f["claim"][:120], "url": f["source_url"]} for f in findings]
        else:
            refs = [{"id": f["id"], "claim": f["claim"], "url": f["source_url"], "title": f["source_title"]}
                    for f in findings]
        prompt = (f"Mission:\n{project['mission']}\n\nRefined problem:\n{plan['refined_problem']}\n\n"
                  f"Landscape:\n{dump(landscape)}\n\nGaps, ranked, with critic reviews:\n{dump(ranked['gaps'])}\n\n"
                  f"Findings you can cite:\n{dump(refs)}\n\n"
                  f"Signals worth monitoring:\n{dump(plan['monitoring_signals'])}")
        if prior:
            prompt += f"\n\nPrevious run finished {prior['date']} with these gaps:\n{dump(prior['gaps'])}"
        return {"markdown": self.llm.ask("reporter", prompts.reporter(), prompt).text}

    # --- output ----------------------------------------------------------------------------------

    @staticmethod
    def _findings_text(findings: list[dict], brief: bool = False) -> str:
        def flag(f):
            return {"unverified": ", UNVERIFIED SOURCE", "disputed": ", SOURCE DOES NOT SUPPORT THIS",
                    "verified": ", confirmed by source"}.get(f.get("grounding"), "")

        if brief:
            return "\n".join(f"{f['id']} ({f['kind']}, {f['confidence']}{flag(f)}): {f['claim']}" for f in findings)
        return "\n".join(
            f"{f['id']} ({f['question_id']}, {f['kind']}, {f['confidence']} confidence{flag(f)}): {f['claim']}"
            f" | evidence: {f['evidence']} | source: {f['source_title']} {f['source_url']} {f['published']}"
            for f in findings)

    @classmethod
    def _solution_md(cls, solved: dict) -> str:
        cell = lambda text: str(text).replace("|", "/").replace("\n", " ")  # noqa: E731
        gap, review, plan = solved["gap"], solved["review"], solved["plan"]
        lines = [f"## Solving {gap['id']}: {gap['title']}", "", gap["description"], "",
                 f"**Chosen approach: {solved['best']}.** {review['rationale']}", "",
                 "### Candidate solutions", "",
                 "| Solution | Already exists? | Feasibility /10 | Impact /10 | Verdict | Main weakness |",
                 "|---|---|---|---|---|---|"]
        prior = solved.get("prior_art") or []
        for s in solved["solutions"]:
            r = next((r for r in review["reviews"] if cls._same_name(s["name"], r["name"])), {})
            p = next((p for p in prior if cls._same_name(s["name"], p["name"])), None)
            exists = p["status"].replace("_", " ") if p else "not checked"
            lines.append(f"| {cell(s['name'])} | {exists} | {r.get('feasibility', '-')} | {r.get('impact', '-')} | "
                         f"{r.get('verdict', '-')} | {cell(r.get('main_weakness', ''))} |")
        if prior:
            lines += ["", "### What already exists", ""]
            for p in prior:
                lines.append(f"**{p['name']}**: {p['status'].replace('_', ' ')}."
                             + (f" Still new: {p['still_new']}" if p["still_new"] else ""))
                lines += [f"- [{m['title']}]({m['url']}) ({m['kind'].replace('_', ' ')}): {m['how_close']}"
                          + ("" if m.get("grounded", True) else " _(unverified link)_") for m in p["closest"]]
                lines.append("")
        for s in solved["solutions"]:
            lines += ["", f"**{s['name']}** ({s['effort']} effort, {s['novelty']} novelty). {s['core_idea']}", "",
                      f"- How it works: {s['how_it_works']}", f"- Builds on: {s['builds_on']}",
                      f"- Why it could work: {s['why_it_could_work']}", f"- Risks: {'; '.join(s['risks'])}"]
        lines += ["", f"### Experiment plan: {solved['best']}", "",
                  f"- **Hypothesis:** {plan['hypothesis']}", f"- **Baseline to beat:** {plan['baseline']}",
                  f"- **Success metric:** {plan['success_metric']}",
                  f"- **Smallest useful test:** {plan['minimum_viable_test']}",
                  f"- **Compute:** {plan['compute']}", f"- **Time:** {plan['estimated_time']}",
                  f"- **Cost:** {plan['estimated_cost']}", "", "**Data needed**", ""]
        lines += [f"- {d}" for d in plan["data_needed"]]
        lines += ["", "**Steps**", ""]
        for i, st in enumerate(plan["steps"], 1):
            title = "" if st["step"].strip().rstrip(".").isdigit() else f"**{st['step']}**: "
            lines.append(f"{i}. {title}{st['detail']} _(tools: {st['tools']})_")
        lines += ["", "**Stop if**", ""] + [f"- {k}" for k in plan["kill_criteria"]]
        lines += ["", f"**If it works:** {plan['if_it_works']}"]
        return "\n".join(lines)

    def _write_report(self, run_id: int, project, markdown: str, findings: list[dict]) -> str:
        folder = os.path.join(config.REPORTS_DIR, project["slug"])
        os.makedirs(folder, exist_ok=True)
        sources = {}
        for f in findings:
            if f["source_url"]:
                sources.setdefault(f["source_url"], f["source_title"] or f["source_url"])
        # Open models sometimes cite as 【F1†https://...】; turn those into ordinary Markdown links.
        markdown = re.sub(r"【(F\d+)†(https?://[^】\s]+)】", r" [\1](\2)", markdown)
        markdown = re.sub(r"【(F\d+)[^】]*】", r" (\1)", markdown)
        markdown = re.sub(r"【([^】]*)】", r" (\1)", markdown)
        for solved in self.store.stages_like(run_id, "solution:%"):
            markdown += "\n\n---\n\n" + self._solution_md(solved)
        grades = [(k, sum(f.get("grounding") == k for f in findings))
                  for k in ("verified", "read", "seen", "unverified", "disputed")]
        markdown += ("\n\n---\n\n**Evidence quality:** " + ", ".join(f"{n} {k}" for k, n in grades if n)
                     + ". _verified_: the farm re-read the page and it supports the claim; _read_: the agent opened "
                       "the page; _seen_: only in search results; _unverified_: the link never appeared in the "
                       "agent's searches; _disputed_: the page doesn't support the claim.")
        cost = self._base_cost + self.llm.meter.cost
        body = (f"{markdown}\n\n---\n\n## All sources\n\n"
                + "\n".join(f"- [{title}]({url})" for url, title in sources.items())
                + f"\n\n_Research farm run {run_id} for `{project['slug']}`, "
                  f"{datetime.now():%Y-%m-%d %H:%M}, approx. cost ${cost:.2f}._\n")
        path = os.path.join(folder, f"{datetime.now():%Y-%m-%d_%H%M}_run{run_id}.md")
        for target in (path, os.path.join(folder, "latest.md")):
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(body)
        return path
