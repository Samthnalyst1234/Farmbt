"""Command line for the research farm. Run `python -m farm --help`."""
import argparse
import re
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from . import config  # noqa: E402  (reads env vars, so it must load after .env)
from .llm import LLM, Meter  # noqa: E402
from .orchestrator import Farm  # noqa: E402
from .store import Store  # noqa: E402


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "project"


def make_farm(args, store: Store) -> Farm:
    llm = LLM(Meter(args.max_cost), mock=args.mock, log=log)
    log("provider: mock (no API calls)" if args.mock else f"provider: {config.PROVIDER}, model {config.DEFAULT_MODEL}")
    return Farm(store, llm, questions=args.questions, workers=args.workers, solve=args.solve, log=log)


def cmd_research(args, store: Store) -> None:
    slug = slugify(args.name or args.mission)
    project = store.project(slug)
    if project and project["mission"] != args.mission:
        sys.exit(f"Project '{slug}' already exists with a different mission. Use --name to pick another name.")
    project = project or store.create_project(slug, args.mission)
    make_farm(args, store).run(store.create_run(project["id"]))


def cmd_monitor(args, store: Store) -> None:
    project = store.project(args.name)
    if not project:
        sys.exit(f"No project named '{args.name}'. See: python -m farm list")
    make_farm(args, store).run(store.create_run(project["id"]))


def cmd_resume(args, store: Store) -> None:
    if not store.run(args.run_id):
        sys.exit(f"No run {args.run_id}.")
    make_farm(args, store).run(args.run_id)


def cmd_solve(args, store: Store) -> None:
    project = store.project(args.name)
    run = project and store.latest_run(project["id"])
    if not run:
        sys.exit(f"No finished run for '{args.name}'.")
    if args.redo:
        if not args.gap:
            sys.exit("--redo needs --gap, e.g. --gap G1")
        gap = args.gap.upper()
        removed = store.delete_stages(run["id"], f"solve:{gap}:%") + store.delete_stages(run["id"], f"solution:{gap}")
        log(f"cleared {removed} saved solver steps for {gap}")
    make_farm(args, store).solve(run["id"], args.gap)


def cmd_dashboard(args, store: Store) -> None:
    from .dashboard import serve

    serve(args.port, open_browser=not args.no_browser)


def cmd_list(args, store: Store) -> None:
    rows = store.projects()
    if not rows:
        print('No projects yet. Start one with: python -m farm research "<mission>"')
    for p in rows:
        print(f"{p['slug']:<42} runs: {p['runs']:<3} last: run {p['last_run']} ({p['last_status']})")
        print(f"    {p['mission'][:110]}")


def cmd_gaps(args, store: Store) -> None:
    project = store.project(args.name)
    run = project and store.latest_run(project["id"])
    if not run:
        sys.exit(f"No finished run for '{args.name}'.")
    gaps = store.stage(run["id"], "critique")["gaps"]
    print(f"Gaps from run {run['id']} ({run['finished_at']}), best first. Report: {run['report_path']}\n")
    for g in gaps:
        r = g["review"]
        print(f"{g['id']:<4}{r['verdict']:<15}impact {g['impact']:<7}difficulty {g['difficulty']:<7}[{g['status']}]")
        print(f"    {g['title']}")
        if r["what_to_verify"]:
            print(f"    verify first: {r['what_to_verify']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="farm", description="AI research farm: research an area, map how it "
                                     "works, and find the gaps worth attacking.")
    sub = parser.add_subparsers(dest="command", required=True)

    def run_options(p):
        p.add_argument("--questions", type=int, default=config.DEFAULT_QUESTIONS,
                       help="research questions (one researcher agent each)")
        p.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS,
                       help="researchers running at the same time")
        p.add_argument("--max-cost", type=float, default=config.DEFAULT_MAX_COST,
                       help="stop and save progress after spending this many USD (0 = no limit)")
        p.add_argument("--solve", type=int, default=config.DEFAULT_SOLVE,
                       help="top gaps to design solutions and experiment plans for (0 = skip)")
        p.add_argument("--mock", action="store_true", help="dry run with fake agents; no API key or cost")

    p = sub.add_parser("research", help="start a new research project and run it")
    p.add_argument("mission", help="the area, problem or pain point to investigate")
    p.add_argument("--name", help="short project name (default: derived from the mission)")
    run_options(p)
    p.set_defaults(fn=cmd_research)

    p = sub.add_parser("monitor", help="re-run a project to see what changed and which gaps are still open")
    p.add_argument("name")
    run_options(p)
    p.set_defaults(fn=cmd_monitor)

    p = sub.add_parser("resume", help="continue a paused or failed run")
    p.add_argument("run_id", type=int)
    run_options(p)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("solve", help="design solutions and an experiment plan for a gap in the latest run")
    p.add_argument("name")
    p.add_argument("--gap", help="gap ID such as G2 (default: the best-ranked open gap)")
    p.add_argument("--redo", action="store_true", help="discard this gap's saved solutions and design them again")
    run_options(p)
    p.set_defaults(fn=cmd_solve)

    sub.add_parser("list", help="list projects").set_defaults(fn=cmd_list)

    p = sub.add_parser("dashboard", help="open the web dashboard")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("gaps", help="show the ranked gaps from a project's latest run")
    p.add_argument("name")
    p.set_defaults(fn=cmd_gaps)

    args = parser.parse_args()
    args.fn(args, Store(config.DB_PATH))


if __name__ == "__main__":
    main()
