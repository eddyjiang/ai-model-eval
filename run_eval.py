#!/usr/bin/env python3
"""
CLI entry point for the Abstraction Collapse eval.

Usage examples:
  # Pilot run (10 groups, 3 templates, claude+gpt, 2 trials)
  python run_eval.py run --pilot

  # Full run with specific models and groups
  python run_eval.py run --run-id my-run --models claude gpt gemini \\
      --groups racial political --templates comedy critical --trials 5

  # Score already-collected responses
  python run_eval.py score --run-id my-run

  # Show DB stats
  python run_eval.py status
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from eval import storage
from eval.utils import load_config, generate_run_id


def resolve_groups(groups_cfg: dict, category_keys: list[str]) -> list[str]:
    """Expand category keys (e.g. ['racial', 'political']) to group names."""
    key_map = {
        "racial": "racial_ethnic",
        "racial_ethnic": "racial_ethnic",
        "political": "political",
        "religious": "religious",
        "gender": "gender_sexuality",
        "gender_sexuality": "gender_sexuality",
        "socioeconomic": "socioeconomic",
        "intersectional": "intersectional",
    }
    names: list[str] = []
    for k in category_keys:
        yaml_key = key_map.get(k.lower())
        if yaml_key and yaml_key in groups_cfg["groups"]:
            names.extend(groups_cfg["groups"][yaml_key]["members"])
        else:
            print(f"Warning: unknown group category '{k}', skipping.")
    return list(dict.fromkeys(names))  # dedup, preserve order


def cmd_run(args) -> None:
    groups_cfg, templates_cfg, models_cfg = load_config()

    # --- Resolve models ---
    if args.models:
        model_keys = args.models
    elif args.pilot:
        model_keys = models_cfg["pilot_models"]
    else:
        model_keys = list(models_cfg["models"].keys())

    unknown_models = [m for m in model_keys if m not in models_cfg["models"]]
    if unknown_models:
        print(f"Error: unknown model(s): {unknown_models}")
        print(f"Valid models: {list(models_cfg['models'].keys())}")
        sys.exit(1)

    # --- Resolve templates ---
    if args.templates:
        template_ids = []
        for t in args.templates:
            if t in templates_cfg["templates"]:
                template_ids.append(t)
            else:
                matched = [
                    tid for tid, td in templates_cfg["templates"].items()
                    if td["category"] == t
                ]
                if matched:
                    template_ids.extend(matched)
                else:
                    print(f"Warning: unknown template/category '{t}', skipping.")
    elif args.pilot:
        template_ids = templates_cfg["pilot_templates"]
    else:
        template_ids = list(templates_cfg["templates"].keys())

    # --- Resolve groups ---
    if args.groups:
        if args.groups == ["all"]:
            group_names = [
                name
                for cat in groups_cfg["groups"].values()
                for name in cat["members"]
            ]
        else:
            group_names = resolve_groups(groups_cfg, args.groups)
    elif args.pilot:
        group_names = groups_cfg["pilot_groups"]
    else:
        group_names = [
            name
            for cat in groups_cfg["groups"].values()
            for name in cat["members"]
        ]

    # --- Resolve trials ---
    if args.trials is not None:
        n_trials = args.trials
    elif args.pilot:
        n_trials = 2
    else:
        n_trials = 5

    # --- Resolve run ID ---
    run_id = args.run_id or generate_run_id()

    # --- Summary ---
    total = len(model_keys) * len(template_ids) * len(group_names) * n_trials
    print(f"\nRun ID      : {run_id}")
    print(f"Models      : {model_keys}")
    print(f"Templates   : {template_ids}")
    print(f"Groups      : {len(group_names)} groups")
    print(f"Trials      : {n_trials}")
    print(f"Total prompts: {total}")
    if args.dry_run:
        print("\n[dry-run] Not executing.")
        return

    if args.retry_errors and args.run_id:
        deleted = storage.delete_errored_responses(run_id=args.run_id)
        if deleted:
            print(f"Cleared {deleted} errored response(s) for retry.")
    print()

    from eval.runner import run
    stats = run(run_id, model_keys, template_ids, group_names, n_trials)
    print(f"\nDone. Responses: {stats['total']} | Errors: {stats['errors']}")
    print(f"Run ID: {run_id}  (use this with 'score' and 'status' subcommands)")

    if not args.no_score:
        print("\nStarting scoring pass...")
        from eval.scorer import score
        score_stats = score(run_id=run_id, model_keys=model_keys)
        print(f"Scored: {score_stats['scored']} | Errors: {score_stats['errors']}")


def cmd_score(args) -> None:
    from eval.scorer import score
    model_keys = args.models if args.models else None
    stats = score(run_id=args.run_id, model_keys=model_keys, rescore_errors=args.rescore_errors)
    print(f"Scored: {stats['scored']} | Errors: {stats['errors']}")


def cmd_status(args) -> None:
    storage.create_tables()
    run_ids = storage.get_run_ids()
    if not run_ids:
        print("No runs found in the database.")
        return
    print(f"{'Run ID':<12} {'Total':>8} {'Scored':>8} {'Errors':>8}")
    print("-" * 40)
    for rid in run_ids:
        s = storage.get_run_stats(rid)
        print(f"{rid:<12} {s['total']:>8} {s['scored']:>8} {s['errors']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Abstraction Collapse AI Model Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Collect model responses")
    p_run.add_argument("--run-id", default=None, help="Unique run identifier (auto-generated if omitted)")
    p_run.add_argument("--pilot", action="store_true", help="Quick pilot run (10 groups, 3 templates, claude+gpt, 2 trials)")
    p_run.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                       help="Models to run (claude gpt gemini grok llama). Default: all.")
    p_run.add_argument("--templates", nargs="+", default=None, metavar="TEMPLATE",
                       help="Template IDs or categories (comedy critical positive). Default: all.")
    p_run.add_argument("--groups", nargs="+", default=None, metavar="GROUP",
                       help="Group categories (racial political religious gender socioeconomic intersectional) or 'all'. Default: all.")
    p_run.add_argument("--trials", type=int, default=None, help="Trials per combination (default: 5, or 2 with --pilot)")
    p_run.add_argument("--no-score", action="store_true", help="Skip automatic scoring after collection")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    p_run.add_argument("--retry-errors", action="store_true",
                       help="Delete API-error rows for this run-id and retry them")

    # --- score ---
    p_score = sub.add_parser("score", help="Run judge scoring on collected responses")
    p_score.add_argument("--run-id", default=None, help="Score only this run (default: all unscored)")
    p_score.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                         help="Score only responses from these models")
    p_score.add_argument("--rescore-errors", action="store_true",
                         help="Delete errored scores and re-judge them with the fixes applied")

    # --- status ---
    sub.add_parser("status", help="Show run statistics from the database")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
