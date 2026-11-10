import argparse
import json
from pathlib import Path
from .models import Request, Result, Policy, Round, HumanResolution
from .engine import evaluate_round


def main():
    p = argparse.ArgumentParser(description="Settle independent results; never grants human art acceptance")
    p.add_argument("input", type=Path, help="JSON round inputs, optionally including history")
    p.add_argument("output", type=Path, help="new immutable round JSON")
    args = p.parse_args()
    try:
        data = json.loads(args.input.read_text())
        for key, model in (("request", Request), ("policy", Policy), ("primary", Result),
                           ("secondary", Result), ("adjudication", Result), ("human_resolution", HumanResolution)):
            if data.get(key) is not None:
                data[key] = model.model_validate(data[key])
        data["history"] = tuple(Round.model_validate(r) for r in data.get("history", []))
        result = evaluate_round(**data)
        with args.output.open("x") as f:
            f.write(result.model_dump_json(indent=2) + "\n")
    except (ValueError, TypeError, OSError) as exc:
        p.exit(2, f"Invalid or unavailable input/output: {exc}\n")
    return 0 if result.outcome.state == "pass" else 10

if __name__ == "__main__":
    raise SystemExit(main())
