from __future__ import annotations

import json
from pathlib import Path

sports_templates = [
    ("The team finished the season with", "The company finished the quarter with"),
    ("In the final game, the player scored", "In the final chapter, the writer explained"),
    ("After the trade deadline, the coach said", "After the policy deadline, the minister said"),
    ("The league announced a new rule before", "The publisher announced a new edition before"),
    ("Fans expected the team to make the playoffs after", "Readers expected the novel to become popular after"),
    ("A veteran forward led the team in", "A veteran scholar led the committee in"),
    ("The manager praised the defense after", "The editor praised the argument after"),
    ("The club rebuilt its roster during", "The studio rebuilt its pipeline during"),
    ("The coach adjusted the lineup after", "The curator adjusted the exhibit after"),
    ("The player returned from injury and helped", "The author returned from leave and finished"),
]

endings = [
    "the difficult stretch.",
    "a narrow victory.",
    "months of uncertainty.",
    "the opening week.",
]

pairs = []
idx = 1
for a, b in sports_templates:
    for e in endings:
        pairs.append({"id": f"p{idx}", "a": f"{a} {e}", "b": f"{b} {e}"})
        idx += 1

payload = {
    "targets": ["game", "season", "team", "league", "player", "score", "coach", "playoffs"],
    "pairs": pairs,
}

out = Path("outputs/features/feature64_minimal_pairs_sports_40.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2))
print(f"Wrote {out} with {len(pairs)} pairs")
