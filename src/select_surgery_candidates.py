from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

THEMES = {
    "music": {"album", "song", "band", "devin", "released", "tour", "track"},
    "aviation": {"aviation", "aircraft", "raaf", "squadron", "flight", "airport"},
    "sports": {"game", "season", "team", "all", "star", "league", "nba", "jordan"},
    "history_politics": {"treaty", "assassination", "war", "president", "national", "dominican"},
    "literature": {"novel", "edition", "author", "book", "llosa", "vargas"},
    "geography": {"london", "city", "river", "province", "county", "town"},
}


def theme_scores(keywords: list[str]) -> dict[str, int]:
    kw = set([k.strip().lower() for k in keywords if k.strip()])
    scores = {}
    for t, lex in THEMES.items():
        scores[t] = len(kw & lex)
    return scores


def choose_theme(keywords: list[str]) -> tuple[str, int]:
    scores = theme_scores(keywords)
    best_theme, best = max(scores.items(), key=lambda kv: kv[1])
    if best == 0:
        return "mixed", 0
    return best_theme, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-json", required=True)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--pick", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.ranking_json).read_text())
    rows = data["rows"][: args.top]
    cards = data["feature_cards"]

    candidates = []
    for r in rows:
        f = str(int(r["feature"]))
        kws = [k.strip() for k in r.get("keywords", "").split(",")]
        theme, strength = choose_theme(kws)
        contexts = cards.get(f, {}).get("top_contexts", [])[:8]

        # crude disentanglement: repeated key phrases across contexts
        tokens = []
        for c in contexts:
            toks = [w.strip(".,:;!?()[]{}\"'`").lower() for w in c.split()]
            toks = [w for w in toks if len(w) >= 4]
            tokens.extend(toks)
        cnt = Counter(tokens)
        dominance = 0.0
        if cnt:
            top_count = cnt.most_common(1)[0][1]
            dominance = top_count / max(sum(cnt.values()), 1)

        score = (
            float(r["brain_surgery_score"])
            + 0.25 * strength
            + 0.20 * float(r.get("coherence", 0.0))
            + 0.15 * float(r.get("entity_ratio", 0.0))
            - 0.20 * float(r.get("genericity", 0.0))
            - 0.50 * dominance
        )

        candidates.append(
            {
                "feature": int(r["feature"]),
                "theme": theme,
                "theme_strength": strength,
                "score": score,
                "brain_surgery_score": float(r["brain_surgery_score"]),
                "frequency": float(r["frequency"]),
                "genericity": float(r["genericity"]),
                "entity_ratio": float(r.get("entity_ratio", 0.0)),
                "coherence": float(r.get("coherence", 0.0)),
                "keywords": r.get("keywords", ""),
                "sample_context": contexts[0] if contexts else "",
            }
        )

    # pick diverse themes first
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    used_themes = set()

    for c in candidates:
        if c["theme"] not in used_themes and c["theme"] != "mixed":
            selected.append(c)
            used_themes.add(c["theme"])
        if len(selected) >= args.pick:
            break

    if len(selected) < args.pick:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= args.pick:
                break

    out = {
        "source": args.ranking_json,
        "top_considered": args.top,
        "selected": selected,
        "top_candidates": candidates[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Surgery candidate shortlist (wiki SAE)\n\n")
        f.write("## Selected 5 feature knobs\n\n")
        for i, c in enumerate(selected, 1):
            f.write(
                f"{i}. **Feature {c['feature']}** — theme: `{c['theme']}`; score={c['score']:.3f}\n"
                f"   - keywords: {c['keywords']}\n"
                f"   - freq={c['frequency']:.4f}, genericity={c['genericity']:.3f}, coherence={c['coherence']:.3f}, entity_ratio={c['entity_ratio']:.3f}\n"
                f"   - sample context: {c['sample_context'][:260].replace(chr(10),' ')}\n\n"
            )

        f.write("## Next step\n")
        f.write("Run alpha sweeps (1.0, 0.75, 0.5, 0.25, 0.0) on these features first, then compare against matched random controls.\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
