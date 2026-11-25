---
name: conference-brand-template-lock
description: Lock the RelationalAI brand/template before any Topic 4 rebuild.
---

# Conference Brand Template Lock

Use this skill before any Topic 4 presentation rebuild begins. The brand lock
must pass before source board, storyboard, render planning, or any other
Phase 02–10 work starts.

## Required Contract

- Run `scripts/presentations/lock_brand_template.py --workspace <workspace> --deck-dir <deck-dir> --reference-deck <reference-deck> --output-md 01-BRAND-REFERENCE-LOCK.md --output-json 01-BRAND-REFERENCE-LOCK.json`.
- The script computes SHA-256 from `assets/brand/relationalai/{logo_white.png, title_decorative_art.png, bug_orange.png}` and cross-checks against the reference deck's embedded copies.
- Write `01-BRAND-REFERENCE-LOCK.md` and `01-BRAND-REFERENCE-LOCK.json` to the phase dir.
- `01-BRAND-REFERENCE-LOCK.json` must contain `status: pass` or list all blockers in `expected_blockers`.
- All brand asset hashes must be SHA-256 bound — no un-hashed assets.
- Treat the brand-lock JSON as an input to the v2.2 digest chain; if a locked asset or reference deck changes, rerun downstream storyboard compile, render spec, rendering, video, and produced-artifact review.
- The `presentation-template` meta tag must be present in the reference deck.

## Rejected Evidence

- Fake inline `title-logo` SVG (blocker code: `fake_inline_title_logo_svg`).
- Text-only `RAI` brand bug — must be image, not text (blocker code: `text_only_rai_brand_bug`).
- Missing `presentation-template` meta tag (blocker code: `template_metadata_missing`).
- Missing `presentation-template-sha256` meta tag (blocker code: `template_sha256_missing`).
- Un-hashed brand assets — any brand asset without a confirmed SHA-256 (blocker code: `brand_asset_hash_missing`).
- Scaffold prose on any slide (blocker code: `scaffold_prose_present`).
- `lock_brand_template.py` run in `--dry-run` mode presented as a passing production receipt.
- `01-BRAND-REFERENCE-LOCK.json` with `status: blocked` used as a downstream gate pass.

## Verification

- `01-BRAND-REFERENCE-LOCK.json` exists in the phase dir.
- `01-BRAND-REFERENCE-LOCK.json` has `status: pass` OR all blockers are listed under `expected_blockers` and are acknowledged by the human approver.
- `01-BRAND-REFERENCE-LOCK.md` exists in the phase dir.
- Downstream v2.2 receipts reference current brand-lock inputs.
- No bulk presentation regeneration before approved `04_safety_alignment_privacy` canary.
