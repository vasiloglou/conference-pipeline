# Public BYOK Cost Safety

The v3.0 public tooling is BYOK or self-hosted by default. Public users may run
browser Q&A/deep-search with their own provider key, or run the deck, report,
GPT-Researcher, podcastfy, and NotebookLM tooling locally. The v3.0 public site
does not deploy a maintainer-funded hosted-key LLM endpoint.

## Current v3.0 Rule

- User-initiated LLM calls use the user's key or local model.
- The browser BYOK key is held only for the page lifetime.
- No public server receives, stores, proxies, or logs user keys.
- Deep search is corpus-only and does not enable live web search.
- Maintainer credentials stay in the private operator environment and never in
  the public bundle, public repository, browser assets, or receipts.

## Future Hosted-Key Requirement

Any future milestone that adds a hosted-key public path MUST implement both:

1. A hard per-session cap for public generation spend.
2. A global daily circuit-breaker mapped to the same `budget_cap_usd` guardrail
   used by private provider-cost enforcement.

The `budget_cap_usd` value is never auto-raised. When either the per-session cap
or global daily circuit-breaker is reached, the system must halt public
generation, write a blocker receipt, and require explicit operator approval
before service resumes.

## Required Failure Behavior

- Cap breach: halt generation before the next provider call.
- Receipt: record the cap, observed spend, session/global scope, and
  `halt_status`.
- User-facing state: show a cost cap reached state, not a retry loop.
- Operator flow: review spend evidence, choose a new cap manually if justified,
  and record approval before resuming.

No implementation may silently fall back from BYOK to maintainer-funded keys.

