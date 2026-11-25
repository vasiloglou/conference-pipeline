# Security

## Reporting a vulnerability

Please report it privately, through GitHub's private vulnerability reporting: on this
repository, open **Security**, then **Report a vulnerability**. Do not open a public issue
for it.

Say what you found, how to reproduce it, and what it affects. You will get an answer, and
a fix or a reason, as soon as the report has been reproduced.

## What counts

- A way for the code to send a key, a private file or a personal path to a public
  surface: a release bundle, a published site, a dataset or this repository.
- A way around a gate that should refuse: the export gate, the private-marker scan,
  `release publish` without `--approved-by`, a cost or budget cap.
- A way for a published site to spend someone else's model credits. The sites are
  bring-your-own-key only; see
  [docs/operations/public-byok-cost-safety.md](docs/operations/public-byok-cost-safety.md).

## Keys

The code reads provider keys from `.env` at the repository root, or from the
environment, and nowhere else. Receipts record that keys were used, never their values
(`secret_values_recorded: false`).
