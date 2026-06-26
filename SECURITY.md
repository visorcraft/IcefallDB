# Security Policy

We take the security of IcefallDB seriously. Thank you for helping keep it and
its users safe.

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues, pull requests,
or discussions.** Public disclosure before a fix is available puts users at risk.

Instead, report privately through GitHub:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

This opens a confidential channel visible only to the maintainers.

When you report, please include as much of the following as you can:

- A description of the issue and why you believe it is a security problem.
- The version, commit, or release affected.
- Step-by-step instructions to reproduce it, ideally with a minimal example.
- The impact you foresee (for example, data disclosure, data corruption, or
  denial of service).

## What to expect

- We will acknowledge your report within a few business days.
- We will investigate, keep you updated on progress, and let you know when a fix
  is planned or released.
- We will credit you for the discovery when the fix ships, unless you prefer to
  remain anonymous.

Please give us a reasonable opportunity to release a fix before any public
disclosure.

## Supported versions

IcefallDB is pre-1.0 software under active development. Security fixes are
applied to the latest release line. We recommend always running the most recent
version.

## Scope

IcefallDB reads and writes plain files on a filesystem (and reads from
S3-compatible object storage). Encryption protects table contents at rest, but
key management is the operator's responsibility - keys must be kept outside the
table directory and out of version control. The HTTP server ships without
built-in authentication or TLS and is intended to run on a trusted network or
behind a reverse proxy that provides them. Reports about these documented
behaviors are welcome as hardening suggestions, but they are known and
intentional defaults rather than vulnerabilities.
