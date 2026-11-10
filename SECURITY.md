# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: open this repository's
**Security** tab and choose **Report a vulnerability**. Please do not open a
public issue, pull request or discussion for a security problem.

Include what you found, how to reproduce it, and what an attacker could do with
it. We aim to acknowledge a report within a week and will keep you updated
until it is fixed. Tell us if you would like to be credited.

Test only against your own installation and your own machines.

## Supported versions

Only the latest engine release (`engine-v*` tags) and the latest commit on the
default branch are supported.

## What we treat as security issues

- **The engine runner's boundary.** `aismith-engine serve` binds to loopback by
  default, takes one job at a time, caps request sizes, confines each job to its
  own work directory and never reaches the network. A way to read or write
  outside a job's directory, to make a stage reach the network, or to leave a
  process running after its job ends is a vulnerability.
- **Pinned inputs.** A way to make the engine load a model file or a source
  that does not match `vendor/weights.lock.json` or `vendor/sources.lock.json`
  while reporting that it matched.
- **The studio app's API.** It has no authentication and is meant for one user
  on their own machine or a trusted network; never expose it to the internet.
  That missing authentication is known. A way for a web page the user visits to
  drive the API through their browser is a vulnerability.
- **Secrets.** A Hugging Face token or any other credential that is logged,
  written to an artifact or returned by an API.

Bugs in third-party models, ComfyUI, Blender or other upstream programs belong
with their own projects (see [NOTICE](NOTICE)), unless the engine's use of them
creates the problem.
