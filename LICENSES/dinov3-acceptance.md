# DINOv3 licence acceptance record

The engine's mesh stage conditions TRELLIS.2 on DINOv3 image features. The
weights come from a gated Hugging Face repository under Meta's DINOv3 License,
which is a custom licence, not an OSI-approved one. `vendor/weights.lock.json`
pins them, and its `dinov3-vitl16` entry points here from its
`licence_acceptance` block.

| Field | Value |
| --- | --- |
| Repository | `facebook/dinov3-vitl16-pretrain-lvd1689m` on Hugging Face |
| Pinned revision | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |
| Files | `config.json`, `model.safetensors`, `preprocessor_config.json` (sizes and sha256 in the lock) |
| Licence | DINOv3 License, as shown on the repository's Hugging Face page |
| Licensee | Boundless Games, LLC |
| Accepted on Hugging Face, recorded at | 2026-11-07T00:17:48Z |

## What this record covers

Boundless Games, LLC accepted the DINOv3 License for this repository on
Hugging Face. The acceptance covers Boundless Games, LLC only. It is kept here
as provenance for the pinned hashes. It grants nothing to anyone else.

## What you need to do

The weights are not in this repository, and `aismith-engine weights install`
downloads them from Hugging Face with your own credentials. Before you install
them:

1. Read the DINOv3 License on the repository's Hugging Face page.
2. Accept it there with your own Hugging Face account, and wait for access to
   be granted.
3. Make a token from that account available as `HF_TOKEN` when you run
   `aismith-engine weights install`.

Among other things, the licence:
- governs any redistribution of the weights or of models derived from them;
- asks for an acknowledgement in research publications;
- carries trade-control terms and a list of prohibited uses.

Read the full text rather than relying on this summary. This record is not
legal advice.
