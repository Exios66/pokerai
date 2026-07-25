---
name: posit-connect-publish
description: >-
  Render the Poker AI Quarto website and publish a NEW content instance to the
  JackJBurleson Posit Connect Cloud account. Use when the user asks to
  publish/render/deploy the Quarto site, update Posit Connect Cloud, or refresh
  the Poker AI Connect content. Never overwrite the psych755 manuscript content
  id unless the user explicitly requests that id.
---

# Posit Connect Cloud publish (JackJBurleson · Poker AI)

End-to-end workflow for this repo (`pokerai`): **update docs → `quarto render` → publish NEW instance → verify**.

## Canonical deployment

| Field | Value |
|---|---|
| Account | `jackjburleson` |
| Content ID | `019f9a68-2304-5291-83c1-e2b9574e723d` |
| Dashboard | https://connect.posit.cloud/jackjburleson/content/019f9a68-2304-5291-83c1-e2b9574e723d |
| Public share URL | https://019f9a68-2304-5291-83c1-e2b9574e723d.share.connect.posit.cloud/ |
| Protected (do not overwrite) | `019f9a10-ebb9-d1d5-839f-97e794bfd0ca` (psych755 manuscript) |
| Config | `_publish.yml`, `_quarto.yml` |

## Workflow

### 1. Update content

Edit `index.qmd` and `docs/*.qmd` as needed. Keep navbar/sidebar/`project.render` in `_quarto.yml` in sync.

### 2. Render

```bash
quarto check
quarto render    # writes _site/
```

Requires Quarto ≥ 1.10 (`posit-connect-cloud` provider).

### 3. Authenticate

Prefer env vars / cached tokens:

- `POSIT_CONNECT_CLOUD_ACCESS_TOKEN`
- `POSIT_CONNECT_CLOUD_REFRESH_TOKEN`
- `POSIT_CONNECT_CLOUD_ACCOUNT_ID`
- or `/tmp/posit-tokens.json` from device-code OAuth

Device-code flow (client_id `quarto-cli`, scope `vivid`):

1. `POST https://login.posit.cloud/oauth/device/authorize`
2. Show `verification_uri_complete` + `user_code`
3. Poll `https://login.posit.cloud/oauth/token`
4. Confirm authorized account is **`jackjburleson`**

### 4. Publish

Update the existing Poker AI deployment (preferred after first publish):

```bash
python scripts/publish_posit_pokerai.py --content-id 019f9a68-2304-5291-83c1-e2b9574e723d
# or skip re-render:
python scripts/publish_posit_pokerai.py --skip-render --content-id 019f9a68-2304-5291-83c1-e2b9574e723d
```

Omit `--content-id` only when intentionally creating another new instance. The helper:

1. Creates (`POST /v1/contents`) or updates (`PATCH …?new_bundle=true`) content
2. Uploads `_site` as a gzip bundle
3. `POST /v1/contents/{id}/publish`
4. Writes `_publish.yml`
5. Verifies the `.share.connect.posit.cloud` URL

### 5. Verify

Fetch the share URL (not only the dashboard SPA) and assert title/body markers
(`Poker AI`, `GPT-2`, `SoelMgd/Poker_Dataset`). Screenshot to
`/opt/cursor/artifacts/pokerai-connect-cloud-published.png` when available.

Add the public share URL to `README.md`.

## Hard rules

- Do **not** publish onto content id `019f9a10-ebb9-d1d5-839f-97e794bfd0ca`.
- Prefer creating a new instance unless the user names an existing Poker AI id.
- Verify via `https://{content-id}.share.connect.posit.cloud/`.
