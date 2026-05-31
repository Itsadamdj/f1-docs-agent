# F1 Documents Scraper

Polls the FIA Formula 1 decision-documents page, downloads new PDFs to
`~/Desktop/F1 Documents/<Event>/`, and sends a phone push (via ntfy) whenever a
**penalty or steward decision** is published.

No third-party Python packages — standard library only.

## Phone notifications (one-time setup)

1. Install the **ntfy** app: [iOS](https://apps.apple.com/app/ntfy/id1625396347) / [Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy).
2. In the app, **Subscribe to topic** and enter exactly:

   ```
   f1-docs-adamj-9f3kx2
   ```

   (This is the `NTFY_TOPIC` value in `f1_docs_scraper.py` — change both places
   if you want a different/private topic. Anyone who knows the topic can read it,
   so keep it secret.)
3. Test it: `python3 f1_docs_scraper.py --test-notify` — a push should arrive.

## What counts as "important"

Title/filename contains any of: decision, penalty, infringement, offence,
summons, reprimand, disqualif, protest, right of review, stewards. Edit
`IMPORTANT_KEYWORDS` in the script to tune.

## Running manually

```bash
python3 f1_docs_scraper.py            # one check
python3 f1_docs_scraper.py --watch    # loop every 120s
python3 f1_docs_scraper.py --watch 60 # loop every 60s
```

## Background agent (already installed)

A launchd agent runs the scraper every 120 seconds and at login.

| Action | Command |
|---|---|
| Status | `launchctl list \| grep f1docs` |
| Stop / disable | `launchctl unload ~/Library/LaunchAgents/com.adamj.f1docs.plist` |
| Start / enable | `launchctl load ~/Library/LaunchAgents/com.adamj.f1docs.plist` |
| Live log | `tail -f "scraper.log"` |

- `state.json` — record of every document already seen/downloaded. Delete it to
  re-download everything (and re-seed without notifications).
- First run seeds existing docs **without** sending pushes; only docs that
  appear afterward trigger a notification.
- **Retention:** PDFs older than `RETENTION_DAYS` (default 30) are deleted on
  every run, and empty event folders are pruned. State entries are kept so old
  docs are never re-downloaded. Change `RETENTION_DAYS` in the script to adjust.

## If it stops finding documents

The scraper depends on the FIA page's HTML layout. If FIA changes it, the log
will show `No documents parsed` — the regexes in `parse_documents()` need
updating.
