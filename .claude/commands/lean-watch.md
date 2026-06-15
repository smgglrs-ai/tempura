# Tech Watch

Analyze external sources against the current project, classify findings,
and route actionable items into the plan.

Two modes:
- **Reactive**: `$ARGUMENTS` contains URLs → analyze those specific sources
- **Proactive**: `$ARGUMENTS` is empty or contains topics → discover and
  analyze sources autonomously via `/deep-research`

## Step 1: Load project context

Read these files to understand current state:
- `CLAUDE.md` — architecture, conventions, design decisions
- `.lean/plan.yml` — active work items (what's planned, in progress)
- `.lean/project.yml` — stack, item_prefix, next_item_id
- `docs/ecosystem-positioning.md` — current competitive landscape (if exists)
- `.lean/watches/` — previous watch records (if any, for continuity)

## Step 2: Determine mode and gather sources

### If `$ARGUMENTS` contains URLs → Reactive mode

Proceed to Step 3 with the provided URLs.

### If `$ARGUMENTS` is empty or contains only topics → Proactive mode

Build a research question from the project context:

1. From CLAUDE.md, extract: project domain, stack, key technologies
2. From plan.yml, extract: in-progress and planned items (what's active)
3. From ecosystem-positioning.md, extract: named competitors, threat areas
4. From previous watches, extract: items marked `watch` or `investigate`
   (these are follow-ups from prior watches)

Compose a `/deep-research` prompt. Example structure:

```
What are the latest developments (last 2-3 weeks) in:
- <core technology areas from CLAUDE.md>
- <competitor names from ecosystem-positioning.md>
- <topics from plan items marked investigate/watch>
- <the project's differentiators — what could threaten them>

Focus on: new tools, papers, security vulnerabilities, spec changes,
breaking API changes, and competitive threats.
Include both mainstream and niche sources.
Prioritize actionable findings over survey-level coverage.
```

If `$ARGUMENTS` contains topics (not URLs), weave them into the
research prompt alongside the auto-generated topics.

Invoke `/deep-research` with this prompt. Wait for the results.
Then proceed to Step 3 treating the research output as the sources.

## Step 3: Analyze sources

For each source (URLs from reactive mode, or findings from deep-research),
determine:
- **What it is** — one-line summary
- **Impact** — HIGH / MEDIUM / LOW for this project
- **Relationship** — validates, threatens, complements, or irrelevant
- **What it validates** — which existing design decisions or capabilities
- **What it threatens** — which assumptions or differentiators
- **Actionable** — does it create work? (see action types below)

Go beyond the articles themselves. Follow references, explore related
work, find connections the articles don't make explicit.

## Step 4: Classify actions

For each finding with actionable implications, classify:

| Action | Meaning | Destination |
|--------|---------|-------------|
| `plan` | Ready to implement or evaluate | New `.lean/items/{PREFIX}-{N}.yml` |
| `investigate` | Needs deeper analysis before committing | Note in watch record, follow up next watch |
| `enrich` | Adds context to existing plan item | Append to existing `.lean/items/{id}.yml` notes |
| `gate` | External dependency on existing item | Update `gate` field in existing item |
| `watch` | Too early, monitor in future watches | Note in watch record only |
| `dismiss` | Evaluated, not relevant | Note in watch record with reason |
| `update-positioning` | Changes competitive landscape | Update `docs/ecosystem-positioning.md` |

## Step 5: Present findings

First, present a TLDR newsletter-style briefing for the user to
scan quickly. This is especially valuable in proactive mode where
the user hasn't read the sources:

```
# Tech Watch — YYYY-MM-DD

## Headlines
- **MCP spec update**: sessions removed from July RC, default
  transport flip confirmed for July 28
- **New IFC competitor**: FIDES v2 achieves F1=0.72 with gateway
  integration — closer to navra's approach than v1

## Threats
- IBM ContextForge now offers gateway-level IFC (was library-only)

## Opportunities
- GLiGuard 300M: Apache 2.0 safety model, ONNX-ready, 87.7 F1
- Cedar OWASP policy templates published by AWS

## Follow-ups from previous watches
- LFM2.5-Audio: still no ONNX export (keep watching)
- Agent Skills standard: now 36+ adopters (investigate threshold met)
```

Then, present the detailed findings grouped by source and action:

```
## Source: "Building Secure Agent Runtimes with OpenClaw" (marktechpost.com)
Impact: MEDIUM
Relationship: validates gateway architecture

- VALIDATES: navra's deny-wins ACL model (OpenClaw has no equivalent)
- ACTION plan: "Evaluate OpenClaw's skill marketplace pattern" (HIGH)
- ACTION update-positioning: OpenClaw security section needs CVE update

## Source: "LFM2.5-Audio ONNX Export" (huggingface.co)
Impact: HIGH
Relationship: complements voice pipeline

- ACTION plan: "Evaluate LFM2.5-Audio as ASR+TTS replacement" (P1, M effort)
- ACTION enrich: NAVRA-041 — add ONNX export details
```

Ask the user which actions to apply.

## Step 6: Apply accepted actions

### New plan items (`plan`)

For each accepted `plan` action:
1. Read `next_item_id` from `.lean/project.yml`
2. Create `.lean/items/{PREFIX}-{N}.yml` with:
   - id, title, status: planned, priority, effort
   - source: "tech-watch-YYYY-MM-DD"
   - description from the finding
   - depends_on if it relates to existing items
3. Increment `next_item_id` in project.yml
4. Regenerate plan.yml:
   ```bash
   bash ~/.claude/lean/scripts/generate-plan.sh .lean
   ```

### Enrich existing items (`enrich`)

Append to the `notes` field in the existing item's `.yml` file.

### Gate updates (`gate`)

Update the `gate` field in the existing item's `.yml` file.
Regenerate plan.yml.

### Positioning updates (`update-positioning`)

Edit `docs/ecosystem-positioning.md` with the new competitive data.

### Watch record

Write `.lean/watches/YYYY-MM-DD.yml`:

```yaml
date: "YYYY-MM-DD"
mode: proactive  # or reactive
sources:
  - url: "https://..."
    title: "Article Title"
    impact: HIGH
    relationship: validates

findings:
  - source: "https://..."
    title: "Finding title"
    action: plan
    item_id: "NAVRA-073"  # if plan action was applied
    detail: "One-line summary"

  - source: "https://..."
    title: "Finding title"
    action: watch
    detail: "Too early, monitor for GA"

  - source: "https://..."
    title: "Finding title"
    action: investigate
    questions:
      - "Does it scale beyond single-node?"
      - "What's the license?"
    detail: "Promising but needs deeper analysis"

  - source: "https://..."
    title: "Finding title"
    action: dismiss
    detail: "Research prototype, no production path"

follow_ups_resolved:
  - from_watch: "2026-05-28"
    title: "LFM2.5-Audio ONNX export"
    resolution: "Still no export, keep watching"

summary: |
  N sources analyzed, M new plan items, K existing items enriched.
```

### Activity log

Append to `.lean/activity.log`:
```yaml
- timestamp: "<ISO8601>"
  actor: claude
  action: tech-watch
  summary: "N sources, M new items (PREFIX-X through PREFIX-Y), K enriched"
```

## Step 7: Report

```
Tech watch complete:
  Sources: N analyzed
  New items: PREFIX-073, PREFIX-074 (planned)
  Enriched: PREFIX-041 (added ONNX export details)
  Positioning: updated OpenClaw section
  Watch: 3 items for follow-up
  Dismissed: 2 items
```
