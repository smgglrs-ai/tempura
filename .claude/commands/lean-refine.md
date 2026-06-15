# Refine Plan Items

Take backlog or planned items and add missing detail: descriptions,
acceptance criteria, scope, effort estimates. Uses agents to read
the codebase and propose refinements.

Use `$ARGUMENTS` to control scope:
- No args: refine all items with warnings from lint
- Item ID: refine a specific item (e.g., `9aa`)
- Status filter: `--backlog`, `--planned`, `--blocked`
- Prefix filter: `--prefix 11` (all items starting with 11)

## Step 1: Identify items to refine

Run the lint script to find items with missing fields:
```bash
bash ~/.claude/lean/scripts/lint-plan.sh .lean
```

Parse warnings for items missing description, acceptance criteria,
or scope. If `$ARGUMENTS` specifies a filter, apply it.

If a specific item ID was given, refine only that item.
If a prefix was given, refine all items matching that prefix.

## Step 2: Read the items

For each item to refine, read its `.lean/items/{id}.yml` file.
Note what fields are present and what's missing.

## Step 3: Analyze the codebase

For each item, use the Agent tool to read the relevant codebase
and propose refinements. Spawn one agent per item (or per cluster
if items are related).

Each agent receives this prompt:

> Read the codebase to understand what implementing this item
> would involve. The item is:
>
> Title: {title}
> Current description: {description or "none"}
> Current scope: {scope or "none"}
>
> Based on the code, propose:
>
> 1. **description** — what needs to be built/changed and why
>    (2-4 sentences, markdown block scalar)
> 2. **acceptance** — 3-6 verifiable criteria for done
> 3. **scope** — list of files/directories/crates affected
> 4. **effort** — S (<1d), M (1-2d), L (3-5d), XL (5+d)
> 5. **verification.quick** — command to quick-test this item
>
> Return as structured YAML.

Use the schema for structured output:

```json
{
  "type": "object",
  "properties": {
    "description": { "type": "string" },
    "acceptance": { "type": "array", "items": { "type": "string" } },
    "scope": { "type": "array", "items": { "type": "string" } },
    "effort": { "type": "string" },
    "verification_quick": { "type": "string" }
  },
  "required": ["description", "acceptance", "scope", "effort"]
}
```

For more than 5 items, use the Workflow tool to parallelize:

```javascript
export const meta = {
  name: 'lean-refine',
  description: 'Refine plan items with descriptions, acceptance criteria, and scope',
  phases: [
    { title: 'Refine', detail: 'Analyze codebase and propose item details' }
  ]
}

phase('Refine')
log(`Refining ${args.items.length} items`)

const REFINE_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    description: { type: 'string' },
    acceptance: { type: 'array', items: { type: 'string' } },
    scope: { type: 'array', items: { type: 'string' } },
    effort: { type: 'string' },
    verification_quick: { type: 'string' }
  },
  required: ['id', 'description', 'acceptance', 'scope', 'effort']
}

const results = await pipeline(
  args.items,
  (item) => agent(
    `Read the codebase at ${args.projectDir} to understand what
implementing this item would involve.

Item ID: ${item.id}
Title: ${item.title}
Current description: ${item.description || 'none'}
Current scope: ${JSON.stringify(item.scope || [])}

Propose:
1. description — what needs to be built/changed and why (2-4 sentences)
2. acceptance — 3-6 verifiable criteria for done
3. scope — files/directories/crates affected
4. effort — S (<1d), M (1-2d), L (3-5d), XL (5+d)
5. verification_quick — command to quick-test this item

Set id to "${item.id}".`,
    { label: 'refine:' + item.id, phase: 'Refine', schema: REFINE_SCHEMA }
  )
)

return { refinements: results.filter(Boolean) }
```

Pass `args`:
- `items`: array of {id, title, description, scope} for items to refine
- `projectDir`: current working directory

## Step 4: Present refinements

For each item, show the proposed changes:

```
## 9aa — MCP 2026-07-28 default flip

description: |
  Implement MCP 2026-07-28 transport default behind a feature gate.
  When the spec finalizes on July 28, flip the gate to make Streamable
  HTTP the default transport instead of stdio.

acceptance:
  - "Feature gate config option exists in config.toml"
  - "Streamable HTTP works when gate is enabled"
  - "stdio remains default when gate is disabled"
  - "cargo test --workspace passes with both gate states"

scope:
  - navra-core/src/transport/
  - navra-server/src/config.rs

effort: L

verification:
  quick: "cargo test -p navra-core --test transport"

Accept? [Y/n/edit]
```

Allow the user to accept, reject, or edit each refinement.

## Step 5: Apply and regenerate

For each accepted refinement, update the item's `.lean/items/{id}.yml`:
- Merge new fields into existing (don't overwrite fields the item
  already has unless the user explicitly accepted the change)
- Preserve existing fields not covered by the refinement

Then regenerate plan.yml:
```bash
bash ~/.claude/lean/scripts/generate-plan.sh .lean
```

Run lint to verify:
```bash
bash ~/.claude/lean/scripts/lint-plan.sh .lean
```

Report: "Refined N items. M warnings remaining."
