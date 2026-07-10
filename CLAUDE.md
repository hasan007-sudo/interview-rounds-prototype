# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev          # start Next.js dev server
npm run build        # production build
npm run test         # run all tests once (vitest)
npm run test:watch   # vitest in watch mode
npm run db:migrate   # create + apply a migration from schema changes (prisma migrate dev)
npm run db:deploy    # apply pending migrations (prisma migrate deploy)
npm run db:status    # show migration status
npm run db:import    # import/re-embed jobs from source (tsx prisma/import-jobs.ts)
```

To run a single test file: `npx vitest run lib/__tests__/search.ranking.test.ts`

## Required Environment Variables

- `DATABASE_URL` — PostgreSQL/Aurora connection string
- `AWS_REGION` — for Amazon Bedrock (embeddings); credentials resolve from the standard AWS chain
- `OPENROUTER_API_KEY` — resume parsing LLM calls via OpenRouter (OpenAI-compatible endpoint)

### LiveKit voice interview (optional; required only for the in-round "Start interview" flow)
- `LIVEKIT_URL` — LiveKit server `wss://…` URL (used by both the server SDK and returned to the client)
- `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — server-side token minting, room creation, agent dispatch
- `LIVEKIT_AGENT_NAME` — dispatch name the agent worker registers under (defaults to `intervoo-agent`)
- `LIVEKIT_AGENT_ID` — agent profile selected via room metadata (defaults to `diagnostic_v2`, the no-ChromaDB flow)

## Architecture

**Two routes, two design eras:**
- `/` — legacy job search page (plain CSS, left untouched unless asked)
- `/onboarding` — new candidate onboarding flow (governed by the design system below)

**Data layer:** PostgreSQL (Amazon Aurora) via Prisma with `pgvector` and `pg_trgm` extensions. Models: `Company`, `Job`, plus the match-scoring tables `Skill` (deduped catalog; embedding is `embed(gloss)`, never the bare token), `JobSkill` (thin join, no vector), and `JobCapability` (one embedded responsibility statement per row). `Job.embedding` is a `vector(512)` column — 512-dim normalized vectors from AWS Bedrock Titan Text Embeddings V2, computed at import time over `title + roleType + summary` (skills are embedded per-token in the Skill catalog instead). Schema changes flow through Prisma migrations (`prisma/migrations/`); the SQL Prisma can't express — the HNSW index DDL and the trigram GIN indexes — lives hand-written in the migrations. When `prisma migrate dev` generates a new migration touching these, review it with `--create-only` first.

**Search (`lib/search.ts`):** Resolves `{ companyText, roleText, skills, experienceMinYears, experienceMaxYears, projectTexts, sort }` to a `JobCard[]`. The model has three parts:
1. **Candidate set** — union of company ∪ role ∪ skill ∪ project(capability) matched jobs, retrieved concurrently and bounded on every path. Title matching is a 3-tier union: exact → trigram (`pg_trgm`) → vector ANN, top-20 per tier. Company matching is exact → trigram. Skill path: top `MAX_SKILL_CANDIDATES = 1000` jobs by coverage ratio (semi-joined to jobs with ≥1 covered skill first). Project path: ANN top-50 capabilities per project vector. Experience is the one hard filter.
2. **Match %** — `matchScore = 65% skills + 35% projects`. `skills%` = share of the job's required skills covered by the candidate, resolved once per search against the Skill catalog (exact normalized token OR gloss-embedding cosine ≥ `SEMANTIC_SKILL_MIN = 0.3`), then set membership over `JobSkill`. `projects%` = AVG over the job's capabilities of the best project↔capability cosine, rescaled through the 0.10–0.35 evidence window. Computed entirely in a single SQL CTE (project vectors pre-cast once in a materialized `proj_vecs` CTE) — no post-SQL re-ranking.
3. **Sort** — each candidate gets a `tier` (0 company, 1 role, 2 everything else). `default` shows the company tier first (even at 0% match, hard-capped at `TIER0_CAP = 10`), reserves `TIER1_RESERVE = 10` slots for the role tier, and backfills the rest by match %; `score` is pure top-N by match %.

**Resume processing (`lib/resume.ts`):** Supports PDF (`unpdf`), DOCX (`mammoth`), TXT. Extracts text then calls an LLM via OpenRouter to return a structured `OnboardingProfile`. The profile drives search via `buildSearchInput()` in `lib/onboarding.ts`.

**Rounds (`lib/rounds.ts`):** Rounds are **not stored per-row** — they are parsed at read time. `parseRounds` splits `Job.focusRoundPattern` on `+`. `buildRounds` reads the 4 fixed round fields (`roundScreening`, `roundBehavioural`, `roundTechnical`, `roundCultureFit`) whose values are `;`-separated competency topics.

**Embeddings (`lib/embeddings.ts`):** Stateless — Bedrock only generates vectors; storage lives in Postgres. Query vectors are LRU-cached (max 500 entries per process).

**Components:** `components/shadcn/` holds lightly-adapted shadcn base primitives; `components/ui/` holds project-specific UI components styled to the design system.

**Tests:** All in `lib/__tests__/`, focused on search logic. Run with `vitest`.

## UI / Design System (mandatory)

Whenever you create, edit, or restyle any UI — components, pages, layouts, CSS —
you **must** read and follow `.interface-design/system.md` first. It is the
source of truth for color, typography, spacing, borders, radius, and component
patterns.

Rules:
- **Scope:** the design system applies to the `/onboarding` route and any new UI
  built in its style. The legacy `/` page uses an older plain-CSS system and is
  left untouched unless explicitly asked.
- **Tokens over guesses:** use the colors, spacing scale, radius ladder, and
  letter-spacing values defined in `system.md`. Do not introduce off-palette
  colors, off-scale spacing, off-ladder radii, or new accent hues.
- **Accent discipline:** indigo is the only brand color; emerald is reserved for
  the parse-success check only.
- **Depth:** borders-first. No shadows on cards/inputs/buttons — shadows only for
  true overlays (menus, modals).
- **Readability:** never track body/running text; only track uppercase
  eyebrows/labels per the letter-spacing config.

After any UI change, sanity-check it against `system.md` (the
`/interface-design:audit` skill can do this). If a design need genuinely isn't
covered by the system, extend `system.md` first, then build — don't drift
silently.

## Stack
- Next.js (App Router) + TypeScript + Tailwind CSS v4 + Prisma.
- Icons: `lucide-react`.
