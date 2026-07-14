# Job recommendation architecture

This document defines the approved target architecture for personalized job
recommendations. It replaces the current single-job-vector and equal-blend scoring
model.

The system keeps two concerns separate:

1. **Preference ordering** decides which group of jobs appears first.
2. **Match score** measures how well the candidate's skills and projects fit a job.

Company or role preference never increases the match percentage.

---

## Result flow

```text
Candidate profile
  ├─ selected company (optional)
  ├─ requested role (optional)
  ├─ skills
  └─ projects
        │
        ▼
Retrieve candidates from company, role, skills, and projects
        │
        ▼
Calculate requiredSkillScore and projectEvidenceScore
        │
        ▼
Reserve result slots by preference tier
        │
        ▼
Display each tier ordered by matchScore
```

Experience remains a hard eligibility filter. Education is not included in the
initial match score.

---

## Database tables and embeddings

The target model uses five tables. Skills are deduplicated into a shared `Skill`
catalog because the system must scale to millions of jobs: distinct skill tokens
number in the tens of thousands, not jobs × 7. Each distinct skill is glossed and
embedded exactly once, and per-job links are thin join rows.

```text
Company
  └── Job
       ├── JobSkill ──► Skill   (deduped catalog: gloss + embedding)
       └── JobCapability
```

### `Company`

Stores the company name and its jobs.

```text
Company
  id
  name
```

No company embedding is stored. Company search remains exact-name followed by
trigram matching.

### `Job`

Stores the main job posting.

```text
Job
  id
  companyId
  jobTitle
  roleSummary
  requiredSkills
  keyResponsibilities
  experienceMinYears
  experienceMaxYears
  roleEmbedding vector(512)
```

`roleEmbedding` is generated from:

```text
job title + role type + role summary
```

It is used for semantic role-title retrieval, such as:

```text
"Data Science" → "Data Science Analyst"
"SDE"          → "Software Engineer"
```

It is not used directly as the displayed match percentage.

### `Skill` (deduped catalog) and `JobSkill` (thin join)

```text
Skill
  id
  token       -- normalized: lowercased, [^a-z0-9] stripped, unique
  label       -- display name as first seen, e.g. "Machine learning"
  gloss       -- LLM-generated one-line description; the embedding input
  embedding vector(512)   -- embed(gloss), never embed(token)

JobSkill
  jobId
  skillId
  -- composite primary key, no vector; millions of rows stay cheap
```

Example:

```text
Data Science Analyst
  ├─ Python
  ├─ C
  ├─ C++
  ├─ Machine learning
  ├─ Data analytics
  └─ Communication
```

Each skill embedding is generated separately. This allows candidate skills such as
`Data Visualisation` or `Data Cleansing` to provide semantic evidence for
`Data analytics` without pretending they match unrelated requirements such as `C++`.

#### Why glosses are mandatory

Bare-token Titan embeddings are not separable — measured on Titan v2 (512-dim):

```text
bare tokens:   aws ↔ cloud computing        0.23   (true synonym)
               c++ ↔ python                 0.26   (unrelated!)
               machine learning ↔ deep learning 0.29

glossed:       aws ↔ cloud computing        0.62
               communication ↔ team collaboration 0.43
               kafka ↔ rabbitmq             0.41
               data analytics ↔ data visualisation 0.40
               noise pairs                  0.00–0.15
```

With bare tokens the signal band overlaps the noise band, so no threshold can both
accept `aws ↔ cloud computing` and reject `c++ ↔ python`. With glosses the bands
separate cleanly. A gloss is one short LLM-written line, e.g.:

```text
token: "aws"
gloss: "AWS (Amazon Web Services): cloud computing platform, cloud infrastructure services"
```

#### Why dedupe (scale)

At millions of jobs, per-(job, skill) vectors mean re-glossing and re-embedding
`python` millions of times and storing duplicate vectors. With the catalog,
semantic similarity is computed once per (candidate skill × distinct skill); job
coverage afterwards is plain set membership over indexed `JobSkill` joins — no
per-job vector math for skills. New jobs only pay for tokens never seen before.

### `JobCapability`

Stores one responsibility or capability per row.

```text
JobCapability
  id
  jobId
  text
  embedding vector(512)
```

Capabilities are built from the **union** of the job fields that describe what
the role actually does and tests — not from `keyResponsibilities` alone:

```text
keyResponsibilities   → one capability per ';'-separated statement
roleSummary           → one capability (always, not only as fallback)
roundTechnical        → one capability (technical interview competencies)
```

`requiredSkills` is **excluded** from capabilities. Skills already own 65% of
the blend; including them here would make the formula effectively
"65% skills + 35% mixture of projects and skills", inflating candidates who
have matching skills but no project evidence. Project evidence must measure
projects only.

Example (Tiger Analytics — Data Science Analyst):

```text
  ├─ Engage with clients to understand their business context
  ├─ Translate business problems … into technical requirements
  ├─ Trainee analyst working on … data analytics and machine learning …   (roleSummary)
  └─ Python/C/C++ programming; Data analytics; Machine learning …         (roundTechnical)
```

The union is required, not optional: measured against a real ML project, Tiger's
two responsibilities score 0.04–0.10 (client-engagement statements — dead), while
the signal lives in roundTechnical (0.190) and roleSummary (0.127). With
responsibilities alone, project evidence is ~0 for this job. Known trade-off,
accepted deliberately: dead capabilities dilute the AVG.

Candidate project embeddings are compared with these individual capability
embeddings. This avoids diluting a relevant project by comparing it with one large
embedding containing the entire job description.

### Candidate embeddings

Candidate skill and project embeddings are generated at search time and cached by
the existing embedding cache. They are not permanently stored initially.

The resume parser emits each skill as `{ name, gloss }` in the same LLM call that
extracts it, and it **includes soft skills** (e.g. `Communication`,
`Team Collaboration`) so soft-skill job requirements are coverable. Candidate
skills never touch the `Skill` table — a brand-new skill works on day one because
its gloss is embedded at query time and compared by cosine; there is no catalog
lookup or taxonomy to maintain.

Profiles parsed before glosses existed fall back to keyword-only matching for
those skills (bare-token vectors are too noisy to trust).

---

## Candidate retrieval

A job can enter the candidate set through any of these paths:

```text
company match
OR role match
OR required-skill match
OR project-to-capability match
```

Role matching keeps the existing hybrid approach:

```text
exact title ∪ trigram title ∪ role embedding
```

Skill retrieval resolves covered skills against the `Skill` catalog first
(small, independent of job count), then joins:

```text
coveredSkillIds = Skill rows WHERE token matches a candidate token
                  OR max cosine(Skill.embedding, candidate skill vectors) >= SEMANTIC_SKILL_MIN
skill path      = top ~1,000 jobs with ≥1 JobSkill.skillId IN coveredSkillIds,
                  ordered by required-skill coverage (covered / required)
```

The skill path is **bounded**: a common skill such as `Python` alone could match
tens of thousands of jobs, so the shortlist is capped (500–1,000) and ordered by
coverage ratio before full scoring. The candidate set is first narrowed by a
semi-join to jobs with ≥1 covered skill (via the `JobSkill.skillId` index), then
the coverage ordering is a GROUP BY over the indexed `JobSkill` join with the
experience hard filter applied — this aggregation-before-LIMIT is intentional
and must not be optimized away, or the cap would keep arbitrary low-coverage
jobs.

Project retrieval uses HNSW ANN over `JobCapability.embedding`, **capped top-K
per project vector** (like the title tier's top-20 cap) so a permissive
similarity can never flood the candidate set:

```text
candidate-project ↔ JobCapability ANN, top-K (~50) per project vector
```

Experience remains a hard filter:

```text
candidate years must fit the job's experience range
```

---

## Match score

Every candidate job receives the same suitability score:

```text
matchScore =
  65% requiredSkillScore
  35% projectEvidenceScore
```

### Required skill score

For each required job skill:

```text
exact candidate skill match      → full coverage
semantic candidate skill match   → coverage when above calibrated threshold
no supported match               → not covered
```

```text
requiredSkillScore =
  covered required skills / total required skills × 100
```

Coverage is computed as set membership (`JobSkill.skillId IN coveredSkillIds`),
so scoring cost does not grow with per-job vector comparisons.

The semantic threshold must be calibrated against real positive and negative skill
pairs. It must not be chosen from a single resume. Starting point from measured
gloss calibration: `SEMANTIC_SKILL_MIN = 0.30` (signal pairs 0.33–0.62, noise
0.00–0.15); validate with `scripts/probe-similarity.ts` before production.

### Project evidence score

Each candidate project is compared with each `JobCapability`.

```text
capabilityEvidence =
  best matching candidate project for that capability

projectEvidenceScore =
  average capabilityEvidence across the job's capabilities
```

Only similarities above the calibrated project-evidence threshold count as evidence.
The current `0.40` floor is not retained automatically: the Aswathy/Tiger Analytics
case showed relevant Titan similarities around `0.20–0.24`, while unrelated values
can also be positive. Calibration is required before selecting the production floor.

### Missing score components

The weights remain fixed:

```text
65% skills + 35% projects
```

If a resume contains no projects, `projectEvidenceScore` is `0`. The UI must show
the breakdown so users can understand why the match score is lower.

---

## Preference tiers and slot reservation

The default result list returns at most 30 jobs.

```text
Tier 0: selected company's jobs
Tier 1: other jobs matching requested role
Tier 2: jobs matching through skills/projects
```

A job belongs to its highest-priority matching tier:

```text
company + role match → Tier 0
role + skills match  → Tier 1
skills/projects only → Tier 2
```

### Default ordering

```text
Tier 0 company:       maximum 10 slots
Tier 1 role:          reserve min(10, available role jobs) slots first
Tier 2 personalized:  remaining slots
```

Within every tier:

```text
matchScore DESC, createdAt ASC
```

Tier 1 semantics, precisely: the first `min(10, available)` role jobs are
**reserved** — they can never be crowded out by Tier 2 — and additional role
jobs beyond 10 still compete in backfill by matchScore. Role jobs are neither
capped at 10 nor unconditionally guaranteed 10 slots when fewer exist.

Unused slots are backfilled by the next available highest-match jobs. The company
cap prevents a large company from consuming the entire result list. The role
reservation ensures requested-role jobs remain visible before personalized
skills/project-only jobs.

Example:

```text
Company = Tiger Analytics
Role    = Data Science

1. Tiger Analytics jobs, ordered by matchScore          (up to 10)
2. Other Data Science role matches, ordered by score    (at least 10 if available)
3. Remaining skills/project matches, ordered by score
```

### Match-score ordering

The optional `score` sort ignores preference tiers:

```text
all candidate jobs ORDER BY matchScore DESC
```

---

## Aswathy acceptance case

Resume:

```text
/Users/mohammedhasan/Downloads/Resume Aswathy B - Aswathy Balan.pdf
```

Target job:

```text
Tiger Analytics — Data Science Analyst
cmq7qxif80009h605q5si77dg
```

The current implementation produces `17%` because:

```text
required skills: 2 / 6 = 33%
project score:           0%
old equal blend:         17%
```

The target implementation must:

- preserve exact matches for `Python` and `Machine learning`;
- recognize only defensible semantic skill relationships, such as evidence toward
  `Data analytics` (via `Data Visualisation`/`SQL`) and `Communication` (via the
  parser's soft-skill extraction, e.g. `Team Collaboration`, gloss cosine 0.43);
- not falsely match `C` or `C++` (expected requiredSkillScore ≈ 4/6 ≈ 67%);
- compare the machine-learning project with individual job capabilities;
- calculate the final score using `65% requiredSkillScore + 35% projectEvidenceScore`;
- keep Tiger Analytics jobs in Tier 0 when the company is selected;
- keep Data Science role matches in Tier 1 when the role is entered.

The acceptance target is not a hardcoded percentage. The breakdown must be
explainable and clearly higher than the current broken `17%` when calibrated
semantic evidence supports it.

Verified with `scripts/match-score-probe.ts` (real glosses + embeddings,
starting thresholds `SEMANTIC_SKILL_MIN = 0.30`, evidence window `0.10–0.35`,
requiredSkills excluded from capabilities):
requiredSkillScore 66.7 (4/6), projectEvidenceScore 12.0, **matchScore 48%**.

---

## Deferred work

Intentionally excluded from the initial match-score implementation:

- storing candidate/resume embeddings (currently embedded at query time and
  held only in the in-memory LRU cache);
- education scoring;
- work-experience scoring beyond the hard eligibility filter;
- Bedrock Rerank;
- learned recommendation models;
- calibrated probability-of-hire claims.

---

## Main implementation files

| File | Responsibility |
|---|---|
| `prisma/schema.prisma` | Add `Skill`, `JobSkill`, `JobCapability`; repurpose role embedding |
| `prisma/migrations/*` | Create vector tables and HNSW indexes |
| `prisma/import-jobs.ts` | Split skills, gloss + embed new `Skill` tokens, embed capabilities |
| `prisma/seed-jd-files.ts` | Seed the same target embedding data |
| `scripts/backfill-job-match-embeddings.ts` | Resumable backfill for existing jobs |
| `lib/embeddings.ts` | Generate and cache Titan embeddings (batch helpers) |
| `lib/resume.ts` | Parser emits `{ name, gloss }` skills including soft skills |
| `lib/onboarding.ts` | Pass glossed skills + project texts through `SearchInput` |
| `lib/search.ts` | Resolve covered skills, retrieve candidates, score, reserve tier slots |
| `lib/types.ts` | Return match-score breakdown |
| `app/api/search/route.ts` | Validate search input |
| `app/api/jobs/[jobId]/match/route.ts` | Calculate one job's candidate-relative score |
| `app/page.tsx` | Display the score breakdown |

Note: glossing makes `OPENROUTER_API_KEY` an import-time dependency
(`import-jobs.ts` / backfill were previously Bedrock-only).
