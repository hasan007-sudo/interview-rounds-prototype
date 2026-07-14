// Job-centric search. Experience stays a hard filter, and the candidate set is
// the UNION of company ∪ role ∪ skill ∪ project(capability) matched jobs. Every
// candidate gets a preference tier (0 company, 1 role, 2 personalized) which
// orders the result groups; the match % never includes preference.
//
//   matchScore = 65% requiredSkillScore + 35% projectEvidenceScore
//
//   • requiredSkillScore — % of the job's required skills covered by the
//     candidate. Coverage is resolved ONCE per search against the deduped
//     Skill catalog (exact normalized token OR gloss-embedding cosine ≥
//     SEMANTIC_SKILL_MIN); per-job scoring is then set membership over the
//     indexed JobSkill join — no per-job vector math.
//   • projectEvidenceScore — AVG over the job's capabilities of the best
//     project↔capability cosine, rescaled through the evidence window.
//
// Retrieval is bounded on every path: title ANN top-20, skill path top
// MAX_SKILL_CANDIDATES by coverage ratio, capability ANN top
// MAX_CAPABILITY_MATCHES per project vector.
// Rounds are parsed at read time from focusRoundPattern.

import { Prisma } from "@prisma/client";
import { prisma } from "./prisma";
import { embed, toPgVectorLiteral } from "./embeddings";
import { glossSkills, normalizeSkillToken } from "./glosses";
import { parseRounds } from "./rounds";
import type { JobCard, Seniority } from "./types";

const MIN_TITLE_TRIGRAM_SIMILARITY = 0.3;
const MIN_COMPANY_TRIGRAM_SIMILARITY = 0.4;
const MAX_TITLE_MATCHES_PER_TIER = 20;
const MAX_RESULTS = 30;

// Blend weights (fixed): skills carry the score, projects support it.
const SKILL_WEIGHT = 0.65;
const PROJECT_WEIGHT = 0.35;

// Gloss-embedding cosine floor for a candidate skill to cover a job skill.
// Calibrated on glossed Titan pairs: signal 0.33–0.62, noise 0.00–0.15
// (see docs/ARCHITECTURE.md). Validate with scripts/probe-similarity.ts.
const SEMANTIC_SKILL_MIN = 0.3;

// Project↔capability evidence window: cosine below MIN counts as zero evidence,
// MIN..MAX rescales linearly to 0–100. Measured noise 0.04–0.10, signal 0.13+.
const PROJECT_EVIDENCE_MIN = 0.1;
const PROJECT_EVIDENCE_MAX = 0.35;

// Bounded retrieval: a common covered skill ("python") could match tens of
// thousands of jobs, so the skill path shortlists by coverage ratio. The
// capability path is ANN top-K per project vector (mirrors the title tier cap).
const MAX_SKILL_CANDIDATES = 1000;
const MAX_CAPABILITY_MATCHES = 50;

// Default-sort slot reservation: Tier 0 (selected company) is hard-capped;
// Tier 1 (requested role) gets min(TIER1_RESERVE, available) reserved slots,
// extras compete in backfill by matchScore.
const TIER0_CAP = 10;
const TIER1_RESERVE = 10;

function normalize(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s+#./-]/g, "");
}

export function deriveSeniority(experienceYears: number | null): Seniority {
  if (experienceYears === null) return "entry";
  if (experienceYears <= 2) return "entry";
  if (experienceYears <= 6) return "mid";
  return "senior";
}

// Title candidates: the UNION of exact, trigram, and semantic (vector) tiers, so a
// generic query like "Software Engineer" also surfaces related real titles. Title
// only decides membership (it never ranks the result), so we return ids only — the
// per-tier similarity lives in each tier's ORDER BY to pick the top MAX_TITLE_MATCHES_PER_TIER.
async function matchTitleIds(text: string): Promise<string[]> {
  const norm = normalize(text);
  if (!norm) return [];

  // The three tiers are independent — run them concurrently.
  const [exact, trigram, vector] = await Promise.all([
    // Tier 1 — exact: case-folded title equality.
    prisma.$queryRaw<{ id: string }[]>`
      SELECT id FROM "Job"
      WHERE lower("jobTitle") = ${norm}
      LIMIT ${MAX_TITLE_MATCHES_PER_TIER}
    `,
    // Tier 2 — trigram: pg_trgm fuzzy match (catches typos/partials). `%` uses the
    // GIN index; keep only rows above the similarity floor, most similar first.
    prisma.$queryRaw<{ id: string }[]>`
      SELECT id FROM "Job"
      WHERE "jobTitle" % ${text} AND similarity("jobTitle", ${text}) >= ${MIN_TITLE_TRIGRAM_SIMILARITY}
      ORDER BY similarity("jobTitle", ${text}) DESC
      LIMIT ${MAX_TITLE_MATCHES_PER_TIER}
    `,
    // Tier 3 — vector: semantic ANN (catches synonyms like "SDE" ↔ "Software
    // Engineer"). `<=>` is cosine distance over the HNSW index; nearest first.
    embed(text).then((vec) => {
      const vecLit = toPgVectorLiteral(vec);
      return prisma.$queryRaw<{ id: string }[]>`
        SELECT id FROM "Job"
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> ${vecLit}::vector
        LIMIT ${MAX_TITLE_MATCHES_PER_TIER}
      `;
    }),
  ]);

  const ids = new Set<string>();
  for (const rows of [exact, trigram, vector]) rows.forEach((r) => ids.add(r.id));
  return [...ids];
}

// Company candidates: exact then trigram. No vector — company names are typed near-exact.
async function matchCompanyNameIds(text: string): Promise<string[]> {
  const norm = normalize(text);
  if (!norm) return [];

  // Tier 1 — exact: case-folded name equality. If it hits, trust it and stop.
  const exact = await prisma.$queryRaw<{ id: string }[]>`
    SELECT id FROM "Company" WHERE lower(name) = ${norm} LIMIT 5
  `;
  if (exact.length > 0) return exact.map((r) => r.id);

  // Tier 2 — trigram: fuzzy fallback, best first. Caller keeps it only if the
  // top score clears MIN_COMPANY_TRIGRAM_SIMILARITY (short names → strict floor).
  const trigram = await prisma.$queryRaw<{ id: string; score: number }[]>`
    SELECT id, similarity(name, ${text})::float AS score FROM "Company"
    WHERE name % ${text}
    ORDER BY score DESC
    LIMIT 5
  `;
  if (trigram.length > 0 && trigram[0].score >= MIN_COMPANY_TRIGRAM_SIMILARITY) {
    return trigram.map((r) => r.id);
  }
  return [];
}

async function matchCompanyIds(text: string): Promise<string[]> {
  const companyNames = text
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);

  const matches = await Promise.all(companyNames.map(matchCompanyNameIds));
  return [...new Set(matches.flat())];
}

export type SortMode = "default" | "score";

// One candidate skill. The gloss (one-line description from the resume parser)
// is the embedding input for semantic coverage; skills without one — manually
// typed or from pre-gloss profiles — match by exact normalized token only
// (bare-token embeddings are too noisy to trust).
export type SkillQuery = { name: string; gloss?: string | null };

export type SearchInput = {
  companyText: string;
  roleText: string;
  skills: SkillQuery[];
  experienceMinYears: number | null;
  experienceMaxYears: number | null;
  // Per-project evidence strings (description + keywords). Each is embedded and
  // compared against JobCapability vectors.
  projectTexts?: string[];
  // "default" tiers company/role matches first; "score" ranks purely by matchScore.
  sort?: SortMode;
};

type ScoreRow = {
  jobId: string;
  jobTitle: string;
  companyName: string;
  experienceMinYears: number | null;
  experienceMaxYears: number | null;
  focusRoundPattern: string;
  covered: number;
  required: number;
  skillsPct: number | null;
  projectsPct: number | null;
  score: number | null;
  tier: number;
};

// Guardrails for gloss-on-miss: a public search API can send junk strings;
// cap what we'll gloss + store per request so the catalog can't be flooded.
const MAX_NEW_GLOSSES_PER_REQUEST = 20;
const MAX_SKILL_NAME_LENGTH = 60;

// Gloss-on-miss: glossless skills whose tokens the catalog has never seen get
// glossed (one batched LLM call), embedded, and upserted into the Skill
// catalog WITHOUT JobSkill links — link-less rows are invisible to scoring
// denominators, so the catalog doubles as a global gloss/embedding cache.
// First request mentioning a new skill pays the gloss+embed latency; every
// later request (any caller) is a catalog hit. Failures degrade to exact-only
// matching for those skills — search must not 500 because OpenRouter is down.
async function ensureCatalogGlosses(skills: SkillQuery[]): Promise<void> {
  const labelByToken = new Map<string, string>();
  for (const s of skills) {
    const name = s.name.trim();
    if (s.gloss?.trim() || !name || name.length > MAX_SKILL_NAME_LENGTH) continue;
    const token = normalizeSkillToken(name);
    if (token && !labelByToken.has(token)) labelByToken.set(token, name);
  }
  if (labelByToken.size === 0) return;

  try {
    const existing = await prisma.skill.findMany({
      where: { token: { in: [...labelByToken.keys()] } },
      select: { token: true },
    });
    for (const e of existing) labelByToken.delete(e.token);

    const entries = [...labelByToken.entries()].slice(0, MAX_NEW_GLOSSES_PER_REQUEST);
    if (entries.length === 0) return;

    const glosses = await glossSkills(entries.map(([, label]) => label));
    for (const [token, label] of entries) {
      const gloss = glosses.get(label) ?? label;
      const vec = toPgVectorLiteral(await embed(gloss));
      // Upsert: token is unique, so a concurrent request creating the same
      // skill is safe — the loser reuses the winner's row.
      await prisma.skill.upsert({
        where: { token },
        create: { token, label, gloss },
        update: {},
      });
      await prisma.$executeRawUnsafe(
        `UPDATE "Skill" SET embedding = $1::vector WHERE token = $2 AND embedding IS NULL`,
        vec,
        token,
      );
    }
  } catch (err) {
    console.warn("gloss-on-miss failed; affected skills match exact-only:", err);
  }
}

// Resolve which catalog skills the candidate covers. A catalog skill is
// covered when any of:
//   1. exact — its token equals a candidate token;
//   2. gloss semantic — its embedding is ≥ SEMANTIC_SKILL_MIN close to a
//      candidate skill's gloss embedding (resume-parsed skills carry a gloss);
//   3. catalog-fallback semantic — for candidate skills WITHOUT a gloss
//      (manually typed chips, pre-gloss profiles): if the typed token exists
//      in the catalog, that row's stored gloss embedding stands in as the
//      query vector (self-join). Free — no LLM or Bedrock call.
// We never embed a bare skill name: bare-token vectors are too noisy to
// threshold (see docs/ARCHITECTURE.md). Catalog-sized work, independent of
// job count; per-job coverage afterwards is set membership.
async function resolveCoveredSkillIds(skills: SkillQuery[]): Promise<string[]> {
  const tokens = skills
    .map((s) => normalizeSkillToken(s.name))
    .filter(Boolean);
  if (tokens.length === 0) return [];

  // Gloss + cache catalog-new glossless skills first, so the catalog-fallback
  // branch below can pick them up in the same request.
  await ensureCatalogGlosses(skills);

  const glosses = [...new Set(skills.map((s) => s.gloss?.trim()).filter((g): g is string => Boolean(g)))];
  const vecLits = await Promise.all(glosses.map((g) => embed(g).then(toPgVectorLiteral)));
  const hasVecs = vecLits.length > 0;

  const glosslessTokens = skills
    .filter((s) => !s.gloss?.trim())
    .map((s) => normalizeSkillToken(s.name))
    .filter(Boolean);

  // Materialized CTEs so each query vector is cast/fetched ONCE, not re-parsed
  // per catalog row inside the correlated EXISTS. Empty arrays → empty CTEs →
  // EXISTS is false, so no separate guards are needed.
  const rows = await prisma.$queryRaw<{ id: string }[]>`
    WITH cand_vecs AS MATERIALIZED (
      SELECT v::vector AS vec FROM unnest(${vecLits}::text[]) v
    ),
    fallback_vecs AS MATERIALIZED (
      SELECT t.embedding AS vec FROM "Skill" t
      WHERE t.token = ANY(${glosslessTokens}::text[]) AND t.embedding IS NOT NULL
    )
    SELECT s.id FROM "Skill" s
    WHERE s.token = ANY(${tokens}::text[])
       OR (
         ${hasVecs} AND s.embedding IS NOT NULL AND EXISTS (
           SELECT 1 FROM cand_vecs v
           WHERE (1 - (s.embedding <=> v.vec)) >= ${SEMANTIC_SKILL_MIN}
         )
       )
       OR (
         s.embedding IS NOT NULL AND EXISTS (
           SELECT 1 FROM fallback_vecs t
           WHERE (1 - (s.embedding <=> t.vec)) >= ${SEMANTIC_SKILL_MIN}
         )
       )
  `;
  return rows.map((r) => r.id);
}

// Bounded skill-path retrieval: top MAX_SKILL_CANDIDATES jobs having ≥1 covered
// skill, ordered by coverage ratio. The semi-join narrows to jobs with ≥1
// covered skill FIRST (via the skillId index), so the coverage-ratio
// aggregation runs only over qualifying jobs instead of the whole table; the
// aggregation-before-LIMIT is still intentional — without it the cap would
// keep arbitrary low-coverage jobs.
async function matchSkillJobIds(
  coveredSkillIds: string[],
  experienceMinYears: number | null,
  experienceMaxYears: number | null,
): Promise<string[]> {
  if (coveredSkillIds.length === 0) return [];
  const rows = await prisma.$queryRaw<{ jobId: string }[]>`
    SELECT js."jobId"
    FROM "JobSkill" js
    JOIN "Job" j ON j.id = js."jobId"
    WHERE js."jobId" IN (
        SELECT "jobId" FROM "JobSkill" WHERE "skillId" = ANY(${coveredSkillIds}::text[])
      )
      AND (
        (${experienceMinYears}::int IS NULL AND ${experienceMaxYears}::int IS NULL)
        OR (
          COALESCE(${experienceMinYears}::int, 0) <= COALESCE(j."experienceMaxYears", 99)
          AND COALESCE(${experienceMaxYears}::int, 99) >= COALESCE(j."experienceMinYears", 0)
        )
      )
    GROUP BY js."jobId"
    ORDER BY COUNT(*) FILTER (WHERE js."skillId" = ANY(${coveredSkillIds}::text[]))::float / COUNT(*) DESC
    LIMIT ${MAX_SKILL_CANDIDATES}
  `;
  return rows.map((r) => r.jobId);
}

// Bounded project-path retrieval: ANN top-K capabilities per project vector
// (HNSW, one concurrent query per vector), unioned and deduped to job ids.
async function matchProjectJobIds(projVecLits: string[]): Promise<string[]> {
  const perVec = await Promise.all(
    projVecLits.map(
      (vec) => prisma.$queryRaw<{ jobId: string }[]>`
        SELECT DISTINCT "jobId" FROM (
          SELECT "jobId" FROM "JobCapability"
          WHERE embedding IS NOT NULL
          ORDER BY embedding <=> ${vec}::vector
          LIMIT ${MAX_CAPABILITY_MATCHES}
        ) nearest
      `,
    ),
  );
  const ids = new Set<string>();
  for (const rows of perVec) rows.forEach((r) => ids.add(r.jobId));
  return [...ids];
}

// Materialized CTE holding the candidate's project vectors, cast from text
// literals ONCE per query. The scoring lateral references it per capability
// row — without this, every 512-dim literal would be re-parsed for every
// (candidate job × capability × project) combination. Every query embedding
// scoringLaterals must also embed this CTE.
function projVecsCte(projVecLits: string[]) {
  return Prisma.sql`
    proj_vecs AS MATERIALIZED (
      SELECT v::vector AS vec FROM unnest(${projVecLits}::text[]) v
    )
  `;
}

// Shared scoring fragment: requiredSkillScore via JobSkill set membership,
// projectEvidenceScore via AVG of per-capability best project cosine rescaled
// through the evidence window. Used by both searchJobs and scoreJobMatch so the
// list and the job page can never disagree. Expects a `proj_vecs` CTE
// (projVecsCte) in the surrounding query.
function scoringLaterals(coveredSkillIds: string[], projVecLits: string[]) {
  const hasProjVecs = projVecLits.length > 0;
  return Prisma.sql`
    -- Lateral 1: "cov" — required-skill coverage for job j.
    -- Pure set membership against the pre-resolved coveredSkillIds — no
    -- vector math here. LEFT join: a job with no JobSkill rows still
    -- produces a row (required = 0) instead of dropping out of results.
    LEFT JOIN LATERAL (
      SELECT
        COUNT(*)::int AS required,                                               -- job's total required skills
        COUNT(*) FILTER (WHERE js."skillId" = ANY(${coveredSkillIds}::text[]))::int AS covered  -- how many the candidate covers
      FROM "JobSkill" js
      WHERE js."jobId" = j.id
    ) cov ON true
    -- Lateral 2: "pe" — project evidence for job j.
    -- Per capability: best = MAX cosine across the candidate's project
    -- vectors (the capability is evidenced by the MOST relevant project).
    -- Then rescale each best through the evidence window (MIN → 0, MAX → 100)
    -- and AVG across the job's capabilities (a project must cover the role's
    -- breadth, not just one line). LEFT join: no embedded capabilities or no
    -- project vectors → evidence NULL, job still returned.
    LEFT JOIN LATERAL (
      SELECT AVG(
        GREATEST(0, LEAST(1,
          (cap.best - ${PROJECT_EVIDENCE_MIN}) / ${PROJECT_EVIDENCE_MAX - PROJECT_EVIDENCE_MIN}  -- rescale window 0.10–0.35 → 0–1
        )) * 100
      )::float AS evidence
      FROM (
        SELECT (
          SELECT MAX((1 - (jc.embedding <=> pv.vec))::float)                     -- <=> is cosine distance; 1 - d = similarity
          FROM proj_vecs pv
        ) AS best
        FROM "JobCapability" jc
        WHERE ${hasProjVecs} AND jc."jobId" = j.id AND jc.embedding IS NOT NULL
      ) cap
    ) pe ON true
  `;
}

// Sub-scores + blend, computed from the lateral outputs. Both sub-scores are
// 0–100 (missing evidence scores 0 — fixed 65/35 weights, per ARCHITECTURE.md).
// Only when the candidate supplied neither skills nor projects is the score
// null (nothing to measure → badge shows "—").
function scoreProjection(hasSkills: boolean, hasProjVecs: boolean) {
  return Prisma.sql`
    CASE WHEN ${hasSkills} AND cov.required > 0
         THEN round(cov.covered::numeric / cov.required * 100)::int
         WHEN ${hasSkills} THEN 0
         ELSE NULL END AS "skillsPct",
    CASE WHEN ${hasProjVecs}
         THEN COALESCE(round(pe.evidence::numeric)::int, 0)
         ELSE NULL END AS "projectsPct",
    CASE WHEN NOT ${hasSkills} AND NOT ${hasProjVecs} THEN NULL
         ELSE round(
           ${SKILL_WEIGHT} * (CASE WHEN ${hasSkills} AND cov.required > 0
                                   THEN cov.covered::numeric / cov.required * 100
                                   ELSE 0 END)
           + ${PROJECT_WEIGHT} * COALESCE(
               CASE WHEN ${hasProjVecs} THEN pe.evidence ELSE 0 END, 0)
         )::int END AS score
  `;
}

export async function searchJobs(input: SearchInput): Promise<JobCard[]> {
  const sort = input.sort ?? "default";
  const skills = input.skills.filter((s) => s.name.trim());
  const projectTexts = (input.projectTexts ?? [])
    .map((t) => t.trim())
    .filter(Boolean);
  const hasSkills = skills.length > 0;
  const hasProjVecs = projectTexts.length > 0;

  // Nothing to constrain on → don't dump the whole table.
  if (!input.roleText && !input.companyText && !hasSkills && !hasProjVecs) {
    return [];
  }

  // Everything up to the scoring query is independent except for two chains
  // (project embeds → project path, covered skills → skill path), so the
  // title/company queries overlap with the Bedrock embeds and skill resolution
  // instead of waiting behind them.
  const projVecsPromise = Promise.all(
    projectTexts.map((t) => embed(t).then(toPgVectorLiteral)),
  );
  const coveredSkillIdsPromise = resolveCoveredSkillIds(skills);

  const [projVecLits, coveredSkillIds, titleIds, companyIds, skillJobIds, projectJobIds] =
    await Promise.all([
      projVecsPromise,
      coveredSkillIdsPromise,
      input.roleText ? matchTitleIds(input.roleText) : Promise.resolve([] as string[]),
      input.companyText ? matchCompanyIds(input.companyText) : Promise.resolve([] as string[]),
      coveredSkillIdsPromise.then((ids) =>
        matchSkillJobIds(ids, input.experienceMinYears, input.experienceMaxYears),
      ),
      projVecsPromise.then(matchProjectJobIds),
    ]);

  if (
    titleIds.length === 0 &&
    companyIds.length === 0 &&
    skillJobIds.length === 0 &&
    projectJobIds.length === 0
  ) {
    return [];
  }

  // Candidate set = union of the four paths (experience hard-filtered), scored
  // and tier-tagged in one query so the LIMIT is a correct top-N.
  //   raw    → union + laterals (coverage, project evidence) + sub-scores +
  //            the 65/35 blend + tier
  //   ranked → row number within tier (for the cap/reservation)
  //   pick   → default: Tier 0 hard-capped at TIER0_CAP, Tier 1 reserves
  //            min(TIER1_RESERVE, available), rest backfills by score;
  //            score: global top-N by matchScore.
  const rows = await prisma.$queryRaw<ScoreRow[]>`
    WITH ${projVecsCte(projVecLits)},
    raw AS (
      SELECT
        j.id AS "jobId",
        j."jobTitle",
        c.name AS "companyName",
        j."experienceMinYears",
        j."experienceMaxYears",
        j."focusRoundPattern",
        j."createdAt",
        COALESCE(cov.covered, 0) AS covered,
        COALESCE(cov.required, 0) AS required,
        ${scoreProjection(hasSkills, hasProjVecs)},
        CASE
          WHEN cardinality(${companyIds}::text[]) > 0 AND j."companyId" = ANY(${companyIds}::text[]) THEN 0
          WHEN cardinality(${titleIds}::text[]) > 0 AND j.id = ANY(${titleIds}::text[]) THEN 1
          ELSE 2
        END AS tier
      FROM "Job" j
      JOIN "Company" c ON c.id = j."companyId"
      ${scoringLaterals(coveredSkillIds, projVecLits)}
      WHERE (
          (${input.experienceMinYears}::int IS NULL AND ${input.experienceMaxYears}::int IS NULL)
          OR (
            COALESCE(${input.experienceMinYears}::int, 0) <= COALESCE(j."experienceMaxYears", 99)
            AND COALESCE(${input.experienceMaxYears}::int, 99) >= COALESCE(j."experienceMinYears", 0)
          )
        )
        AND (
          (cardinality(${companyIds}::text[]) > 0 AND j."companyId" = ANY(${companyIds}::text[]))
          OR (cardinality(${titleIds}::text[]) > 0 AND j.id = ANY(${titleIds}::text[]))
          OR (cardinality(${skillJobIds}::text[]) > 0 AND j.id = ANY(${skillJobIds}::text[]))
          OR (cardinality(${projectJobIds}::text[]) > 0 AND j.id = ANY(${projectJobIds}::text[]))
        )
    ),
    ranked AS (
      SELECT raw.*,
        ROW_NUMBER() OVER (
          PARTITION BY tier
          ORDER BY score DESC NULLS LAST, "createdAt" ASC
        ) AS rn_in_tier
      FROM raw
    ),
    pick AS (
      SELECT * FROM ranked
      WHERE ${sort} != 'default' OR tier != 0 OR rn_in_tier <= ${TIER0_CAP}
      ORDER BY
        CASE WHEN ${sort} = 'default' THEN
          CASE WHEN tier = 0 THEN 0
               WHEN tier = 1 AND rn_in_tier <= ${TIER1_RESERVE} THEN 1
               ELSE 2 END
          ELSE 0 END ASC,
        score DESC NULLS LAST,
        "createdAt" ASC
      LIMIT ${MAX_RESULTS}
    )
    SELECT
      "jobId", "jobTitle", "companyName",
      "experienceMinYears", "experienceMaxYears", "focusRoundPattern",
      covered, required, "skillsPct", "projectsPct", score, tier
    FROM pick
    ORDER BY
      CASE WHEN ${sort} = 'default' THEN tier ELSE 0 END ASC,
      score DESC NULLS LAST,
      "createdAt" ASC
  `;

  return rows.map((row) => {
    const rounds = parseRounds(row.focusRoundPattern);
    return {
      jobId: row.jobId,
      jobTitle: row.jobTitle,
      companyName: row.companyName,
      seniority: deriveSeniority(row.experienceMinYears),
      experienceMinYears: row.experienceMinYears,
      experienceMaxYears: row.experienceMaxYears,
      roundCount: rounds.length,
      rounds,
      score: row.score,
      skillsPct: row.skillsPct,
      projectsPct: row.projectsPct,
      roleOrCompanyMatched: row.tier < 2,
      matchedSkills: hasSkills ? row.covered : null,
      totalSkills: row.required,
    };
  });
}

export type JobMatch = {
  score: number | null;
  skillsPct: number | null;
  projectsPct: number | null;
  matchedSkills: number | null;
  totalSkills: number;
};

// Match score for a single job, using the exact same covered-skill resolution
// and scoring SQL as searchJobs, scoped to one job — the job detail page and
// the search list can never disagree. Returns null if the job doesn't exist.
export async function scoreJobMatch(
  jobId: string,
  input: { skills: SkillQuery[]; projectTexts?: string[] },
): Promise<JobMatch | null> {
  const skills = input.skills.filter((s) => s.name.trim());
  const projectTexts = (input.projectTexts ?? [])
    .map((t) => t.trim())
    .filter(Boolean);
  const hasSkills = skills.length > 0;
  const hasProjVecs = projectTexts.length > 0;

  const [projVecLits, coveredSkillIds] = await Promise.all([
    Promise.all(projectTexts.map((t) => embed(t).then(toPgVectorLiteral))),
    resolveCoveredSkillIds(skills),
  ]);

  const rows = await prisma.$queryRaw<
    {
      covered: number;
      required: number;
      skillsPct: number | null;
      projectsPct: number | null;
      score: number | null;
    }[]
  >`
    WITH ${projVecsCte(projVecLits)}
    SELECT
      COALESCE(cov.covered, 0) AS covered,
      COALESCE(cov.required, 0) AS required,
      ${scoreProjection(hasSkills, hasProjVecs)}
    FROM "Job" j
    ${scoringLaterals(coveredSkillIds, projVecLits)}
    WHERE j.id = ${jobId}
  `;

  const row = rows[0];
  if (!row) return null;
  return {
    score: row.score,
    skillsPct: row.skillsPct,
    projectsPct: row.projectsPct,
    matchedSkills: hasSkills ? row.covered : null,
    totalSkills: row.required,
  };
}
