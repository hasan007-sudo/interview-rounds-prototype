// Group 6 — Combined inputs
// Tests role+company, role+skills, role+experience, and all-four together.
//
// All retrieval paths are kicked off concurrently (title tiers included), so
// the $queryRaw calls interleave in deterministic microtask-FIFO order: the
// synchronous kickoffs first (title exact, title trigram, company exact), then
// the awaited chains (title vector after embed; covered skills → skill path):
//
// role+company (no skills):
//   CALL 1 → title exact        CALL 2 → title trigram
//   CALL 3 → company exact      CALL 4 → title vector ANN
//   LAST   → final scoring SQL
//
// role+skills (no company):
//   CALL 1 → title exact        CALL 2 → title trigram
//   CALL 3 → title vector ANN   CALL 4 → covered-skill resolution
//   CALL 5 → skill-path candidates
//   LAST   → final scoring SQL

import { describe, it, expect, beforeEach, type Mock } from "vitest";
import { vi } from "vitest";
import { searchJobs } from "../search";
import { prisma } from "../prisma";
import { embed, toPgVectorLiteral } from "../embeddings";
import {
  FIXED_VEC,
  FIXED_VEC_LIT,
  makeRow,
  makeMatch,
  makeCompanyExact,
  makeSkillId,
  makeSkillJob,
} from "./helpers";

vi.mock("../prisma", () => ({ prisma: { $queryRaw: vi.fn() } }));
vi.mock("../embeddings", () => ({ embed: vi.fn(), toPgVectorLiteral: vi.fn() }));

const q = () => prisma.$queryRaw as Mock;

beforeEach(() => {
  vi.clearAllMocks();
  (embed as Mock).mockResolvedValue(FIXED_VEC);
  (toPgVectorLiteral as Mock).mockReturnValue(FIXED_VEC_LIT);
});

describe("role + company", () => {
  it("returns jobs that matched on role or company (role/company tier)", async () => {
    // role and company both contribute candidate ids to the UNION; tier < 2
    // maps to roleOrCompanyMatched.
    q()
      .mockResolvedValueOnce([makeMatch("job-1", 1.0)]) // title exact
      .mockResolvedValueOnce([]) // title trigram
      .mockResolvedValueOnce([makeCompanyExact("company-1")]) // company exact
      .mockResolvedValueOnce([]) // title vector
      .mockResolvedValueOnce([
        makeRow({ jobId: "job-1", companyName: "Google", tier: 0 }),
      ]); // final

    const result = await searchJobs({
      roleText: "Engineer",
      companyText: "Google",
      skills: [],
      experienceMinYears: null,
      experienceMaxYears: null,
    });

    expect(result).toHaveLength(1);
    expect(result[0]!.companyName).toBe("Google");
    expect(result[0]!.roleOrCompanyMatched).toBe(true);
  });

  it("makes exactly 5 $queryRaw calls for role+company (exact company match)", async () => {
    // Exact company hit short-circuits the company trigram fallback. Call
    // sequence: title-exact, title-trigram, company-exact, title-vector,
    // final = 5 calls.
    q()
      .mockResolvedValueOnce([makeMatch("job-1", 1.0)]) // title exact
      .mockResolvedValueOnce([]) // title trigram
      .mockResolvedValueOnce([makeCompanyExact("company-1")]) // company exact
      .mockResolvedValueOnce([]) // title vector
      .mockResolvedValueOnce([makeRow()]); // final

    await searchJobs({
      roleText: "Engineer",
      companyText: "Acme",
      skills: [],
      experienceMinYears: null,
      experienceMaxYears: null,
    });

    expect(q()).toHaveBeenCalledTimes(5);
  });
});

describe("role + skills", () => {
  it("resolves covered skills, runs 3 title tiers + skill path + final, and maps coverage through", async () => {
    q()
      .mockResolvedValueOnce([makeMatch("job-1", 1.0)]) // title exact
      .mockResolvedValueOnce([]) // title trigram
      .mockResolvedValueOnce([]) // title vector
      .mockResolvedValueOnce([makeSkillId("skill-react")]) // covered skills
      .mockResolvedValueOnce([makeSkillJob("job-1")]) // skill-path candidates
      .mockResolvedValueOnce([
        makeRow({ covered: 1, required: 2, skillsPct: 50, score: 33 }),
      ]); // final

    const result = await searchJobs({
      roleText: "Frontend Engineer",
      companyText: "",
      skills: [{ name: "React" }],
      experienceMinYears: null,
      experienceMaxYears: null,
    });

    expect(q()).toHaveBeenCalledTimes(6);
    expect(result[0]!.matchedSkills).toBe(1);
    expect(result[0]!.totalSkills).toBe(2);
    expect(result[0]!.skillsPct).toBe(50);
  });
});

describe("role + experience", () => {
  it("does not apply an experience filter when both range bounds are null", async () => {
    // min=max=null → the overlap predicate is skipped in SQL. JS-side this
    // is just a passthrough path; verify it still runs the role-only call sequence.
    q()
      .mockResolvedValueOnce([makeMatch("job-1", 1.0)])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([makeRow({ jobId: "job-1" })]);

    const result = await searchJobs({
      roleText: "Software Engineer",
      companyText: "",
      skills: [],
      experienceMinYears: null,
      experienceMaxYears: null,
    });

    expect(result).toHaveLength(1);
    expect(result[0]!.jobId).toBe("job-1");
  });
});

describe("all four inputs combined", () => {
  it("makes 7 calls and returns a single ranked result for role+company+skills+experience", async () => {
    // title-exact, title-trigram, company-exact sync; then title-vector,
    // covered-skills, skill-path, final.
    q()
      .mockResolvedValueOnce([makeMatch("job-1", 1.0)]) // title exact
      .mockResolvedValueOnce([]) // title trigram
      .mockResolvedValueOnce([makeCompanyExact("company-1")]) // company exact
      .mockResolvedValueOnce([]) // title vector
      .mockResolvedValueOnce([makeSkillId("skill-react")]) // covered skills
      .mockResolvedValueOnce([makeSkillJob("job-1")]) // skill-path candidates
      .mockResolvedValueOnce([makeRow({ jobId: "job-1" })]); // final

    const result = await searchJobs({
      roleText: "Software Engineer",
      companyText: "Acme",
      skills: [{ name: "React" }, { name: "TypeScript" }],
      experienceMinYears: 3,
      experienceMaxYears: 3,
    });

    expect(q()).toHaveBeenCalledTimes(7);
    expect(result).toHaveLength(1);
  });
});
