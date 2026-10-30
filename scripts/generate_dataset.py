"""Seeded generator for the synthetic resume dataset and job descriptions.

Produces, under ``data/``:

- 36 resumes (6 roles x 6 people, a junior/mid/senior spread), split across
  PDF, DOCX, TXT and MD so Step 3's loader has to handle every format.
- 6 hand-written job descriptions.
- ``resume_specs.json``: per-resume ground truth (name, skills, years,
  education) used to measure Step 3's extraction accuracy.
- ``ground_truth.json``: graded relevance (0/1/2) of every resume against
  every job description, used to measure Step 4's retrieval accuracy.

Deliberately stresses the Step 3 extractor: header wording, date styles and
section order vary per resume; some skills appear only in a bullet rather
than the Skills line; a couple of candidates per role describe a skill in
prose with no matching taxonomy alias at all (a genuine paraphrase miss).
Every resume is round-tripped through M1's ``read_file`` before the script
exits, so a bad template can't silently produce something unreadable.

Run from the repo root: ``python scripts/generate_dataset.py``.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.shared import Pt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402
import m1  # noqa: E402

TODAY = date(2026, 1, 1)

DEGREE_LEVELS = {"Ph.D.": 4, "M.S.": 3, "M.A.": 3, "B.S.": 2, "B.A.": 2}

FIRST_NAMES = [
    "Olivia",
    "Liam",
    "Emma",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "Lucas",
    "Mia",
    "Benjamin",
    "Charlotte",
    "James",
    "Amelia",
    "Elijah",
    "Harper",
    "Oliver",
    "Evelyn",
    "Alexander",
    "Abigail",
    "Michael",
    "Emily",
    "Daniel",
    "Elizabeth",
    "Matthew",
    "Sofia",
    "Henry",
    "Avery",
    "Jackson",
    "Ella",
    "Sebastian",
    "Scarlett",
    "Jack",
    "Grace",
    "Owen",
    "Chloe",
    "Samuel",
    "Victoria",
    "Nathan",
    "Priya",
    "Wei",
    "Carlos",
    "Fatima",
    "Dmitri",
    "Aisha",
    "Hiroshi",
    "Ingrid",
    "Kwame",
    "Mei",
    "Diego",
    "Yuki",
    "Amara",
    "Luca",
    "Nadia",
    "Omar",
    "Sven",
    "Zara",
]
LAST_NAMES = [
    "Johnson",
    "Williams",
    "Brown",
    "Davis",
    "Miller",
    "Wilson",
    "Moore",
    "Taylor",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Garcia",
    "Martinez",
    "Robinson",
    "Clark",
    "Rodriguez",
    "Lewis",
    "Lee",
    "Walker",
    "Hall",
    "Allen",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Green",
    "Baker",
    "Adams",
    "Nelson",
    "Hill",
    "Campbell",
    "Mitchell",
    "Roberts",
    "Carter",
    "Phillips",
    "Evans",
    "Turner",
    "Parker",
    "Collins",
    "Edwards",
    "Stewart",
    "Flores",
    "Morris",
    "Nguyen",
    "Murphy",
    "Rivera",
    "Cook",
    "Patel",
    "Kim",
    "Chen",
    "Singh",
    "Khan",
    "Ali",
    "Novak",
    "Haddad",
    "Okafor",
]
COMPANIES = [
    "Brightwave Analytics",
    "Northstar Robotics",
    "Vertex Cloud Systems",
    "Lumen Data Labs",
    "Fathom AI",
    "Cobalt Logic",
    "Pinecrest Software",
    "Meridian Health Tech",
    "Axion Freight",
    "Cedarline Retail",
    "Harbor Fintech",
    "Quantum Byte",
    "Skyline Mobility",
    "Trailhead Robotics",
    "Oakmont Insurance Group",
    "Fern Analytics",
    "Granite Peak Systems",
    "Bluewing Media",
    "Solace Health",
    "Riverton Logistics",
    "Crestview Labs",
    "Marigold Commerce",
    "Orbital Devices",
    "Hollowbrook Energy",
    "Westbridge Capital",
    "Verdant Agritech",
    "Ironclad Security",
    "Nimbus Data",
    "Clearcut Media",
    "Pivotal Works",
]
AREA_CODES = [206, 312, 415, 512, 617, 702, 804, 919]

SUMMARY_HEADERS = ["Summary", "Profile", "Objective"]
EXPERIENCE_HEADERS = ["EXPERIENCE", "Work History", "Professional Experience:", "Employment"]
EDUCATION_HEADERS = ["Education", "EDUCATION"]
SKILLS_HEADERS = ["Skills", "Technical Skills", "Core Competencies", "SKILLS"]
PROJECTS_HEADER = "Projects"
CERTIFICATIONS_HEADER = "Certifications"


@dataclass
class JobEntry:
    title: str
    company: str
    start: date
    end: date | None  # None means "Present"
    bullets: list[str]
    skills_used: list[str]
    date_style: str  # "en_dash" | "short_dash" | "slash"


@dataclass
class CandidateSpec:
    candidate_id: str
    first_name: str
    last_name: str
    role: str
    seniority: str
    email: str = ""
    phone: str = ""
    skills: list[str] = field(default_factory=list)
    jobs: list[JobEntry] = field(default_factory=list)
    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    certifications: list[str] = field(default_factory=list)
    include_projects: bool = False
    inline_skills: bool = False
    explicit_years_stated: bool = False
    skills_header_word: str = "Skills"
    summary_header: str = "Summary"
    experience_header: str = "EXPERIENCE"
    education_header: str = "Education"
    summary_text: str = ""
    header_labels: dict[str, str] = field(default_factory=dict)
    file_format: str = "txt"
    skills_demo: str | None = None

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}"


ROLE_DEFS: dict[str, dict] = {
    "ml_engineer": {
        "title_base": "ML Engineer",
        "role_label": "ML Engineer",
        "core_skills": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "PyTorch",
            "Scikit-learn",
        ],
        "bullet_only_skill": "MLflow",
        "paraphrase_skill": "Computer Vision",
        "nice_extra": ["Spark", "NLP", "MLOps"],
        "degree_options": [
            ("B.S.", "Computer Science"),
            ("M.S.", "Machine Learning"),
            ("M.S.", "Computer Science"),
            ("Ph.D.", "Artificial Intelligence"),
        ],
        "certs": ["AWS Certified Machine Learning - Specialty", "TensorFlow Developer Certificate"],
        "summary_with_years": (
            "{title} with {years}+ years building and shipping machine learning systems "
            "in Python, from research prototypes through production deployment."
        ),
        "summary_no_years": (
            "{title} focused on building and deploying machine learning models in Python, "
            "with hands-on experience across the full model lifecycle."
        ),
        "bullets": [
            "Designed and trained deep learning models in PyTorch for a real-time "
            "recommendation system, improving click-through rate by 18%.",
            "Built data pipelines in Python to preprocess and validate training data at scale.",
            "Deployed TensorFlow models to production behind a REST API with automated "
            "retraining on a schedule.",
            "Used Scikit-learn to prototype and compare classical ML baselines before "
            "committing to a deep learning approach.",
            "Tracked experiments and model versions with MLflow across a team of four engineers.",
            "Partnered with product managers to translate business goals into measurable "
            "model objectives.",
            "Built systems that let machines interpret and classify photographs for a "
            "quality-inspection product.",
        ],
        "bullet_only_text": "Tracked experiments and model versions with MLflow across a "
        "team of four engineers.",
        "paraphrase_text": "Built systems that let machines interpret and classify "
        "photographs for a quality-inspection product.",
        "projects": [
            "Built an end-to-end fraud-detection pipeline, from feature engineering to a "
            "deployed PyTorch model.",
            "Trained a transformer-based text classifier and published the model card "
            "internally for reuse.",
        ],
    },
    "data_scientist": {
        "title_base": "Data Scientist",
        "role_label": "Data Scientist",
        "core_skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Statistics"],
        "bullet_only_skill": "Spark",
        "paraphrase_skill": "A/B Testing",
        "nice_extra": ["Machine Learning", "Deep Learning", "Tableau", "A/B Testing"],
        "degree_options": [
            ("B.S.", "Statistics"),
            ("M.S.", "Data Science"),
            ("M.S.", "Applied Mathematics"),
            ("Ph.D.", "Statistics"),
        ],
        "certs": [
            "Google Data Analytics Certificate",
            "Microsoft Certified: Azure Data Scientist Associate",
        ],
        "summary_with_years": (
            "{title} with {years}+ years turning SQL and Python analysis into decisions, "
            "using Pandas and Scikit-learn to model and explain customer behavior."
        ),
        "summary_no_years": (
            "{title} who pairs SQL and Python with statistical rigor to find and explain "
            "patterns in messy, real-world data."
        ),
        "bullets": [
            "Queried and joined large SQL tables to build analysis-ready datasets for "
            "weekly business reviews.",
            "Used Pandas and NumPy to clean and transform multi-source data before modeling.",
            "Built Scikit-learn models to predict customer churn, presenting findings with "
            "clear statistics to non-technical stakeholders.",
            "Ran large joins and aggregations on a Spark cluster when a dataset outgrew a "
            "single machine.",
            "Designed controlled experiments to compare product variants and measured the "
            "impact on conversion before any change shipped broadly.",
            "Built dashboards summarizing key metrics for the leadership team.",
        ],
        "bullet_only_text": "Ran large joins and aggregations on a Spark cluster when a "
        "dataset outgrew a single machine.",
        "paraphrase_text": "Designed controlled experiments to compare product variants "
        "and measured the impact on conversion before any change shipped broadly.",
        "projects": [
            "Built a churn-prediction model in Scikit-learn and tracked its lift in a "
            "subsequent experiment.",
            "Ran a cohort analysis in SQL and Pandas that reshaped the onboarding roadmap.",
        ],
    },
    "python_backend": {
        "title_base": "Backend Engineer",
        "role_label": "Python Backend Engineer",
        "core_skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "Git"],
        "bullet_only_skill": "Redis",
        "paraphrase_skill": "Microservices",
        "nice_extra": ["Flask", "MongoDB", "REST APIs", "Celery"],
        "degree_options": [
            ("B.S.", "Computer Science"),
            ("B.S.", "Software Engineering"),
            ("M.S.", "Computer Science"),
        ],
        "certs": ["AWS Certified Developer - Associate"],
        "summary_with_years": (
            "{title} with {years}+ years building Python services with FastAPI and Django "
            "on top of PostgreSQL, deployed with Docker."
        ),
        "summary_no_years": (
            "{title} building Python services with FastAPI and Django, backed by "
            "PostgreSQL and shipped in Docker containers."
        ),
        "bullets": [
            "Built and maintained REST endpoints in FastAPI serving millions of requests per day.",
            "Designed PostgreSQL schemas and wrote migrations for a growing Django monolith.",
            "Containerized services with Docker and wrote the Git branching workflow the "
            "team still uses.",
            "Cached expensive query results in Redis to cut p95 latency by half.",
            "Split a monolithic application into independently deployable services that "
            "scaled under load without a full rewrite.",
            "Wrote integration tests covering the most critical API endpoints.",
        ],
        "bullet_only_text": "Cached expensive query results in Redis to cut p95 latency by half.",
        "paraphrase_text": "Split a monolithic application into independently deployable "
        "services that scaled under load without a full rewrite.",
        "projects": [
            "Rebuilt a billing service in FastAPI with a PostgreSQL-backed ledger.",
            "Introduced Docker Compose for local development, cutting onboarding time "
            "from a day to an hour.",
        ],
    },
    "react_frontend": {
        "title_base": "Frontend Engineer",
        "role_label": "React Frontend Engineer",
        "core_skills": ["JavaScript", "TypeScript", "React", "HTML", "CSS", "Git"],
        "bullet_only_skill": "Redux",
        "paraphrase_skill": "Next.js",
        "nice_extra": ["Node.js", "GraphQL", "Jest"],
        "degree_options": [
            ("B.S.", "Computer Science"),
            ("B.A.", "Web Design"),
            ("B.S.", "Information Technology"),
        ],
        "certs": ["Meta Front-End Developer Certificate"],
        "summary_with_years": (
            "{title} with {years}+ years building React applications in TypeScript, from "
            "component libraries to full product surfaces."
        ),
        "summary_no_years": (
            "{title} building accessible, responsive React applications in TypeScript."
        ),
        "bullets": [
            "Built reusable React components in TypeScript shared across three product teams.",
            "Rewrote a jQuery page in React, cutting first-paint time by 40%.",
            "Wrote semantic HTML and CSS that kept the app usable with a screen reader.",
            "Managed complex UI state with Redux across a multi-step checkout flow.",
            "Built server-rendered pages that loaded fast and ranked well in search, "
            "without a separate backend team.",
            "Set up CI checks in Git so a broken build never reached the main branch.",
        ],
        "bullet_only_text": "Managed complex UI state with Redux across a multi-step "
        "checkout flow.",
        "paraphrase_text": "Built server-rendered pages that loaded fast and ranked well "
        "in search, without a separate backend team.",
        "projects": [
            "Built a design-system component library in React and TypeScript, adopted by "
            "two other teams.",
            "Migrated a legacy jQuery dashboard to React with no downtime.",
        ],
    },
    "devops_cloud": {
        "title_base": "DevOps Engineer",
        "role_label": "DevOps/Cloud Engineer",
        "core_skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux"],
        "bullet_only_skill": "Ansible",
        "paraphrase_skill": "Prometheus",
        "nice_extra": ["Jenkins", "Azure", "GCP", "Grafana"],
        "degree_options": [
            ("B.S.", "Computer Engineering"),
            ("B.S.", "Information Systems"),
            ("M.S.", "Cloud Computing"),
        ],
        "certs": [
            "AWS Certified Solutions Architect - Associate",
            "Certified Kubernetes Administrator (CKA)",
        ],
        "summary_with_years": (
            "{title} with {years}+ years running Kubernetes on AWS and codifying "
            "infrastructure in Terraform."
        ),
        "summary_no_years": (
            "{title} running containerized workloads on Kubernetes and AWS, with "
            "infrastructure defined in Terraform."
        ),
        "bullets": [
            "Ran production workloads on Kubernetes across multiple AWS regions.",
            "Wrote Terraform modules that stood up a full environment in one command.",
            "Built CI/CD pipelines that cut deploy time from an hour to under ten minutes.",
            "Administered a fleet of Linux hosts, automating patching with Ansible.",
            "Set up dashboards and alerts that caught production issues before customers "
            "noticed, cutting the average time to detect an outage in half.",
            "Containerized a legacy application with Docker so it could run on Kubernetes.",
        ],
        "bullet_only_text": "Administered a fleet of Linux hosts, automating patching "
        "with Ansible.",
        "paraphrase_text": "Set up dashboards and alerts that caught production issues "
        "before customers noticed, cutting the average time to detect an outage in half.",
        "projects": [
            "Migrated a fleet of EC2 instances to a Kubernetes cluster defined entirely in "
            "Terraform.",
            "Built a CI/CD pipeline that deploys to AWS on every merge to main.",
        ],
    },
    "data_analyst": {
        "title_base": "Data Analyst",
        "role_label": "Data Analyst",
        "core_skills": ["SQL", "Excel", "Tableau", "Python", "Statistics"],
        "bullet_only_skill": "Power BI",
        "paraphrase_skill": "dbt",
        "nice_extra": ["Power BI", "Looker", "ETL"],
        "degree_options": [
            ("B.A.", "Economics"),
            ("B.S.", "Business Analytics"),
            ("B.S.", "Statistics"),
            ("M.S.", "Business Analytics"),
        ],
        "certs": ["Google Data Analytics Certificate", "Tableau Desktop Specialist"],
        "summary_with_years": (
            "{title} with {years}+ years turning SQL queries and Excel models into clear "
            "Tableau dashboards for business stakeholders."
        ),
        "summary_no_years": (
            "{title} turning SQL queries and Excel models into clear, actionable reporting."
        ),
        "bullets": [
            "Wrote SQL queries joining five source tables into a single weekly sales report.",
            "Built Excel models that forecasted quarterly revenue within 5% of actuals.",
            "Built Tableau dashboards used by the executive team in weekly reviews.",
            "Used Python and Statistics to test whether a marketing change actually moved "
            "the metrics it targeted.",
            "Built repeatable transformation pipelines that turned raw tables into trusted, "
            "reporting-ready datasets without a single manual export.",
            "Published a Power BI dashboard tracking support-ticket volume by region.",
        ],
        "bullet_only_text": "Published a Power BI dashboard tracking support-ticket "
        "volume by region.",
        "paraphrase_text": "Built repeatable transformation pipelines that turned raw "
        "tables into trusted, reporting-ready datasets without a single manual export.",
        "projects": [
            "Built a Tableau dashboard that replaced a weekly manual Excel report.",
            "Wrote a SQL-based cohort report that is still used to track retention.",
        ],
    },
}

JD_DEFS = [
    {
        "id": "senior_ml_engineer",
        "filename": "jd_senior_ml_engineer.txt",
        "title": "Senior Machine Learning Engineer",
        "overview": (
            "We are hiring a Senior Machine Learning Engineer to take models from "
            "research notebooks to production systems serving millions of requests."
        ),
        "must_have": [
            "5+ years of experience with Python",
            "3+ years of experience with TensorFlow or PyTorch",
            "Bachelor's degree in Computer Science, Machine Learning, or a related field",
        ],
        "nice_to_have": [
            "Experience with MLOps tooling such as MLflow",
            "Familiarity with Spark for large-scale data processing",
            "Background in Computer Vision or NLP",
        ],
        "target_roles": ["ml_engineer"],
        "adjacent_roles": ["data_scientist"],
        "min_years": 5,
        "required_skill": "Python",
    },
    {
        "id": "junior_data_analyst",
        "filename": "jd_junior_data_analyst.txt",
        "title": "Junior Data Analyst",
        "overview": (
            "We are looking for a Junior Data Analyst to help turn raw data into clear, "
            "actionable reporting for the business team."
        ),
        "must_have": [
            "1+ years of experience with SQL and Excel",
            "Familiarity with Python for data analysis",
            "Bachelor's degree in Statistics, Economics, Business Analytics, or a related field",
        ],
        "nice_to_have": [
            "Experience building Tableau or Power BI dashboards",
            "Exposure to dbt or ETL pipelines",
        ],
        "target_roles": ["data_analyst"],
        "adjacent_roles": ["data_scientist"],
        "min_years": 0,
        "required_skill": None,
    },
    {
        "id": "backend_python_fastapi",
        "filename": "jd_backend_python_fastapi.txt",
        "title": "Backend Software Engineer (Python / FastAPI)",
        "overview": (
            "We are hiring a Backend Software Engineer to build and operate the Python "
            "services behind our core product."
        ),
        "must_have": [
            "3+ years of professional experience with Python",
            "Hands-on experience with FastAPI or Django",
            "Experience with PostgreSQL and Docker",
        ],
        "nice_to_have": [
            "Experience with Redis caching",
            "Familiarity with Celery for background jobs",
            "Exposure to a microservices architecture",
        ],
        "target_roles": ["python_backend"],
        "adjacent_roles": ["devops_cloud"],
        "min_years": 3,
        "required_skill": "Python",
    },
    {
        "id": "react_frontend",
        "filename": "jd_react_frontend.txt",
        "title": "Frontend Engineer (React)",
        "overview": (
            "We are hiring a Frontend Engineer to build the React application our "
            "customers use every day."
        ),
        "must_have": [
            "2+ years of experience with JavaScript and React",
            "Experience with TypeScript",
        ],
        "nice_to_have": [
            "Experience with Next.js or server-side rendering",
            "Familiarity with GraphQL",
            "Experience writing tests with Jest",
        ],
        "target_roles": ["react_frontend"],
        "adjacent_roles": [],
        "min_years": 0,
        "required_skill": None,
    },
    {
        "id": "devops_kubernetes_aws",
        "filename": "jd_devops_kubernetes_aws.txt",
        "title": "DevOps Engineer (Kubernetes & AWS)",
        "overview": (
            "We are hiring a DevOps Engineer to run our containerized workloads on "
            "Kubernetes and AWS, and to codify our infrastructure as code."
        ),
        "must_have": [
            "3+ years of professional experience with AWS",
            "Hands-on experience with Kubernetes and Docker in production",
            "Experience with Terraform for infrastructure as code",
        ],
        "nice_to_have": [
            "Experience building CI/CD pipelines",
            "Familiarity with Prometheus and Grafana for monitoring",
            "Exposure to Azure or GCP",
        ],
        "target_roles": ["devops_cloud"],
        "adjacent_roles": ["python_backend"],
        "min_years": 3,
        "required_skill": "AWS",
    },
    {
        "id": "mlops_cross_role",
        "filename": "jd_mlops_cross_role.txt",
        "title": "MLOps Engineer",
        "overview": (
            "We are hiring an MLOps Engineer to own the path from a trained model to a "
            "reliable production deployment on Kubernetes."
        ),
        "must_have": [
            "3+ years of professional experience with Python",
            "Experience deploying machine learning models to production with Docker and Kubernetes",
            "Experience with AWS or GCP cloud infrastructure",
        ],
        "nice_to_have": [
            "Experience with MLflow for experiment tracking",
            "Familiarity with TensorFlow or PyTorch",
            "Background in CI/CD for machine learning pipelines",
        ],
        "target_roles": ["ml_engineer", "devops_cloud"],
        "adjacent_roles": ["data_scientist"],
        "min_years": 3,
        # None: ml_engineer and devops_cloud don't share one required skill, so the
        # must-have gate for this cross-role JD falls back to total years only.
        "required_skill": None,
    },
]


def slugify(first: str, last: str) -> str:
    return f"{first.lower()}_{last.lower()}"


def merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def years_between(intervals: list[tuple[date, date]]) -> float:
    merged = merge_intervals(intervals)
    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)


def build_job_dates(
    total_years: float, num_jobs: int, overlap: bool, rng: random.Random
) -> list[tuple[date, date | None]]:
    """Return ``num_jobs`` (start, end) spans covering exactly ``total_years`` up to
    TODAY, with the last span open-ended ("Present"). When ``overlap`` is set, one
    internal boundary is pulled earlier so two spans overlap, without changing the
    merged total."""
    total_days = round(total_years * 365.25)
    start = TODAY - timedelta(days=total_days)
    if num_jobs == 1:
        return [(start, None)]
    fracs = sorted(rng.uniform(0.2, 0.8) for _ in range(num_jobs - 1))
    boundaries = [start] + [start + timedelta(days=round(f * total_days)) for f in fracs] + [TODAY]
    spans: list[list[date]] = [[boundaries[i], boundaries[i + 1]] for i in range(num_jobs)]
    if overlap:
        idx = rng.randrange(num_jobs - 1)
        overlap_days = rng.randint(30, 120)
        spans[idx + 1][0] = spans[idx + 1][0] - timedelta(days=overlap_days)
    result: list[tuple[date, date | None]] = [(s, e) for s, e in spans[:-1]]
    result.append((spans[-1][0], None))
    return result


def title_sequence(seniority: str, num_jobs: int) -> list[str]:
    order = ["junior", "mid", "senior"]
    end_idx = order.index(seniority)
    start_idx = max(0, end_idx - (num_jobs - 1))
    seq = [order[i] for i in range(start_idx, end_idx + 1)]
    while len(seq) < num_jobs:
        seq.insert(0, seq[0])
    return seq[-num_jobs:]


def format_date(d: date, style: str) -> str:
    if style == "short_dash":
        return str(d.year)
    if style == "slash":
        return f"{d.month:02d}/{d.year}"
    return d.strftime("%b %Y")


def format_date_range(start: date, end: date | None, style: str) -> str:
    end_text = "Present" if end is None else format_date(end, style)
    start_text = format_date(start, style)
    if style == "short_dash":
        return f"{start_text}-{end_text}"
    if style == "slash":
        return f"{start_text} - {end_text}"
    return f"{start_text} – {end_text}"


def sanitize_for_pdf(text: str) -> str:
    replacements = {
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_candidates(rng: random.Random) -> list[CandidateSpec]:
    first_pool = rng.sample(FIRST_NAMES, 36)
    last_pool = rng.sample(LAST_NAMES, 36)
    candidates: list[CandidateSpec] = []
    cursor = 0
    for role, role_def in ROLE_DEFS.items():
        seniorities = rng.sample(["junior", "junior", "mid", "mid", "senior", "senior"], 6)
        for idx in range(6):
            first, last = first_pool[cursor], last_pool[cursor]
            cursor += 1
            seniority = seniorities[idx]
            candidate = CandidateSpec(
                candidate_id=slugify(first, last),
                first_name=first,
                last_name=last,
                role=role,
                seniority=seniority,
            )
            candidate.email = f"{first.lower()}.{last.lower()}@example.com"
            area = rng.choice(AREA_CODES)
            candidate.phone = f"({area}) 555-01{rng.randint(0, 99):02d}"
            degree, field_of_study = rng.choice(role_def["degree_options"])
            candidate.degree = degree
            candidate.field_of_study = field_of_study
            candidate.institution = rng.choice(
                [
                    "State University",
                    "Northfield University",
                    "Lakeview College",
                    "Cedar Tech Institute",
                    "Brightmoor University",
                    "Westlake University",
                    "Riverstone College",
                    "Hamilton Institute of Technology",
                    "Grand Valley University",
                    "Pinecrest College",
                ]
            )
            candidate.inline_skills = rng.random() < 0.2
            candidate.explicit_years_stated = rng.random() < 0.5
            candidate.skills_header_word = rng.choice(SKILLS_HEADERS)
            candidate.summary_header = rng.choice(SUMMARY_HEADERS)
            candidate.experience_header = rng.choice(EXPERIENCE_HEADERS)
            candidate.education_header = rng.choice(EDUCATION_HEADERS)
            candidate.include_projects = rng.random() < 0.5

            cert_pool = role_def["certs"]
            if seniority == "senior" and rng.random() < 0.7:
                candidate.certifications = rng.sample(cert_pool, k=min(2, len(cert_pool)))
            elif seniority == "mid" and rng.random() < 0.3:
                candidate.certifications = [rng.choice(cert_pool)]

            # Skills line: the role's core skills, plus an occasional extra.
            skills = list(role_def["core_skills"])
            if rng.random() < 0.4:
                extra = rng.choice(role_def["nice_extra"])
                if extra not in skills:
                    skills.append(extra)
            candidate.skills = skills

            # Build job entries.
            if seniority == "junior":
                total_years = round(rng.uniform(0.5, 2.5), 1)
                num_jobs = 1
            elif seniority == "mid":
                total_years = round(rng.uniform(3.0, 5.5), 1)
                num_jobs = 2
            else:
                total_years = round(rng.uniform(6.0, 11.0), 1)
                num_jobs = rng.choice([2, 3])
            overlap = num_jobs >= 2 and rng.random() < 0.35
            spans = build_job_dates(total_years, num_jobs, overlap, rng)
            titles = title_sequence(seniority, num_jobs)
            companies = rng.sample(COMPANIES, k=num_jobs)
            date_style = rng.choice(["en_dash", "short_dash", "slash"])

            include_bullet_only = idx == 0
            include_paraphrase = idx == 1
            bonus_bullets = []
            if include_bullet_only:
                bonus_bullets.append(role_def["bullet_only_text"])
            if include_paraphrase:
                bonus_bullets.append(role_def["paraphrase_text"])

            jobs: list[JobEntry] = []
            for job_idx, (start, end) in enumerate(spans):
                title = f"{titles[job_idx].capitalize()} {role_def['title_base']}"
                if titles[job_idx] == "mid":
                    title = role_def["title_base"]
                bullet_pool = role_def["bullets"]
                job_bullets = list(rng.sample(bullet_pool, k=min(3, len(bullet_pool))))
                if job_idx == len(spans) - 1:
                    for bonus in bonus_bullets:
                        if bonus not in job_bullets:
                            job_bullets.append(bonus)
                skills_used = list(role_def["core_skills"])
                if job_idx == 0 and len(spans) > 1:
                    skills_used = skills_used[: max(1, len(skills_used) // 2)]
                jobs.append(
                    JobEntry(
                        title=title,
                        company=companies[job_idx],
                        start=start,
                        end=end,
                        bullets=job_bullets,
                        skills_used=skills_used,
                        date_style=date_style,
                    )
                )
            candidate.jobs = jobs

            years_text = str(int(total_years))
            if candidate.explicit_years_stated:
                template_key = "summary_with_years"
            else:
                template_key = "summary_no_years"
            candidate.summary_text = role_def[template_key].format(
                title=role_def["role_label"], years=years_text
            )

            if include_bullet_only:
                candidate.skills_demo = "bullet_only"
            if include_paraphrase:
                candidate.skills_demo = "paraphrase"

            candidates.append(candidate)
    return candidates


def assign_formats(candidates: list[CandidateSpec], rng: random.Random) -> None:
    formats = ["pdf"] * 12 + ["docx"] * 10 + ["txt"] * 10 + ["md"] * 4
    rng.shuffle(formats)
    for candidate, fmt in zip(candidates, formats, strict=True):
        candidate.file_format = fmt


def render_blocks(candidate: CandidateSpec) -> tuple[str, list[tuple[str | None, list[str]]]]:
    role_def = ROLE_DEFS[candidate.role]
    header_labels: dict[str, str] = {}
    contact_line = f"{candidate.email} | {candidate.phone}"
    blocks: list[tuple[str | None, list[str]]] = [(None, [contact_line])]

    header_labels["summary"] = candidate.summary_header
    blocks.append((candidate.summary_header, [candidate.summary_text]))

    header_labels["experience"] = candidate.experience_header
    exp_lines: list[str] = []
    for job in reversed(candidate.jobs):
        exp_lines.append(f"{job.title}, {job.company}")
        exp_lines.append(format_date_range(job.start, job.end, job.date_style))
        for bullet in job.bullets:
            exp_lines.append(f"- {bullet}")
    blocks.append((candidate.experience_header, exp_lines))

    header_labels["education"] = candidate.education_header
    edu_line = f"{candidate.degree} in {candidate.field_of_study}, {candidate.institution}"
    blocks.append((candidate.education_header, [edu_line]))

    header_labels["skills"] = candidate.skills_header_word
    skills_text = ", ".join(candidate.skills)
    if candidate.inline_skills:
        blocks.append((None, [f"{candidate.skills_header_word}: {skills_text}"]))
    else:
        blocks.append((candidate.skills_header_word, [skills_text]))

    if candidate.include_projects:
        header_labels["projects"] = PROJECTS_HEADER
        blocks.append((PROJECTS_HEADER, [f"- {p}" for p in role_def["projects"]]))

    if candidate.certifications:
        header_labels["certifications"] = CERTIFICATIONS_HEADER
        blocks.append((CERTIFICATIONS_HEADER, [f"- {c}" for c in candidate.certifications]))

    candidate.header_labels = header_labels
    return candidate.name, blocks


def write_txt_or_md(path: Path, name: str, blocks: list[tuple[str | None, list[str]]]) -> None:
    lines = [name, ""]
    for header, body in blocks:
        if header:
            lines.append(header)
        lines.extend(body)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_docx(path: Path, name: str, blocks: list[tuple[str | None, list[str]]]) -> None:
    doc = Document()
    name_paragraph = doc.add_paragraph()
    name_run = name_paragraph.add_run(name)
    name_run.bold = True
    name_run.font.size = Pt(14)
    for header, body in blocks:
        if header:
            header_paragraph = doc.add_paragraph()
            header_run = header_paragraph.add_run(header)
            header_run.bold = True
        for line in body:
            doc.add_paragraph(line)
    doc.save(str(path))


def _pdf_line(pdf: FPDF, height: float, text: str) -> None:
    pdf.multi_cell(0, height, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def write_pdf(path: Path, name: str, blocks: list[tuple[str | None, list[str]]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=14)
    _pdf_line(pdf, 8, sanitize_for_pdf(name))
    pdf.set_font("Helvetica", size=11)
    for header, body in blocks:
        pdf.ln(2)
        if header:
            pdf.set_font("Helvetica", style="B", size=12)
            _pdf_line(pdf, 6, sanitize_for_pdf(header))
            pdf.set_font("Helvetica", size=11)
        for line in body:
            _pdf_line(pdf, 6, sanitize_for_pdf(line))
    pdf.output(str(path))


def write_resumes(candidates: list[CandidateSpec]) -> None:
    for candidate in candidates:
        name, blocks = render_blocks(candidate)
        filename = f"{candidate.candidate_id}.{candidate.file_format}"
        path = config.RESUMES_DIR / filename
        if candidate.file_format in {"txt", "md"}:
            write_txt_or_md(path, name, blocks)
        elif candidate.file_format == "docx":
            write_docx(path, name, blocks)
        else:
            write_pdf(path, name, blocks)


def round_trip_check(candidates: list[CandidateSpec], jd_defs: list[dict]) -> None:
    problems: list[str] = []
    for candidate in candidates:
        filename = f"{candidate.candidate_id}.{candidate.file_format}"
        path = config.RESUMES_DIR / filename
        result = m1.read_file(str(path))
        if not result.get("success"):
            problems.append(f"{filename}: read_file failed: {result.get('error')}")
            continue
        content = result["content"]
        if candidate.last_name not in content:
            problems.append(f"{filename}: last name {candidate.last_name!r} not found")
        for section, label in candidate.header_labels.items():
            if label not in content:
                problems.append(f"{filename}: {section} header {label!r} not found")
    for jd in jd_defs:
        path = config.JOB_DESCRIPTIONS_DIR / jd["filename"]
        result = m1.read_file(str(path))
        if not result.get("success"):
            problems.append(f"{jd['filename']}: read_file failed: {result.get('error')}")
            continue
        if "Must-have" not in result["content"]:
            problems.append(f"{jd['filename']}: missing 'Must-have' section")
    if problems:
        raise RuntimeError("Round-trip check failed:\n" + "\n".join(problems))
    print(
        f"Round-trip check passed for {len(candidates)} resumes and "
        f"{len(jd_defs)} job descriptions."
    )


def build_resume_specs(candidates: list[CandidateSpec]) -> list[dict]:
    specs = []
    for candidate in candidates:
        intervals = [(job.start, job.end or TODAY) for job in candidate.jobs]
        total_years = years_between(intervals)
        skill_years = {}
        for skill in candidate.skills:
            skill_intervals = [
                (job.start, job.end or TODAY) for job in candidate.jobs if skill in job.skills_used
            ]
            if skill_intervals:
                skill_years[skill] = years_between(skill_intervals)
        skills = set(candidate.skills)
        role_def = ROLE_DEFS[candidate.role]
        if candidate.skills_demo == "bullet_only":
            skill = role_def["bullet_only_skill"]
            skills.add(skill)
            skill_years[skill] = total_years
        elif candidate.skills_demo == "paraphrase":
            skill = role_def["paraphrase_skill"]
            skills.add(skill)
            skill_years[skill] = total_years
        filename = f"{candidate.candidate_id}.{candidate.file_format}"
        specs.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "email": candidate.email,
                "phone": candidate.phone,
                "resume_path": str((config.RESUMES_DIR / filename).relative_to(config.ROOT_DIR)),
                "file_format": candidate.file_format,
                "role": candidate.role,
                "seniority": candidate.seniority,
                "total_years": total_years,
                "skills": sorted(skills),
                "skill_years": skill_years,
                "education": {
                    "degree": candidate.degree,
                    "field": candidate.field_of_study,
                    "institution": candidate.institution,
                    "level": DEGREE_LEVELS[candidate.degree],
                },
                "certifications": candidate.certifications,
            }
        )
    return specs


def jd_body_text(jd: dict) -> str:
    lines = [jd["title"], "", jd["overview"], "", "Must-have:"]
    lines.extend(f"- {item}" for item in jd["must_have"])
    lines.append("")
    lines.append("Nice-to-have:")
    lines.extend(f"- {item}" for item in jd["nice_to_have"])
    return "\n".join(lines) + "\n"


def write_job_descriptions() -> None:
    for jd in JD_DEFS:
        path = config.JOB_DESCRIPTIONS_DIR / jd["filename"]
        path.write_text(jd_body_text(jd), encoding="utf-8")


def grade_relevance(spec: dict, jd: dict) -> int:
    role = spec["role"]
    if role in jd["target_roles"]:
        base = 2
    elif role in jd["adjacent_roles"]:
        base = 1
    else:
        return 0
    if base == 2 and jd["required_skill"]:
        years_for_skill = spec["skill_years"].get(jd["required_skill"], 0.0)
        if years_for_skill < jd["min_years"]:
            return 1
    elif base == 2 and spec["total_years"] < jd["min_years"]:
        return 1
    return base


def build_ground_truth(specs: list[dict]) -> dict[str, dict[str, int]]:
    ground_truth: dict[str, dict[str, int]] = {}
    for jd in JD_DEFS:
        graded: dict[str, int] = {}
        for spec in specs:
            resume_file = Path(spec["resume_path"]).name
            graded[resume_file] = grade_relevance(spec, jd)
        ground_truth[jd["id"]] = graded
    return ground_truth


def reset_output_dirs() -> None:
    for pattern in ("*.pdf", "*.docx", "*.txt", "*.md"):
        for path in config.RESUMES_DIR.glob(pattern):
            path.unlink()
    for path in config.JOB_DESCRIPTIONS_DIR.glob("*.txt"):
        path.unlink()


def print_summary(specs: list[dict], ground_truth: dict[str, dict[str, int]]) -> None:
    by_role: dict[str, int] = {}
    by_format: dict[str, int] = {}
    for spec in specs:
        by_role[spec["role"]] = by_role.get(spec["role"], 0) + 1
        by_format[spec["file_format"]] = by_format.get(spec["file_format"], 0) + 1
    print(f"Generated {len(specs)} resumes.")
    print(f"  by role: {by_role}")
    print(f"  by format: {by_format}")
    for jd_id, graded in ground_truth.items():
        counts = {0: 0, 1: 0, 2: 0}
        for grade in graded.values():
            counts[grade] += 1
        print(f"  ground truth for {jd_id}: {counts}")


def main() -> None:
    rng = random.Random(config.DATASET_SEED)
    config.RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    config.JOB_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    reset_output_dirs()

    candidates = build_candidates(rng)
    assign_formats(candidates, rng)
    write_resumes(candidates)
    write_job_descriptions()
    round_trip_check(candidates, JD_DEFS)

    specs = build_resume_specs(candidates)
    config.RESUME_SPECS_PATH.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")

    ground_truth = build_ground_truth(specs)
    config.GROUND_TRUTH_PATH.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")

    print_summary(specs, ground_truth)


if __name__ == "__main__":
    main()
