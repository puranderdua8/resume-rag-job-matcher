"""Skills taxonomy: canonical skill name -> known aliases.

A starter set covering the role families in the plan (ML, data, backend,
frontend, DevOps/cloud). Step 2 grows this as the dataset generator's
content pools reveal more real-world phrasing.
"""

from __future__ import annotations

SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python"],
    "SQL": ["sql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural networks", "neural nets"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react", "react.js", "reactjs"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment"],
    "Git": ["git"],
}


def canonical_skills() -> list[str]:
    """All canonical skill names, in insertion order."""
    return list(SKILL_ALIASES)
