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
    # Added in Step 2 for the dataset generator's role content (data analyst,
    # DevOps/cloud and frontend phrasing not covered by the Step 1 starter set).
    "Statistics": ["statistics", "statistical analysis"],
    "A/B Testing": ["a/b testing", "ab testing"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Excel": ["excel", "ms excel", "microsoft excel"],
    "Looker": ["looker"],
    "dbt": ["dbt"],
    "ETL": ["etl", "extract transform load"],
    "Spark": ["spark", "apache spark", "pyspark"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "MLOps": ["mlops"],
    "MLflow": ["mlflow"],
    "Redis": ["redis"],
    "Microservices": ["microservices", "microservice architecture"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis"],
    "Celery": ["celery"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Redux": ["redux"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "GraphQL": ["graphql"],
    "Jest": ["jest"],
    "Jenkins": ["jenkins"],
    "Ansible": ["ansible"],
    "Linux": ["linux"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
}


def canonical_skills() -> list[str]:
    """All canonical skill names, in insertion order."""
    return list(SKILL_ALIASES)
