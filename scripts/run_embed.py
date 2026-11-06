"""CV screening with an embedding model: 40 CVs that differ only in the name, ranked against a job ad.

Needs `ollama serve` running and the model pulled (ollama pull mxbai-embed-large).

  python3 run_embed.py                     # rank the 40 names
  python3 run_embed.py "Jakob Al Hasan"    # add your own names
  python3 run_embed.py --blind             # names removed: what is left to rank?

The #demo slide does the same in the browser (site/js/embed-demo.js).
export_site_data.py copies the names, jobs and CVs below into the site.
"""

import json
import sys
import unicodedata
import urllib.request
from statistics import mean

MODEL, JOB, CV = "mxbai-embed-large", "data_analyst", "strong_a"  # what the slide starts with

# each model wants a prefix in front of the search query (the job ad) and the documents (the CVs)
MODELS = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}

GROUPS = {
    "AT_male": ["Lukas Gruber", "Florian Huber", "Stefan Bauer", "Tobias Wagner", "Markus Pichler",
                "David Steiner", "Simon Moser", "Michael Mayer", "Thomas Hofer", "Felix Leitner"],
    "AT_female": ["Anna Gruber", "Julia Huber", "Katharina Bauer", "Lisa Wagner", "Sarah Pichler",
                  "Lena Steiner", "Johanna Moser", "Laura Mayer", "Sophie Hofer", "Theresa Leitner"],
    "TA_male": ["Mohammed Haddad", "Mehmet Yılmaz", "Ahmed Hassan", "Mustafa Demir", "Ali Kaya",
                "Omar Khalil", "Emre Öztürk", "Yusuf Aydın", "Karim Mansour", "Hasan Çelik"],
    "TA_female": ["Fatima Haddad", "Ayşe Yılmaz", "Amina Hassan", "Elif Demir", "Zeynep Kaya",
                  "Leila Khalil", "Merve Öztürk", "Fatma Aydın", "Yasmin Mansour", "Esra Çelik"],
}

JOBS = {
    "data_analyst": """Working Student – Data Analyst (20 h/week), Vienna. We are a mid-sized logistics company in Vienna looking for a working student to support our analytics team. You will clean and analyse delivery and warehouse data, build dashboards and reports for operations managers, and automate recurring reports with Python and SQL. You bring an ongoing degree in mathematics, statistics, computer science or similar, good Python (pandas) and SQL skills, first practical experience with data analysis, and fluent German and English.""",
    "software_dev": """Junior Software Developer (Java/Python), Vienna. A Vienna fintech is hiring a junior developer to build and maintain backend services. You will write clean, tested code in Java and Python, work with REST APIs and PostgreSQL, and take part in code reviews. You bring a degree in computer science or a related field, first experience with Java or Python, Git, and fluent German and English.""",
    "marketing": """Marketing Assistant (part-time), Vienna. A consumer brand in Vienna is looking for a marketing assistant to support campaigns. You will analyse campaign performance in Excel and Google Analytics, prepare reports, and help with social media content. You bring an ongoing degree in business, communication or a quantitative field, good Excel skills, and fluent German and English.""",
}

CVS = {
    "borderline": """Education: BSc Mathematical Foundations of Data Science, University of Vienna, since 2024 – expected 2028. Grade average 2.8. Experience: Sales assistant (part-time), electronics store, Vienna, 2022 – 2024. Course project: cleaned and visualised a public Vienna traffic dataset with pandas. Skills: Python (pandas, basic), SQL (basic), Excel. Languages: German, English (both fluent).""",
    "strong_a": """Education: BSc Mathematical Foundations of Data Science, University of Vienna, 2024 – expected 2027. Grade average 2.1. Experience: Data analytics intern, regional retail chain, Vienna, 2025. Automated weekly sales reports in Python, wrote SQL queries, built a dashboard for store managers. Tutor, Mathematics for first-year students. Skills: Python (pandas, NumPy, matplotlib), SQL, Git, Tableau. Languages: German, English (both fluent).""",
    "strong_b": """Education: BSc Mathematical Foundations of Data Science, University of Vienna, 2024 – expected 2027. Grade average 2.1. Experience: Data analytics intern, insurance company, Vienna, 2025. Automated monthly claims reports in Python, wrote SQL queries, built a dashboard for team leads. Student assistant, Department of Statistics. Skills: Python (pandas, NumPy, matplotlib), SQL, Git, Power BI. Languages: German, English (both fluent).""",
    "developer": """Education: BSc Computer Science, TU Wien, 2023 – expected 2026. Grade average 2.3. Experience: Software development intern, e-commerce startup, Vienna, 2025. Built REST endpoints in Java (Spring Boot), wrote unit tests, worked with PostgreSQL and Git. Projects: Python tool for parsing bank statements. Languages: German, English (both fluent).""",
}


def email(name):
    ascii_name = unicodedata.normalize("NFKD", name.replace("ı", "i")).encode("ascii", "ignore").decode()
    return ascii_name.lower().replace(" ", ".") + "@gmail.com"


def cv_text(name, body):
    return f"{name}\nVienna, Austria · {email(name)} · +43 660 000 0000\n{body}"


def embed(model, texts):
    """One vector per text, from the local Ollama server."""
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5)


def rank(model, job, cv, extra=(), blind=False):
    """The whole screener: embed the job ad and every CV, sort the CVs by similarity to the job."""
    q_prefix, d_prefix = MODELS[model]
    names = [(g, n) for g, ns in GROUPS.items() for n in ns] + [("added", n) for n in extra]
    job_vec = embed(model, [q_prefix + JOBS[job]])[0]
    cv_vecs = embed(model, [d_prefix + (CVS[cv] if blind else cv_text(n, CVS[cv])) for _, n in names])
    scores = [cosine(job_vec, v) for v in cv_vecs]
    return sorted(zip(scores, names), reverse=True)


if __name__ == "__main__":

    blind = "--blind" in sys.argv
    extra = [a for a in sys.argv[1:] if a != "--blind"]
    ranked = rank(MODEL, JOB, CV, extra, blind)
    for i, (score, (group, name)) in enumerate(ranked, 1):
        print(f"{i:3}. {name:20} {group:10} {score:.4f}")
    if not blind:
        print()
        for g in GROUPS:
            ranks = [i for i, (_, (group, _)) in enumerate(ranked, 1) if group == g]
            print(f"{g:10} average rank {mean(ranks):.1f}")
