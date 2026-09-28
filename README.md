# Halo Apex

KI-gestützte Code-Task-Pipeline mit entkoppelter Architektur: Vektor-Retrieval,
AST-basiertes Patching und optionales Streaming arbeiten als unabhängige
Komponenten zusammen, die erst durch den Orchestrator verdrahtet werden.

## Architektur

| Komponente | Package | Aufgabe |
|---|---|---|
| **Halo Vector** | `halo_vector/` | Embeddings (bge-m3) → ChromaDB-Retrieval |
| **Halo Blade** | `halo_blade/` | AST-Skeleton, Git-Apply-Patching, Executor |
| **Halo Stream** | `halo_stream/` | Routing-Stub (deaktiviert = Direct Pass-through) |
| **Orchestrator** | `runner.py`, `main.py` | Task-Pipeline: Index → LLM → Apply → Gate → Commit |

Die Sub-Packages sind 100 % standalone und importieren niemals die zentrale
Config — `main.py` injiziert die Werte per Konstruktor-Parameter
(Pydantic-Settings aus `.env` / `HALO_*`-Variablen).

## Sicherheit

Jeder Task läuft auf einem eigenen Git-Branch (`halo/<task-slug>`); der
aktuelle Branch wird nie direkt gepatcht. Review mit:

```bash
git diff <branch>...halo/<slug>
```

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # HALO_*-Variablen setzen
```

## Verwendung

```bash
python cli.py health                      # End-to-End-Selbsttest
python cli.py index <pfad>               # Repo in Vektor-Index aufnehmen
python cli.py task "<auftrag>" --file <datei>
```

## Benchmark

`bench.py` misst den Performance-Zugewinn der Pipeline (Latenzen,
First-Try-Rate, Hybrid-Prompt vs. Volldump) auf einem frischen,
opferbaren Test-Repo — der produktive Index unter
`~/.local/share/halo_apex` bleibt unangetastet.

```bash
./venv/bin/python bench.py
```
