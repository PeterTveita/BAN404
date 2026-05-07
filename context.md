# Context: BAN404 — Introduction to Machine Learning

**Last updated:** 2026-05-07
**Status:** In progress

---

## Overview
BAN404 course at NHH. Uses ISLP (Introduction to Statistical Learning with Python) as textbook. Exercises are done in Jupyter notebooks. Pensum-boken ligger i `Pensum/ISLP.pdf`.

## Goals / Research Questions
- Forstå og implementere statistiske læringsmetoder i Python
- Løse øvingsoppgaver fra ISLP kapittel 2, 3 og videre

## Current Status
- Kapittel 2 (Statistical Learning): konseptuelle oppgaver 1–7 gjennomgått og forstått
- Exercise 2.8 (College-datasettet): (a)–(g) fullført, (h) gjenstår
- Lecture 2 (Linear Regression + KNN): gjennomgått og forklart
- Exercise 3.8 (Auto-datasettet): del (a) fullført med tolkning, (b)–(c) gjenstår

## Progress Log
| Date | Update |
|------|--------|
| 2026-05-07 | Kontekstfil opprettet |
| 2026-05-07 | Gikk gjennom konseptene fra kap. 2: flexible/inflexible models, bias-variance tradeoff, parametric vs non-parametric, KNN |
| 2026-05-07 | Løste Exercise 2.8 (a)–(g) i ex28_solution.ipynb |
| 2026-05-07 | Gikk gjennom Lecture 2: polynomial regression, train/test MSE, overfitting, KNN for regression |
| 2026-05-07 | Startet Exercise 3.8 i ex38.ipynb — del (a) fullført |

## Decisions Made
- Bruker `index_col=0` eller `college3`-metoden for å sette universitetsnavnene som radnavn i College-datasettet
- Bruker `MS` (ModelSpec) fra ISLP for å bygge designmatrise i regresjonsoppgaver
- Auto-data lastes med `load_data('Auto')` fra ISLP-biblioteket

## Open Questions / Blockers
- Exercise 2.8 (h): utforsk datasettet fritt og kommenter funn — ikke påbegynt
- Exercise 3.8 (b): scatter + regresjonslinje med `ax.axline()`
- Exercise 3.8 (c): diagnostikkplott (residualer og QQ-plott) — ikke påbegynt
- Exercise 3.9 og 3.10 ikke påbegynt

## Key Files & Resources
- `Pensum/ISLP.pdf` — pensumbok
- `Foreleser/01/ex28_solution.ipynb` — løsning Exercise 2.8 (College)
- `Foreleser/01/ex28.py` — foreleserens løsning Exercise 2.8
- `Foreleser/02/ex38.ipynb` — løsning Exercise 3.8 (Auto) — under arbeid
- `Foreleser/02/ex38.py` — foreleserens løsning Exercise 3.8
- `Foreleser/02/lecture2.pdf` — forelesning om linear regression og KNN
- `ISLP Repo/Auto.csv` — Auto-datasettet (også tilgjengelig via `load_data('Auto')`)
- `Foreleser/01/College.csv` — College-datasettet

## Notes
- Studenten er nybegynner i Python og statistisk læring — foretrekker grundige forklaringer på norsk
- Foretrekker å løse oppgavene selv med veiledning fremfor å få ferdig kode
- Viktige konsepter forstått: bias-variance tradeoff, overfitting/underfitting, MSE, p-verdier, R², KI vs. PI
