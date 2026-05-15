# Context: BAN404 — Introduction to Machine Learning

**Last updated:** 2026-05-15 (session 7)
**Status:** In progress

---

## Overview
BAN404 course at NHH. Uses ISLP (Introduction to Statistical Learning with Python) as textbook. Exercises are done in Jupyter notebooks. Pensum-boken ligger i `Pensum/ISLP.pdf`.

## Goals / Research Questions
- Forstå og implementere statistiske læringsmetoder i Python
- Løse øvingsoppgaver fra ISLP kapittel 2, 3 og videre

## Current Status
- Kapittel 2 (Statistical Learning): konseptuelle oppgaver 1–7 gjennomgått og forstått
- Exercise 2.8 (College-datasettet): (a)–(g) fullført ✅, (h) gjenstår
- Lecture 2 (Linear Regression + KNN): gjennomgått og forklart ✅
- Exercise 3.8 (Auto-datasettet): (a)–(c) fullført ✅
- Tutorial 1, Task 1 (a)–(e): fullført ✅
- Tutorial 1, Task 2: ikke påbegynt
- Exercise 4.13 (Weekly-datasettet): (a)–(h) fullført ✅, (i) og (j) gjenstår
- Exercise 5.5 (Default-datasettet): (a)–(d) fullført ✅
- Exercise 6.11 (Boston-datasettet): del a) fullført ✅, del b) og c) (tolkningsceller) gjenstår
- Exercise 7.6 (Wage-datasettet): (a) og (b) fullført ✅

## Progress Log
| Date | Update |
|------|--------|
| 2026-05-07 | Kontekstfil opprettet |
| 2026-05-07 | Gikk gjennom konseptene fra kap. 2: flexible/inflexible models, bias-variance tradeoff, parametric vs non-parametric, KNN |
| 2026-05-07 | Løste Exercise 2.8 (a)–(g) i ex28_solution.ipynb |
| 2026-05-07 | Gikk gjennom Lecture 2: polynomial regression, train/test MSE, overfitting, KNN for regression |
| 2026-05-07 | Startet Exercise 3.8 i ex38.ipynb — del (a) fullført |
| 2026-05-07 | Exercise 3.8 (b) og (c) fullført — scatter+regresjonslinje og diagnostikkplott |
| 2026-05-07 | Startet Tutorial 1 (Exercise set 1) — Task 1 (a) og (b) fullført |
| 2026-05-07 | Tutorial 1, Task 1 (c): KNN-prediksjon for K=1,3,5 fullført |
| 2026-05-07 | Tutorial 1, Task 1 (d): gått gjennom knn()-funksjonen linje for linje |
| 2026-05-07 | Tutorial 1, Task 1 (e): diskutert metodevalg — lineær regresjon foretrukket ved lineært mønster |
| 2026-05-07 | Grundig gjennomgang av begreper: varians, standardavvik, nullhypotese, p-verdi, statistisk signifikans, konfidensintervall |
| 2026-05-07 | Opprettet oppsummeringsfil: `Foreleser/Tutorial 1/statistikk_begreper.md` |
| 2026-05-14 | Exercise 4.13 (a): lastet Weekly-datasettet, korrelasjonsmatrise og plott fullført |
| 2026-05-14 | Exercise 4.13 (b): logistisk regresjon med alle lag + Volume — kun Lag2 signifikant (p=0.030) |
| 2026-05-14 | Exercise 4.13 (c): confusion matrix og accuracy (56.11%) — modellen sier nesten alltid Up |
| 2026-05-14 | Exercise 4.13 (d): logistisk regresjon med Lag2, train 1990–2008, test 2009–2010 → 62.50% |
| 2026-05-14 | Exercise 4.13 (e): LDA med Lag2 → 62.50% (identisk med logistisk regresjon) |
| 2026-05-14 | Exercise 4.13 (f): QDA med Lag2 → 58.65% |
| 2026-05-14 | Exercise 4.13 (g): KNN K=1 med Lag2 → 50.00% (overfitting) |
| 2026-05-14 | Exercise 4.13 (h): Naive Bayes med Lag2 → 59.00% |
| 2026-05-14 | Gjennomgått Positron-innstillinger: inline suggestions, extension bisect, conda-miljø BAN404 |
| 2026-05-15 | Exercise 5.5 (a)–(d) fullført: validation set approach på Default-datasettet |
| 2026-05-15 | Gjennomgått teori: logistisk vs. lineær regresjon, sigmoid, log-odds, hva det betyr å fitte en modell |
| 2026-05-15 | Gjennomgått validation set approach: splitting, treningssett vs. valideringssett, feilratens variabilitet |
| 2026-05-15 | Forstår nå: avhengig vs. uavhengig variabel, prob vs. pred, threshold 0.5, random seed og reproduserbarhet |
| 2026-05-15 | Gjennomgått forelesning 6 (linear model selection) og 7 (bootstrap) grundig |
| 2026-05-15 | Exercise 6.11 del a) fullført: OLS (MSE 40.22), Ridge (MSE 40.51, α=107.19), Lasso (MSE 40.47, α=0.26) |
| 2026-05-15 | Koeffisientsammenligning kjørt: Lasso setter chas, nox og rm til null |
| 2026-05-15 | Forstår OLS vs Ridge vs Lasso: straff, shrinkage, variabelutvelgelse, L1/L2 |
| 2026-05-15 | Forstår bias-variance tradeoff, støy, overfitting, og hvorfor vi deler i train/test |
| 2026-05-15 | Forstår X/y-split og train/test-split — og hvorfor sklearn krever separat X og y |
| 2026-05-15 | Exercise 7.6 (a): polynomregresjon på Wage-datasettet, validation set approach — optimal grad = 7 |
| 2026-05-15 | Exercise 7.6 (b): trappefunksjon på Wage-datasettet, validation set approach — optimalt antall kutt = 10 |
| 2026-05-15 | Forstår polynomregresjon: np.vander, Vandermonde-matrise, grad vs. antall kolonner |
| 2026-05-15 | Forstår trappefunksjoner: pd.cut, kategorisk variabel, dummyvariabler, plt.step |
| 2026-05-15 | Forstår 0-indeksering og ±1-justeringer i MSE-arrays (k-1 ved lagring, +1/+2 ved avlesning) |
| 2026-05-15 | Forstår livssyklus-hypotesen for lønn: humankapital, kurvet alder-lønn-sammenheng |

## Decisions Made
- Bruker `index_col=0` eller `college3`-metoden for å sette universitetsnavnene som radnavn i College-datasettet
- Bruker `MS` (ModelSpec) fra ISLP for å bygge designmatrise i regresjonsoppgaver
- Auto-data lastes med `load_data('Auto')` fra ISLP-biblioteket
- Bruker sklearn (RidgeCV, LassoCV) for Ridge og Lasso — statsmodels har ikke regulariseringsmetoder
- Bruker `np.logspace(-4, 4, 200)` som kandidatverdier for alpha i kryss-validering
- Bruker `sm.add_constant` for OLS når statsmodels brukes uten formelnotasjon
- Bruker `np.vander(age, k+1, increasing=True)` for polynombasis i smf.ols-formler
- Bruker `pd.cut(data['age'], k)` for å lage trappefunksjoner (k like store bins)
- For å predikere med trappefunksjon på testsettet: `test.assign(age_cut=pd.cut(test['age'], k))`

## Open Questions / Blockers
- Exercise 2.8 (h): utforsk datasettet fritt og kommenter funn — ikke påbegynt
- Exercise 3.9 og 3.10 ikke påbegynt
- Tutorial 1, Task 2: ikke påbegynt (College-datasettet, train/test split, MSE, KNN)
- Exercise 4.13 (i): sammenligne alle metoder og velge beste — gjenstår
- Exercise 4.13 (j): eksperimentere med ulike prediktor-kombinasjoner og K-verdier — gjenstår

## Key Files & Resources
- `Pensum/ISLP.pdf` — pensumbok
- `Foreleser/03/ex4.13.ipynb` — løsning Exercise 4.13 (Weekly) — (a)–(h) ferdig
- `Foreleser/01/ex28_solution.ipynb` — løsning Exercise 2.8 (College)
- `Foreleser/01/ex28.py` — foreleserens løsning Exercise 2.8
- `Foreleser/02/ex38.ipynb` — løsning Exercise 3.8 (Auto) — under arbeid
- `Foreleser/02/ex38.py` — foreleserens løsning Exercise 3.8
- `Foreleser/02/lecture2.pdf` — forelesning om linear regression og KNN
- `ISLP Repo/Auto.csv` — Auto-datasettet (også tilgjengelig via `load_data('Auto')`)
- `Foreleser/01/College.csv` — College-datasettet
- `Foreleser/Tutorial 1/Tutorial1.ipynb` — løsning Tutorial 1 — Task 1 ferdig, Task 2 gjenstår
- `Foreleser/Tutorial 1/tutorial1.pdf` — oppgavetekst Tutorial 1
- `Foreleser/Tutorial 1/statistikk_begreper.md` — oppsummering av statistiske begreper med case
- `Foreleser/05/ex55.ipynb` — løsning Exercise 5.5 (Default) — ferdig ✅
- `Foreleser/05/ex55.py` — foreleserens løsning Exercise 5.5
- `Foreleser/07/ex611.ipynb` — løsning Exercise 6.11 (Boston) — del a) ferdig ✅
- `Foreleser/07/ex611.py` — foreleserens løsning Exercise 6.11
- `Foreleser/06/lecture6.pdf` — forelesning om linear model selection (subset, Ridge, Lasso)
- `Foreleser/07/lecture7.pdf` — forelesning om bootstrap
- `Foreleser/08/ex76_solution.ipynb` — løsning Exercise 7.6 (Wage) — ferdig ✅
- `Foreleser/08/ex76.ipynb` — foreleserens løsning Exercise 7.6
- `Foreleser/08/lecture8ho.pdf` — forelesning om Moving Beyond Linearity (kap. 7)

## Notes
- Studenten er nybegynner i Python og statistisk læring — foretrekker grundige forklaringer på norsk
- Foretrekker å løse oppgavene selv med veiledning fremfor å få ferdig kode
- Viktige konsepter forstått: bias-variance tradeoff, overfitting/underfitting, MSE, p-verdier, R², KI vs. PI
- Forstår nå: varians, standardavvik, nullhypotese, p-verdi, statistisk signifikans, konfidensintervall, korrelasjon ≠ kausalitet
- Forstår forskjellen på array, liste og DataFrame — og hvorfor NumPy vektorisering brukes
- Forstår intuisjonen bak KNN (lav vs. høy K, overfit/underfit, outliers, ekstrapolasjon)
- Forstår logistisk regresjon: sigmoid-funksjon, log-odds, koeffisienter, p-verdier, konfidensintervall, std err
- Forstår confusion matrix og accuracy
- Forstår forskjellen mellom in-sample og out-of-sample evaluering
- Forstår LDA, QDA, KNN, Naive Bayes på konseptuelt nivå
- Forstår validation set approach: splitting, feilratens variabilitet, og hvorfor vi ikke tester på treningsdata
- Forstår logistisk regresjon grundig: sigmoid, log-odds, fitting, prob vs. pred, threshold, avhengig/uavhengig variabel
- Forstår random seed og reproduserbarhet i Python (np.random.default_rng)
- Forstår overfitting (lav K) vs underfitting (høy K) i KNN
- Forstår OLS vs Ridge vs Lasso: straff (L1/L2), shrinkage, variabelutvelgelse
- Forstår bias (for enkel modell) vs varians (for kompleks modell) og tradeoff
- Forstår hva støy er og hva det betyr å "lære støy"
- Forstår MSE og test-MSE som evalueringsmetrikk
- Forstår alpha (λ) som straffeparameter og hvordan kryss-validering velger optimal α
- Forstår hvorfor sklearn krever separat X og y, og når statsmodels vs sklearn brukes
- Forstår polynomregresjon: np.vander, Vandermonde-matrise, grad k krever k+1 kolonner
- Forstår trappefunksjoner: pd.cut, kategorisk variabel, dummyvariabler, plt.step
- Forstår 0-indeksering og ±1-justeringer i MSE-loops
- Forstår livssyklus-hypotesen for lønn og hvorfor lineær modell ikke holder for alder-lønn
- Forstår at np.random.default_rng og np.random.seed bruker ulike RNG-systemer — gir ulik split med samme seed-tall
- BAN404-miljø i conda er konfigurert og verifisert — bruker `conda activate BAN404`
- Quarto-extension deaktivert i Positron (forårsaket backspace-problem)
- Inline suggestions slått av for workspace i Positron
