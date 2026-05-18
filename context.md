# Context: BAN404 — Introduction to Machine Learning

**Last updated:** 2026-05-18 (session 17)
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
- Lecture 9 (GAM + backfitting): gjennomgått og forstått ✅
- Lecture 10 (Tree-based methods): gjennomgått og forstått ✅
- Exercise 8.8 (Carseats): (a)–(f) fullført ✅
- Tutorial 2, Task 1 (a)–(e): fullført ✅ — logistisk regresjon + KNN fra scratch
- Tutorial 2, Task 4 (a)–(d): fullført ✅ — validation set + LOO på Auto-datasettet
- Tutorial 3, Task 1a: split 50/50 med np.setdiff1d fullført ✅
- Tutorial 3, Task 1b: OLS, Ridge og LASSO på Apps ~ Accept + Top10perc + Expend ✅
- Tutorial 3, Task 1c: RidgeCV og LassoCV med cv=5 ✅
- Tutorial 3, Task 1d: optimal α, koeffisientsammenligning i tabell ✅
- Tutorial 3, Task 2a: scatter av data ✅
- Tutorial 3, Task 2b: polynomregresjon K=7 med np.vander ✅
- Tutorial 3, Task 2c: regression spline med 2 knutepunkter (ξ=3, ξ=6) manuelt ✅
- Tutorial 3, Task 2d: BSpline fra ISLP.transforms — identisk med c) ✅
- Tutorial 3, Task 2e: LOO-kryss-validering for ulike knutepunktkombinasjoner ✅ — beste: (3.5, 5.5)
- Tutorial 4: alle oppgaver fullført ✅ (Task 1a–b: clustering, Task 2a–d: regression trees, Task 3a–c: RF + boosting)
- Tutorial 2 (exercise2.2.ipynb — studentens fil): debugget og forstått ✅
- Tutorial 3 (exercise3.2.ipynb — studentens fil): påbegynt — Task 1a (split) og delvis 1b (Ridge) i gang

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
| 2026-05-16 | Gjennomgått Lecture 9: GAM via backfitting med KNN-smoother (gam_example.py og .ipynb) |
| 2026-05-16 | Forstår GAM: additiv modell (y = b0 + f1(x1) + f2(x2)), ikke-parametrisk estimering, backfitting-algoritmen |
| 2026-05-16 | Forstår begreper: parametrisk vs. ikke-parametrisk, konfundert, partiell residual, konvergens |
| 2026-05-16 | Gjennomgått Lecture 10: tree-based methods (regresjonstrær, pruning, bagging, random forest, BART) |
| 2026-05-16 | Forstår rekursiv binær splitting: RSS-minimering, grådighet, stoppbetingelse |
| 2026-05-16 | Forstår tree pruning: g(T) = RSS + alpha*|T|, kryssvalidering av alpha |
| 2026-05-16 | Forstår bagging: bootstrap + gjennomsnitt av mange trær, reduserer varians |
| 2026-05-16 | Forstår random forest: m tilfeldige prediktorer per kutt, ukorrelerte trær, effekt av m |
| 2026-05-16 | Exercise 8.8 (a): train/test split 50/50 på Carseats |
| 2026-05-16 | Exercise 8.8 (b): regresjontre med min_samples_leaf=30, MSE=6.6492 |
| 2026-05-16 | Exercise 8.8 (c): pruning med cost_complexity_pruning_path + GridSearchCV (5-fold CV) |
| 2026-05-16 | Exercise 8.8 (d): bagging med RandomForestRegressor(max_features=p) |
| 2026-05-16 | Exercise 8.8 (e): random forests, loop over m-verdier, plott av MSE vs m |
| 2026-05-16 | Exercise 8.8 (f): BART fra ISLP.bart |
| 2026-05-16 | Forstår fit/predict/evaluate-syklusen grundig |
| 2026-05-16 | Forstår at y_test kun brukes til evaluering, aldri til prediksjon |
| 2026-05-16 | Forstår sklearn vs statsmodels: sklearn krever manuell X/y-split, statsmodels bruker formel med ~ |
| 2026-05-16 | Forstår MSE-tolkning: sammenligne RMSE mot std og mean i y_test |
| 2026-05-16 | Tutorial 2 Task 1: logistisk regresjon med sm.GLM/Binomial, KNN fra scratch, confusion matrix, plotting |
| 2026-05-16 | Tutorial 2 Task 4: validation set 50/50 + LOO på Auto — age og weight som signifikante prediktorer |
| 2026-05-16 | Tutorial 3 Task 1 (a)–(d): split med np.setdiff1d, OLS/Ridge/Lasso, RidgeCV/LassoCV (cv=5), koeffisientsammenligning |
| 2026-05-16 | Tutorial 3 Task 2 (a)–(e): polynomregresjon K=7, regression spline manuelt, BSpline fra ISLP, LOO for knutepunkter — beste: (3.5, 5.5) |
| 2026-05-17 | Tutorial 4 Task 1a påbegynt: K-means clustering (k=3) på Iris — StandardScaler, KMeans, pd.crosstab for sammenligning mot faktiske arter |
| 2026-05-18 | Tutorial 4 Task 1b fullført: hierarkisk clustering (complete linkage), dendrogram, cut_tree — lignende resultater som K-means |
| 2026-05-18 | Tutorial 4 Task 2a fullført: manuell RSS-minimering (minimize_scalar) for Years og Hits — Years vinner med s≈4.5 |
| 2026-05-18 | Tutorial 4 Task 2b fullført: DecisionTreeRegressor(min_samples_leaf=15, max_depth=3), plot_tree |
| 2026-05-18 | Tutorial 4 Task 2c fullført: prediksjon ny obs (Hits=125, Years=2.5) med np.exp for å konvertere fra logSalary |
| 2026-05-18 | Tutorial 4 Task 2d fullført: R² manuelt (TSS/RSS) og .score() for tree1 og tree2 med get_dummies |
| 2026-05-18 | Tutorial 4 Task 3a fullført: testMAD for tree1 (0.367) og tree2 (0.449) på testsettet |
| 2026-05-18 | Tutorial 4 Task 3b fullført: RandomForestRegressor(n_estimators=500, max_features=6) — MAD=0.299, variable importance plot |
| 2026-05-18 | Tutorial 4 Task 3c fullført: GradientBoostingRegressor(n_estimators=1000, lr=0.01, max_depth=2) — MAD=0.293 |
| 2026-05-18 | Eksamensinformasjon fra foreleser: neural networks = kun MLPRegressor fra sklearn (ikke torch, ikke CNN/RNN) |
| 2026-05-18 | Eksamensinformasjon fra foreleser: deskriptiv statistikk er kontekstavhengig — scatter, boxplot, krysstabell |
| 2026-05-18 | Mål: lage ett cheatsheet med all faglig informasjon for å printe og ta med på eksamen |
| 2026-05-18 | Laget Positron code snippets for alle BAN404-metoder — samlet i BAN404.code-snippets (15 snippets) |
| 2026-05-18 | Lagt til confusion_matrix og ConfusionMatrixDisplay i BAN404-snippet og exercise3.2.ipynb |
| 2026-05-18 | Debugget exercise2.2.ipynb: familiy→family, print(m1.summary)→print(m1.summary()), K=20 for liten data, list→np.array |
| 2026-05-18 | Forstår sm.add_constant-regler: påkrevd i sm.OLS/GLM/GLS, ikke i smf.ols (formel-API) og aldri i sklearn |
| 2026-05-18 | Forstår polynomregresjon (np.vander, K=7 med n=8 = perfekt fit, ekstrem overfitting) |
| 2026-05-18 | Forstår kubisk regresjonsspline: h(x,ξ)-funksjon, np.column_stack, 4+K parametere, BSpline fra ISLP identisk |
| 2026-05-18 | Startet exercise3.2.ipynb — fikset ytrain = train.iloc[:, 0].values (ikke 0:) |
| 2026-05-18 | Gjennomgått typisk eksamensinnhold: train/test + Ridge/Lasso + bootstrap + GAM er kjernen |
| 2026-05-18 | Eksamenstrening oppgave 1 (2025, Python-versjon): Ridge, LOO-CV, bootstrap, GAM gjennomgått |
| 2026-05-18 | Forstår Ridge: objektivfunksjon, demeaning, scipy.optimize.minimize vs analytisk løsning |
| 2026-05-18 | Forstår LOO-CV for Ridge: g_fast med np.linalg.solve, subsett for ytelse |
| 2026-05-18 | Forstår Bootstrap: trekk med tilbakelegging, samplingsfordeling, persentil-KI |
| 2026-05-18 | Forstår GAM med BSpline: identifisere ikke-linearitet med scatter + delta R², spline basis |
| 2026-05-18 | Forstår bias-variance trade-off dypt: NHH-analogi, bias=systematisk feil, varians=spredning mellom datasett |
| 2026-05-16 | Forstår OLS/Ridge/Lasso: RSS, L1/L2-straff, marginal kostnad, shrinkage, variabelseleksjon, bias-variance tradeoff |
| 2026-05-16 | Forstår CV: k-fold deler treningsdata i k deler, roterer valideringsdel, velger beste α uten å røre testsettet |
| 2026-05-16 | Forstår regression splines: cubic spline basis, h-funksjon, knutepunkter, 4+K parametere |
| 2026-05-16 | Forstår BSpline fra ISLP: intercept=True nødvendig for å matche manuell basis |
| 2026-05-16 | Forstår GLM: samlebetegnelse for modeller — Binomial=logistisk, Gaussian=OLS, Poisson=telledata |
| 2026-05-16 | Forstår maximum likelihood: finn β som gjør observerte data mest sannsynlige (vs. OLS som minimerer RSS) |
| 2026-05-16 | Forstår LOO: tren n ganger med n-1 obs, test på 1 — hver runde uavhengig, ingen hukommelse mellom runder |
| 2026-05-16 | Forstår KNN fra scratch: np.abs for avstand, np.argsort for indekser, np.mean for sannsynlighet |
| 2026-05-16 | Forstår pd.crosstab for confusion matrix — forelesers metode, bruk .values for å unngå indeks-feil |

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
- Logistisk regresjon (statsmodels): `X = sm.add_constant(x)` → `sm.GLM(y, X, family=sm.families.Binomial()).fit()`
- Predikere med logistisk modell: `prob = m1.predict(X)` → `pred = (prob > 0.5).astype(int)`
- Manuell sigmoid: `prob = 1 / (1 + np.exp(-a0 - a1 * x0))` der `a0 = m1.params[0]`, `a1 = m1.params[1]`
- KNN fra scratch: `d = np.abs(x - x0)` → `o = np.argsort(d)[:K]` → `yprob = np.mean(y[o])`
- Train/test split (korrekt metode): `train_idx = np.random.choice(n, n//2, replace=False)` → `test_idx = np.setdiff1d(np.arange(n), train_idx)` → `train = df.iloc[train_idx]`, `test = df.iloc[test_idx]`. Merk: `~train_idx` på heltallsarray gir bitvis NOT (feil!) — kun `~` på boolsk maske er korrekt
- y fra kategorisk: `y_train = (train['y'] == 'high').astype(int)`
- Confusion matrix (forelesers metode): `pd.crosstab(y_test.values, pred.values, rownames=['Actual'], colnames=['Predicted'])`
- Accuracy: `np.mean(y_test == pred)`
- LOO-loop: `for i in range(n): train = auto[np.arange(n) != i]; test = auto.iloc[[i]]` — tren og prediker én og én
- `X_test.reindex(columns=X_train.columns, fill_value=1)` — sikrer samme kolonner (inkl. const) i LOO

## Exam Cheat-Sheet Topics
*(Samles her — skal formateres til ferdig printbart ark)*
**Mål:** Ett kompakt ark med all faglig informasjon som kan printes og tas med på eksamen.


### Logistisk regresjon — teori
- Formel: P(Y=1|X=x) = 1 / (1 + e^(−β0 − β1·x))
- Brukes når Y er binær (0/1) — ikke kontinuerlig
- Estimeres med maximum likelihood (ikke OLS/minste kvadrat)
- Koeffisienter: positiv β1 → høyere x gir høyere P(Y=1)
- Prediksjon: P > 0.5 → Y=1, P ≤ 0.5 → Y=0 (standard terskel)
- sm.GLM med Binomial() og Logit link-funksjon

### KNN — teori
- P(Y=1|X=x0) = (1/K) * antall naboer med y=1
- Lav K → overfit (følger støy), høy K → underfit (for glatt)
- Bryter uavgjort med lavest indeks (argsort-rekkefølge)
- Gjennomsnittet av 0-er og 1-er er alltid mellom 0 og 1 → kan tolkes som sannsynlighet

### ⚠️ Når må du legge til `sm.add_constant()`?

| Bibliotek | Metode | add_constant nødvendig? |
|---|---|---|
| statsmodels | `sm.OLS`, `sm.GLM`, `sm.GLS` | **JA** — alltid |
| statsmodels | `smf.ols('y ~ x', data=df)` | **NEI** — formula API gjør det automatisk |
| sklearn | Ridge, Lasso, KNN, LDA, QDA, MLPRegressor, osv. | **NEI** — `fit_intercept=True` er default |

**Viktig:** Manglende `add_constant` gir ingen feilmelding — modellen kjører, men gir **gale koeffisienter** uten advarsel.

```python
# RIKTIG — statsmodels uten formel:
X = sm.add_constant(x)
m = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# RIKTIG — statsmodels med formel (ingen add_constant):
m = smf.ols('y ~ x', data=df).fit()

# RIKTIG — sklearn (ingen add_constant):
ridge = skm.Ridge(alpha=1000)
ridge.fit(X_train, y_train)
```

### GLM — teori
- Generalized Linear Model: samlebetegnelse
- Binomial() → logistisk regresjon (sigmoid/logit link)
- Gaussian() → vanlig OLS (identity link)
- Poisson() → telledata
- sm.add_constant() må kalles eksplisitt (sklearn gjør det automatisk)

### Confusion matrix
```
               Predikert 0   Predikert 1
Faktisk 0      TN (riktig)   FP (feil)
Faktisk 1      FN (feil)     TP (riktig)
```
- Accuracy = (TN + TP) / totalt
- pd.crosstab(y_true.values, pred.values) — forelesers metode

### OLS vs Ridge vs LASSO — teori
- OLS: minimerer kun RSS → ingen begrensning på koeffisienter → kan gi store, motvirkende β ved korrelerte variabler → overfitting
- Ridge: minimerer RSS + α·Σβₖ² (L2-straff). Straffen vokser raskere jo større β er (marginal kostnad øker) → koeffisienter krymper mot null, men aldri nøyaktig null
- LASSO: minimerer RSS + α·Σ|βₖ| (L1-straff). Straffen vokser likt per enhet uansett størrelse (konstant marginal kostnad) → koeffisienter kan bli nøyaktig null → automatisk variabelseleksjon
- Hvorfor LASSO setter til null: |β| koster like mye å gå fra 0.01→0 som fra 2→1.99. Ridge koster nesten ingenting å gå fra 0.01→0 (0.01²≈0), så den lar det være
- α=0 → identisk med OLS for begge
- Optimal α velges med kryss-validering (RidgeCV / LassoCV)

### Bias-variance tradeoff
- **Varians**: hvor mye modellen endrer seg ved nye treningsdata. Kompleks modell (lav α) lærer støy → ustabile koeffisienter → høy varians
- **Bias**: hvor systematisk feil modellen er. Enkel modell (høy α) ignorerer ekte mønstre → konsistent feil → høy bias
- Høy α → koeffisienter → 0 → enkel modell → høy bias, lav varians
- Lav α → frie koeffisienter → kompleks modell → lav bias, høy varians
- Mål: finn α som minimerer testfeil (balanse mellom bias og varians)
- Gjelder generelt: OLS har lav bias men høy varians; Ridge/LASSO introduserer litt bias mot lavere varians

### Validation set approach
- Del data tilfeldig 50/50 i train/test
- Tren på train, evaluer på test
- Enkel men variabel — avhenger av hvilken split man fikk

### LOO (Leave-One-Out)
- Tren n modeller — hver gang holder ut én observasjon
- Alle prediksjoner er "ekte" testprediksjoner (aldri sett under trening)
- Mer konservativ enn 50/50-split, men tregere (n ganger treningskostnad)
- `train = auto[np.arange(n) != i]`, `test = auto.iloc[[i]]`

### Nøkkelkode — logistisk regresjon
```python
X = sm.add_constant(x)
m1 = sm.GLM(y, X, family=sm.families.Binomial()).fit()
prob = m1.predict(X)                      # sannsynligheter
pred = (prob > 0.5).astype(int)          # prediksjoner
# Manuell sigmoid for ett punkt:
prob = 1 / (1 + np.exp(-a0 - a1 * x0))
```

### Nøkkelkode — KNN fra scratch
```python
def knn(x0, x, y, K=20):
    d = np.abs(x - x0)         # avstander
    o = np.argsort(d)[:K]      # indekser til K nærmeste
    return np.mean(y[o])       # sannsynlighet
```

### Nøkkelkode — train/test split
```python
np.random.seed(123)
n = len(df)
train_idx = np.random.choice(n, size=n//2, replace=False)
test_idx  = np.setdiff1d(np.arange(n), train_idx)   # korrekt komplement
train = df.iloc[train_idx]
test  = df.iloc[test_idx]
```

### Nøkkelkode — LOO
```python
pred = np.zeros(n, dtype=int)
for i in range(n):
    train = auto[np.arange(n) != i]
    test  = auto.iloc[[i]]
    y_train = (train['y'] == 'high').astype(int)
    X_train = sm.add_constant(train[['age', 'weight']])
    X_test  = test[['age', 'weight']].reindex(columns=X_train.columns, fill_value=1)
    m1 = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    pred[i] = int(m1.predict(X_test).iloc[0] > 0.5)
y_true = (auto['y'] == 'high').astype(int)
pd.crosstab(y_true.values, pred, rownames=['Actual'], colnames=['Predicted'])
```

### Nøkkelkode — np.where for kategorisk y
```python
auto['y'] = np.where(auto['mpg'] > auto['mpg'].median(), 'high', 'low')
auto['y'] = auto['y'].astype('category')
```

### Clustering — teori
- **K-means:** velg K, tilordne til nærmeste sentroid, oppdater sentroider, gjenta. Krever StandardScaler. Ikke-deterministisk (n_init=10 for stabilitet)
- **Hierarkisk (agglomerativ):** start med n clusters, slå sammen to nærmeste om gangen. Resulterer i dendrogram. Ingen K på forhånd
- **Complete linkage:** avstand mellom clusters = max avstand mellom punktpar. Gir kompakte grupper
- **Dendrogram:** y-akse = avstand ved sammenslåing. Horisontalt kutt ved K gir K clusters. Store hopp = naturlige skiller
- **cut_tree:** kutter dendrogram ved ønsket K, returnerer cluster-tilhørighet for hver obs

### Clustering — nøkkelkode
```python
# K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3, random_state=1, n_init=10)
km.fit(X_scaled); labels = km.labels_
pd.crosstab(labels, y)

# Hierarkisk
hc = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
hc.fit(X); linkage_mat = compute_linkage(hc)
dendrogram(linkage_mat, no_labels=True)
clusters = cut_tree(linkage_mat, n_clusters=3).reshape(-1)
```

### Regresjons-trær — teori
- Splitter prediktorrommets rekursivt i rektangler. Prediksjon = gjennomsnitt av y i region
- **Første split:** finn variabel X_j og kuttepunkt s som minimerer RSS = Σ(y−ȳ_R1)² + Σ(y−ȳ_R2)²
- Ubegrenset tre → overfitter (samples=1 i blader, R²=1)
- Begrens med `min_samples_leaf` og `max_depth`
- **R²** = (TSS−RSS)/TSS. TSS = Σ(y−ȳ)², RSS = Σ(y−ŷ)²

### Regresjons-trær — nøkkelkode
```python
# Manuell første split
def RSS(s, x1, y):
    yR1 = y[x1 < s].mean(); yR2 = y[x1 >= s].mean()
    return ((y[x1<s]-yR1)**2).sum() + ((y[x1>=s]-yR2)**2).sum()
res = minimize_scalar(RSS, bounds=(0,50), args=(x1, y), method='bounded')

# DecisionTreeRegressor
tree = DecisionTreeRegressor(min_samples_leaf=15, max_depth=3, random_state=0)
tree.fit(X_train, y_train)
tree.predict(X_test)
tree.score(X_train, y_train)  # R²

# R² manuelt
TSS = ((y - y.mean())**2).sum()
RSS = ((y - tree.predict(X))**2).sum()
R2 = (TSS - RSS) / TSS
```

### MAD (Mean Absolute Deviation)
- MAD = mean(|y − ŷ|) — gjennomsnittlig absolutt avvik
- Mindre sensitiv for ekstreme feil enn MSE
```python
def MAD(y, pred): return np.mean(np.abs(y - pred))
```

### Random Forest og Boosting — teori
| Metode | Trær | Strategi | Nøkkelparameter |
|---|---|---|---|
| Enkelt tre | 1 | Rekursiv splitting | min_samples_leaf, max_depth |
| Bagging | Mange | Bootstrap + gjennomsnitt | n_estimators |
| Random Forest | Mange | Bootstrap + tilfeldig m prediktorer | max_features |
| Boosting | Mange | Sekvensielt på residualer | learning_rate, n_estimators |

- **Random Forest:** `max_features=m` → kun m prediktorer vurderes per split → trærne ukorrelerte
- **Boosting:** lav learning_rate + mange svake trær → gradvis forbedring

### Random Forest og Boosting — nøkkelkode
```python
# Forbered data med dummies
X2train = train.drop(columns=['Salary','logSalary'])
X2train = pd.get_dummies(X2train, drop_first=True)

# Random Forest
rf = RandomForestRegressor(n_estimators=500, max_features=6, random_state=123)
rf.fit(X2train, y_train)
importances = pd.Series(rf.feature_importances_, index=X2train.columns)
importances.sort_values().plot.barh()

# Boosting
boost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,
                                   max_depth=2, random_state=123)
boost.fit(X2train, y_train)
```

### Ridge-regresjon — teori
- Minimerer: RSS + α · sum(β²)  ←  L2-straff
- Høy α → koeffisienter krympes mot null (men aldri eksakt null)
- α=0 → identisk med OLS
- sklearn håndterer standardisering og intercept automatisk (fit_intercept=True som default)
- Optimal α velges med RidgeCV (k-fold CV)
- Bias-variance: α opp → mer bias, mindre varians → lavere test-MSE når data er støyfull

### Ridge-regresjon — kode (forelesers metode)
```python
import sklearn.linear_model as skm

# Fit med gitt alpha
Ridge = skm.Ridge(alpha=10000)
Ridge.fit(X_train, y_train)
print(Ridge.coef_)                          # Koeffisienter (uten intercept)

result_Ridge = Ridge.predict(X_test)
MSE_Ridge = ((y_test - result_Ridge)**2).mean()

# Finn optimal alpha med RidgeCV (k-fold)
alphas    = np.logspace(-6, 6, 100)
Ridge_CV  = skm.RidgeCV(alphas=alphas, cv=5)
Ridge_CV.fit(X_train, y_train)
print(f"Optimal alpha: {Ridge_CV.alpha_:.4f}")

# Fit med optimal alpha
Ridge_opt = skm.Ridge(alpha=Ridge_CV.alpha_)
Ridge_opt.fit(X_train, y_train)
MSE_opt   = ((y_test - Ridge_opt.predict(X_test))**2).mean()
```

### Ridge-regresjon — manuell utregning uten pakker
```python
# Analytisk løsning: b = (X'X + αI)^{-1} X'y
# (sklearn gjør dette internt — her vises mekanismen)
Xc = np.column_stack([np.ones(len(X)), X])  # legg til intercept-kolonne
alpha = 10
p  = Xc.shape[1]
A  = Xc.T @ Xc + alpha * np.eye(p)         # (X'X + αI)
b  = Xc.T @ y                              # X'y
b_ridge = np.linalg.solve(A, b)            # Løs lineært system
y_pred  = Xc @ b_ridge
mse     = np.mean((y - y_pred)**2)
```

### Bootstrap — teori
- Simulerer samplingsfordeling uten ny data
- Trekk n obs med tilbakelegging fra datasettet → beregn statistikken → gjenta B ganger
- Gir B verdier av statistikken → dette ER samplingsfordelingen
- 95% KI: persentil-metoden → [2.5%, 97.5%] av bootstrap-verdiene
- ddof=1 i np.var → S² (stikkprøvevarians, deler på n-1, upartisk)

### Bootstrap — kode (for varians)
```python
np.random.seed(42)
n = len(y);  B = 1000
boot_vars = np.array([
    np.var(y[np.random.choice(n, n, replace=True)], ddof=1)
    for _ in range(B)
])
ci_low, ci_high = np.percentile(boot_vars, [2.5, 97.5])
plt.hist(boot_vars, bins=40)
plt.axvline(np.var(y, ddof=1), color='red', linestyle='--')
```

### Bootstrap — manuell utregning
```python
# Én bootstrap-iterasjon for hånd:
idx      = np.random.choice(n, n, replace=True)   # Trekk med tilbakelegging
y_boot   = y[idx]                                  # Bootstrap-stikkprøve
s2_boot  = np.sum((y_boot - np.mean(y_boot))**2) / (n - 1)  # S² manuelt
# Gjenta B ganger og samle s2_boot-verdiene
```

### GAM med BSpline — teori
- Generalisert additiv modell: y = b0 + f1(x1) + f2(x2) + ... + fq(xq)
- Hver fj kan være ikke-lineær (spline, polynom, KNN-smoother)
- Identifiser ikke-lineære variabler: sammenlign lineær R² vs. kvadratisk R²
  - Stor delta R² → variabelen er ikke-lineær → bruk spline
- BSpline: cubic spline basis med interne knutepunkter
- GAM gir lavere trenings-MSE enn OLS hvis ikke-lineære sammenhenger finnes

### GAM med BSpline — kode (forelesers metode)
```python
from ISLP.transforms import BSpline

# Identifiser ikke-linearitet
for i, col in enumerate(var_names):
    xi     = X[:, i]
    r2_lin  = sm.OLS(y, sm.add_constant(xi)).fit().rsquared
    r2_quad = sm.OLS(y, sm.add_constant(
                  np.column_stack([xi, xi**2]))).fit().rsquared
    print(f"{col}: delta R² = {r2_quad - r2_lin:.3f}")

# Splines for ikke-lineære variabler (f.eks. x2 og x3)
bs2 = BSpline(internal_knots=[-1, 0, 1], degree=3, intercept=False)
X2_sp = bs2.fit_transform(X[:, 1].reshape(-1, 1))

# Sett sammen GAM-designmatrise
X_gam = np.column_stack([np.ones(len(y)), X[:, 0], X2_sp, X[:, 2]])
gam   = sm.OLS(y, X_gam).fit()

# MSE og R²
mse_gam = np.mean((y - gam.predict(X_gam))**2)
r2_gam  = 1 - mse_gam / np.var(y, ddof=1)
```

### GAM — manuell utregning uten pakker
```python
# Kubisk spline manuelt (ett knutepunkt ξ):
def h(x, xi): return np.maximum((x - xi)**3, 0)

xi = 0.0
X_spline = np.column_stack([
    np.ones(len(x)), x, x**2, x**3, h(x, xi)
])
model = sm.OLS(y, X_spline).fit()
mse   = np.mean((y - model.predict(X_spline))**2)
r2    = 1 - mse / np.var(y, ddof=1)
```

### Polynomregresjon — designmatrise med np.vander
```python
# Grad K=7 → 8 kolonner: [1, x, x², x³, x⁴, x⁵, x⁶, x⁷]
X = np.vander(x, K + 1, increasing=True)  # K+1 kolonner, inkl. intercept
model = sm.OLS(y, X).fit()                # intercept allerede i X → ingen add_constant
y_pred = model.predict(X)
```
- `np.vander(x, 8, increasing=True)` gir Vandermonde-matrise med 8 kolonner
- Intercept er inkludert i kolonnen av enere → **ikke** bruk `sm.add_constant()`
- K=7 med n=8 → 8 parametere, 8 obs → **perfekt fit** (R²=1, RSS=0) — ekstrem overfitting

### Kubisk regresjonsspline — manuell og BSpline

**Manuelt (h-funksjon):**
```python
def h(x, xi):
    return np.where(x > xi, (x - xi)**3, 0)  # truncated cubic

# To knutepunkter ξ₁=3, ξ₂=6 → 6 kolonner: [1, x, x², x³, h(x,3), h(x,6)]
X_spline = np.column_stack([
    np.ones_like(x), x, x**2, x**3,
    h(x, 3), h(x, 6)
])
model_spline = sm.OLS(y, X_spline).fit()
y_pred = model_spline.predict(X_spline)
```

**Med BSpline fra ISLP (gir identiske prediksjoner):**
```python
from ISLP.transforms import BSpline
bs = BSpline(internal_knots=[3, 6], degree=3, intercept=True)
X_bs = bs.fit_transform(x.reshape(-1, 1))
model_bs = sm.OLS(y, X_bs).fit()

# Verifiser at de er like:
print(np.allclose(model_spline.predict(X_spline), model_bs.predict(X_bs)))  # True
```
- Kubisk spline med K knutepunkter → **4 + K parametere**
- `intercept=True` i BSpline er nødvendig for å matche manuell basis
- Intercept er inkludert i basis → **ikke** bruk `sm.add_constant()`

### Neural Networks (MLPRegressor) — teori
- Kun `MLPRegressor` fra `sklearn.neural_network` er pensum (ikke torch, ikke CNN/RNN)
- Skjult lag: K lineærkombinasjoner kjørt gjennom aktivasjonsfunksjon (ReLU eller sigmoid)
- ReLU: g(z) = max(0, z). Sigmoid: g(z) = e^z / (1+e^z)
- Minimerer RSS(β, ω) via gradient descent — ikke-konveks problem → lokal minimum

### Deskriptiv statistikk — eksamensmønster
- "Finn lovende prediktorer" → velg plot basert på variabeltyper:
  - Kontinuerlig Y + kontinuerlig X → **scatterplot**
  - Kontinuerlig Y + kategorisk X → **boxplot** (Y for hver kategori)
  - Kategorisk Y + kontinuerlig X → **boxplot** (X for hver kategori)
  - Kategorisk Y + kategorisk X → **krysstabell** (gjerne normalisert: P(Y|X))

## Open Questions / Blockers
- Exercise 2.8 (h): utforsk datasettet fritt og kommenter funn — ikke påbegynt
- Exercise 3.9 og 3.10 ikke påbegynt
- Tutorial 1, Task 2: ikke påbegynt (College-datasettet, train/test split, MSE, KNN)
- Exercise 4.13 (i): sammenligne alle metoder og velge beste — gjenstår
- Exercise 4.13 (j): eksperimentere med ulike prediktor-kombinasjoner og K-verdier — gjenstår
- Exercise 6.11 del b) og c) (tolkningsceller) — gjenstår
- Tutorial 3, Task 1e: evaluere OLS/Ridge/Lasso med testMSE — gjenstår

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
- `Foreleser/09/gam_example.py` — forelesers GAM-eksempel med backfitting og KNN
- `Foreleser/09/gam_example.ipynb` — notebook-versjon av GAM-eksempel (startpunkt)
- `Foreleser/10/ex88ac.ipynb` — forelesers startkode for Exercise 8.8 (a)–(c)
- `Foreleser/10/ex88_solution.ipynb` — løsning Exercise 8.8 (a)–(f) ferdig ✅
- `Foreleser/Tutorial 2/exercise2.ipynb` — løsning Tutorial 2 — Task 1 og Task 4 ferdig ✅
- `Foreleser/Tutorial 2/exercise2.pdf` — oppgavetekst Tutorial 2
- `Foreleser/Tutorial 2/exercise2_solutions.pdf` — forelesers løsningsforslag Tutorial 2
- `Foreleser/Tutorial 3/exercise3.ipynb` — løsning Tutorial 3 — Task 1 og Task 2 ferdig ✅
- `Foreleser/Tutorial 3/exercise3.pdf` — oppgavetekst Tutorial 3
- `Foreleser/Tutorial 3/exercise3_solutions.pdf` — forelesers løsningsforslag Tutorial 3
- `Foreleser/Tutorial 4/exercise4.ipynb` — løsning Tutorial 4 — alle tasks fullført ✅
- `Foreleser/Tutorial 4/exercise4.pdf` — oppgavetekst Tutorial 4
- `Foreleser/Tutorial 4/iris.csv` — Iris-datasettet (150 obs, 4 mål + type)
- `Foreleser/Previous exam - in R/2025/exam_task1.ipynb` — Python-versjon av eksamen 2025, oppgave 1 (student fyller inn)
- `Foreleser/Previous exam - in R/2025/2025.ipynb` — løsninger til eksamen 2025, oppgave 1
- `Data for exam/task1_data.csv` — eksamensdata: 5000 obs, x1–x6 + y (ikke-lineært: x2 og x3)
- `~/Library/Application Support/Positron/User/snippets/BAN404.code-snippets` — Positron-snippets for alle metoder (prefix: ban404, ols, ridge, lasso, logreg, knn, lda, qda, nb, tree, rf, boost, mlp, gam, kmeans, hclust)

## Jupyter → PDF-konvertering
```bash
# Enkel konvertering (notebook må være kjørt på forhånd):
jupyter nbconvert --to pdf "Foreleser/13/ames.ipynb"

# Kjør alle celler automatisk og konverter:
jupyter nbconvert --to pdf --execute "Foreleser/13/ames.ipynb"

# Alternativ via HTML (sikrere, unngår LaTeX-feil):
jupyter nbconvert --to html --execute "Foreleser/13/ames.ipynb"
# Åpne HTML i nettleser → Cmd+P → Save as PDF
```
- Alltid oppgi full relativ sti fra BAN404-roten
- `--execute` kjører cellene under konverteringen (krever ikke lagret output)
- HTML-ruten anbefales hvis PDF-konverteringen kutter tidlig (LaTeX-feil)

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
- Forstår GAM og backfitting: additiv modell, KNN-smoother, partiell residual, konvergens
- Forstår Ridge dypt: objektivfunksjon, demeaning, scipy vs analytisk, bias-variance mekanisme
- Forstår bootstrap: tilbakelegging, samplingsfordeling, persentil-KI, ddof=1
- Forstår GAM med BSpline: delta R² for å identifisere ikke-linearitet, spline basis, sammenligning MSE
- Forstår LOO for Ridge: mask-teknikken, g_fast med linalg.solve, subsettstrategi for store n
- Forstår bias-variance dypt: NHH-analogi (bias=spørre på biblioteket, varians=få studenter)
- Forstår regresjonstrær: rekursiv binær splitting, RSS-minimering, bladprediksjon, overfitting
- Forstår pruning: complexity penalty alpha, GridSearchCV for valg av alpha
- Forstår bagging vs. random forest: bagging bruker alle p prediktorer, RF bruker m < p
- Forstår fit/predict/evaluate-syklusen og train/test-logikken fullt ut
- Forstår at y_test brukes kun til evaluering — aldri som input til predict
- Forstår sklearn vs statsmodels: sklearn alltid manuell X/y-split, statsmodels via formel
- Forstår GLM/Binomial grundig: sigmoid holder P mellom 0 og 1, IRLS finner β via maximum likelihood
- Forstår maximum likelihood: multipliser P(faktisk utfall) for alle obs — finn β som maksimerer produktet
- Forstår KNN for klassifikasjon: gjennomsnittet av K nærmeste y-verdier (0/1) er en sannsynlighet
- Forstår LOO vs. validation set: LOO mer konservativ (bruker n-1 til trening), ingen tilfeldig variabilitet
- Forstår at np.argsort returnerer indekser, ikke verdier — nødvendig for å koble avstand til y
- Forstår at pd.crosstab trenger .values når y har navngitt indeks (f.eks. bilnavn fra Auto)
- BAN404-miljø i conda er konfigurert og verifisert — bruker `conda activate BAN404`
- Quarto-extension deaktivert i Positron (forårsaket backspace-problem)
- Inline suggestions slått av for workspace i Positron
