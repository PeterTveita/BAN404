# Statistiske begreper — BAN404

> En oversikt over sentrale begreper i statistikk og regresjonsanalyse, med eksempler og en gjennomgående case.

---

## Innhold

1. [Varians og standardavvik](#1-varians-og-standardavvik)
2. [Nullhypotesen](#2-nullhypotesen-h)
3. [P-verdi](#3-p-verdi)
4. [Statistisk signifikans](#4-statistisk-signifikans)
5. [Konfidensintervall](#5-konfidensintervall)
6. [Sammenhengen mellom begrepene](#6-sammenhengen-mellom-begrepene)
7. [Case — påvirker studietimer karakteren?](#7-case--påvirker-studietimer-karakteren)

---

## 1. Varians og standardavvik

### Hva måler vi?

Vi vil ha ett tall som beskriver **hvor spredt dataen er rundt gjennomsnittet**.

```
Gruppe A:  [9, 10, 11]       → tett rundt 10  →  liten spredning
Gruppe B:  [1, 10, 19]       → langt fra 10   →  stor spredning
```

Begge har snitt = 10, men er fundamentalt forskjellige. Vi trenger et mål som skiller dem.

---

### Hvorfor kvadrere avvikene?

Et naturlig forsøk er å summere avvikene fra snittet — men positive og negative avvik kansellerer hverandre og gir alltid 0. Løsningen er å kvadrere:

```
Avvik:           -1,  0,  1
Kvadrert:         1,  0,  1   ← alltid positivt
```

Kvadrering har en ekstra fordel: **store avvik straffes hardere enn små.**

```
Avvik på 1  →  kvadrert: 1    (lite ekstrastraff)
Avvik på 5  →  kvadrert: 25   (mye ekstrastraff)
```

---

### Formlene

**Varians** — gjennomsnittet av de kvadrerte avvikene:

$$\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}$$

**Standardavvik** — kvadratroten av variansen, slik at enheten er den samme som dataen:

$$\sigma = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}}$$

---

### Eksempel

```
Data:           [9, 10, 11]      snitt = 10
Avvik:          -1,  0,  1
Kvadrert:        1,  0,  1

Varians       = (1 + 0 + 1) / 3  =  0.67
Standardavvik = √0.67             ≈  0.82
```

> **Tolkning:** et typisk datapunkt ligger omtrent **±0.82 fra snittet på 10**.

| Mål | Formel | Ulempe | Bruk |
|---|---|---|---|
| Varians | Snitt av kvadrerte avvik | Kvadrerte enheter — vanskelig å tolke | Matematiske beregninger |
| Standardavvik | √varians | — | Kommunikasjon og tolkning |

---

## 2. Nullhypotesen (H₀)

### Ideen

Før du ser på data setter du opp en "ingen effekt"-påstand som utgangspunkt. Du starter alltid med å anta at H₀ er sann, og leter etter bevis mot den.

> **Tenk på H₀ som en skeptiker:** "Bevis det. Inntil da antar jeg at ingenting skjer."

---

### I regresjonssammenheng

Nullhypotesen er at koeffisienten er lik 0 — altså at x ikke forklarer noe av y:

| Hypotese | Påstand |
|---|---|
| **H₀** | α₁ = 0 — x har ingen effekt på y |
| **H₁** | α₁ ≠ 0 — x har en effekt på y |

Hvis α₁ = 0 forsvinner x helt fra modellen:

```
ŷ = α₀ + 0 · x  =  α₀    ← samme svar uansett x
```

---

## 3. P-verdi

### Hva er spørsmålet?

> "Hvor sannsynlig er det å observere disse dataene, **gitt at H₀ er sann**?"

---

### Skalaen

```
p = 0.001  →  0.1%  →  ekstremt usannsynlig tilfeldig  →  svært sterke bevis mot H₀
p = 0.03   →  3%    →  lite sannsynlig tilfeldig        →  sterke bevis mot H₀
p = 0.20   →  20%   →  ganske sannsynlig tilfeldig      →  svake bevis mot H₀
p = 0.80   →  80%   →  veldig sannsynlig tilfeldig      →  ingen bevis mot H₀
```

---

### Den viktigste misforståelsen

| Vanlig feil | Riktig tolkning |
|---|---|
| "Vi er 97% sikre på sammenhengen" | **Feil** |
| "3% sjanse for disse dataene hvis H₀ er sann" | **Riktig** |

P-verdien sier noe om **dataen gitt H₀** — ikke om **H₀ gitt dataen**. Det er subtilt, men viktig.

---

## 4. Statistisk signifikans

### Grensen

Forskere har blitt enige om en standard grense på **p < 0.05**:

```
p = 0.03  <  0.05  →  signifikant      →  forkast H₀
p = 0.08  >  0.05  →  ikke signifikant →  behold H₀
```

---

### Ulike nivåer

| Signifikansnivå | Grense | Krav til bevis |
|---|---|---|
| 10%-nivå | p < 0.10 | Lavt |
| **5%-nivå** | **p < 0.05** | **Standard** |
| 1%-nivå | p < 0.01 | Strengt |

---

### Advarsel

> Signifikant betyr **ikke** at effekten er stor eller viktig — bare at den sannsynligvis er **reell**.

En bitte liten effekt kan være høyst signifikant med nok data, men likevel være uvesentlig i praksis.

---

## 5. Konfidensintervall

### Hva er det?

Når vi estimerer en koeffisient fra data, får vi aldri den eksakte sanne verdien — bare et estimat med usikkerhet. Konfidensintervallet beskriver denne usikkerheten.

---

### Riktig tolkning av 95% KI

> "Hvis vi gjentok undersøkelsen 100 ganger, ville **95 av de 100 intervallene** inneholde den sanne verdien."

Den sanne verdien er fast — det er **intervallet** som varierer fra datasett til datasett.

---

### Dartspilleranalogien

Tenk på en dartspiller som treffer blinken 95% av kastene. Etter ett kast vet du ikke om akkurat *det* kastet traff — men du vet at **metoden er 95% pålitelig**.

---

### Koblingen til signifikans

```
α₁ = 1.702,   95% KI: [0.8, 2.6]

Inneholder intervallet 0?  →  Nei  →  Signifikant ✓
```

| Intervall | Tolkning |
|---|---|
| Smalt | Lite usikkerhet — mye data eller liten spredning |
| Bredt | Stor usikkerhet — lite data eller stor spredning |
| Inneholder ikke 0 | Statistisk signifikant |
| Inneholder 0 | Ikke statistisk signifikant |

---

## 6. Sammenhengen mellom begrepene

```
                    Standardavvik
                          │
              Beskriver spredning i dataen
                          │
                          ▼
         Høy spredning → mer usikkerhet i estimatene
                          │
                          ▼
              Bredere konfidensintervall
                          │
                          ▼
     P-verdien tester om koeffisienten er så langt fra 0
     at det ikke kan forklares av usikkerheten alene
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
         p < 0.05               p > 0.05
        Signifikant           Ikke signifikant
        Forkast H₀             Behold H₀
```

---

## 7. Case — påvirker studietimer karakteren?

Du er forsker og vil undersøke om antall studietimer per uke påvirker karakteren studenter får på eksamen. Du samler inn data fra **100 studenter**.

---

### Steg 1 — Forstå dataen

```
Karakterer:   [2, 3, 4, 4, 5, 6, 3, 5, 4, ...]
Snitt:        4.0 karakterpoeng
```

**Beregn varians:**
```
(2-4)² + (3-4)² + (4-4)² + ...
─────────────────────────────── = 1.44
              100
```

**Beregn standardavvik:**
```
√1.44 = 1.2 karakterpoeng
```

> Et typisk datapunkt ligger ±1.2 karakterpoeng fra snittet — de fleste mellom 2.8 og 5.2. Dette virker rimelig.

---

### Steg 2 — Sett opp hypoteser

| | |
|---|---|
| **H₀** | Studietimer har **ingen** effekt på karakter — α₁ = 0 |
| **H₁** | Studietimer har en effekt på karakter — α₁ ≠ 0 |

---

### Steg 3 — Kjør regresjon

```
karakter = 2.1 + 0.8 · studietimer
```

| Koeffisient | Navn | Verdi | Tolkning |
|---|---|---|---|
| α₀ | Intercept / konstantledd | 2.1 | Forventet karakter uten studietimer |
| α₁ | Stigningstall / slope | 0.8 | Hver ekstra studietime øker karakter med 0.8 |

---

### Steg 4 — Vurder usikkerheten

Det høye standardavviket (1.2) betyr mye støy i dataen, som skaper usikkerhet i estimatene.

**95% konfidensintervall for α₁: [0.5, 1.1]**

```
Inneholder 0?  Nei  →  studietimer har sannsynligvis en reell effekt ✓
```

Hadde standardavviket vært større → bredere intervall → mer usikkerhet.

---

### Steg 5 — Tolk p-verdien

**p-verdi = 0.002**

```
Hvis H₀ er sann (ingen effekt):
sannsynlighet for å observere disse dataene = 0.2%
```

α₁ = 0.8 er så langt fra 0 at det neppe skyldes tilfeldigheter alene.

---

### Steg 6 — Konklusjon

```
p = 0.002  <  0.05  →  statistisk signifikant  →  forkast H₀
```

> *"Vi forkaster nullhypotesen på 5%-nivå. Studietimer har en statistisk signifikant positiv effekt på karakter. For hver ekstra studietime øker forventet karakter med 0.8 poeng. Vi er rimelig sikre på at den sanne effekten ligger mellom 0.5 og 1.1 poeng."*

---

### Hva vi ikke kan si

| Påstand | Status |
|---|---|
| "Vi er 99.8% sikre på sammenhengen" | Feil — p-verdien sier ikke dette |
| "Studietimer forårsaker bedre karakterer" | Kan ikke konkluderes — bare korrelasjon |

> **Korrelasjon ≠ kausalitet.** Kanskje er flinke studenter de som både studerer mye *og* får gode karakterer — uavhengig av hverandre.

---

## Vedlegg — rettssaksanalogien

| Rettssak | Statistikk |
|---|---|
| Tiltalte er uskyldig inntil motsatt er bevist | H₀ er sann inntil motsatt er bevist |
| Bevisene mot tiltalte | Dataen din |
| Styrken på bevisene | P-verdien |
| Sterk nok bevis → skyldig | p < 0.05 → forkast H₀ |
| Frikjent ≠ uskyldig | Beholder H₀ ≠ H₀ er sann |
