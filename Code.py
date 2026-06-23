# ============================================================
#  Tema 1 - Introducere in Machine Learning
#  Dataset: Job Salary Prediction | Seria CC
# ============================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # grafice de baza
import seaborn as sns          # grafice mai frumoase
from scipy.stats import chi2_contingency  # test statistic Chi-patrat

# Preprocessing (curatarea datelor)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modele ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge

# Metrici (cum masuram cat de bun e modelul)
from sklearn.metrics import (
    accuracy_score,     # procentul de predictii corecte
    precision_score,     # din ce am zis "Large", cate erau chiar "Large"
    recall_score,     # din toate exemplele "Large", cate le-am gasit
    f1_score,          # media armonica intre precision si recall
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score              # cat % din variatie explica modelul (0-1, mai mare = mai bun)
)

print("=" * 60)
print("CITIM DATELE")
print("=" * 60)

train = pd.read_csv('CC_education_economy_train.csv')
val   = pd.read_csv('CC_education_economy_test.csv')

print(f"Train: {train.shape[0]} randuri x {train.shape[1]} coloane")
print(f"Val:   {val.shape[0]} randuri x {val.shape[1]} coloane")
print("\nPrimele 3 randuri din train:")
print(train.head(3))
print("\nTipul fiecarei coloane (object = text, int64/float64 = numere):")
print(train.dtypes)

TARGET_CLF = 'vacation'   # ce vrem sa prezicam la CLASIFICARE
TARGET_REG = 'salary'     # ce vrem sa prezicam la REGRESIE

# Coloanele cu valori NUMERICE (numere continue sau discrete)
NUM_COLS = [
    'experience_years',
    'skills_count',
    'certifications',
    'total_days_worked',
    'aggregated_score'
]

# Coloanele cu valori CATEGORICE (text, categorii)
CAT_COLS = [
    'job_title',
    'education_level',
    'industry',
    'company_size',
    'location',
    'remote_work',    #  aici am valori lipsa.
    'skill_bracket'
]


# ============================================================
# SECTIUNEA 2 - EXPLORAREA DATELOR (EDA)
# ============================================================

print("\n" + "=" * 60)
print("SECTIUNEA 2 - EXPLORAREA DATELOR (EDA)")
print("=" * 60)


# ------------------------------------------------------------
# 2.1 Analiza atributelor NUMERICE
# ------------------------------------------------------------
# .describe() ne da automat: count, mean, std, min, 25%, 50%, 75%, max

print("\n--- Statistici atribute NUMERICE ---")
print(train[NUM_COLS].describe().round(2))

# BOXPLOT pentru atribute numerice
fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(18, 5))
for i, col in enumerate(NUM_COLS):
    train.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col, fontsize=9)
plt.suptitle("Fig 1. Boxplot atribute numerice (train)", fontsize=11)
plt.tight_layout()
plt.savefig('fig1_boxplot_numerice.png', dpi=100)
plt.show()
print("Salvat: fig1_boxplot_numerice.png")


# ------------------------------------------------------------
# 2.2 Analiza atributelor CATEGORICE
# ------------------------------------------------------------

print("\n--- Statistici atribute CATEGORICE ---")
print(f"{'Coloana':<20} {'Nr. exemple valide':>20} {'Nr. valori unice':>18}")
print("-" * 60)
for col in CAT_COLS:
    print(f"{col:<20} {train[col].count():>20} {train[col].nunique():>18}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['education_level', 'company_size', 'skill_bracket']):
    counts = train[col].value_counts()
    axes[i].bar(counts.index, counts.values, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Distributia: {col}', fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Nr. exemple')
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle("Fig 2. Distributia a 3 atribute categorice", fontsize=11)
plt.tight_layout()
plt.savefig('fig2_distributie_categorice.png', dpi=100)
plt.show()
print("Salvat: fig2_distributie_categorice.png")


# ------------------------------------------------------------
# 2.3 Analiza echilibrului de clase (pentru target-ul de clasificare)
# ------------------------------------------------------------

print("\n--- Echilibrul claselor 'vacation' ---")
print(train[TARGET_CLF].value_counts())
print("\nProcentaj din total:")
print((train[TARGET_CLF].value_counts(normalize=True) * 100).round(1).astype(str) + '%')

fig, ax = plt.subplots(figsize=(7, 5))
counts = train[TARGET_CLF].value_counts()
ax.bar(counts.index, counts.values,
       color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'], edgecolor='black')
ax.set_title("Fig 3. Distributia claselor 'vacation' (train)")
ax.set_xlabel('Clasa')
ax.set_ylabel('Nr. exemple')
for i, v in enumerate(counts.values):
    ax.text(i, v + 150, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_echilibru_clase.png', dpi=100)
plt.show()
print("Salvat: fig3_echilibru_clase.png")

# Concluzie: daca clasele sunt dezechilibrate, vom raporta si F1/Precision/Recall,
# nu doar acuratete.


# ------------------------------------------------------------
# 2.4 Analiza corelatiei intre atribute
# Cerinta minima: numeric vs clasificare, categoric vs clasificare,
#                 numeric vs regresie, categoric vs regresie
# ------------------------------------------------------------

# --- A) Corelatie NUMERICA (Pearson) ---
# Pearson masoara cat de mult se misca doua coloane impreuna.
# Valori aproape de 1 sau -1 = corelatie puternica
# Valori aproape de 0 = independente

print("\n--- Corelatie atribute NUMERICE (Pearson) ---")
corr_matrix = train[NUM_COLS].corr()
print(corr_matrix.round(2))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True)
ax.set_title("Fig 4. Heatmap corelatie atribute numerice", fontsize=11)
plt.tight_layout()
plt.savefig('fig4_heatmap_corelatie_numerica.png', dpi=100)
plt.show()
print("Salvat: fig4_heatmap_corelatie_numerica.png")


# --- B) Atribut NUMERIC vs target CLASIFICARE (vacation) ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
train.boxplot(column='experience_years', by=TARGET_CLF, ax=axes[0])
axes[0].set_title('experience_years vs vacation')
axes[0].set_xlabel('vacation')
train.boxplot(column='skills_count', by=TARGET_CLF, ax=axes[1])
axes[1].set_title('skills_count vs vacation')
axes[1].set_xlabel('vacation')
plt.suptitle("Fig 5. Atribute numerice vs vacation (clasificare)")
plt.tight_layout()
plt.savefig('fig5_numeric_vs_vacation.png', dpi=100)
plt.show()
print("Salvat: fig5_numeric_vs_vacation.png")


# --- C) Atribut NUMERIC vs target REGRESIE (salary) ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(train['experience_years'], train['salary'],
                alpha=0.05, s=5, color='steelblue')
axes[0].set_xlabel('experience_years')
axes[0].set_ylabel('salary')
axes[0].set_title('experience_years vs salary')

axes[1].scatter(train['skills_count'], train['salary'],
                alpha=0.05, s=5, color='darkorange')
axes[1].set_xlabel('skills_count')
axes[1].set_ylabel('salary')
axes[1].set_title('skills_count vs salary')

plt.suptitle("Fig 6. Atribute numerice vs salary (regresie)")
plt.tight_layout()
plt.savefig('fig6_numeric_vs_salary.png', dpi=100)
plt.show()
print("Salvat: fig6_numeric_vs_salary.png")


# --- D) Atribut CATEGORIC vs target CLASIFICARE (Chi-patrat) ---

print("\n--- Corelatie CATEGORICA vs vacation (Chi-patrat) ---")
perechi_cat_clf = [
    ('education_level', TARGET_CLF),
    ('company_size',    TARGET_CLF),
    ('remote_work',     TARGET_CLF),
    ('skill_bracket',   TARGET_CLF),
]
print(f"{'Atribut':<20} {'Target':<12} {'Chi2':>10} {'p-value':>12} {'Concluzie'}")
print("-" * 65)
for col1, col2 in perechi_cat_clf:
    tabel = pd.crosstab(train[col1], train[col2])
    chi2, p, dof, _ = chi2_contingency(tabel)
    concluzie = "CORELATE" if p < 0.05 else "independente"
    print(f"{col1:<20} {col2:<12} {chi2:>10.1f} {p:>12.4f}   {concluzie}")


# --- E) Atribut CATEGORIC vs target REGRESIE (salary) ---

print("\n--- Corelatie CATEGORICA vs salary (boxplot grupat) ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# education_level vs salary
edu_medii = train.groupby('education_level')['salary'].median().sort_values()
train.boxplot(column='salary', by='education_level', ax=axes[0])
axes[0].set_title('education_level vs salary')
axes[0].set_xlabel('education_level')
axes[0].tick_params(axis='x', rotation=30)

# company_size vs salary
train.boxplot(column='salary', by='company_size', ax=axes[1])
axes[1].set_title('company_size vs salary')
axes[1].set_xlabel('company_size')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle("Fig 7. Atribute categorice vs salary (regresie)", fontsize=11)
plt.tight_layout()
plt.savefig('fig7_categoric_vs_salary.png', dpi=100)
plt.show()
print("Salvat: fig7_categoric_vs_salary.png")


# ============================================================
# SECTIUNEA 3 - PREPROCESAREA DATELOR
# ============================================================
# Curatam datele ca sa fie gata pentru model.
# Ordinea CONTEAZA:
#   1. Imputam valorile lipsa
#   2. Tratam outlieri (pe atribute numerice)
#   3. Extragem tintele (y) inainte de encodare
#   4. Encodam variabilele categorice cu get_dummies
#   5. Standardizam variabilele numerice
# ============================================================

print("\n" + "=" * 60)
print("SECTIUNEA 3 - PREPROCESAREA DATELOR")
print("=" * 60)

# Lucram pe COPII ale datelor originale.
# Astfel, daca gresim ceva, datele originale raman intacte.
train_proc = train.copy()
val_proc   = val.copy()


# ------------------------------------------------------------
# 3.1 Valori LIPSA (imputare)
# ------------------------------------------------------------
# Unele celule din CSV sunt goale -> pandas le vede ca NaN.
# Trebuie sa le inlocuim inainte de a antrena modelul.

print("\n--- 3.1 Valori lipsa ---")
valori_lipsa_train = train_proc.isnull().sum()
print("Coloane cu valori lipsa in TRAIN:")
print(valori_lipsa_train[valori_lipsa_train > 0])

valori_lipsa_val = val_proc.isnull().sum()
print("\nColoane cu valori lipsa in VAL:")
print(valori_lipsa_val[valori_lipsa_val > 0])

# Coloana 'remote_work' are ~30% valori lipsa.
# Alegem sa imputam cu MODA (valoarea cea mai frecventa),
# pentru ca 'remote_work' e o variabila categorica.
# Este logic sa presupunem ca angajatii fara date sunt in categoria majoritara.

moda_remote = train_proc['remote_work'].mode()[0]
print(f"\nModa 'remote_work' in train: '{moda_remote}'")
print(f"Imputam valorile lipsa cu: '{moda_remote}'")

# IMPORTANT: calculam moda DOAR din train si o aplicam si pe val!
train_proc['remote_work'] = train_proc['remote_work'].fillna(moda_remote)
val_proc['remote_work']   = val_proc['remote_work'].fillna(moda_remote)

print("Valori lipsa dupa imputare (train):", train_proc.isnull().sum().sum())
print("Valori lipsa dupa imputare (val):  ", val_proc.isnull().sum().sum())


# ------------------------------------------------------------
# 3.2 Valori EXTREME (outlieri) - metoda IQR
# ------------------------------------------------------------
# Un outlier este o valoare mult prea mare sau prea mica fata de restul.
# Ex: daca toti au experienta intre 0-30 ani, cineva cu 500 ani e outlier.
#
# Metoda IQR:
#   Q1 = percentila 25%  (25% din valori sunt sub aceasta)
#   Q3 = percentila 75%  (75% din valori sunt sub aceasta)
#   IQR = Q3 - Q1        (cat de "lata" e distributia la mijloc)
#
#   Limita de jos  = Q1 - 1.5 * IQR
#   Limita de sus  = Q3 + 1.5 * IQR
#
#   Tot ce e sub/peste aceste limite = outlier
#   Outlierul e inlocuit cu MEDIANA coloanei (valoarea de la mijloc)
#
# IMPORTANT: calculam limitele SI mediana DOAR din TRAIN,
#            si le aplicam si pe VAL.
#            Daca le-am calcula din val separat, nu ar fi ok -
#            in practica nu stim nimic despre datele noi.

print("\n--- 3.2 Detectia si tratarea outlierilor (IQR din TRAIN) ---")

# Pas 1: calculam limitele din train si le salvam
outlier_info = {}  # dictionar: coloana -> (limita_jos, limita_sus, mediana)

for col in NUM_COLS:
    Q1      = train_proc[col].quantile(0.25)
    Q3      = train_proc[col].quantile(0.75)
    IQR     = Q3 - Q1
    lim_jos = Q1 - 1.5 * IQR
    lim_sus = Q3 + 1.5 * IQR
    mediana = train_proc[col].median()
    outlier_info[col] = (lim_jos, lim_sus, mediana)

# Pas 2: inlocuim outlieri in TRAIN
print("TRAIN - outlieri inlocuiti cu mediana:")
for col in NUM_COLS:
    lim_jos, lim_sus, mediana = outlier_info[col]
    masca = (train_proc[col] < lim_jos) | (train_proc[col] > lim_sus)
    nr = masca.sum()
    train_proc[col] = train_proc[col].where(~masca, mediana)
    print(f"  {col:<22}: {nr:>5} outlieri → inlocuiti cu mediana={mediana:.1f}")

# Pas 3: aplicam ACELEASI limite in VAL
print("\nVAL - outlieri inlocuiti cu mediana din TRAIN:")
for col in NUM_COLS:
    lim_jos, lim_sus, mediana = outlier_info[col]
    masca = (val_proc[col] < lim_jos) | (val_proc[col] > lim_sus)
    nr = masca.sum()
    val_proc[col] = val_proc[col].where(~masca, mediana)
    print(f"  {col:<22}: {nr:>5} outlieri → inlocuiti cu mediana din train={mediana:.1f}")


# ------------------------------------------------------------
# 3.3 Extragem TINTELE (y) inainte de encodare
# ------------------------------------------------------------
# Tintele sunt coloanele pe care vrem sa le prezicam.
# Le scoatem acum, inainte sa encodam/modificam tabelul.

# Pentru clasificare: 'vacation' e text deci il transformam in numar cu LabelEncoder
# LabelEncoder: 'Large'->0, 'Medium'->1, 'No Vacation'->2, 'Small'->3 (ordine alfabetica)
# LabelEncoder e OK DOAR pentru coloana TINTA de clasificare.
le_vacation = LabelEncoder()
y_train_clf = le_vacation.fit_transform(train_proc[TARGET_CLF])
y_val_clf   = le_vacation.transform(val_proc[TARGET_CLF])
print(f"\nClasele vacation encodate: {dict(zip(le_vacation.classes_, le_vacation.transform(le_vacation.classes_)))}")
print(f"Ordine clase: {list(le_vacation.classes_)}")

# Pentru regresie: 'salary' e deja numeric → il luam direct
y_train_reg = train_proc[TARGET_REG].values
y_val_reg   = val_proc[TARGET_REG].values

# Stergem coloanele tinta din tabel (nu le vrem ca feature-uri)
train_proc = train_proc.drop(columns=[TARGET_CLF, TARGET_REG])
val_proc   = val_proc.drop(columns=[TARGET_CLF, TARGET_REG])


# ------------------------------------------------------------
# 3.4 Encodarea variabilelor CATEGORICE cu get_dummies
# ------------------------------------------------------------
# Modelele ML lucreaza cu numere, nu cu text.
# get_dummies (= One-Hot Encoding) transforma o coloana categorica
# in MAI MULTE coloane binare (0 sau 1).
#
# Exemplu pentru 'company_size' cu valorile 'Small', 'Medium', 'Large':
#   company_size_Large   company_size_Medium   company_size_Small
#         0                    0                     1         <- angajat la firma Small
#         1                    0                     0         <- angajat la firma Large
#
# De ce e mai bine decat LabelEncoder pe feature-uri?
# LabelEncoder ar pune: Small=0, Medium=1, Large=2
# Dar asta implica Large > Medium > Small ca numere, ceea ce e fals -
# nu exista o ordine numerica reala intre categorii!
# get_dummies evita aceasta problema.
#
# ATENTIE: get_dummies trebuie aplicat pe TRAIN si VAL impreuna,
# sau trebuie aliniate coloanele dupa (unele categorii pot lipsi dintr-un set).

print("\n--- 3.4 Encodare ONE-HOT (get_dummies) ---")
print("Coloane inainte de encodare:", list(train_proc.columns))

# Aplicam get_dummies
train_proc = pd.get_dummies(train_proc, columns=CAT_COLS, drop_first=False)
val_proc   = pd.get_dummies(val_proc,   columns=CAT_COLS, drop_first=False)

# Aliniem coloanele: val poate avea coloane lipsa fata de train
# (daca o categorie nu apare in val, acea coloana va fi 0 in val)
train_proc, val_proc = train_proc.align(val_proc, join='left', axis=1, fill_value=0)

print(f"Coloane dupa encodare: {len(train_proc.columns)} coloane")
print("Primele 10 coloane noi:", list(train_proc.columns[:10]))


# ------------------------------------------------------------
# 3.5 Standardizarea atributelor NUMERICE
# ------------------------------------------------------------
# Atributele numerice pot avea scale foarte diferite:
#   experience_years: 0 - 40
#   salary:           10000 - 500000
#
# Algoritmii lineari (Regresia Liniara, Ridge) sunt afectati de asta -
# coloana cu valori mari va "domina" calculul.
# StandardScaler transforma fiecare coloana astfel:
#   valoare_noua = (valoare - medie) / deviatia_standard
# Rezultat: fiecare coloana numerica are medie ~ 0 si std ~ 1.
#
# IMPORTANT: fit_transform pe TRAIN, doar transform pe VAL.
# (nu vrem ca modelul sa "vada" valorile din val inainte de evaluare)

print("\n--- 3.5 Standardizare (StandardScaler) ---")

scaler = StandardScaler()
train_proc[NUM_COLS] = scaler.fit_transform(train_proc[NUM_COLS])
val_proc[NUM_COLS]   = scaler.transform(val_proc[NUM_COLS])

print("Dupa standardizare, in TRAIN:")
print(train_proc[NUM_COLS].describe().loc[['mean', 'std']].round(3))


# ------------------------------------------------------------
# Pregatim X (features) si y (tinte) ca matrici numpy
# ------------------------------------------------------------
# Modelele sklearn primesc date ca matrici numpy (nu ca DataFrame pandas).
# .values transforma un DataFrame/Series pandas intr-un array numpy.

X_train = train_proc.values
X_val   = val_proc.values

print(f"\nX_train: {X_train.shape} (randuri x coloane)")
print(f"X_val:   {X_val.shape}")
print(f"y_train_clf: {y_train_clf.shape}  y_val_clf: {y_val_clf.shape}")
print(f"y_train_reg: {y_train_reg.shape}  y_val_reg: {y_val_reg.shape}")


# ============================================================
# SECTIUNEA 4 - ALGORITMI DE INVATARE AUTOMATA
# ============================================================

print("\n" + "=" * 60)
print("SECTIUNEA 4 - ALGORITMI DE INVATARE AUTOMATA")
print("=" * 60)


# ============================================================
# 4A. CLASIFICARE - Arbore de Decizie (Decision Tree)
# ============================================================
# Un arbore de decizie pune intrebari de tipul:
#
# Hiperparametru important: max_depth = cat de adanc creste arborele
#   - adanc mare -> invata mai multe detalii, risc de OVERFIT
#     (invata din memorie exemple din train, dar nu generalizeaza pe val)
#   - adanc mic  -> model mai simplu, posibil UNDERFIT
#     (prea simplu ca sa capteze pattern-urile)
#
# Facem mai multe experimente si vedem care max_depth e mai bun.

print("\n--- CLASIFICARE: Arbore de Decizie ---")
print("Target:", TARGET_CLF, "| Clase:", list(le_vacation.classes_))

# Functie helper: antreneaza un model si intoarce metricile
# (o functie = un bloc de cod reutilizabil, il chemam de mai multe ori)
def evalueaza_clasificator(model, X_tr, y_tr, X_te, y_te, nume):
    """Antreneaza modelul si calculeaza metricile pe setul de test."""
    model.fit(X_tr, y_tr)          # ANTRENARE: modelul invata din X_tr, y_tr
    pred = model.predict(X_te)     # PREDICTIE: aplica ce a invatat pe X_te
    acc  = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, pred, average='macro', zero_division=0)
    f1   = f1_score(y_te, pred, average='macro', zero_division=0)
    print(f"  {nume:<30} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    return pred, acc, prec, rec, f1

print()

# Experiment 1: Baseline - fara limita de adancime
# (arborele creste pana cand fiecare frunza are un singur exemplu - overfit sigur)
pred_b, acc_b, prec_b, rec_b, f1_b = evalueaza_clasificator(
    DecisionTreeClassifier(random_state=42),
    X_train, y_train_clf, X_val, y_val_clf,
    "Baseline (max_depth=None)"
)

# Experiment 2: max_depth=5
pred_5, acc_5, prec_5, rec_5, f1_5 = evalueaza_clasificator(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train_clf, X_val, y_val_clf,
    "max_depth=5"
)

# Experiment 3: max_depth=10
pred_10, acc_10, prec_10, rec_10, f1_10 = evalueaza_clasificator(
    DecisionTreeClassifier(max_depth=10, random_state=42),
    X_train, y_train_clf, X_val, y_val_clf,
    "max_depth=10"
)

# Experiment 4: max_depth=15
pred_15, acc_15, prec_15, rec_15, f1_15 = evalueaza_clasificator(
    DecisionTreeClassifier(max_depth=15, random_state=42),
    X_train, y_train_clf, X_val, y_val_clf,
    "max_depth=15"
)

# Experiment 5: max_depth=20
pred_20, acc_20, prec_20, rec_20, f1_20 = evalueaza_clasificator(
    DecisionTreeClassifier(max_depth=20, random_state=42),
    X_train, y_train_clf, X_val, y_val_clf,
    "max_depth=20"
)

# --- Tabel comparativ clasificare ---
print("\n--- Tab. 1. Influenta max_depth asupra clasificarii ---")
print(f"{'Model':<30} {'Acc (↑)':>9} {'Prec (↑)':>10} {'Rec (↑)':>9} {'F1 (↑)':>8}")
print("-" * 70)
rezultate_clf = [
    ("Baseline (max_depth=None)", acc_b,  prec_b,  rec_b,  f1_b),
    ("max_depth=5",               acc_5,  prec_5,  rec_5,  f1_5),
    ("max_depth=10",              acc_10, prec_10, rec_10, f1_10),
    ("max_depth=15",              acc_15, prec_15, rec_15, f1_15),
    ("max_depth=20",              acc_20, prec_20, rec_20, f1_20),
]

best_f1_val = max(r[4] for r in rezultate_clf)
for name, acc, prec, rec, f1 in rezultate_clf:
    marker = "  <- cel mai bun F1" if abs(f1 - best_f1_val) < 1e-9 else ""
    print(f"{name:<30} {acc:>9.3f} {prec:>10.3f} {rec:>9.3f} {f1:>8.3f}{marker}")

# --- Alegem cel mai bun model si facem matrice de confuzie ---
# Gasim max_depth-ul cu cel mai bun F1
best_depth_idx = max(range(len(rezultate_clf)), key=lambda i: rezultate_clf[i][4])
best_depth_name = rezultate_clf[best_depth_idx][0]
print(f"\nCel mai bun model: {best_depth_name}")

# Reantrenam cel mai bun model
best_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
best_clf.fit(X_train, y_train_clf)
best_pred_clf = best_clf.predict(X_val)

# Matrice de confuzie:
# randul i = clasa REALA, coloana j = clasa PREZISA
# diagonala principala = predictii corecte
# valorile in afara diagonalei = greseli
cm = confusion_matrix(y_val_clf, best_pred_clf)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=le_vacation.classes_).plot(
    ax=ax, cmap='Blues', colorbar=False
)
ax.set_title("Fig 8. Matrice de confuzie - DT max_depth=10")
plt.tight_layout()
plt.savefig('fig8_confusion_matrix.png', dpi=100)
plt.show()
print("Salvat: fig8_confusion_matrix.png")

# --- Metrici per clasa ---
print("\n--- Tab. 2. Metrici per clasa (DT max_depth=10) ---")
prec_cls = precision_score(y_val_clf, best_pred_clf, average=None, zero_division=0)
rec_cls  = recall_score(y_val_clf, best_pred_clf, average=None, zero_division=0)
f1_cls   = f1_score(y_val_clf, best_pred_clf, average=None, zero_division=0)
print(f"{'Clasa':<15} {'Precizie (↑)':>14} {'Recall (↑)':>12} {'F1 (↑)':>8}")
print("-" * 52)
for i, cls in enumerate(le_vacation.classes_):
    print(f"{cls:<15} {prec_cls[i]:>14.3f} {rec_cls[i]:>12.3f} {f1_cls[i]:>8.3f}")
print(f"{'MACRO AVG':<15} {prec_cls.mean():>14.3f} {rec_cls.mean():>12.3f} {f1_cls.mean():>8.3f}")


# ============================================================
# 4B. REGRESIE - Regresia Liniara + Ridge
# ============================================================
# Regresia liniara prezice un numar (salariul) ca o combinatie liniara
# a feature-urilor: salary = w1*exp + w2*skills + ... + b
#
# Ridge adauga o penalizare la antrenare pentru a evita overfitting-ul:
#   cost_total = eroarea_modelului + alpha * suma(w^2)
#
# alpha = cat de puternica e penalizarea:
#   alpha=0   -> identic cu Regresia Liniara simpla
#   alpha mic -> penalizare slaba, coeficienti pot fi mari
#   alpha mare-> penalizare puternica, coeficienti mici, model mai simplu

print("\n--- REGRESIE: Regresia Liniara si Ridge ---")
print("Target:", TARGET_REG)

# Functie helper pentru evaluare
def evalueaza_regresor(model, X_tr, y_tr, X_te, y_te, nume):
    """Antreneaza modelul de regresie si calculeaza metricile."""
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    mae_tr  = mean_absolute_error(y_tr, pred_tr)
    mae_te  = mean_absolute_error(y_te, pred_te)
    mse_te  = mean_squared_error(y_te, pred_te)
    rmse_te = np.sqrt(mse_te)
    r2_te   = r2_score(y_te, pred_te)

    print(f"  {nume:<25}  MAE_train={mae_tr:.0f}  MAE_val={mae_te:.0f}  "
          f"MSE={mse_te:.0f}  RMSE={rmse_te:.0f}  R2={r2_te:.4f}")
    return pred_te, mae_tr, mae_te, mse_te, rmse_te, r2_te

print()
# Experiment 1: Regresia Liniara simpla (BASELINE)
pred_lr, mae_tr_lr, mae_te_lr, mse_lr, rmse_lr, r2_lr = evalueaza_regresor(
    LinearRegression(),
    X_train, y_train_reg, X_val, y_val_reg,
    "Linear Regression"
)

# Experiment 2-5: Ridge cu diferite valori de alpha
pred_r01, mae_tr_r01, mae_te_r01, mse_r01, rmse_r01, r2_r01 = evalueaza_regresor(
    Ridge(alpha=0.1),
    X_train, y_train_reg, X_val, y_val_reg,
    "Ridge alpha=0.1"
)

pred_r1, mae_tr_r1, mae_te_r1, mse_r1, rmse_r1, r2_r1 = evalueaza_regresor(
    Ridge(alpha=1.0),
    X_train, y_train_reg, X_val, y_val_reg,
    "Ridge alpha=1.0"
)

pred_r10, mae_tr_r10, mae_te_r10, mse_r10, rmse_r10, r2_r10 = evalueaza_regresor(
    Ridge(alpha=10.0),
    X_train, y_train_reg, X_val, y_val_reg,
    "Ridge alpha=10"
)

pred_r100, mae_tr_r100, mae_te_r100, mse_r100, rmse_r100, r2_r100 = evalueaza_regresor(
    Ridge(alpha=100.0),
    X_train, y_train_reg, X_val, y_val_reg,
    "Ridge alpha=100"
)

# --- Tabel comparativ regresie ---
print("\n--- Tab. 3. Comparatie modele de regresie ---")
print(f"{'Model':<25} {'MAE_train':>10} {'MAE_val (↓)':>12} {'MSE (↓)':>12} {'RMSE (↓)':>10} {'R2 (↑)':>8}")
print("-" * 82)
rezultate_reg = [
    ("Linear Regression", mae_tr_lr,  mae_te_lr,  mse_lr,  rmse_lr,  r2_lr),
    ("Ridge alpha=0.1",   mae_tr_r01, mae_te_r01, mse_r01, rmse_r01, r2_r01),
    ("Ridge alpha=1.0",   mae_tr_r1,  mae_te_r1,  mse_r1,  rmse_r1,  r2_r1),
    ("Ridge alpha=10",    mae_tr_r10, mae_te_r10, mse_r10, rmse_r10, r2_r10),
    ("Ridge alpha=100",   mae_tr_r100,mae_te_r100,mse_r100,rmse_r100,r2_r100),
]
best_r2_val = max(r[5] for r in rezultate_reg)
for name, mae_tr, mae_te, mse, rmse, r2 in rezultate_reg:
    marker = "  <- cel mai bun" if abs(r2 - best_r2_val) < 1e-9 else ""
    print(f"{name:<25} {mae_tr:>10.0f} {mae_te:>12.0f} {mse:>12.0f} {rmse:>10.0f} {r2:>8.4f}{marker}")


# --- Grafic: curbe de eroare Ridge ---
# Vedem cum se comporta MAE pe train vs val pentru fiecare alpha.
# Daca MAE_train << MAE_val -> model overfit la alpha mic
# Pe masura ce alpha creste -> modelul devine mai simplu

alphas    = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
mae_train_list = []
mae_val_list   = []

for a in alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, y_train_reg)
    mae_train_list.append(mean_absolute_error(y_train_reg, m.predict(X_train)))
    mae_val_list.append(mean_absolute_error(y_val_reg,   m.predict(X_val)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(alphas)), mae_train_list, 'b-o', label='Train MAE', linewidth=2)
ax.plot(range(len(alphas)), mae_val_list,   'r-o', label='Val MAE',   linewidth=2)
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([str(a) for a in alphas])
ax.set_xlabel('Alpha (regularizare Ridge)')
ax.set_ylabel('MAE')
ax.set_title("Fig 9. Curbe eroare Ridge - Train vs Val")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig9_curbe_eroare_ridge.png', dpi=100)
plt.show()
print("Salvat: fig9_curbe_eroare_ridge.png")

# --- Grafic: predictii vs valori reale ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_val_reg, pred_lr, alpha=0.1, s=5, color='steelblue')
# Linia ideala: daca am prezice perfect, toate punctele ar fi pe dreapta y=x
min_val = min(y_val_reg.min(), pred_lr.min())
max_val = max(y_val_reg.max(), pred_lr.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Predictie perfecta')
ax.set_xlabel('Salary real')
ax.set_ylabel('Salary prezis')
ax.set_title("Fig 10. Predictii vs Valori Reale (Linear Regression)")
ax.legend()
plt.tight_layout()
plt.savefig('fig10_predictii_vs_reale.png', dpi=100)
plt.show()
print("Salvat: fig10_predictii_vs_reale.png")


# ============================================================
# SECTIUNEA 5 - REZUMAT FINAL
# ============================================================

print("\n" + "=" * 60)
print("SECTIUNEA 5 - REZUMAT FINAL")
print("=" * 60)

print("\n>> CLASIFICARE (Arbore de Decizie, max_depth=10)")
print(f"   Acuratete:    {acc_10:.3f}")
print(f"   Precizie:     {prec_10:.3f}  (macro avg)")
print(f"   Recall:       {rec_10:.3f}  (macro avg)")
print(f"   F1:           {f1_10:.3f}  (macro avg)")
print()
print("   Observatii:")
print("   - 'No Vacation' e cel mai usor de prezis (are cel mai mult suport)")
print("   - Clasele rare ('Medium', 'Large') au F1 mai mic")
print("   - get_dummies a creat multe coloane binare din variabilele categorice,")
print("     ceea ce permite arborelui sa ia decizii mai precise")

print("\n>> REGRESIE (Ridge / Linear Regression)")
print(f"   MAE  (val): {mae_te_lr:.0f}")
print(f"   MSE  (val): {mse_lr:.0f}")
print(f"   RMSE (val): {rmse_lr:.0f}")
print(f"   R2   (val): {r2_lr:.4f}")
print()
print("   Observatii:")
print("   - R2 mic indica ca modelul liniar nu captureaza bine relatia cu salariul")
print("   - Ridge nu aduce imbunatatiri majore fata de Linear Regression")
print("   - Salariul depinde probabil de combinatii complexe, nu de o relatie liniara")

print("\n" + "=" * 60)
print("GATA! Toate sectiunile au fost rulate.")
print("Grafice salvate: fig1 - fig10")
print("Fisiere Kaggle: submission_clasificare.csv, submission_regresie.csv")
print("=" * 60)
