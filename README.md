# 🚢 Titanic – Machine Learning from Disaster

> **This is my first Kaggle challenge.** 🎉
> Competition: [Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
> Author: [nex0rasdatascientist](https://www.kaggle.com/nex0rasdatascientist)
> Notebook: [`titanic-project.ipynb`](titanic-project.ipynb)

## 1. The Problem

The goal of the competition is to build a **binary classification model** that predicts whether a passenger **survived (1) or did not survive (0)** the sinking of the Titanic, using passenger information such as class, sex, age, family on board, ticket fare, cabin and port of embarkation.

- **Train set** (`train.csv`): 891 passengers with the known `Survived` label.
- **Test set** (`test.csv`): 418 passengers without the label — this is what gets submitted.
- **Metric:** accuracy.

## 2. How I Attacked the Problem

Since this is my first challenge, I kept the approach simple, step-by-step and easy to explain:

1. **Load the data** and look at its structure (`head`, `describe`).
2. **Analyse every feature one by one**, comparing the survival rate between its groups to decide whether it carries signal.
3. **Pick the features** that show a clear relation with survival, and drop the ones that don't.
4. **Preprocess**: create a few simple engineered features, fill missing ages, one-hot encode categories and scale numbers.
5. **Split** the training data (80% train / 20% validation) to measure accuracy on unseen passengers.
6. **Train a Logistic Regression** model as a simple, interpretable baseline.
7. **Predict the test set** and generate `submission.csv`.

> The encoder and the scaler are **fitted on the training split only** and then reused on the validation and test data, so no information leaks from the validation/test sets.

## 3. Feature Analysis

For each column, I looked at the survival rate (`Survived.mean()`) per group. If the groups have very different survival rates, the feature is useful to the model. If not (or if the column is just an identifier), it is left out.

| Feature | Used? | Why |
|---|---|---|
| `PassengerId` | ❌ | Only a unique row number; no real relation with survival, the model could only memorise it. |
| `Pclass` | ✅ categorical | Survival drops clearly with class: **63% (1st) → 47% (2nd) → 24% (3rd)**. Treated as a category because classes are labels, not a real quantity. |
| `Name` | ❌ | Every name is unique, so it can't be generalised as-is. (It contains titles like *Mr / Miss / Master*, but `Sex` and `Age` already cover much of that.) |
| `Sex` | ✅ | Strongest signal: **74% of women survived vs 19% of men** ("women and children first"). |
| `SibSp` | ✅ numeric | Passengers with at least one sibling/spouse aboard survived more (**47% vs 35%**). |
| `Parch` | ✅ numeric | Same idea as `SibSp`: having family aboard changes the survival rate. |
| `Ticket` | ❌ (explored only) | Mostly unique values. I started extracting a `TicketPrefix` but left it out of the final model: it's noisy and I preferred a smaller set of clearer features for a first baseline. |
| `Fare` | ✅ numeric + `FareOverMean` | Survivors tend to have paid higher fares. Also created a binary feature, `FareOverMean`: **60% survival above the average fare vs 32% below**. |
| `Cabin` | ✅ as `HasCabin` (yes/no) | Many missing values and the exact code is too sparse, but *having a cabin recorded* is informative: **67% survival vs 30%** (it mostly correlates with wealthier passengers). |
| `Embarked` | ✅ categorical | **Cherbourg 55%, Queenstown 39%, Southampton 34%**. |
| `Age` | ✅ numeric | Missing values filled with the mean. |

## 4. Final Features & Preprocessing

| Type | Features | Handling |
|---|---|---|
| Numerical | `Age`, `Fare`, `SibSp`, `Parch` | Missing `Age` filled with the mean, then standardised |
| Categorical | `Sex`, `Embarked`, `Pclass`, `HasCabin`, `FareOverMean` | One-hot encoded |
| Dropped | `PassengerId`, `Name`, `Ticket`, raw `Cabin` | Identifiers / too sparse for a first baseline |

Small reusable functions handle loading, feature creation, missing values, encoding and scaling, so **exactly the same transformations** are applied to the training, validation and test data.

## 5. Train / Validation Split

I hold out **20%** of the training data (179 passengers) to evaluate the model on passengers it has never seen, and train on the remaining **80%** (712 passengers).

## 6. Model & Result

I used **Logistic Regression**: a simple, fast and interpretable baseline well suited to a binary target and to my first competition.

**Validation accuracy: 80.45%**

## 7. Submission

The 418 passengers of `test.csv` are predicted with the same preprocessing, and the result is saved in the format Kaggle requires (`PassengerId`, `Survived`) as `submission.csv`.

## 8. Conclusion & Next Steps

This was my **first Kaggle challenge**, and a simple Logistic Regression on a handful of well-understood features reached **80.45% accuracy** on my validation split.

Ideas to try next:
- Extract the **title** from `Name` (Mr, Miss, Mrs, Master…) as a feature.
- Use the **ticket prefix** and family size.
- Try tree-based models (Random Forest, Gradient Boosting) and cross-validation.

Thanks for reading! 🙌 Follow my work: [nex0rasdatascientist on Kaggle](https://www.kaggle.com/nex0rasdatascientist).
