# Predicting User Churn at Waze: A PM Case Study

Churn prediction treated as a **product decision**, not just a modeling exercise. This project asks who churns, whether we can spot them early, whether acting on it pays off, and what to build and test next.

> **Data note:** this uses the practice Waze dataset from Google's Advanced Data Analytics Certificate (~15,000 users, one month of activity). The dataset is not included in this repository; it's available through the course. Findings describe this dataset, not the live Waze product.

---

## TL;DR

- **14,299 labeled users, 17.7% churned** in the month.
- **Habit is the biggest lever.** Users who never drove churned at **40%**, users who drove 20+ days at **5%**.
- **The risk is intense but infrequent users, not heavy ones.** "Occasional long-haul" users (fewer than 10 driving days, 590+ km per driving day) are **20% of users but 36% of all churners**. Professional drivers churn the least (**7.6%**).
- **XGBoost beats a one-line rule by 20% on PR-AUC** (0.38 vs 0.32; random guessing is 0.18). But to catch 70% of churners, it nudges **41%** of users vs **45%** for the rule. Whether that gap justifies an ML pipeline is a product call.
- **On unseen users, the model catches 72% of churners by nudging 39% of users**, about 2 unneeded nudges per churner caught. So the intervention must be cheap.
- **Next step:** a holdout A/B test (~3,200 flagged users per arm) of low-cost in-app nudges, targeted by segment.

---

## 1. Problem framing

Waze is crowd-sourced: every active driver both uses the map and improves it with live traffic reports. Churn costs twice: we lose the user, and the product gets slightly worse for everyone who stays.

| | |
|---|---|
| **Goal** | Flag likely churners early enough to act, and learn which behaviors signal churn |
| **Success criteria** | Catch at least 70% of churners at a precision where intervening still makes sense |
| **Key trade-off** | Missing a churner costs more than nudging a loyal user, so the model optimizes for recall |
| **Non-goals** | Predicting the exact churn date; explaining why each individual leaves (that needs user research) |

## 2. Key findings

### Driving frequency is the strongest churn signal
![Churn rate by driving days](charts/01_churn_by_driving_days.png)

Churn falls steadily as driving days increase. The product question becomes *"how do we get users to open Waze on more days?"* rather than *"how do we win back users who already left?"*

### Where the risk is differs from where the volume is

![Churn rate by segment](charts/03_churn_by_segment.png)

| Segment | Definition | Share of users | Churn rate | Share of churners |
|---|---|---|---|---|
| Dormant | Didn't drive this month | 7% | **40%** | 16% |
| Occasional long-haul | < 10 driving days, 590+ km per driving day | 20% | 31% | **36%** |
| Occasional | < 10 driving days, normal distances | 13% | 23% | 17% |
| Regular | 10+ driving days | 43% | 10% | 24% |
| Professional | 60+ drives on 15+ days | 17% | 8% | 7% |

Dormant users are the riskiest, but occasional long-haul users are the biggest share of the problem.

### A model helps, but only modestly at the decision point
![Model comparison](charts/04_model_comparison.png)

| Model | PR-AUC | Users nudged to catch 70% of churners | Precision |
|---|---|---|---|
| **XGBoost** | **0.378** | **41%** | **31%** |
| Random forest | 0.370 | 41% | 31% |
| Logistic regression | 0.361 | 41% | 30% |
| Simple rule: fewest driving days | 0.316 | 45% | 28% |
| Random guessing | 0.177 | n/a | 18% |

*Validation set. PR-AUC is used instead of accuracy because of class imbalance: predicting "nobody churns" would score 82% accuracy.*

## 3. Recommendations and experiment plan

| Segment | Hypothesis | Intervention | Primary metric |
|---|---|---|---|
| **Dormant / Occasional** | Waze isn't part of their routine | Prompt to save Home/Work, then "time to leave" commute alerts | Driving days per user, 30-day retention |
| **Occasional long-haul** | They use Waze only for long trips and forget it in between | Trip-planning features, post-trip "save this destination" | Segment retention |
| **Professional** | Waze already fits their routine | No retention spend; study what keeps them and apply it to long-haul users | Segment retention (guardrail) |
| **All flagged users** | A timely, cheap nudge prevents some churn | Weekly risk score; in-app prompt for top-risk users | Churn among flagged users vs a holdout |

**Experiment sizing:** detecting a 10% relative drop in churn among flagged users needs about **3,162 users per arm** (95% confidence, 80% power).
**Guardrails:** notification opt-out rate, uninstall rate, sessions among non-targeted users.

## 4. Limitations

- **One month, one snapshot.** No weekly trends, so a user whose activity is falling looks the same as one with steady low use.
- **Correlation, not cause.** Few driving days may be a symptom of churn rather than a cause; only experiments can tell.
- **Modest signal.** With aggregate usage alone, any model misses some churners or flags many loyal users.
- **Illustrative economics.** The cost and value figures in the notebook's threshold analysis are placeholders to show the reasoning.

**Data I'd ask for next:** daily or weekly activity, trip purpose and time of day, in-app feedback, notification response history, exit surveys.

## 5. Approach

1. Data-quality checks (missing labels, implausible values)
2. Feature engineering: behavior ratios such as km per driving day and drives per driving day
3. Hypothesis-driven EDA with effect sizes, not just p-values
4. Rule-based segmentation into five personas
5. Models: simple rule baseline, logistic regression, random forest, XGBoost (60/20/20 train/validation/test split, class weights for imbalance)
6. Threshold chosen for a recall target, with the cost trade-off made explicit
7. Permutation importance for churn drivers
8. A/B test sizing for the proposed intervention

## 6. How to run

1. Open `waze_churn_case_study.ipynb` in [Google Colab](https://colab.research.google.com) or Jupyter.
2. Place `waze_dataset.csv` (from the course) in the same folder as the notebook (in Colab: the `content` folder).
3. Run all cells (~2–3 minutes). Charts are saved to `charts/`.

**Tools:** Python, pandas, scikit-learn, XGBoost, statsmodels, matplotlib, seaborn.

---

**Author:** Danish Khan · https://www.linkedin.com/in/danish-khan01/
