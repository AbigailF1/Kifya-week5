# Buy-Now-Pay-Later Credit Scoring Model

This repository contains the end-to-end development of a credit scoring model for Bati Bank's new Buy-Now-Pay-Later (BNPL) service. As an Analytics Engineer, the goal is to leverage customer behavioral data from our eCommerce partner to build a reliable and automated system for assessing credit risk.

## Credit Scoring Business Understanding


### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord fundamentally links a bank's capital requirements to its underlying risks. For credit risk, it allows banks to use the Internal Ratings-Based (IRB) approach, where we can use our own internal models to estimate risk parameters like the Probability of Default (PD).

This presents a significant business advantage: a more accurate, risk-sensitive model can lead to more efficient capital allocation compared to the standardized approach. However, this advantage comes with a critical regulatory obligation. Under Basel II's Pillar 2 (Supervisory Review Process), regulators require that any internal model be **robust, transparent, and thoroughly documented**.

Therefore, our need for an interpretable model is not merely a technical preference; it is a **business and compliance necessity**. We must be able to:
* **Explain to regulators** how the model arrives at its conclusions.
* **Validate the logic** and demonstrate that the chosen features and their weights are sensible.
* **Ensure fairness and non-discrimination** by clearly understanding the model's decision-making process.

A "black box" model, regardless of its predictive power, would fail regulatory scrutiny and jeopardize our ability to use the IRB approach. This drives our focus on building a model that is both accurate and explainable.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

The eCommerce dataset contains rich transactional data but lacks a direct, historical `default` label, which is the ground truth needed for a standard supervised classification model. To overcome this, we must **engineer a proxy variable**—an observable feature that we hypothesize is strongly correlated with the true risk of default.

**Necessity of the Proxy:**
For this project, we will use Recency, Frequency, and Monetary (RFM) analysis to identify a segment of customers who are "disengaged" (e.g., have not transacted recently and have a low transaction frequency/value). The core business assumption is that **customer disengagement is a proxy for higher credit risk**. A customer with low loyalty or declining activity on the platform is presumed to be less likely to honor a future credit obligation.

**Potential Business Risks:**
Our model's success hinges on how accurately this proxy represents actual default behavior. There are two primary risks associated with this assumption:

* **Risk of False Positives (Lost Opportunity):** The model might incorrectly flag a creditworthy customer as "high-risk" simply because they are a new or infrequent shopper. By denying them credit, we create a negative customer experience and **lose potential revenue and customer lifetime value**.
* **Risk of False Negatives (Financial Loss):** The model might incorrectly classify a genuinely high-risk customer as "low-risk" because their transactional behavior does not fit the "disengaged" profile (e.g., a fraudster on a spending spree). Approving credit for this segment could lead to **direct financial losses** when they inevitably default.

Managing these risks requires continuous model monitoring and validation once real-world default data becomes available.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice of modeling algorithm in a regulated financial context involves a critical trade-off between predictive performance and model interpretability.

**Simple Model: Logistic Regression (with Weight of Evidence - WoE)**

* **Strengths:**
    * **High Interpretability:** Each feature's coefficient directly translates into its contribution to the final score, making it easy to create credit scorecards and explain decisions to stakeholders and regulators.
    * **Regulatory Acceptance:** It is a well-understood and widely accepted methodology in the financial industry.
    * **Fairness Audits:** The transparency makes it easier to audit the model for bias against protected classes.

* **Weaknesses:**
    * **Lower Predictive Power:** It assumes a linear relationship between features and the outcome, and may fail to capture complex, non-linear patterns in the data, potentially leading to lower accuracy.

**Complex Model: Gradient Boosting (e.g., XGBoost, LightGBM)**

* **Strengths:**
    * **High Predictive Power:** These models are highly effective at capturing complex, non-linear interactions between features, often resulting in superior performance metrics (e.g., higher AUC).
    * **Potential for Lower Losses:** Higher accuracy can translate directly into better risk identification, reducing default rates and improving profitability.

* **Weaknesses:**
    * **"Black Box" Nature:** It is inherently difficult to explain the exact reasoning behind a specific prediction, which poses a major challenge for regulatory approval and internal governance.
    * **Risk of Overfitting:** Without careful tuning, these models can easily overfit the training data.
    * **Complexity in Auditing:** Assessing fairness and bias is more challenging.

**The Key Trade-Off:**
For Bati Bank, the central decision is balancing **performance vs. transparency**. We must weigh the potential increase in predictive accuracy from a complex model against the absolute need for a model that is transparent, defensible to regulators, and easily explainable. Our initial approach will likely involve building both types of models to quantify this trade-off precisely.
