# Statistics & Probability Interview Questions

## Table of Contents
1. [Probability Fundamentals](#probability-fundamentals)
2. [Distributions](#distributions)
3. [Statistical Inference](#statistical-inference)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Bayesian Statistics](#bayesian-statistics)
6. [A/B Testing](#ab-testing)
7. [Coding Problems](#coding-problems)

---

## Probability Fundamentals

### Q1: Explain conditional probability and Bayes' theorem
```
Conditional Probability:
P(A|B) = P(A ∩ B) / P(B)
"Probability of A given B occurred"

Bayes' Theorem:
P(A|B) = P(B|A) × P(A) / P(B)

Components:
- P(A|B): Posterior - probability after seeing evidence
- P(B|A): Likelihood - probability of evidence given hypothesis
- P(A): Prior - initial belief
- P(B): Evidence - normalizing constant

Example: Disease testing
- P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
- Even with 99% accurate test, if disease is rare (0.1%), 
  most positives are false positives!
```

### Q2: What is the Law of Large Numbers vs Central Limit Theorem?
```
Law of Large Numbers (LLN):
- Sample mean converges to population mean as n → ∞
- Justifies using sample statistics to estimate population parameters
- X̄_n → μ as n → ∞

Central Limit Theorem (CLT):
- Distribution of sample means approaches normal distribution
- Regardless of population distribution (with finite variance)
- X̄ ~ N(μ, σ²/n) for large n

Key differences:
- LLN: About convergence of VALUE
- CLT: About convergence of DISTRIBUTION

Practical implications:
- CLT allows us to construct confidence intervals
- CLT justifies many statistical tests
- Rule of thumb: n ≥ 30 for CLT to apply
```

### Q3: Explain independence vs conditional independence
```
Independence:
P(A ∩ B) = P(A) × P(B)
Equivalently: P(A|B) = P(A)

Conditional Independence:
P(A ∩ B | C) = P(A|C) × P(B|C)
A and B are independent GIVEN C

Example - Naive Bayes assumption:
Features X₁, X₂ are conditionally independent given class Y
P(X₁, X₂ | Y) = P(X₁|Y) × P(X₂|Y)

Important: Independence ≠ Conditional Independence
- Two events can be independent but not conditionally independent
- Two events can be conditionally independent but not independent
```

### Q4: What is expected value and variance?
```python
# Expected Value (Mean)
E[X] = Σ x × P(X=x)  # Discrete
E[X] = ∫ x × f(x) dx  # Continuous

# Properties:
E[aX + b] = a×E[X] + b
E[X + Y] = E[X] + E[Y]  # Always true
E[XY] = E[X]×E[Y]  # Only if independent

# Variance
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

# Properties:
Var(aX + b) = a²×Var(X)
Var(X + Y) = Var(X) + Var(Y)  # Only if independent

# Standard Deviation
σ = √Var(X)

# Covariance and Correlation
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
Corr(X,Y) = Cov(X,Y) / (σₓ × σᵧ)  # Range: [-1, 1]
```

---

## Distributions

### Q5: Compare common probability distributions
```
Discrete Distributions:

Bernoulli(p):
- Single trial, success/failure
- E[X] = p, Var(X) = p(1-p)

Binomial(n, p):
- Number of successes in n trials
- E[X] = np, Var(X) = np(1-p)

Poisson(λ):
- Count of events in fixed interval
- E[X] = λ, Var(X) = λ
- Approximates Binomial when n large, p small

Geometric(p):
- Number of trials until first success
- E[X] = 1/p, Var(X) = (1-p)/p²

Continuous Distributions:

Uniform(a, b):
- E[X] = (a+b)/2, Var(X) = (b-a)²/12

Normal(μ, σ²):
- Bell curve, symmetric
- 68-95-99.7 rule for 1, 2, 3 standard deviations

Exponential(λ):
- Time between Poisson events
- E[X] = 1/λ, Var(X) = 1/λ²
- Memoryless property: P(X > s+t | X > s) = P(X > t)
```

### Q6: When do you use each distribution?
```
Bernoulli/Binomial:
- Coin flips, click/no-click, conversion
- Fixed number of independent trials

Poisson:
- Website visits per hour
- Defects per unit
- Emails received per day
- When counting rare events

Exponential:
- Time until next customer arrives
- Time between failures
- Any "waiting time" scenario

Normal:
- Heights, weights, test scores
- Sum of many independent variables (CLT)
- Errors in measurements

Log-Normal:
- Stock prices, income distribution
- When log(X) is normally distributed
- Always positive, right-skewed

Beta:
- Modeling probabilities
- Bayesian prior for Bernoulli parameter
- Click-through rates, conversion rates
```

### Q7: What is the relationship between Poisson and Exponential?
```
Connection:
- If events occur according to Poisson process with rate λ
- Time between events follows Exponential(λ)

Poisson(λ): Number of events in unit time
Exponential(λ): Time until next event

Example:
- Customers arrive at rate 5/hour (Poisson)
- Time between customers is Exponential(5)
- Expected wait = 1/5 hour = 12 minutes

Memoryless property (Exponential):
P(X > s + t | X > s) = P(X > t)
- No matter how long you've waited, expected remaining wait is same
- Only Exponential (continuous) and Geometric (discrete) have this property
```

---

## Statistical Inference

### Q8: Explain confidence intervals
```
Confidence Interval:
Range of values that likely contains the true parameter

For mean with known variance:
CI = X̄ ± z_{α/2} × (σ/√n)

For mean with unknown variance:
CI = X̄ ± t_{α/2, n-1} × (s/√n)

Interpretation:
- 95% CI means: If we repeated this 100 times, 
  ~95 intervals would contain true parameter
- NOT: "95% probability true parameter is in this interval"

Width depends on:
- Confidence level (higher = wider)
- Sample size (larger = narrower)
- Variance (higher = wider)

For proportions:
CI = p̂ ± z_{α/2} × √(p̂(1-p̂)/n)
```

### Q9: What is the difference between Type I and Type II errors?
```
Type I Error (False Positive):
- Reject H₀ when H₀ is true
- Probability = α (significance level)
- "Crying wolf"

Type II Error (False Negative):
- Fail to reject H₀ when H₀ is false
- Probability = β
- "Missing a real effect"

Power = 1 - β
- Probability of correctly rejecting false H₀
- Ability to detect a real effect

Trade-off:
- Decreasing α increases β (and vice versa)
- Only way to decrease both: increase sample size

Example - Medical test:
- Type I: Healthy person diagnosed as sick
- Type II: Sick person diagnosed as healthy
- Which is worse depends on context!
```

### Q10: Explain p-value and its common misinterpretations
```
P-value definition:
Probability of observing data as extreme as ours,
assuming null hypothesis is true.

p = P(Data | H₀)

Common misinterpretations (WRONG):
✗ Probability that H₀ is true
✗ Probability results are due to chance
✗ Probability of making an error if you reject H₀
✗ Size of the effect

Correct interpretation:
If null hypothesis were true, how surprising is our data?

P-value thresholds:
- p < 0.05: "Statistically significant"
- p < 0.01: "Highly significant"
- These are arbitrary conventions!

Problems with p-values:
- P-hacking: testing many hypotheses
- Publication bias: only significant results published
- Significant ≠ Important (large n → small p even for tiny effects)

Better approach:
- Report effect sizes with confidence intervals
- Consider practical significance
```

---

## Hypothesis Testing

### Q11: Compare parametric vs non-parametric tests
```
Parametric Tests:
- Assume specific distribution (usually normal)
- More powerful when assumptions met
- Examples: t-test, ANOVA, F-test

Non-parametric Tests:
- No distribution assumptions
- Use ranks instead of raw values
- Less powerful but more robust
- Examples: Mann-Whitney, Wilcoxon, Kruskal-Wallis

When to use non-parametric:
- Small sample size (can't verify normality)
- Ordinal data
- Heavy outliers
- Skewed distributions

Parametric → Non-parametric equivalents:
- One-sample t-test → Wilcoxon signed-rank
- Two-sample t-test → Mann-Whitney U
- Paired t-test → Wilcoxon signed-rank
- One-way ANOVA → Kruskal-Wallis
```

### Q12: Explain ANOVA and when to use it
```
ANOVA (Analysis of Variance):
Compare means across 3+ groups

One-way ANOVA:
- One factor (e.g., compare treatments A, B, C)
- H₀: μ₁ = μ₂ = μ₃ = ...
- F-statistic = Between-group variance / Within-group variance

Two-way ANOVA:
- Two factors (e.g., treatment AND gender)
- Can test main effects and interaction

Assumptions:
1. Independence of observations
2. Normality within groups
3. Homogeneity of variances (Levene's test)

Post-hoc tests (if ANOVA significant):
- Tukey HSD: All pairwise comparisons
- Bonferroni: Adjusted for multiple comparisons
- Dunnett: Compare all to control

Why not multiple t-tests?
- Family-wise error rate increases
- 3 groups, 3 comparisons: 1-(0.95)³ = 14.3% Type I error
```

### Q13: What is the multiple comparisons problem?
```
Problem:
With many tests, some will be significant by chance.
- 20 tests at α=0.05 → expect 1 false positive

Solutions:

Bonferroni Correction:
- Divide α by number of tests
- α_adjusted = α / m
- Very conservative (high Type II error)

Holm-Bonferroni:
- Step-down procedure
- Less conservative than Bonferroni

False Discovery Rate (FDR):
- Control expected proportion of false positives
- Benjamini-Hochberg procedure
- Better for exploratory analysis

When to use each:
- Few comparisons, need certainty: Bonferroni
- Many comparisons, exploratory: FDR
- Pre-specified contrasts: No correction needed
```

---

## Bayesian Statistics

### Q14: Explain Bayesian vs Frequentist approaches
```
Frequentist:
- Parameters are fixed, unknown constants
- Probability = long-run frequency
- Uses p-values, confidence intervals
- No prior information

Bayesian:
- Parameters have probability distributions
- Probability = degree of belief
- Uses posterior distributions, credible intervals
- Incorporates prior knowledge

Comparison:

                    Frequentist         Bayesian
Parameter           Fixed constant      Random variable
Prior knowledge     Not used            Explicitly used
Uncertainty         Confidence interval Credible interval
Interpretation      Frequency           Probability
Small samples       Problematic         Handles well

Bayesian advantages:
- Intuitive probability statements
- Incorporates prior knowledge
- Better for small samples
- Natural for sequential updating

Bayesian challenges:
- Subjective prior selection
- Computational complexity
- Prior sensitivity
```

### Q15: What are conjugate priors?
```
Conjugate Prior:
Prior and posterior belong to same distribution family.
Makes computation tractable.

Common conjugate pairs:

Likelihood      Prior           Posterior
---------       -----           ---------
Bernoulli       Beta            Beta
Binomial        Beta            Beta
Poisson         Gamma           Gamma
Normal (μ)      Normal          Normal
Normal (σ²)     Inverse-Gamma   Inverse-Gamma
Exponential     Gamma           Gamma
Multinomial     Dirichlet       Dirichlet

Example - Beta-Binomial:
Prior: Beta(α, β)
Data: k successes in n trials
Posterior: Beta(α + k, β + n - k)

Why useful:
- Closed-form posterior
- Easy to interpret (pseudocounts)
- Sequential updating is simple
```

---

## A/B Testing

### Q16: Design an A/B test
```
Steps:

1. Define hypothesis and metrics
   - Primary metric (e.g., conversion rate)
   - Secondary metrics
   - Guardrail metrics (shouldn't get worse)

2. Calculate sample size
   n = 2 × (z_α + z_β)² × σ² / δ²
   
   Where:
   - α: significance level (typically 0.05)
   - β: 1 - power (typically 0.2)
   - σ²: variance
   - δ: minimum detectable effect

3. Randomization
   - User-level (not session-level)
   - Hash-based for consistency
   - Check for balance

4. Run experiment
   - Don't peek! (or use sequential testing)
   - Run for full duration
   - Beware day-of-week effects

5. Analyze results
   - Check for sample ratio mismatch
   - Statistical significance
   - Practical significance
   - Segment analysis (carefully)
```

### Q17: Common A/B testing pitfalls
```
1. Peeking problem
   - Checking results repeatedly inflates Type I error
   - Solutions: Sequential testing, fixed horizon

2. Multiple comparisons
   - Testing many metrics/variants
   - Solution: Bonferroni correction or FDR control

3. Simpson's paradox
   - Overall effect opposite of segment effects
   - Solution: Check segment-level results

4. Network effects
   - Users influence each other
   - Solution: Cluster randomization

5. Novelty/primacy effects
   - Initial behavior differs from long-term
   - Solution: Run longer, measure retention

6. Sample ratio mismatch
   - Unequal traffic allocation
   - Indicates implementation bug

7. Survivorship bias
   - Only analyzing users who complete journey
   - Solution: Intent-to-treat analysis

8. Low power
   - Can't detect real effects
   - Solution: Larger sample, variance reduction
```

### Q18: What is multi-armed bandit vs A/B testing?
```
A/B Testing (Fixed Horizon):
- Equal allocation to all variants
- Run for predetermined time
- Analyze at end
- Optimal for learning

Multi-Armed Bandit (Adaptive):
- Dynamically allocate traffic
- Exploit best variant while exploring others
- Continuous optimization
- Optimal for earning

Bandit Algorithms:
- Epsilon-greedy: Random exploration with probability ε
- UCB (Upper Confidence Bound): Optimism in face of uncertainty
- Thompson Sampling: Sample from posterior, pick best

When to use each:

A/B Testing:
- Need clear statistical conclusions
- Effects are subtle
- Long-term impact matters
- Can afford exploration cost

Bandit:
- High cost of showing inferior variant
- Short-term optimization
- Continuous deployment
- Many variants to test
```

---

## Coding Problems

### Q19: Implement bootstrap for confidence interval
```python
import numpy as np

def bootstrap_ci(data, statistic_func, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap confidence interval for any statistic.
    
    Args:
        data: Original sample
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
    
    Returns:
        (lower, upper) confidence interval
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    # Percentile method
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    
    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower, upper

# Example usage
data = np.random.exponential(scale=2, size=100)
ci_mean = bootstrap_ci(data, np.mean)
ci_median = bootstrap_ci(data, np.median)
print(f"Mean 95% CI: {ci_mean}")
print(f"Median 95% CI: {ci_median}")
```

### Q20: Implement A/B test analysis
```python
import numpy as np
from scipy import stats

def ab_test_analysis(control, treatment, alpha=0.05):
    """
    Analyze A/B test results for conversion rates.
    
    Args:
        control: Array of 0/1 for control group
        treatment: Array of 0/1 for treatment group
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    n_c, n_t = len(control), len(treatment)
    p_c = np.mean(control)
    p_t = np.mean(treatment)
    
    # Pooled proportion under H0
    p_pooled = (np.sum(control) + np.sum(treatment)) / (n_c + n_t)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_c + 1/n_t))
    
    # Z-statistic
    z = (p_t - p_c) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Confidence interval for difference
    se_diff = np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t)
    z_alpha = stats.norm.ppf(1 - alpha/2)
    ci_lower = (p_t - p_c) - z_alpha * se_diff
    ci_upper = (p_t - p_c) + z_alpha * se_diff
    
    # Relative lift
    lift = (p_t - p_c) / p_c * 100
    
    return {
        'control_rate': p_c,
        'treatment_rate': p_t,
        'absolute_difference': p_t - p_c,
        'relative_lift': lift,
        'z_statistic': z,
        'p_value': p_value,
        'significant': p_value < alpha,
        'confidence_interval': (ci_lower, ci_upper)
    }

# Example
np.random.seed(42)
control = np.random.binomial(1, 0.10, 1000)    # 10% conversion
treatment = np.random.binomial(1, 0.12, 1000)  # 12% conversion

results = ab_test_analysis(control, treatment)
for key, value in results.items():
    print(f"{key}: {value}")
```

### Q21: Calculate sample size for A/B test
```python
from scipy import stats
import numpy as np

def sample_size_proportion(p1, mde, alpha=0.05, power=0.8):
    """
    Calculate required sample size per group for proportion test.
    
    Args:
        p1: Baseline conversion rate
        mde: Minimum detectable effect (absolute)
        alpha: Significance level
        power: Statistical power (1 - beta)
    
    Returns:
        Required sample size per group
    """
    p2 = p1 + mde
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Pooled standard deviation
    p_avg = (p1 + p2) / 2
    
    # Sample size formula
    n = 2 * ((z_alpha + z_beta) ** 2) * p_avg * (1 - p_avg) / (mde ** 2)
    
    return int(np.ceil(n))

def sample_size_continuous(sigma, mde, alpha=0.05, power=0.8):
    """
    Calculate required sample size for continuous metric.
    
    Args:
        sigma: Standard deviation
        mde: Minimum detectable effect (absolute)
        alpha: Significance level
        power: Statistical power
    
    Returns:
        Required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
    
    return int(np.ceil(n))

# Examples
print("Proportion test:")
print(f"Baseline 5%, detect 1% lift: {sample_size_proportion(0.05, 0.01)}")
print(f"Baseline 10%, detect 1% lift: {sample_size_proportion(0.10, 0.01)}")

print("\nContinuous test:")
print(f"σ=10, detect effect of 1: {sample_size_continuous(10, 1)}")
```

### Q22: Simulate probability problems
```python
import numpy as np

def birthday_paradox(n_people, n_simulations=10000):
    """
    Probability that at least 2 people share birthday.
    """
    matches = 0
    for _ in range(n_simulations):
        birthdays = np.random.randint(0, 365, n_people)
        if len(birthdays) != len(set(birthdays)):
            matches += 1
    return matches / n_simulations

# Analytical solution
def birthday_analytical(n):
    prob_no_match = 1
    for i in range(n):
        prob_no_match *= (365 - i) / 365
    return 1 - prob_no_match

print(f"23 people - Simulated: {birthday_paradox(23):.3f}")
print(f"23 people - Analytical: {birthday_analytical(23):.3f}")


def monty_hall(switch, n_simulations=10000):
    """
    Monty Hall problem simulation.
    
    Args:
        switch: Whether to switch doors
    
    Returns:
        Win probability
    """
    wins = 0
    for _ in range(n_simulations):
        # Prize behind random door
        prize = np.random.randint(0, 3)
        # Initial choice
        choice = np.random.randint(0, 3)
        
        # Host opens a door (not prize, not choice)
        available = [d for d in range(3) if d != prize and d != choice]
        opened = np.random.choice(available)
        
        if switch:
            # Switch to remaining door
            choice = [d for d in range(3) if d != choice and d != opened][0]
        
        if choice == prize:
            wins += 1
    
    return wins / n_simulations

print(f"\nMonty Hall - Stay: {monty_hall(switch=False):.3f}")
print(f"Monty Hall - Switch: {monty_hall(switch=True):.3f}")
```

### Q23: Bayesian inference example
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def bayesian_ab_test(successes_a, trials_a, successes_b, trials_b, 
                     prior_alpha=1, prior_beta=1, n_samples=100000):
    """
    Bayesian A/B test using Beta-Binomial conjugacy.
    
    Args:
        successes_a, trials_a: Results for variant A
        successes_b, trials_b: Results for variant B
        prior_alpha, prior_beta: Beta prior parameters
        n_samples: Number of posterior samples
    
    Returns:
        Dictionary with Bayesian analysis results
    """
    # Posterior parameters
    alpha_a = prior_alpha + successes_a
    beta_a = prior_beta + trials_a - successes_a
    
    alpha_b = prior_alpha + successes_b
    beta_b = prior_beta + trials_b - successes_b
    
    # Sample from posteriors
    samples_a = np.random.beta(alpha_a, beta_a, n_samples)
    samples_b = np.random.beta(alpha_b, beta_b, n_samples)
    
    # Probability B > A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Expected loss if choosing A when B is better
    diff = samples_b - samples_a
    expected_loss_a = np.mean(np.maximum(diff, 0))
    expected_loss_b = np.mean(np.maximum(-diff, 0))
    
    # Credible intervals
    ci_a = np.percentile(samples_a, [2.5, 97.5])
    ci_b = np.percentile(samples_b, [2.5, 97.5])
    ci_diff = np.percentile(diff, [2.5, 97.5])
    
    return {
        'posterior_mean_a': alpha_a / (alpha_a + beta_a),
        'posterior_mean_b': alpha_b / (alpha_b + beta_b),
        'prob_b_better': prob_b_better,
        'expected_loss_a': expected_loss_a,
        'expected_loss_b': expected_loss_b,
        'ci_a': ci_a,
        'ci_b': ci_b,
        'ci_diff': ci_diff
    }

# Example
results = bayesian_ab_test(
    successes_a=100, trials_a=1000,  # 10% conversion
    successes_b=120, trials_b=1000   # 12% conversion
)

print("Bayesian A/B Test Results:")
for key, value in results.items():
    if isinstance(value, (list, np.ndarray)):
        print(f"{key}: [{value[0]:.4f}, {value[1]:.4f}]")
    else:
        print(f"{key}: {value:.4f}")
```
