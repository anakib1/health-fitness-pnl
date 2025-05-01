import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
months = 36
subscription_price = 20
market_size = 360_000_000  # 0.36B maximum market size
install_to_paid_conversion = 0.1  # Base conversion rate from installs to paid users

# Cohort-specific churn rates based on industry data
# Industry average: 12% monthly churn
# Best-in-class (Sworkit): 7% monthly churn
# We'll model improving from industry average to best-in-class
max_churn_rate = 0.2  # Starting with industry average
min_churn_rate = 0.12  # Best-in-class target
churn_improvement_rate = 0.08  # Slower improvement rate to reflect real-world challenges

# Retention benchmarks
day_30_retention = 0.07  # 7% 30-day retention (2022 benchmark)
day_90_retention = 0.24  # 24% 90-day retention (MyFitnessPal benchmark)

# Costs
base_maintenance_cost = 5000
per_user_maintenance_cost = 2.5  # Increased from 0.5 to account for support, servers, etc.
initial_marketing_budget = 10000
marketing_growth_rate = 0.10
development_cost = 100000

# CPI calculation with diminishing returns
base_cpi = 0.4 * 2.0 + 0.4 * 1.5 + 0.2 * 3.0  # Updated to reflect fitness app industry average
cpi_increase_rate = 0.02  # CPI increases by 2% for each 100k users

# App Store Commission
store_commission_rate = 0.30

# Inflation
inflation_rate_annual = 0.04
monthly_discount_rate = (1 + inflation_rate_annual) ** (1 / 12) - 1

# === MARKETING AND ACQUISITION ===
marketing_budgets = [initial_marketing_budget * ((1 + marketing_growth_rate) ** i) for i in range(months)]

# Calculate total users to determine CPI
total_users = 0
installs = []
new_paid_users = []

for i in range(months):
    # Calculate current CPI with diminishing returns
    current_cpi = base_cpi * (1 + (total_users / 100000) * cpi_increase_rate)
    
    # Calculate potential installs
    potential_installs = int(marketing_budgets[i] / current_cpi)
    
    # Check against market size limit
    remaining_market = market_size - total_users
    actual_installs = min(potential_installs, remaining_market)
    
    # Calculate paid users with diminishing conversion rate
    conversion_rate = install_to_paid_conversion * (1 - (total_users / market_size) * 0.5)
    new_paid = int(actual_installs * conversion_rate)
    
    installs.append(actual_installs)
    new_paid_users.append(new_paid)
    total_users += new_paid

# === COHORT REVENUE MODEL ===
cohort_matrix = np.zeros((months, months))
for cohort in range(months):
    users = new_paid_users[cohort]
    for m in range(cohort, months):
        # Calculate cohort-specific churn rate with retention benchmarks
        months_retained = m - cohort
        if months_retained == 0:
            retention_rate = 1.0
        elif months_retained == 1:
            retention_rate = day_30_retention
        elif months_retained == 3:
            retention_rate = day_90_retention
        else:
            # Calculate churn rate that improves over time
            cohort_churn_rate = max(
                min_churn_rate,
                max_churn_rate * (1 - churn_improvement_rate) ** months_retained
            )
            retention_rate = (1 - cohort_churn_rate) ** months_retained
        
        retained = int(users * retention_rate)
        cohort_matrix[cohort][m] = retained * subscription_price

monthly_revenue = cohort_matrix.sum(axis=0)

# === COST AND PROFIT CALCULATIONS ===
active_users = [int(np.sum(cohort_matrix[:, i] / subscription_price)) for i in range(months)]
maintenance_costs = [base_maintenance_cost + per_user_maintenance_cost * u for u in active_users]
development_costs = [development_cost if i == 0 else 0 for i in range(months)]
total_cost = np.array(marketing_budgets) + np.array(maintenance_costs) + np.array(development_costs)

store_commission = monthly_revenue * store_commission_rate
net_revenue = monthly_revenue - store_commission

operating_profit_net = net_revenue - total_cost
ebitda = net_revenue - np.array(marketing_budgets) - np.array(maintenance_costs)
cumulative_profit_net = np.cumsum(operating_profit_net)

inflation_factors = [(1 / ((1 + monthly_discount_rate) ** i)) for i in range(months)]
inflation_adjusted_profit = operating_profit_net * inflation_factors

required_investment = np.where(cumulative_profit_net < 0, total_cost, 0)

# === FINAL SLIDE DATAFRAME ===
slide_df = pd.DataFrame({
    "Month": range(1, months + 1),
    "Net Revenue ($M)": net_revenue / 1e6,
    "Total Cost ($M)": total_cost / 1e6,
    "Net Profit ($M)": operating_profit_net / 1e6,
    "EBITDA ($M)": ebitda / 1e6,
    "EBITDA Margin (%)": (ebitda / net_revenue * 100).round(1),
    "Inflation Adjusted Profit ($M)": inflation_adjusted_profit / 1e6,
    "Cumulative Profit ($M)": cumulative_profit_net / 1e6,
    "Required Investment ($M)": required_investment / 1e6,
    "Active Users (K)": [u/1000 for u in active_users],
    "New Users (K)": [u/1000 for u in new_paid_users]
})

# Calculate LTV and CAC for each cohort
cohort_metrics = []
for cohort in range(months):
    users = new_paid_users[cohort]
    if users > 0:  # Avoid division by zero
        # Calculate total revenue for this cohort
        total_revenue = np.sum(cohort_matrix[cohort])
        ltv = total_revenue / users
        
        # Calculate CAC for this cohort
        # Marketing spend for this cohort
        marketing_spend = marketing_budgets[cohort]
        # Number of installs needed to get one paid user
        installs_per_paid = 1 / (install_to_paid_conversion * (1 - (total_users / market_size) * 0.5))
        # Current CPI for this cohort
        current_cpi = base_cpi * (1 + (total_users / 100000) * cpi_increase_rate)
        # CAC = CPI * installs needed per paid user
        cac = current_cpi * installs_per_paid
        
        cohort_metrics.append({
            'cohort': cohort + 1,
            'users': users,
            'ltv': ltv,
            'cac': cac,
            'ltv_cac_ratio': ltv / cac
        })

# Convert to DataFrame for better analysis
cohort_df = pd.DataFrame(cohort_metrics)

print("\nLifetime Value (LTV) and Customer Acquisition Cost (CAC) Analysis:")
print("\nCohort-specific metrics:")
print(cohort_df.to_string(index=False))

print("\nSummary Statistics:")
print(f"Average LTV: ${cohort_df['ltv'].mean():.2f}")
print(f"Average CAC: ${cohort_df['cac'].mean():.2f}")
print(f"Average LTV/CAC Ratio: {cohort_df['ltv_cac_ratio'].mean():.2f}x")
print(f"Median LTV/CAC Ratio: {cohort_df['ltv_cac_ratio'].median():.2f}x")

# Add cohort metrics to the main DataFrame
slide_df['CAC'] = cohort_df['cac']
slide_df['LTV'] = cohort_df['ltv']
slide_df['LTV/CAC Ratio'] = cohort_df['ltv_cac_ratio']

slide_df.to_csv("slide_df.csv", index=False)

print("\nMain metrics:")
print(slide_df.to_string(index=False))
slide_df.to_csv("slide_df.csv", index=False)

# === CHARTS ===
months_range = range(1, months + 1)

plt.figure(figsize=(28, 6))

# Chart 1: Profit & Inflation-Adjusted Profit
plt.subplot(1, 4, 1)
plt.plot(months_range, operating_profit_net / 1e6, label="Net Profit ($M)", marker='o')
plt.plot(months_range, inflation_adjusted_profit / 1e6, label="Inflation Adj. Profit ($M)", marker='x')
plt.title("Monthly Profit vs Inflation-Adjusted Profit")
plt.xlabel("Month")
plt.ylabel("USD ($M)")
plt.grid(True)
plt.legend()

# Chart 2: Required Investment
plt.subplot(1, 4, 2)
plt.bar(months_range, required_investment / 1e6, color='orange')
plt.title("Monthly Required Investment")
plt.xlabel("Month")
plt.ylabel("Investment Required ($M)")
plt.grid(True)

# Chart 3: Total Revenue
plt.subplot(1, 4, 3)
plt.plot(months_range, net_revenue / 1e6, label="Net Revenue", marker='o', color='green')
plt.title("Monthly Total Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue ($M)")
plt.grid(True)
plt.legend()

# Chart 4: EBITDA Margin
plt.subplot(1, 4, 4)
plt.plot(months_range, slide_df["EBITDA Margin (%)"], label="EBITDA Margin", marker='o', color='purple')
plt.title("Monthly EBITDA Margin")
plt.xlabel("Month")
plt.ylabel("Margin (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Additional chart for user growth
plt.figure(figsize=(14, 6))
plt.plot(months_range, slide_df["Active Users (K)"], label="Active Users", marker='o', color='blue')
plt.plot(months_range, slide_df["New Users (K)"], label="New Users", marker='x', color='red')
plt.title("User Growth Over Time")
plt.xlabel("Month")
plt.ylabel("Users (K)")
plt.grid(True)
plt.legend()
plt.show()
