import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from economics import (
    months, subscription_price, market_size, install_to_paid_conversion,
    max_churn_rate, min_churn_rate, churn_improvement_rate,
    day_30_retention, day_90_retention,
    base_maintenance_cost, per_user_maintenance_cost,
    initial_marketing_budget, marketing_growth_rate, development_cost,
    base_cpi, cpi_increase_rate, store_commission_rate,
    inflation_rate_annual, monthly_discount_rate
)

# Constants
months = 36
marketing_growth_rate = 0.10
base_maintenance_cost = 5000
per_user_maintenance_cost = 2.5
development_cost = 100000
cpi_increase_rate = 0.02
store_commission_rate = 0.30
inflation_rate_annual = 0.04
monthly_discount_rate = (1 + inflation_rate_annual) ** (1 / 12) - 1
churn_improvement_rate = 0.08

def run_model(
    subscription_price,
    market_size,
    install_to_paid_conversion,
    max_churn_rate,
    min_churn_rate,
    day_90_retention,
    base_cpi,
    initial_marketing_budget
):
    # === MARKETING AND ACQUISITION ===
    marketing_budgets = [initial_marketing_budget * ((1 + marketing_growth_rate) ** i) for i in range(months)]
    
    # Calculate total users to determine CPI
    total_users = 0
    installs = []
    new_paid_users = []
    
    for i in range(months):
        current_cpi = base_cpi * (1 + (total_users / 100000) * cpi_increase_rate)
        potential_installs = int(marketing_budgets[i] / current_cpi)
        remaining_market = market_size - total_users
        actual_installs = min(potential_installs, remaining_market)
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
            months_retained = m - cohort
            if months_retained == 0:
                retention_rate = 1.0
            elif months_retained == 3:
                retention_rate = day_90_retention
            else:
                # Calculate retention rate based on months retained
                if months_retained < 3:
                    # Linear interpolation between 1.0 and day_90_retention for first 3 months
                    retention_rate = 1.0 - (months_retained / 3) * (1.0 - day_90_retention)
                else:
                    # After 90 days, use the churn model
                    cohort_churn_rate = max(
                        min_churn_rate,
                        max_churn_rate * (1 - churn_improvement_rate) ** (months_retained - 3)
                    )
                    retention_rate = day_90_retention * ((1 - cohort_churn_rate) ** (months_retained - 3))
            
            retained = int(users * retention_rate)
            cohort_matrix[cohort][m] = retained * subscription_price

    monthly_revenue = cohort_matrix.sum(axis=0)
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

    # Create DataFrame
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
        if users > 0:
            total_revenue = np.sum(cohort_matrix[cohort])
            ltv = total_revenue / users
            marketing_spend = marketing_budgets[cohort]
            installs_per_paid = 1 / (install_to_paid_conversion * (1 - (total_users / market_size) * 0.5))
            current_cpi = base_cpi * (1 + (total_users / 100000) * cpi_increase_rate)
            cac = current_cpi * installs_per_paid
            
            cohort_metrics.append({
                'cohort': cohort + 1,
                'users': users,
                'ltv': ltv,
                'cac': cac,
                'ltv_cac_ratio': ltv / cac
            })

    cohort_df = pd.DataFrame(cohort_metrics)
    
    # Create charts
    months_range = range(1, months + 1)
    
    # Chart 1: Profit & Inflation-Adjusted Profit
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(months_range, operating_profit_net / 1e6, label="Net Profit ($M)", marker='o')
    plt.plot(months_range, inflation_adjusted_profit / 1e6, label="Inflation Adj. Profit ($M)", marker='x')
    plt.title("Monthly Profit vs Inflation-Adjusted Profit")
    plt.xlabel("Month")
    plt.ylabel("USD ($M)")
    plt.grid(True)
    plt.legend()
    
    # Chart 2: User Growth
    plt.subplot(1, 2, 2)
    plt.plot(months_range, slide_df["Active Users (K)"], label="Active Users", marker='o', color='blue')
    plt.plot(months_range, slide_df["New Users (K)"], label="New Users", marker='x', color='red')
    plt.title("User Growth Over Time")
    plt.xlabel("Month")
    plt.ylabel("Users (K)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    profit_chart = plt.gcf()
    plt.close()

    # Chart 3: EBITDA Margin
    plt.figure(figsize=(14, 6))
    plt.plot(months_range, slide_df["EBITDA Margin (%)"], label="EBITDA Margin", marker='o', color='purple')
    plt.title("Monthly EBITDA Margin")
    plt.xlabel("Month")
    plt.ylabel("Margin (%)")
    plt.grid(True)
    plt.legend()
    margin_chart = plt.gcf()
    plt.close()

    # Chart 4: Required Investment
    plt.figure(figsize=(14, 6))
    plt.bar(months_range, slide_df["Required Investment ($M)"], color='orange')
    plt.title("Monthly Required Investment")
    plt.xlabel("Month")
    plt.ylabel("Investment Required ($M)")
    plt.grid(True)
    investment_chart = plt.gcf()
    plt.close()

    return (
        slide_df.to_html(index=False),
        cohort_df.to_html(index=False),
        profit_chart,
        margin_chart,
        investment_chart
    )

# Create Gradio interface
with gr.Blocks(title="Fitness App Economics Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Fitness App Economics Model
    
    This interactive tool helps model and analyze the economics of a fitness app subscription business.
    Adjust the parameters below to see how they affect key metrics like revenue, user growth, and required investment.
    """)
    
    with gr.Row():
        with gr.Column():
            subscription_price = gr.Slider(
                minimum=5, maximum=50, value=20, step=1,
                label="Subscription Price ($)"
            )
            market_size = gr.Slider(
                minimum=1000000, maximum=1000000000, value=360000000, step=1000000,
                label="Market Size"
            )
            install_to_paid_conversion = gr.Slider(
                minimum=0.01, maximum=0.5, value=0.12, step=0.01,
                label="Install to Paid Conversion Rate"
            )
            max_churn_rate = gr.Slider(
                minimum=0.12, maximum=0.5, value=0.4, step=0.01,
                label="Maximum Churn Rate"
            )
            min_churn_rate = gr.Slider(
                minimum=0.05, maximum=0.2, value=0.12, step=0.01,
                label="Minimum Churn Rate"
            )
        with gr.Column():
            day_90_retention = gr.Slider(
                minimum=0.1, maximum=0.5, value=0.24, step=0.01,
                label="90-Day Retention Rate"
            )
            base_cpi = gr.Slider(
                minimum=0.5, maximum=5.0, value=2.1, step=0.1,
                label="Base CPI ($)"
            )
            initial_marketing_budget = gr.Slider(
                minimum=5000, maximum=50000, value=10000, step=1000,
                label="Initial Marketing Budget ($)"
            )
    
    run_button = gr.Button("Run Model", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Main Metrics")
            main_metrics = gr.HTML()
        with gr.Column():
            gr.Markdown("## Cohort Metrics")
            cohort_metrics = gr.HTML()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Profit and User Growth")
            profit_chart = gr.Plot()
        with gr.Column():
            gr.Markdown("## EBITDA Margin")
            margin_chart = gr.Plot()
    
    with gr.Row():
        gr.Markdown("## Required Investment")
        investment_chart = gr.Plot()
    
    run_button.click(
        fn=run_model,
        inputs=[
            subscription_price, market_size, install_to_paid_conversion,
            max_churn_rate, min_churn_rate, day_90_retention,
            base_cpi, initial_marketing_budget
        ],
        outputs=[main_metrics, cohort_metrics, profit_chart, margin_chart, investment_chart]
    )

# For Hugging Face Spaces deployment
if __name__ == "__main__":
    demo.launch() 