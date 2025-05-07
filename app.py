import gradio as gr
from business_model import BusinessParameters, BusinessModel
import pandas as pd
import tempfile

def run_model(
    monthly_price: float,
    quarterly_price: float,
    yearly_price: float,
    market_size: int,
    install_to_trial_conversion: float,
    trial_to_paid_conversion: float,
    base_cpi: float,
    initial_marketing_budget: float,
    max_marketing_budget: float,
    rebill_rate: float,
    store_payment_percentage: float,
    trial_period_days: int,
    development_period_months: int,
    marketing_team_salary: float,
    marketing_team_per_budget: float
):
    # Create business parameters
    params = BusinessParameters(
        monthly_price=monthly_price,
        quarterly_price=quarterly_price,
        yearly_price=yearly_price,
        market_size=market_size,
        install_to_trial_conversion=install_to_trial_conversion,
        trial_to_paid_conversion=trial_to_paid_conversion,
        base_cpi=base_cpi,
        initial_marketing_budget=initial_marketing_budget,
        max_marketing_budget=max_marketing_budget,
        rebill_rate=rebill_rate,
        store_payment_percentage=store_payment_percentage,
        trial_period_days=trial_period_days,
        development_period_months=development_period_months,
        marketing_team_salary=marketing_team_salary,
        marketing_team_per_budget=marketing_team_per_budget
    )
    
    # Create and run business model
    model = BusinessModel(params)
    main_metrics_df, cohort_df, charts = model.calculate_metrics()
    
    # Calculate average LTV, CAC, and LTV/CAC ratio for the whole period
    avg_ltv = cohort_df['ltv'].mean() if not cohort_df.empty else 0
    avg_cac = cohort_df['cac'].mean() if not cohort_df.empty else 0
    avg_ltv_cac = cohort_df['ltv_cac_ratio'].mean() if not cohort_df.empty else 0
    # Calculate total investment required (sum of Required Investment column)
    total_investment_required = main_metrics_df['Required Investment ($M)'].sum() if not main_metrics_df.empty else 0
    
    # Calculate total profit at 2 years (24 months)
    total_profit_2y = main_metrics_df.loc[23, "Cumulative Profit ($M)"] if len(main_metrics_df) > 23 else 0
    
    # Format DataFrames for display
    main_metrics_df = main_metrics_df.round(2)
    cohort_df = cohort_df.round(2)
    
    # Save CSVs to temp files
    main_metrics_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    cohort_metrics_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    main_metrics_df.to_csv(main_metrics_csv.name, index=False)
    cohort_df.to_csv(cohort_metrics_csv.name, index=False)
    
    return (
        main_metrics_df,
        cohort_df,
        charts[0],  # Profit and User Growth
        charts[1],  # EBITDA Margin
        charts[2],  # Required Investment
        charts[3],  # Marketing and Maintenance Spend
        charts[4],  # Stacked Cost Chart
        charts[5],  # Profit and Required Investment Combined Chart
        avg_ltv,
        avg_cac,
        avg_ltv_cac,
        total_investment_required,
        total_profit_2y,
        main_metrics_csv.name,
        cohort_metrics_csv.name
    )

# Create Gradio interface
with gr.Blocks(title="Fitness App Economics Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Fitness App Economics Model
    
    This interactive tool helps model and analyze the economics of a fitness app subscription business.
    Adjust the parameters below to see how they affect key metrics like revenue, user growth, and required investment.
    
    Note: Subscription tier distribution is fixed at 70% monthly, 20% quarterly, and 10% yearly subscriptions.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Subscription Prices")
            monthly_price = gr.Slider(
                minimum=5, maximum=50, value=24, step=1,
                label="Monthly Subscription Price ($)"
            )
            quarterly_price = gr.Slider(
                minimum=5, maximum=50, value=15, step=1,
                label="Quarterly Subscription Price ($)"
            )
            yearly_price = gr.Slider(
                minimum=5, maximum=50, value=10, step=1,
                label="Yearly Subscription Price ($)"
            )
        with gr.Column():
            gr.Markdown("### Market Parameters")
            market_size = gr.Slider(
                minimum=1000000, maximum=1000000000, value=360000000, step=1000000,
                label="Market Size"
            )
            install_to_trial_conversion = gr.Slider(
                minimum=0.01, maximum=0.5, value=0.08, step=0.01,
                label="Install to Trial Conversion Rate"
            )
            trial_to_paid_conversion = gr.Slider(
                minimum=0.01, maximum=0.7, value=0.4, step=0.01,
                label="Trial to Paid Conversion Rate"
            )
            trial_period_days = gr.Slider(
                minimum=1, maximum=30, value=7, step=1,
                label="Trial Period (days)"
            )
            development_period_months = gr.Slider(
                minimum=1, maximum=12, value=3, step=1,
                label="Development Period (months)"
            )
            base_cpi = gr.Slider(
                minimum=0.5, maximum=10.0, value=1.2, step=0.1,
                label="Base CPI ($)"
            )
            initial_marketing_budget = gr.Slider(
                minimum=40000, maximum=250000, value=40000, step=10000,
                label="Initial Marketing Budget ($)"
            )
            max_marketing_budget = gr.Slider(
                minimum=50000, maximum=2000000, value=500000, step=10000,
                label="Max Marketing Budget Cap ($)"
            )
            marketing_team_salary = gr.Slider(
                minimum=2000, maximum=10000, value=4000, step=500,
                label="Marketing Team Salary ($/month)"
            )
            marketing_team_per_budget = gr.Slider(
                minimum=10000, maximum=100000, value=50000, step=5000,
                label="$ per Marketing Team Member"
            )
            rebill_rate = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.8, step=0.1,
                label="Average Rebill Rate (times)"
            )
            store_payment_percentage = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.33, step=0.1,
                label="Store Payment Percentage"
            )
    
    run_button = gr.Button("Run Model", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Main Metrics")
            main_metrics = gr.Dataframe(
                headers=["Month", "Net Revenue ($M)", "Total Cost ($M)", "Marketing Spend ($M)", 
                        "Development Cost ($M)", "Marketing Team Cost ($M)", "Operational Cost ($M)", "Maintenance Cost ($M)",
                        "Cumulative Marketing ($M)", "Net Profit ($M)", "EBITDA ($M)", 
                        "EBITDA Margin (%)", "Inflation Adjusted Profit ($M)", 
                        "Cumulative Profit ($M)", "Required Investment ($M)",
                        "Active Users (K)", "Active Trials (K)", "New Users (K)", "New Trials (K)"],
                datatype=["number"]*19,
                col_count=(19, "fixed"),
                row_count=(60, "fixed"),
                type="pandas"
            )
            main_metrics_download = gr.File(label="Download Main Metrics CSV")
            gr.Markdown("### Total Investment Required (Sum of Required Investment)")
            total_investment_out = gr.Number(label="Total Investment Required ($M)")
            gr.Markdown("### Total Profit at 2 Years")
            total_profit_2y_out = gr.Number(label="Total Profit at 2 Years ($M)")
            gr.Markdown("### Average LTV, CAC, and LTV/CAC Ratio (Whole Period)")
            avg_ltv_out = gr.Number(label="Average LTV ($)")
            avg_cac_out = gr.Number(label="Average CAC ($)")
            avg_ltv_cac_out = gr.Number(label="Average LTV/CAC Ratio")
        with gr.Column():
            gr.Markdown("## Cohort Metrics")
            cohort_metrics = gr.Dataframe(
                headers=["Cohort", "Users", "Trials", "LTV", "CAC", "LTV/CAC Ratio"],
                datatype=["number"]*6,
                col_count=(6, "fixed"),
                row_count=(60, "fixed"),
                type="pandas"
            )
            cohort_metrics_download = gr.File(label="Download Cohort Metrics CSV")
    
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
    
    with gr.Row():
        gr.Markdown("## Marketing and Maintenance Spend")
        spend_chart = gr.Plot()
    
    with gr.Row():
        gr.Markdown("## Stacked Cost Breakdown")
        stacked_cost_chart = gr.Plot()
    
    with gr.Row():
        gr.Markdown("## Net Profit and Required Investment")
        profit_investment_chart = gr.Plot()
    
    run_button.click(
        fn=run_model,
        inputs=[
            monthly_price, quarterly_price, yearly_price,
            market_size, install_to_trial_conversion,
            trial_to_paid_conversion, base_cpi, initial_marketing_budget,
            max_marketing_budget, rebill_rate, store_payment_percentage, trial_period_days,
            development_period_months, marketing_team_salary, marketing_team_per_budget
        ],
        outputs=[main_metrics, cohort_metrics, profit_chart, margin_chart, investment_chart, spend_chart, stacked_cost_chart, profit_investment_chart, avg_ltv_out, avg_cac_out, avg_ltv_cac_out, total_investment_out, total_profit_2y_out, main_metrics_download, cohort_metrics_download]
    )

if __name__ == "__main__":
    demo.launch() 