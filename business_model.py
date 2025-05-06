import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from charts import ChartCreator

@dataclass(frozen=True)
class SubscriptionTier:
    duration_months: int
    price: float
    distribution: float  # Percentage of users who choose this tier

@dataclass
class BusinessParameters:
    # Required parameters (no default values)
    monthly_price: float
    quarterly_price: float
    yearly_price: float
    rebill_rate: float
    market_size: int
    install_to_trial_conversion: float
    trial_to_paid_conversion: float
    base_cpi: float
    initial_marketing_budget: float
    
    # Optional parameters (with default values)
    months: int = 36  # 5 years projection
    development_period_months: int = 6  # 6 months development period
    marketing_growth_rate: float = 1.0  # 100% monthly growth
    max_marketing_budget: float = 300000  # Cap at 300K per month
    developer_salary: float = 5000
    developer_count: int = 3
    monthly_operational_cost: float = 7000
    per_user_maintenance_cost: float = 0.5  # Reduced to $0.5 per user
    cpi_increase_rate: float = 0.00
    store_commission_rate: float = 0.30
    store_payment_percentage: float = 0.33  # 33% of payments go through store
    inflation_rate_annual: float = 0.04
    trial_period_days: int = 7  # Default 7-day trial period
    marketing_team_salary: float = 2500   # Monthly salary per marketing team member
    marketing_team_per_budget: float = 50000  # $50K/month per marketing team member
    refund_rate: float = 0.03  # 2% default refund rate
    
    # Fixed distribution between tiers (70/20/10)
    monthly_distribution: float = 0.70
    quarterly_distribution: float = 0.20
    yearly_distribution: float = 0.10
    
    seasonality_months: list = field(default_factory=lambda: [1])  # Only January
    seasonality_cac_factor: float = 1.3  # CAC is 2x in January
    seasonality_install_to_trial_factor: float = 1.15  # install-to-trial is 1.15x in January

    def __post_init__(self):
        # Validate subscription distributions sum to 1
        total_distribution = self.monthly_distribution + self.quarterly_distribution + self.yearly_distribution
        if not np.isclose(total_distribution, 1.0):
            raise ValueError("Subscription tier distributions must sum to 1.0")

class BusinessModel:
    def __init__(self, params: BusinessParameters):
        self.params = params
        self.monthly_discount_rate = (1 + params.inflation_rate_annual) ** (1 / 12) - 1
        self.chart_creator = ChartCreator(params.months)
        
        # Initialize subscription tiers
        self.subscription_tiers = [
            SubscriptionTier(1, params.monthly_price, params.monthly_distribution),
            SubscriptionTier(3, params.quarterly_price, params.quarterly_distribution),
            SubscriptionTier(12, params.yearly_price, params.yearly_distribution)
        ]

    def calculate_marketing_and_acquisition(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        # Calculate marketing budgets with cap
        marketing_budgets = []
        current_budget = self.params.initial_marketing_budget
        
        for i in range(self.params.months):
            if i < self.params.development_period_months:
                marketing_budgets.append(0)
            else:
                current_budget = min(
                    current_budget * (1 + self.params.marketing_growth_rate),
                    self.params.max_marketing_budget
                )
                month_in_year = (i % 12) + 1
                if month_in_year in self.params.seasonality_months:
                    marketing_budgets.append(current_budget * 2)
                else:
                    marketing_budgets.append(current_budget)
        
        total_users = 0
        installs = []
        trials = []
        new_paid_users = []
        
        for i in range(self.params.months):
            if i < self.params.development_period_months:
                installs.append(0)
                trials.append(0)
                new_paid_users.append(0)
                continue
            month_in_year = (i % 12) + 1
            if month_in_year in self.params.seasonality_months:
                current_cpi = self.params.base_cpi * self.params.seasonality_cac_factor
                install_to_trial = min(self.params.install_to_trial_conversion * self.params.seasonality_install_to_trial_factor, 1.0)
            else:
                current_cpi = self.params.base_cpi
                install_to_trial = self.params.install_to_trial_conversion
            potential_installs = int(marketing_budgets[i] / current_cpi)
            remaining_market = self.params.market_size - total_users
            actual_installs = min(potential_installs, remaining_market)
            new_trials = int(actual_installs * install_to_trial)
            new_paid = int(new_trials * self.params.trial_to_paid_conversion)
            installs.append(actual_installs)
            trials.append(new_trials)
            new_paid_users.append(new_paid)
            total_users += new_paid
        
        return marketing_budgets, installs, trials, new_paid_users

    def calculate_cohort_matrix(self, new_paid_users: List[int], trials: List[int]) -> np.ndarray:
        cohort_matrix = np.zeros((self.params.months, self.params.months))
        trial_matrix = np.zeros((self.params.months, self.params.months))
        
        # Calculate trial days per month (approximate)
        trial_months = self.params.trial_period_days / 30.44  # Average days in a month
        
        # Calculate churn rate from rebill rate
        churn_rate = 1.0 / (1.0 + self.params.rebill_rate)
        retention_rate = 1.0 - churn_rate
        
        for cohort in range(self.params.months):
            trial_users = trials[cohort]
            paid_users = new_paid_users[cohort]
            
            if trial_users == 0 and paid_users == 0:
                continue
                
            # Distribute users across subscription tiers
            tier_trials = {
                tier: int(trial_users * tier.distribution)
                for tier in self.subscription_tiers
            }
            
            tier_paid = {
                tier: int(paid_users * tier.distribution)
                for tier in self.subscription_tiers
            }
            
            # Calculate trial period
            for tier in self.subscription_tiers:
                trial_user_count = tier_trials[tier]
                if trial_user_count > 0:
                    for m in range(cohort, min(cohort + int(trial_months) + 1, self.params.months)):
                        trial_matrix[cohort][m] += trial_user_count
            
            # Churn-based active user and revenue modeling
            for tier in self.subscription_tiers:
                paid_user_count = tier_paid[tier]
                if paid_user_count == 0:
                    continue
                start_month = cohort + int(trial_months)  # Start after trial period
                for m in range(start_month, self.params.months):
                    months_since_conversion = m - start_month
                    active_users = paid_user_count * (retention_rate ** months_since_conversion)
                    cohort_matrix[cohort][m] += active_users * tier.price
        
        return cohort_matrix, trial_matrix

    def calculate_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[plt.Figure]]:
        marketing_budgets, installs, trials, new_paid_users = self.calculate_marketing_and_acquisition()
        cohort_matrix, trial_matrix = self.calculate_cohort_matrix(new_paid_users, trials)
        
        monthly_revenue = cohort_matrix.sum(axis=0)
        # Subtract refunds
        refunds = monthly_revenue * self.params.refund_rate
        monthly_revenue = monthly_revenue - refunds
        active_trials = trial_matrix.sum(axis=0)
        active_paid_users = [int(np.sum(cohort_matrix[:, i] / self.params.monthly_price)) 
                           for i in range(self.params.months)]
        
        # Calculate development costs (ongoing for all months)
        development_costs = [
            self.params.developer_salary * self.params.developer_count 
            for i in range(self.params.months)
        ]
        
        # Calculate operational costs (start after development period)
        operational_costs = [
            self.params.monthly_operational_cost if i >= self.params.development_period_months else 0
            for i in range(self.params.months)
        ]
        
        # Calculate maintenance costs (including trial users, start after development period)
        maintenance_costs = [
            self.params.per_user_maintenance_cost * (active_paid_users[i] + active_trials[i])
            if i >= self.params.development_period_months else 0
            for i in range(self.params.months)
        ]
        
        # Calculate marketing team size and cost
        marketing_team_size = [
            int(np.ceil(marketing_budgets[i] / self.params.marketing_team_per_budget)) if marketing_budgets[i] > 0 else 0
            for i in range(self.params.months)
        ]
        marketing_team_costs = [
            marketing_team_size[i] * self.params.marketing_team_salary
            for i in range(self.params.months)
        ]
        
        total_cost = (np.array(marketing_budgets) + 
                     np.array(development_costs) + 
                     np.array(operational_costs) + 
                     np.array(maintenance_costs) +
                     np.array(marketing_team_costs))
        
        # Calculate store commission only on store payments
        store_commission = monthly_revenue * self.params.store_commission_rate * self.params.store_payment_percentage
        net_revenue = monthly_revenue - store_commission
        operating_profit_net = net_revenue - total_cost
        ebitda = net_revenue - np.array(marketing_budgets) - np.array(maintenance_costs)
        cumulative_profit_net = np.cumsum(operating_profit_net)
        
        inflation_factors = [(1 / ((1 + self.monthly_discount_rate) ** i)) 
                           for i in range(self.params.months)]
        inflation_adjusted_profit = operating_profit_net * inflation_factors

        # Calculate rolling required investment (cash buffer)
        rolling_required_investment = []
        cumulative_profit = 0
        for i in range(self.params.months):
            cumulative_profit += operating_profit_net[i]
            if cumulative_profit < 0:
                rolling_required_investment.append(-cumulative_profit)
                cumulative_profit = 0
            else:
                rolling_required_investment.append(0)

        # Create main metrics DataFrame
        main_metrics_df = pd.DataFrame({
            "Month": range(1, self.params.months + 1),
            "Net Revenue ($M)": net_revenue / 1e6,
            "Total Cost ($M)": total_cost / 1e6,
            "Marketing Spend ($M)": np.array(marketing_budgets) / 1e6,
            "Development Cost ($M)": np.array(development_costs) / 1e6,
            "Marketing Team Cost ($M)": np.array(marketing_team_costs) / 1e6,
            "Operational Cost ($M)": np.array(operational_costs) / 1e6,
            "Maintenance Cost ($M)": np.array(maintenance_costs) / 1e6,
            "Cumulative Marketing ($M)": np.cumsum(marketing_budgets) / 1e6,
            "Net Profit ($M)": operating_profit_net / 1e6,
            "EBITDA ($M)": ebitda / 1e6,
            "EBITDA Margin (%)": np.round(np.divide(ebitda, net_revenue, out=np.zeros_like(ebitda), where=net_revenue!=0) * 100, 1),
            "Inflation Adjusted Profit ($M)": inflation_adjusted_profit / 1e6,
            "Cumulative Profit ($M)": cumulative_profit_net / 1e6,
            "Required Investment ($M)": np.array(rolling_required_investment) / 1e6,
            "Active Users (K)": [u/1000 for u in active_paid_users],
            "Active Trials (K)": [t/1000 for t in active_trials],
            "New Users (K)": [u/1000 for u in new_paid_users],
            "New Trials (K)": [t/1000 for t in trials]
        })

        # For stacked cost chart
        cost_components = pd.DataFrame({
            "Month": range(1, self.params.months + 1),
            "Marketing": marketing_budgets,
            "Development": development_costs,
            "Marketing Team": marketing_team_costs,
            "Operations": operational_costs,
            "User Maintenance": maintenance_costs,
            "Revenue": monthly_revenue
        })

        # Calculate cohort metrics
        cohort_metrics = []
        for cohort in range(self.params.months):
            users = new_paid_users[cohort]
            if users > 0:
                total_revenue = np.sum(cohort_matrix[cohort])
                ltv = total_revenue / users  # Net revenue per paid user
                marketing_spend = marketing_budgets[cohort]
                cac = marketing_spend / users  # CAC: marketing spend per paid user
                cohort_metrics.append({
                    'cohort': cohort + 1,
                    'users': users,
                    'trials': trials[cohort],
                    'ltv': ltv,
                    'cac': cac,
                    'ltv_cac_ratio': ltv / cac if cac > 0 else 0
                })

        cohort_df = pd.DataFrame(cohort_metrics)
        
        # Create charts using the ChartCreator
        charts = self.chart_creator.create_all_charts(main_metrics_df, cost_components)
        
        return main_metrics_df, cohort_df, charts 