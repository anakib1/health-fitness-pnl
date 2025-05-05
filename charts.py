import matplotlib.pyplot as plt
import numpy as np

class ChartCreator:
    def __init__(self, months: int):
        self.months_range = range(1, months + 1)
        self.figures = []

    def create_all_charts(self, df, cost_components=None) -> list:
        self._create_profit_chart(df)
        self._create_margin_chart(df)
        self._create_investment_chart(df)
        self._create_spend_chart(df)
        if cost_components is not None:
            self._create_stacked_cost_chart(cost_components)
        self._create_profit_investment_chart(df)
        return self.figures

    def _create_profit_chart(self, df):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot profit on primary y-axis
        ax1.plot(self.months_range, df["Net Profit ($M)"], 'b-', label='Net Profit')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Profit ($M)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create secondary y-axis for users
        ax2 = ax1.twinx()
        ax2.plot(self.months_range, df["Active Users (K)"], 'r-', label='Active Users')
        ax2.set_ylabel('Active Users (K)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Profit and User Growth Over Time')
        fig.tight_layout()
        self.figures.append(fig)

    def _create_margin_chart(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(self.months_range, df["EBITDA Margin (%)"], 'g-')
        plt.xlabel('Month')
        plt.ylabel('EBITDA Margin (%)')
        plt.title('EBITDA Margin Over Time')
        plt.grid(True)
        self.figures.append(plt.gcf())

    def _create_investment_chart(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(self.months_range, df["Required Investment ($M)"], 'r-')
        plt.xlabel('Month')
        plt.ylabel('Required Investment ($M)')
        plt.title('Required Investment Over Time')
        plt.grid(True)
        self.figures.append(plt.gcf())

    def _create_spend_chart(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(self.months_range, df["Marketing Spend ($M)"], 'b-', label='Marketing')
        plt.plot(self.months_range, df["Development Cost ($M)"], 'g-', label='Development')
        plt.plot(self.months_range, df["Operational Cost ($M)"], 'y-', label='Operational')
        plt.plot(self.months_range, df["Maintenance Cost ($M)"], 'r-', label='Maintenance')
        plt.xlabel('Month')
        plt.ylabel('Cost ($M)')
        plt.title('Marketing and Maintenance Spend Over Time')
        plt.legend()
        plt.grid(True)
        self.figures.append(plt.gcf())

    def _create_stacked_cost_chart(self, cost_df):
        plt.figure(figsize=(12, 7))
        labels = ["Marketing", "Development", "Marketing Team", "Operations", "User Maintenance"]
        data = [cost_df[label] / 1e6 for label in labels]
        plt.stackplot(cost_df["Month"], data, labels=labels)
        plt.xlabel('Month')
        plt.ylabel('Cost ($M)')
        plt.title('Stacked Cost Breakdown Over Time')
        plt.legend(loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        self.figures.append(plt.gcf())

    def _create_profit_investment_chart(self, df):
        plt.figure(figsize=(12, 7))
        plt.bar(df["Month"], df["Net Profit ($M)"], color='tab:blue', alpha=0.6, label='Net Profit ($M)')
        plt.bar(df["Month"], -df["Required Investment ($M)"], color='tab:red', alpha=0.4, label='Required Investment ($M, negative)')
        plt.xlabel('Month')
        plt.ylabel('Amount ($M)')
        plt.title('Monthly Net Profit and Required Investment')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        self.figures.append(plt.gcf()) 