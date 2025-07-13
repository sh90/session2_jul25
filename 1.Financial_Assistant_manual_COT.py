# pip install openai
"""
High-Level Plan

1. Generate a Step-by-Step Plan through human-in-the-loop planning — and it’s often more reliable, safer, and easier to audit.

2. Iteratively execute each step for reasoning using GenAI model such as GPT. - AI model Executes the Reasoning Step-by-Step

3. Summarize and return a final structured report
"""

from openai import OpenAI
import json
import data_info

class FinancialAdvisorAgent:

    def provide_investment_advice(self, user_profile, investment_goals):
        """Generate personalized investment advice using chain-of-thought reasoning."""

        prompt = f"""
        As a financial advisor, provide investment recommendations for this client basis on only the information from year 2020-2024:

        CLIENT PROFILE:
        {json.dumps(user_profile, indent=2)}

        INVESTMENT GOALS:
        {investment_goals}

        Let's think through this step-by-step:
        1. Analyze the client's risk tolerance based on age, financial situation, and goals
        2. Consider current market conditions and economic factors
        3. Evaluate appropriate asset allocation (stocks, bonds, alternatives)
        4. Recommend specific investment vehicles and explain the rationale
        5. Address potential concerns and provide risk mitigation strategies
        
        """
        client = OpenAI(api_key=data_info.open_ai_key)
        response =  client.responses.create(
            model="gpt-4o-mini",
            input = prompt,
            temperature=0,
        )
        return response.output_text

# Example usage
advisor = FinancialAdvisorAgent()
user_profile = {
    "age": 42,
    "income": 120000,
    "savings": 180000,
    "debt": 220000,  # Mortgage
    "dependents": 2,
    "existing_investments": {
        "stocks": 50000,
        "bonds": 30000,
        "retirement_accounts": 210000
    },
    "risk_tolerance": "moderate"
}


investment_goals = """
I want to save for my children's college education (ages 8 and 10) while also 
growing my retirement fund. I'm concerned about market volatility but want to 
balance growth with reasonable risk. I can invest $1,500 monthly.
"""

advice = advisor.provide_investment_advice(user_profile, investment_goals)
print(advice)
