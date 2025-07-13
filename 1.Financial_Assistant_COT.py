"""
High-Level Plan

1. Generate a Step-by-Step Plan using GenAI model such as GPT - This is called CoT plan

2. Iteratively execute each step for reasoning using GenAI model such as GPT. - AI model Executes the Reasoning Step-by-Step

3. Summarize and return a final structured report
"""

from openai import OpenAI
import data_info
import json

class PlanningFinancialAdvisorAgent:

    def query_gpt(self, prompt: str) -> str:
        client = OpenAI(api_key=data_info.open_ai_key)
        response =  client.responses.create(
            model="gpt-4o-mini",
            input = prompt,
            temperature=0,
        )
        return response.output_text

    def generate_plan(self, user_profile, investment_goals) -> list:
        """Step 1: Generate a plan (a list of reasoning steps)"""
        plan_prompt = f"""
        You are a financial planning assistant. A client has given you a profile and investment goals.
        
        Your task is to generate a step-by-step reasoning plan to create an investment strategy.
        
        CLIENT PROFILE:
        {json.dumps(user_profile, indent=2)}
        
        INVESTMENT GOALS:
        {investment_goals}
        
        Generate a list of numbered reasoning steps to guide investment advice.
        """
        plan_text = self.query_gpt(plan_prompt)
        steps = [step.strip() for step in plan_text.split('\n') if step.strip() and step[0].isdigit()]
        return steps

    def execute_plan(self, user_profile, investment_goals, steps: list) -> list:
        """Step 2: Run each step with reasoning"""
        results = []
        for step in steps:
            reasoning_prompt = f"""
            You are a financial advisor. Using the information below, reason through this step:
            
            CLIENT PROFILE:
            {json.dumps(user_profile, indent=2)}
            
            INVESTMENT GOALS:
            {investment_goals}
            
            Step: {step}
            
            Thought:"""
            answer = self.query_gpt(reasoning_prompt)
            results.append((step, answer))
        return results

    def summarize_recommendation(self, results: list) -> str:
        """Step 3: Compile all thoughts into a final recommendation"""
        compiled = "\n".join([f"{step}\n{thought}" for step, thought in results])
        final_prompt = f"""
        You are a financial planner. Summarize the findings and generate a final, personalized investment strategy based on the following step-by-step reasoning:
        
        {compiled}
        
        Final Answer:"""
        return self.query_gpt(final_prompt)

    def provide_investment_advice(self, user_profile, investment_goals):
        steps = self.generate_plan(user_profile, investment_goals)
        results = self.execute_plan(user_profile, investment_goals, steps)
        final_answer = self.summarize_recommendation(results)

        # Optional: print intermediate steps
        print("PLAN:")
        for step in steps:
            print(" -", step)

        print("\n STEP-BY-STEP THOUGHTS:")
        for i, (step, thought) in enumerate(results, 1):
            print(f"\nStep {i}: {step}\n{thought}")

        print("\n *****FINAL ADVICE*******:")
        return final_answer

advisor = PlanningFinancialAdvisorAgent()

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
