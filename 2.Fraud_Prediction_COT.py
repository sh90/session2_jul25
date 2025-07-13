
import json
from datetime import datetime, timedelta
from openai import OpenAI
import data_info

class CoTPlanningFraudAgent:

    def analyze_transaction(self, transaction, user_history):
        """Main entry point for CoT-style fraud analysis."""
        features = self._extract_features(transaction, user_history)

        # Step 1: Generate a plan
        plan_prompt = f"""You are a financial fraud analyst.
        Your task is to analyze this transaction step-by-step.
        
        CURRENT TRANSACTION:
        {json.dumps(transaction, indent=2)}
        
        USER HISTORY SUMMARY:
        {json.dumps(features, indent=2)}
        
        Write a clear step-by-step plan to assess fraud risk (numbered steps).
        """
        plan = self.query_gpt(plan_prompt)

        # Step 2: Execute each step one by one
        steps = plan.strip().split("\n")
        reasoning_log = ""
        for step in steps:
            if not step.strip(): continue
            reasoning_log += f"\n {step}\n"
            step_reasoning = self.query_gpt(f"""Given the following transaction and user data, perform this step:\nStep: {step}

            TRANSACTION:
            {json.dumps(transaction)}
            
            USER FEATURES:
            {json.dumps(features)}
            """)
            reasoning_log += f" Thought: {step_reasoning.strip()}\n"

        # Step 3: Generate final fraud risk score
        final_prompt = f"""Based on the analysis steps above, summarize the fraud risk as a JSON object with this format:
        
          "fraud_risk_score": <0-100>,
          "risk_level": "<Low|Medium|High>",
          "explanation": "..."

        Analysis Steps: {reasoning_log}
        """
        final_result = self.query_gpt(final_prompt)

        return {
            "plan": plan,
            "step_analysis": reasoning_log,
            "fraud_report": final_result,
            "features": features
        }

    def _extract_features(self, transaction, history):
        """Extract relevant features from transaction history."""
        amounts = [tx["amount"] for tx in history]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        locations = [tx["location"] for tx in history]
        common_locations = set([loc for loc in locations if locations.count(loc) > 1])

        recent_count = 0
        if history:
            current_time = datetime.fromisoformat(transaction["timestamp"])
            for tx in history:
                tx_time = datetime.fromisoformat(tx["timestamp"])
                if current_time - tx_time <= timedelta(hours=24):
                    recent_count += 1

        return {
            "avg_transaction_amount": avg_amount,
            "transaction_velocity_24h": recent_count,
            "common_locations": list(common_locations),
            "usual_merchant_categories": list(set([tx["merchant_category"] for tx in history])),
            "transaction_count_30d": len(history),
            "highest_single_amount": max(amounts) if amounts else 0,
        }

    def query_gpt(self, prompt: str) -> str:
        """Query GPT for a response."""
        client = OpenAI(api_key=data_info.open_ai_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
        )
        return response.output_text


user_history = [
    {"timestamp": "2025-04-18T10:30:00", "amount": 42.15, "merchant": "Starbucks", "merchant_category": "Food", "location": "New York"},
    {"timestamp": "2025-04-17T18:20:00", "amount": 125.30, "merchant": "Whole Foods", "merchant_category": "Grocery", "location": "New York"},
    {"timestamp": "2025-04-15T12:10:00", "amount": 85.00, "merchant": "Amazon", "merchant_category": "Retail", "location": "Online"},
    {"timestamp": "2025-04-12T09:15:00", "amount": 35.50, "merchant": "Starbucks", "merchant_category": "Food", "location": "New York"},
    {"timestamp": "2025-04-10T20:20:00", "amount": 200.00, "merchant": "Nike", "merchant_category": "Retail", "location": "New York"},
]

suspicious_transaction = {
    "timestamp": "2025-04-19T03:45:00",
    "amount": 9999.99,
    "merchant": "Electronics Store",
    "merchant_category": "Electronics",
    "location": "New Delhi"
}

agent = CoTPlanningFraudAgent()
result = agent.analyze_transaction(suspicious_transaction, user_history)

print(" CoT Plan:\n", result["plan"])
print("\n Step-by-step Analysis:\n", result["step_analysis"])
print("\n Fraud Risk Report:\n", result["fraud_report"])
