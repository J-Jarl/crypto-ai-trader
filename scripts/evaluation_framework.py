"""
Evaluation Framework for AI-Driven Cryptocurrency Trading
Uses Qwen3-coder to systematically test and measure trading AI performance
"""

import json
from typing import List, Dict, Any
from datetime import datetime
import subprocess


class TradingEvaluator:
    """Evaluates AI trading decisions against predefined criteria"""
    
    def __init__(self, model_name: str = "qwen3-coder:30b"):
        self.model_name = model_name
        self.evaluation_results = []
    
    def call_ollama(self, prompt: str) -> str:
        """Call Ollama model for evaluation"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error calling model: {str(e)}"
    
    def evaluate_sentiment_analysis(self, news_text: str, expected_sentiment: str, ai_sentiment: str) -> Dict[str, Any]:
        """
        Evaluate how well the AI analyzed sentiment from news
        
        Args:
            news_text: The original news article
            expected_sentiment: Ground truth sentiment (bullish/bearish/neutral)
            ai_sentiment: What your AI predicted
        """
        prompt = f"""Evaluate this sentiment analysis:

News Text: {news_text}

Expected Sentiment: {expected_sentiment}
AI Predicted Sentiment: {ai_sentiment}

Score the AI's sentiment analysis on a scale of 0-100, where:
- 100 = Perfect match with clear reasoning
- 75-99 = Correct but could be more nuanced
- 50-74 = Partially correct or close
- 25-49 = Incorrect but understandable mistake
- 0-24 = Completely wrong

Respond in JSON format:
{{
    "score": <number 0-100>,
    "reasoning": "<explanation of score>",
    "correct": <true/false>
}}
"""
        
        response = self.call_ollama(prompt)
        
        try:
            # Parse JSON response
            eval_result = json.loads(response)
            eval_result["test_type"] = "sentiment_analysis"
            eval_result["timestamp"] = datetime.now().isoformat()
            self.evaluation_results.append(eval_result)
            return eval_result
        except json.JSONDecodeError:
            return {
                "score": 0,
                "reasoning": "Failed to parse evaluation response",
                "correct": False,
                "raw_response": response
            }
    
    def evaluate_trading_decision(self, market_data: Dict, ai_decision: str, actual_outcome: str) -> Dict[str, Any]:
        """
        Evaluate a trading decision after knowing the outcome
        
        Args:
            market_data: Dict with price, volume, indicators, sentiment, etc.
            ai_decision: What the AI recommended (buy/sell/hold with reasoning)
            actual_outcome: What happened (e.g., "price went up 5% in 24h")
        """
        prompt = f"""Evaluate this trading decision:

Market Context:
{json.dumps(market_data, indent=2)}

AI Decision: {ai_decision}

Actual Outcome: {actual_outcome}

Evaluate this decision considering:
1. Was the decision profitable given the outcome?
2. Was the reasoning sound even if outcome was unexpected?
3. Was risk management appropriate?
4. Would you make the same decision with this information?

Score 0-100 and respond in JSON:
{{
    "score": <number>,
    "profitable": <true/false>,
    "sound_reasoning": <true/false>,
    "risk_appropriate": <true/false>,
    "overall_assessment": "<brief assessment>"
}}
"""
        
        response = self.call_ollama(prompt)
        
        try:
            eval_result = json.loads(response)
            eval_result["test_type"] = "trading_decision"
            eval_result["timestamp"] = datetime.now().isoformat()
            self.evaluation_results.append(eval_result)
            return eval_result
        except json.JSONDecodeError:
            return {
                "score": 0,
                "reasoning": "Failed to parse evaluation response",
                "raw_response": response
            }
    
    def evaluate_risk_management(self, trade_details: Dict, ai_position_size: float) -> Dict[str, Any]:
        """
        Evaluate if AI is sizing positions appropriately for risk management
        
        Args:
            trade_details: Portfolio size, trade risk, market volatility, etc.
            ai_position_size: Percentage of portfolio AI wants to risk
        """
        prompt = f"""Evaluate this position sizing decision:

Trade Details:
{json.dumps(trade_details, indent=2)}

AI Recommended Position Size: {ai_position_size}% of portfolio

Evaluate considering:
1. Is this position size appropriate for the risk level?
2. Does it account for market volatility?
3. Is it conservative enough to survive drawdowns?
4. Would a professional trader approve this sizing?

Score 0-100 and respond in JSON:
{{
    "score": <number>,
    "appropriate": <true/false>,
    "risk_level": "<low/medium/high>",
    "recommendation": "<keep/increase/decrease position size>",
    "reasoning": "<explanation>"
}}
"""
        
        response = self.call_ollama(prompt)
        
        try:
            eval_result = json.loads(response)
            eval_result["test_type"] = "risk_management"
            eval_result["timestamp"] = datetime.now().isoformat()
            self.evaluation_results.append(eval_result)
            return eval_result
        except json.JSONDecodeError:
            return {
                "score": 0,
                "reasoning": "Failed to parse evaluation response",
                "raw_response": response
            }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary stats across all evaluations"""
        if not self.evaluation_results:
            return {"message": "No evaluations performed yet"}
        
        scores = [r["score"] for r in self.evaluation_results if "score" in r]
        
        return {
            "total_evaluations": len(self.evaluation_results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "test_types": list(set(r.get("test_type", "unknown") for r in self.evaluation_results))
        }
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save all evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump({
                "evaluations": self.evaluation_results,
                "summary": self.get_summary_statistics()
            }, f, indent=2)
        print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = TradingEvaluator()
    
    # Example 1: Test sentiment analysis
    print("Testing sentiment analysis...")
    result1 = evaluator.evaluate_sentiment_analysis(
        news_text="Bitcoin ETF approval expected to drive institutional adoption",
        expected_sentiment="bullish",
        ai_sentiment="bullish"
    )
    print(f"Sentiment Score: {result1.get('score', 'N/A')}/100")
    print(f"Correct: {result1.get('correct', 'N/A')}\n")
    
    # Example 2: Test trading decision
    print("Testing trading decision...")
    result2 = evaluator.evaluate_trading_decision(
        market_data={
            "asset": "BTC",
            "price": 45000,
            "24h_volume": "high",
            "sentiment": "positive",
            "technical_signal": "oversold RSI"
        },
        ai_decision="BUY - Oversold conditions with positive sentiment",
        actual_outcome="Price increased 8% in next 48 hours"
    )
    print(f"Decision Score: {result2.get('score', 'N/A')}/100")
    print(f"Profitable: {result2.get('profitable', 'N/A')}\n")
    
    # Get summary
    summary = evaluator.get_summary_statistics()
    print("Summary Statistics:")
    print(json.dumps(summary, indent=2))
    
    # Save results
    evaluator.save_results()