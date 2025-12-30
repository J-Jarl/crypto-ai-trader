#!/usr/bin/env python3
"""
Backtesting Framework for Bitcoin Trading AI
============================================

Re-runs predictions on historical data AND evaluates them to test system improvements.

This script:
1. Generates predictions at historical timestamps using the CURRENT system
2. Evaluates those predictions 12 hours later
3. Compares with original system predictions
4. Reports on accuracy and PnL improvements

Author: J-Jarl
Created: December 20, 2025
"""

import json
import os
import argparse
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ccxt
import sys

# Import from trading_ai for prediction generation
sys.path.append(str(Path(__file__).parent))
from trading_ai import (
    BitcoinTradingAdvisor,
    MarketDataFetcher,
    OllamaClient,
    MarketData,
    SentimentAnalysis,
    TradingRecommendation
)
from evaluation_framework import TradingEvaluator
from data_cache import DataCache


class BacktestRunner:
    """Runs backtests by regenerating predictions on historical data"""

    def __init__(self, output_dir: str = "data/backtest_results"):
        """
        Initialize the backtest runner

        Args:
            output_dir: Directory to save backtest results
        """
        self.output_dir = Path(output_dir)
        self.predictions_dir = self.output_dir / "predictions"
        self.evaluations_dir = self.output_dir / "evaluations"
        self.original_predictions_dir = Path("data/analysis_results")

        # Create output directories
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange for historical data
        self.exchange = ccxt.coinbase()

        # Initialize data cache for consistent backtesting
        self.data_cache = DataCache()
        print("📦 Data cache initialized")

        # Initialize evaluator
        self.evaluator = TradingEvaluator(results_dir=str(self.predictions_dir))

        print(f"Backtest output directory: {self.output_dir}")
        print(f"Original predictions: {self.original_predictions_dir}")

    def get_prediction_times(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Generate list of prediction times (9 AM and 1 PM Mon-Fri)

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of datetime objects for prediction times
        """
        prediction_times = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_date:
            # Only weekdays (0=Monday, 6=Sunday)
            if current.weekday() < 5:
                # Add 9 AM prediction
                morning = current.replace(hour=9, minute=0, second=0, microsecond=0)
                if start_date <= morning <= end_date:
                    prediction_times.append(morning)

                # Add 1 PM prediction
                afternoon = current.replace(hour=13, minute=0, second=0, microsecond=0)
                if start_date <= afternoon <= end_date:
                    prediction_times.append(afternoon)

            current += timedelta(days=1)

        return sorted(prediction_times)

    def fetch_historical_ohlcv(self, timestamp: datetime, timeframe: str = '1h', limit: int = 200) -> List:
        """
        Fetch historical OHLCV data at specific timestamp using cache

        Args:
            timestamp: The datetime to fetch data for
            timeframe: Candlestick timeframe (1h, 4h, 1d, etc.)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV data
        """
        try:
            since = int(timestamp.timestamp() * 1000) - (limit * 3600 * 1000)
            # Use cached data for consistent backtesting
            ohlcv = self.data_cache.fetch_ohlcv('BTC/USDT', timeframe, since, limit)
            return ohlcv
        except Exception as e:
            print(f"Error fetching historical OHLCV: {e}")
            return []

    def fetch_historical_fear_greed(self, date: datetime) -> Optional[Dict]:
        """
        Fetch historical Fear & Greed Index for a specific date using cache

        Args:
            date: The date to fetch Fear & Greed for

        Returns:
            Dict with fear_greed_index and classification, or None if unavailable
        """
        try:
            # Use cached data for consistent backtesting
            index = self.data_cache.fetch_fear_greed(date)

            if index is None:
                return None

            # Classify the index value
            if index <= 25:
                classification = "Extreme Fear"
            elif index <= 45:
                classification = "Fear"
            elif index <= 55:
                classification = "Neutral"
            elif index <= 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"

            return {
                'fear_greed_index': index,
                'fear_greed_classification': classification
            }

        except Exception as e:
            print(f"Warning: Could not fetch historical Fear & Greed: {e}")
            return None

    def calculate_technical_indicators(self, ohlcv_data: List, current_price: float) -> Dict:
        """
        Calculate technical indicators from OHLCV data

        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume]
            current_price: Current price

        Returns:
            Dict with technical indicators
        """
        if not ohlcv_data or len(ohlcv_data) < 50:
            return {}

        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]

        indicators = {}

        # RSI 14
        if len(closes) >= 15:
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                gains.append(max(0, change))
                losses.append(max(0, -change))

            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                indicators['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                indicators['rsi_14'] = 100

        # Moving averages
        if len(closes) >= 5:
            indicators['sma_5'] = sum(closes[-5:]) / 5
        if len(closes) >= 10:
            indicators['sma_10'] = sum(closes[-10:]) / 10
        if len(closes) >= 20:
            indicators['sma_20'] = sum(closes[-20:]) / 20
        if len(closes) >= 50:
            indicators['sma_50'] = sum(closes[-50:]) / 50

        # EMA 9
        if len(closes) >= 9:
            multiplier = 2 / (9 + 1)
            ema = closes[0]
            for close in closes[1:]:
                ema = (close * multiplier) + (ema * (1 - multiplier))
            indicators['ema_9'] = ema

        # Price vs SMA
        if 'sma_20' in indicators:
            indicators['price_vs_sma20'] = "above" if current_price > indicators['sma_20'] else "below"
        if 'sma_50' in indicators:
            indicators['price_vs_sma50'] = "above" if current_price > indicators['sma_50'] else "below"

        # Volume analysis
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            indicators['volume_spike'] = indicators['volume_ratio'] > 2.0

        return indicators

    def _create_fallback_recommendation(self, current_price: float, market_data: MarketData,
                                         portfolio_value: float, risk_tolerance: str):
        """
        Create a fallback recommendation when AI generation fails.
        Uses basic technical analysis.
        """

        # Simple scoring based on available indicators
        score = 0
        if market_data.rsi_14:
            if market_data.rsi_14 < 30:
                score += 2  # Oversold
            elif market_data.rsi_14 > 70:
                score -= 2  # Overbought

        if market_data.price_vs_sma20 == "above":
            score += 1
        elif market_data.price_vs_sma20 == "below":
            score -= 1

        # Determine action
        if score >= 2:
            action = "buy"
            confidence = 60.0
        elif score <= -2:
            action = "sell"
            confidence = 60.0
        else:
            action = "hold"
            confidence = 50.0

        # Position sizing
        risk_pct = {'low': 2, 'medium': 5, 'high': 10}.get(risk_tolerance, 5)
        position_size_pct = risk_pct if action != "hold" else 0

        # Stop loss and take profit
        if action == "buy":
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.10
        elif action == "sell":
            stop_loss = current_price * 1.05
            take_profit = current_price * 0.90
        else:
            stop_loss = None
            take_profit = None

        return TradingRecommendation(
            action=action,
            confidence=confidence,
            entry_price=current_price if action != "hold" else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_percentage=position_size_pct,
            reasoning=f"Fallback technical analysis (AI unavailable). Score: {score}, RSI: {market_data.rsi_14}"
        )

    def generate_prediction_at_time(self, timestamp: datetime, portfolio_value: float = 10000.0,
                                     risk_tolerance: str = "medium") -> Optional[Dict]:
        """
        Generate a prediction at a specific historical timestamp using CURRENT system

        Args:
            timestamp: The time to generate prediction for
            portfolio_value: Portfolio value in USD
            risk_tolerance: Risk tolerance level

        Returns:
            Prediction dict in the same format as trading_ai.py output
        """
        print(f"\n  Generating prediction for {timestamp.strftime('%Y-%m-%d %H:%M')}")

        # Fetch historical market data
        print(f"    Fetching historical OHLCV data...")
        ohlcv_1h = self.fetch_historical_ohlcv(timestamp, '1h', 200)

        if not ohlcv_1h:
            print(f"    ❌ Could not fetch historical data for {timestamp}")
            return None

        # Get current price from OHLCV
        current_price = ohlcv_1h[-1][4]  # Close price of most recent candle
        volume_24h = sum([candle[5] for candle in ohlcv_1h[-24:]])
        price_change_24h = current_price - ohlcv_1h[-24][4] if len(ohlcv_1h) >= 24 else 0
        price_change_pct = (price_change_24h / ohlcv_1h[-24][4] * 100) if len(ohlcv_1h) >= 24 else 0

        print(f"    Price: ${current_price:,.2f}")

        # Fetch historical Fear & Greed
        print(f"    Fetching historical Fear & Greed Index...")
        fear_greed = self.fetch_historical_fear_greed(timestamp)

        # Calculate technical indicators using the REAL MarketDataFetcher
        print(f"    Calculating technical indicators...")
        fetcher = MarketDataFetcher()

        # Extract price data from OHLCV
        closes = [candle[4] for candle in ohlcv_1h]

        # Calculate individual indicators using real MarketDataFetcher methods
        rsi_14 = fetcher.calculate_rsi(closes, 14) if len(closes) >= 15 else None
        sma_5 = fetcher.calculate_sma(closes, 5) if len(closes) >= 5 else None
        sma_10 = fetcher.calculate_sma(closes, 10) if len(closes) >= 10 else None
        sma_20 = fetcher.calculate_sma(closes, 20) if len(closes) >= 20 else None
        sma_50 = fetcher.calculate_sma(closes, 50) if len(closes) >= 50 else None
        ema_9 = fetcher.calculate_ema(closes, 9) if len(closes) >= 9 else None

        # Price vs MA comparisons
        price_vs_sma20 = "above" if sma_20 and current_price > sma_20 else "below" if sma_20 else None
        price_vs_sma50 = "above" if sma_50 and current_price > sma_50 else "below" if sma_50 else None

        # Volume analysis (20-day average = 480 hourly candles)
        volume_avg_20d = sum([c[5] for c in ohlcv_1h[-480:]]) / 480 if len(ohlcv_1h) >= 480 else volume_24h
        volume_ratio = volume_24h / volume_avg_20d if volume_avg_20d > 0 else 1.0
        volume_spike = volume_ratio > 2.0

        # Detect Wyckoff patterns using REAL detection system
        print(f"    Detecting multi-timeframe Wyckoff patterns...")
        wyckoff_patterns = fetcher.detect_wyckoff_patterns(ohlcv_1h)

        # Analyze liquidity zones
        liquidity_analysis = fetcher.analyze_liquidity_zones(ohlcv_1h, current_price, volume_24h)

        # Build MarketData object (using the real dataclass)
        market_data = MarketData(
            current_price=current_price,
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            price_change_percentage_24h=price_change_pct,
            fear_greed_index=fear_greed['fear_greed_index'] if fear_greed else None,
            fear_greed_classification=fear_greed['fear_greed_classification'] if fear_greed else None,
            rsi_14=rsi_14,
            sma_5=sma_5,
            sma_10=sma_10,
            sma_20=sma_20,
            sma_50=sma_50,
            ema_9=ema_9,
            price_vs_sma20=price_vs_sma20,
            price_vs_sma50=price_vs_sma50,
            nearest_support=liquidity_analysis.get('nearest_support'),
            nearest_resistance=liquidity_analysis.get('nearest_resistance'),
            distance_to_support_pct=liquidity_analysis.get('distance_to_support_pct'),
            distance_to_resistance_pct=liquidity_analysis.get('distance_to_resistance_pct'),
            volume_ratio=volume_ratio,
            volume_spike=volume_spike,
            potential_liquidity_zone=liquidity_analysis.get('potential_liquidity_zone', False),
            wyckoff_patterns=wyckoff_patterns,
            exchange_available=True,
            exchange_name='Coinbase (Historical)',
            exchange_error=None,
            fear_greed_available=fear_greed is not None,
            fear_greed_error=None
        )

        # Create advisor instance (needed for pattern detection)
        print(f"    Initializing AI advisor...")
        ollama_client = OllamaClient(model="qwen3-coder:30b")
        advisor = BitcoinTradingAdvisor(ollama_client)

        # Detect reversal signals
        print(f"    Detecting reversal signals...")
        reversal_signal = advisor.detect_reversal_conditions(market_data)

        # Detect bounce patterns (HIGHEST PRIORITY)
        print(f"    Detecting bounce patterns (Wyckoff + MA + Volume)...")
        bounce_pattern = advisor.detect_bounce_patterns(market_data)
        if bounce_pattern["pattern"] != "none":
            print(f"      🚨 BOUNCE PATTERN: {bounce_pattern['pattern'].upper()} - {bounce_pattern['signal']}")

        # Store bounce pattern and reversal signal in market_data for validation
        market_data.bounce_pattern = bounce_pattern
        market_data.reversal_signal = reversal_signal

        # Create neutral sentiment (no historical news available)
        sentiment = SentimentAnalysis(
            sentiment='neutral',
            confidence=50.0,
            key_points=['No historical news available - technical analysis only'],
            reasoning='Backtest mode: Using technical indicators and Wyckoff patterns only'
        )

        # Use REAL BitcoinTradingAdvisor to generate recommendation
        print(f"    Generating recommendation using real AI system...")

        try:
            recommendation = advisor.generate_recommendation(
                sentiment=sentiment,
                current_price=current_price,
                portfolio_value=portfolio_value,
                risk_tolerance=risk_tolerance,
                market_data=market_data
            )
        except Exception as e:
            print(f"    ⚠️  AI generation failed: {e}")
            print(f"    Using fallback technical-only mode")
            # If AI fails, create a basic recommendation from market data
            recommendation = self._create_fallback_recommendation(
                current_price, market_data, portfolio_value, risk_tolerance
            )

        # Validate recommendation consistency (same as in run_analysis)
        try:
            validation = advisor.validate_recommendation_consistency(
                recommendation=recommendation,
                bounce_pattern=market_data.bounce_pattern if hasattr(market_data, 'bounce_pattern') else {'pattern': 'none'},
                wyckoff_patterns=market_data.wyckoff_patterns,
                reversal_signal=market_data.reversal_signal if hasattr(market_data, 'reversal_signal') else {'type': 'none'},
                market_data=market_data
            )

            # If validation found critical hallucinations, apply smart override
            if not validation['is_valid'] and len(validation['warnings']) > 0:
                print(f"    ⚠️  Recommendation validation warnings:")
                for warning in validation['warnings']:
                    print(f"        - {warning}")

                critical_hallucination = any('hallucination' in w.lower() for w in validation['warnings'])

                if critical_hallucination:
                    print(f"    🚫 CRITICAL HALLUCINATION - Applying pattern-based override")

                    # Store original recommendation
                    original_action = recommendation.action

                    # SMART OVERRIDE: Use actual detected patterns in priority order
                    override_action = None
                    override_reasoning = ""
                    override_confidence = 65.0  # Medium confidence for overrides

                    bounce_pattern = market_data.bounce_pattern if hasattr(market_data, 'bounce_pattern') else {'pattern': 'none'}
                    reversal_signal = market_data.reversal_signal if hasattr(market_data, 'reversal_signal') else {'type': 'none'}

                    # PRIORITY 1: Bounce Pattern (Highest)
                    if bounce_pattern.get('pattern') != 'none':
                        override_action = bounce_pattern['signal'].lower()
                        override_reasoning = f"Override based on {bounce_pattern['pattern']}: {bounce_pattern['reasoning']}"
                        override_confidence = 75.0 if bounce_pattern['strength'] == 'strong' else 65.0
                        print(f"      → Using BOUNCE PATTERN: {override_action.upper()}")

                    # PRIORITY 2: Reversal Signal
                    elif reversal_signal.get('type') != 'none':
                        if 'bullish' in reversal_signal['type'] or 'oversold' in reversal_signal['type']:
                            override_action = 'buy'
                            override_reasoning = f"Override based on reversal: {reversal_signal['type']}"
                            override_confidence = 70.0
                            print(f"      → Using REVERSAL SIGNAL: BUY")
                        elif 'bearish' in reversal_signal['type'] or 'overbought' in reversal_signal['type']:
                            override_action = 'sell'
                            override_reasoning = f"Override based on reversal: {reversal_signal['type']}"
                            override_confidence = 70.0
                            print(f"      → Using REVERSAL SIGNAL: SELL")

                    # PRIORITY 3: Short-term Wyckoff
                    elif market_data.wyckoff_patterns.get('short_term'):
                        pattern = market_data.wyckoff_patterns['short_term']
                        if pattern['pattern'] in ['accumulation', 'bear_trap']:
                            # Check RSI before BUY override
                            if market_data.rsi_14 and market_data.rsi_14 > 60:
                                print(f"      → SHORT-TERM WYCKOFF suggests BUY but RSI {market_data.rsi_14:.1f} is overbought - SKIPPING")
                                override_action = None  # Skip this override
                            else:
                                override_action = 'buy'
                                override_reasoning = f"Override based on short-term Wyckoff {pattern['pattern']}"
                                override_confidence = 70.0
                                print(f"      → Using SHORT-TERM WYCKOFF: BUY")
                        elif pattern['pattern'] in ['distribution', 'bull_trap']:
                            # Check RSI before SELL override
                            if market_data.rsi_14 and market_data.rsi_14 < 40:
                                print(f"      → SHORT-TERM WYCKOFF suggests SELL but RSI {market_data.rsi_14:.1f} is oversold - SKIPPING")
                                override_action = None  # Skip this override
                            else:
                                override_action = 'sell'
                                override_reasoning = f"Override based on short-term Wyckoff {pattern['pattern']}"
                                override_confidence = 70.0
                                print(f"      → Using SHORT-TERM WYCKOFF: SELL")

                    # PRIORITY 4: Medium-term Wyckoff
                    elif market_data.wyckoff_patterns.get('medium_term'):
                        pattern = market_data.wyckoff_patterns['medium_term']
                        if pattern['pattern'] in ['accumulation', 'bear_trap']:
                            # Check RSI before BUY override
                            if market_data.rsi_14 and market_data.rsi_14 > 60:
                                print(f"      → MEDIUM-TERM WYCKOFF suggests BUY but RSI {market_data.rsi_14:.1f} is overbought - SKIPPING")
                                override_action = None  # Skip this override
                            else:
                                override_action = 'buy'
                                override_reasoning = f"Override based on medium-term Wyckoff {pattern['pattern']}"
                                override_confidence = 65.0
                                print(f"      → Using MEDIUM-TERM WYCKOFF: BUY")
                        elif pattern['pattern'] in ['distribution', 'bull_trap']:
                            # Check RSI before SELL override
                            if market_data.rsi_14 and market_data.rsi_14 < 40:
                                print(f"      → MEDIUM-TERM WYCKOFF suggests SELL but RSI {market_data.rsi_14:.1f} is oversold - SKIPPING")
                                override_action = None  # Skip this override
                            else:
                                override_action = 'sell'
                                override_reasoning = f"Override based on medium-term Wyckoff {pattern['pattern']}"
                                override_confidence = 65.0
                                print(f"      → Using MEDIUM-TERM WYCKOFF: SELL")

                    # DEFAULT: HOLD if no patterns detected OR all overrides blocked by RSI
                    if override_action is None:
                        override_action = 'hold'
                        override_reasoning = "No clear patterns detected after hallucination OR RSI safety blocked all signals - defaulting to HOLD"
                        override_confidence = 0.0
                        print(f"      → No safe patterns available: HOLD")

                    # Calculate stop-loss and take-profit for override
                    override_stop = None
                    override_target = None
                    if override_action == 'buy':
                        override_stop = current_price * 0.98  # 2% stop-loss
                        override_target = current_price * 1.04  # 4% take-profit
                    elif override_action == 'sell':
                        override_stop = current_price * 1.02  # 2% stop-loss
                        override_target = current_price * 0.96  # 4% take-profit

                    # Apply override
                    recommendation = TradingRecommendation(
                        action=override_action,
                        confidence=override_confidence,
                        entry_price=current_price if override_action != 'hold' else None,
                        stop_loss=override_stop,
                        take_profit=override_target,
                        position_size_percentage=5.0 if override_action != 'hold' else 0.0,
                        reasoning=f"OVERRIDDEN: AI hallucinated. Original: {original_action}. {override_reasoning}. Warnings: {validation['warnings']}"
                    )

                    validation['overridden'] = True
                    validation['original_action'] = original_action
                    validation['override_action'] = override_action

        except Exception as e:
            print(f"    ⚠️  Validation failed: {e}")
            validation = {'is_valid': True, 'warnings': [], 'error': str(e)}

        # Calculate hybrid stop-loss with R/R validation (same as live trading)
        print(f"    Calculating hybrid stop-loss and R/R validation...")
        hybrid_stop = None
        blocked_recommendation = None

        if recommendation.action.lower() in ["buy", "sell"]:
            # Check if this is a bounce pattern trade
            is_bounce_pattern = bounce_pattern.get('pattern') != 'none'

            # For bounce patterns, adjust targets BEFORE R/R validation
            print(f"      DEBUG: is_bounce_pattern={is_bounce_pattern}, take_profit={recommendation.take_profit}")
            if is_bounce_pattern:
                if not recommendation.take_profit:
                    print(f"      WARNING: Bounce pattern detected but no take_profit set! Using resistance as target.")
                    # Set initial target to nearest resistance for BUY, support for SELL
                    if recommendation.action.lower() == 'buy':
                        recommendation = TradingRecommendation(
                            action=recommendation.action,
                            confidence=recommendation.confidence,
                            entry_price=recommendation.entry_price or current_price,
                            stop_loss=recommendation.stop_loss,
                            take_profit=market_data.nearest_resistance,
                            position_size_percentage=recommendation.position_size_percentage,
                            reasoning=recommendation.reasoning
                        )
                    elif recommendation.action.lower() == 'sell':
                        recommendation = TradingRecommendation(
                            action=recommendation.action,
                            confidence=recommendation.confidence,
                            entry_price=recommendation.entry_price or current_price,
                            stop_loss=recommendation.stop_loss,
                            take_profit=market_data.nearest_support,
                            position_size_percentage=recommendation.position_size_percentage,
                            reasoning=recommendation.reasoning
                        )

                if recommendation.take_profit:
                    # Pre-calculate ATR for target adjustment
                    atr = fetcher.calculate_atr(ohlcv_1h, period=14, timeframe='1h')

                    if atr and atr > 0:
                        if recommendation.action.lower() == 'buy':
                            # Target should be at least 2x ATR above entry for momentum move
                            atr_target = current_price + (atr * 2)
                            # Or 1.5% gain (whichever is larger)
                            percent_target = current_price * 1.015

                            # Use the larger of: current target, ATR-based, or percentage-based
                            better_target = max(
                                recommendation.take_profit,
                                atr_target,
                                percent_target
                            )

                            if better_target > recommendation.take_profit:
                                print(f"      Bounce pattern detected - adjusting target from ${recommendation.take_profit:,.2f} to ${better_target:,.2f}")
                                recommendation = TradingRecommendation(
                                    action=recommendation.action,
                                    confidence=recommendation.confidence,
                                    entry_price=recommendation.entry_price,
                                    stop_loss=recommendation.stop_loss,
                                    take_profit=better_target,
                                    position_size_percentage=recommendation.position_size_percentage,
                                    reasoning=recommendation.reasoning
                                )

                        elif recommendation.action.lower() == 'sell':
                            # For SELL bounce patterns, target 2x ATR below or 1.5% drop
                            atr_target = current_price - (atr * 2)
                            percent_target = current_price * 0.985

                            better_target = min(
                                recommendation.take_profit,
                                atr_target,
                                percent_target
                            )

                            if better_target < recommendation.take_profit:
                                print(f"      Bounce pattern detected - adjusting target from ${recommendation.take_profit:,.2f} to ${better_target:,.2f}")
                                recommendation = TradingRecommendation(
                                    action=recommendation.action,
                                    confidence=recommendation.confidence,
                                    entry_price=recommendation.entry_price,
                                    stop_loss=recommendation.stop_loss,
                                    take_profit=better_target,
                                    position_size_percentage=recommendation.position_size_percentage,
                                    reasoning=recommendation.reasoning
                                )

            hybrid_stop = fetcher.calculate_hybrid_stop_loss(
                ohlcv_data=ohlcv_1h,
                entry_price=recommendation.entry_price or current_price,
                target_price=recommendation.take_profit,
                action=recommendation.action,
                portfolio_value=portfolio_value,
                timeframe='1h',
                is_bounce_pattern=is_bounce_pattern,
                market_data=market_data
            )

            if hybrid_stop and hybrid_stop.get("atr_value"):
                print(f"      - ATR Stop Loss: ${hybrid_stop['atr_stop_loss']:,.2f}")
                if hybrid_stop.get('risk_reward_ratio'):
                    rr_status = "GOOD" if hybrid_stop['meets_rr_minimum'] else "LOW"
                    print(f"      - Risk/Reward: 1:{hybrid_stop['risk_reward_ratio']} ({rr_status})")

                # Safety filter: Block trades with poor R/R ratio (same as live trading)
                if hybrid_stop.get('risk_reward_ratio') and not hybrid_stop['meets_rr_minimum']:
                    # CAPITULATION OVERRIDE: Lower R/R requirements for Wyckoff bottoms
                    is_capitulation = False
                    if is_bounce_pattern and recommendation.action.lower() == 'buy':
                        # Check for capitulation conditions (Wyckoff bottom forming)
                        rsi = market_data.rsi_14 if market_data.rsi_14 else 50
                        fear_greed = market_data.fear_greed_index if market_data.fear_greed_index else 50

                        # Check for volume decline (confirms accumulation maturing)
                        volume_declining = False
                        if hasattr(market_data, 'wyckoff_patterns'):
                            short_term = market_data.wyckoff_patterns.get('short_term', {})
                            if short_term and short_term.get('pattern') == 'accumulation':
                                volume_declining = short_term.get('volume_declining', False)

                        # Capitulation = accumulation + low RSI + extreme fear + VOLUME DECLINING
                        if rsi < 50 and fear_greed < 30 and volume_declining:
                            is_capitulation = True
                            print(f"      🎯 CAPITULATION DETECTED: RSI {rsi:.1f} + F&G {fear_greed} + Volume Declining")
                            print(f"      → Mature Wyckoff bottom - lowering R/R requirement to 0.75:1")
                        elif rsi < 50 and fear_greed < 30:
                            # Conditions met but volume still high - too early
                            print(f"      ⏳ Early accumulation detected but volume still high - using standard R/R")

                    # Determine minimum R/R based on trade type and capitulation
                    if is_capitulation:
                        min_rr = 0.75  # Relaxed for capitulation bottoms
                        min_rr_label = "0.75:1"
                        trade_type = "capitulation bounce"
                    elif is_bounce_pattern:
                        min_rr = 1.0
                        min_rr_label = "1:1"
                        trade_type = "bounce pattern"
                    else:
                        min_rr = 2.0
                        min_rr_label = "1:2"
                        trade_type = "regular"

                    # Only block if below the adjusted minimum
                    if hybrid_stop['risk_reward_ratio'] < min_rr:
                        print(f"      ⚠️  TRADE BLOCKED: R/R {hybrid_stop['risk_reward_ratio']:.2f}:1 < {min_rr_label} minimum ({trade_type})")

                        # Save original recommendation
                        blocked_recommendation = {
                            "action": recommendation.action,
                            "confidence": recommendation.confidence,
                            "entry_price": recommendation.entry_price,
                            "stop_loss": recommendation.stop_loss,
                            "take_profit": recommendation.take_profit,
                            "reasoning": recommendation.reasoning
                        }

                        # Calculate required target for minimum R/R
                        entry = recommendation.entry_price or current_price
                        atr_stop = hybrid_stop['atr_stop_loss']
                        risk_amount = abs(entry - atr_stop)
                        required_profit = risk_amount * min_rr

                        if recommendation.action.lower() == "buy":
                            required_target = entry + required_profit
                        else:  # sell
                            required_target = entry - required_profit

                        # Check for DISTRIBUTION DOMINANCE when bounce blocked
                        distribution_override = False
                        if is_bounce_pattern and recommendation.action.lower() == 'buy':
                            # Count distribution patterns across timeframes
                            distribution_count = 0
                            accumulation_count = 0

                            for timeframe in ['short_term', 'medium_term', 'long_term']:
                                pattern = market_data.wyckoff_patterns.get(timeframe)
                                if pattern:
                                    if pattern.get('pattern') in ['distribution', 'bull_trap']:
                                        distribution_count += 1
                                    elif pattern.get('pattern') in ['accumulation', 'bear_trap']:
                                        accumulation_count += 1

                            # If 2+ timeframes show distribution, override to SELL
                            if distribution_count >= 2 and accumulation_count < distribution_count:
                                # RSI SAFETY CHECK - Don't SELL into oversold conditions
                                rsi = market_data.rsi_14 if market_data.rsi_14 else 50
                                fear_greed = market_data.fear_greed_index if market_data.fear_greed_index else 50

                                # Block SELL if oversold (RSI < 35) or extreme fear + low RSI
                                if rsi < 35:
                                    print(f"      ⚠️  Distribution dominance detected but RSI {rsi:.1f} too oversold - SKIPPING")
                                    print(f"      → High bounce risk - keeping HOLD")
                                    distribution_override = False
                                elif rsi < 40 and fear_greed < 25:
                                    print(f"      ⚠️  Distribution dominance detected but RSI {rsi:.1f} + F&G {fear_greed} (extreme oversold + fear) - SKIPPING")
                                    print(f"      → Extreme fear zone with low RSI - keeping HOLD")
                                    distribution_override = False
                                else:
                                    print(f"      🎯 DISTRIBUTION DOMINANCE: {distribution_count} timeframes bearish")
                                    print(f"      → Overriding blocked {recommendation.action.upper()} to SELL")
                                    distribution_override = True

                                # Only create SELL if RSI safety allows it
                                if distribution_override:
                                    # Create SELL recommendation with momentum targets
                                    # Target: 2x ATR below or 1.5% drop (whichever is larger)
                                    atr_target = current_price - (atr * 2) if atr and atr > 0 else current_price * 0.985
                                    percent_target = current_price * 0.985  # 1.5% drop
                                    target_price = min(atr_target, percent_target)

                                    recommendation = TradingRecommendation(
                                        action='sell',
                                        confidence=65.0,
                                        entry_price=current_price,
                                        stop_loss=market_data.nearest_resistance,  # Stop at resistance
                                        take_profit=target_price,  # Momentum target
                                        position_size_percentage=5.0,
                                        reasoning=f"Distribution dominance override: {distribution_count} timeframes show distribution. Using momentum target (2x ATR or 1.5%). Original {recommendation.action} blocked by R/R {hybrid_stop['risk_reward_ratio']:.2f}:1"
                                    )

                                    print(f"      - Distribution SELL target: ${target_price:,.2f} (momentum-based)")

                                    # Recalculate R/R for SELL
                                    hybrid_stop = fetcher.calculate_hybrid_stop_loss(
                                        ohlcv_data=ohlcv_1h,
                                        entry_price=current_price,
                                        target_price=target_price,
                                        action='sell',
                                        portfolio_value=portfolio_value,
                                        timeframe='1h',
                                        is_bounce_pattern=False,
                                        market_data=market_data
                                    )

                                print(f"      - New R/R for SELL: {hybrid_stop.get('risk_reward_ratio', 0):.2f}:1")

                                distribution_override = True

                        # Override recommendation to HOLD (only if not overridden by distribution dominance)
                        if not distribution_override:
                            new_reasoning = (
                                f"Trade blocked - Risk/Reward ratio of {hybrid_stop['risk_reward_ratio']}:1 is below the {min_rr_label} minimum threshold for {trade_type} trades. "
                                f"The ATR-based stop loss at ${atr_stop:,.2f} would require a target of ${required_target:,.2f} to meet minimum R/R. "
                                f"Current target of ${recommendation.take_profit:,.2f} is insufficient. Waiting for better setup with improved risk/reward."
                            )

                            recommendation = TradingRecommendation(
                                action="hold",
                                confidence=recommendation.confidence * 0.5,
                                entry_price=None,
                                stop_loss=None,
                                take_profit=None,
                                position_size_percentage=0.0,
                                reasoning=new_reasoning
                            )

                            print(f"      - Recommendation changed to: HOLD")
                    else:
                        # R/R meets the adjusted minimum (possibly due to capitulation override)
                        if is_capitulation:
                            print(f"      ✅ CAPITULATION OVERRIDE: Trade approved with R/R {hybrid_stop['risk_reward_ratio']:.2f}:1")

        # Build prediction in same format as trading_ai.py
        prediction = {
            'timestamp': timestamp.isoformat(),
            'analysis_mode': 'BACKTEST',
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'risk_tolerance': risk_tolerance,
            'articles_analyzed': 0,
            'data_sources': {
                'exchange_available': True,
                'exchange_name': 'Coinbase (Historical)',
                'fear_greed_available': fear_greed is not None,
                'news_sources_count': 0
            },
            'market_data': {
                'current_price': market_data.current_price,
                'volume_24h': market_data.volume_24h,
                'price_change_24h': market_data.price_change_24h,
                'fear_greed_index': market_data.fear_greed_index,
                'fear_greed_classification': market_data.fear_greed_classification,
                'rsi_14': market_data.rsi_14,
                'sma_5': market_data.sma_5,
                'sma_10': market_data.sma_10,
                'sma_20': market_data.sma_20,
                'sma_50': market_data.sma_50,
                'ema_9': market_data.ema_9,
                'price_vs_sma20': market_data.price_vs_sma20,
                'price_vs_sma50': market_data.price_vs_sma50,
                'nearest_support': market_data.nearest_support,
                'nearest_resistance': market_data.nearest_resistance,
                'distance_to_support_pct': market_data.distance_to_support_pct,
                'distance_to_resistance_pct': market_data.distance_to_resistance_pct,
                'volume_ratio': market_data.volume_ratio,
                'volume_spike': market_data.volume_spike,
                'potential_liquidity_zone': market_data.potential_liquidity_zone,
                'wyckoff_patterns': market_data.wyckoff_patterns,
                'bounce_pattern': market_data.bounce_pattern if hasattr(market_data, 'bounce_pattern') else {'pattern': 'none'},
                'reversal_signal': market_data.reversal_signal if hasattr(market_data, 'reversal_signal') else {'type': 'none'}
            },
            'sentiment_analysis': {
                'sentiment': sentiment.sentiment,
                'confidence': sentiment.confidence,
                'key_points': sentiment.key_points,
                'reasoning': sentiment.reasoning
            },
            'recommendation': {
                'action': recommendation.action,
                'confidence': recommendation.confidence,
                'entry_price': recommendation.entry_price,
                'stop_loss': recommendation.stop_loss,
                'take_profit': recommendation.take_profit,
                'position_size_pct': recommendation.position_size_percentage,
                'reasoning': recommendation.reasoning
            },
            'confidence_level': 'HIGH' if recommendation.confidence >= 75 else 'MEDIUM' if recommendation.confidence >= 50 else 'LOW',
            'position_sizing': {
                'btc_amount': (portfolio_value * recommendation.position_size_percentage / 100) / current_price if current_price > 0 else 0,
                'usd_amount': portfolio_value * recommendation.position_size_percentage / 100,
                'stop_loss': recommendation.stop_loss,
                'take_profit': recommendation.take_profit
            },
            'consistency_validation': validation,
            'hybrid_stop': hybrid_stop,
            'blocked_recommendation': blocked_recommendation
        }

        print(f"    ✅ Prediction generated: {recommendation.action.upper()} ({recommendation.confidence:.0f}% confidence)")

        return prediction

    def save_prediction(self, prediction: Dict, timestamp: datetime):
        """Save prediction to backtest predictions directory"""
        filename = f"btc_backtest_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.predictions_dir / filename

        with open(filepath, 'w') as f:
            json.dump(prediction, f, indent=2)

        print(f"    Saved to: {filepath.name}")

    def evaluate_prediction_12h_later(self, prediction: Dict, timestamp: datetime) -> Optional[Dict]:
        """
        Evaluate a prediction 12 hours after it was made

        Args:
            prediction: The prediction to evaluate
            timestamp: When the prediction was made

        Returns:
            Evaluation dict
        """
        print(f"\n  Evaluating prediction from {timestamp.strftime('%Y-%m-%d %H:%M')} (pattern-based window)")

        evaluation = self.evaluator.evaluate_prediction(prediction, hours_forward=12)

        if evaluation:
            print(f"    Result: {'✅ Correct' if evaluation['prediction_correct'] else '❌ Incorrect'}")
            print(f"    Price change: {evaluation['percent_change']:+.2f}%")
            print(f"    PnL: ${evaluation['hypothetical_pnl']:+,.2f}")

            # Save evaluation
            filename = f"eval_backtest_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.evaluations_dir / filename

            with open(filepath, 'w') as f:
                json.dump(evaluation, f, indent=2)

            print(f"    Saved to: {filepath.name}")

        return evaluation

    def load_original_prediction(self, timestamp: datetime) -> Optional[Dict]:
        """
        Load original prediction from data/analysis_results

        Args:
            timestamp: Prediction timestamp

        Returns:
            Original prediction dict or None
        """
        filename = f"btc_analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.original_predictions_dir / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load original prediction {filename}: {e}")

        return None

    def compare_predictions(self, original: Dict, new: Dict, original_eval: Optional[Dict],
                           new_eval: Optional[Dict]) -> Dict:
        """
        Compare original and new predictions

        Args:
            original: Original prediction
            new: New prediction from backtest
            original_eval: Original evaluation (if exists)
            new_eval: New evaluation

        Returns:
            Comparison dict
        """
        comparison = {
            'timestamp': new.get('timestamp'),
            'recommendation_changed': False,
            'wyckoff_changed': False,
            'original_recommendation': None,
            'new_recommendation': None,
            'original_wyckoff': None,
            'new_wyckoff': None,
            'original_correct': None,
            'new_correct': None,
            'original_pnl': None,
            'new_pnl': None,
            'improvement': None
        }

        # Extract recommendations
        orig_rec = original.get('recommendation', {}).get('action', 'UNKNOWN') if original else None
        new_rec = new.get('recommendation', {}).get('action', 'UNKNOWN')

        comparison['original_recommendation'] = orig_rec
        comparison['new_recommendation'] = new_rec
        comparison['recommendation_changed'] = (orig_rec != new_rec) if orig_rec else False

        # Extract Wyckoff patterns
        orig_wyckoff = original.get('market_data', {}).get('wyckoff_pattern') if original else None
        new_wyckoff = new.get('market_data', {}).get('wyckoff_patterns')

        comparison['original_wyckoff'] = orig_wyckoff  # Single pattern
        comparison['new_wyckoff'] = new_wyckoff  # Multi-timeframe patterns
        comparison['wyckoff_changed'] = (orig_wyckoff is not None) != (new_wyckoff is not None)

        # Compare evaluations
        if original_eval:
            comparison['original_correct'] = original_eval.get('prediction_correct')
            comparison['original_pnl'] = original_eval.get('hypothetical_pnl')

        if new_eval:
            comparison['new_correct'] = new_eval.get('prediction_correct')
            comparison['new_pnl'] = new_eval.get('hypothetical_pnl')

        # Determine improvement
        if comparison['original_correct'] is not None and comparison['new_correct'] is not None:
            if comparison['new_correct'] and not comparison['original_correct']:
                comparison['improvement'] = 'better'
            elif not comparison['new_correct'] and comparison['original_correct']:
                comparison['improvement'] = 'worse'
            elif comparison['new_pnl'] is not None and comparison['original_pnl'] is not None:
                if comparison['new_pnl'] > comparison['original_pnl']:
                    comparison['improvement'] = 'better_pnl'
                elif comparison['new_pnl'] < comparison['original_pnl']:
                    comparison['improvement'] = 'worse_pnl'
                else:
                    comparison['improvement'] = 'same'
            else:
                comparison['improvement'] = 'same'

        return comparison

    def run_backtest(self, start_date: datetime, end_date: datetime,
                     portfolio_value: float = 10000.0, risk_tolerance: str = "medium") -> Dict:
        """
        Run complete backtest for date range

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            portfolio_value: Portfolio value in USD
            risk_tolerance: Risk tolerance level

        Returns:
            Backtest results dict
        """
        print("\n" + "=" * 80)
        print("BITCOIN TRADING AI - BACKTESTING FRAMEWORK")
        print("=" * 80)
        print(f"\nDate Range: {start_date.date()} to {end_date.date()}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Risk Tolerance: {risk_tolerance}")
        print()

        # Get prediction times
        prediction_times = self.get_prediction_times(start_date, end_date)
        print(f"Prediction times (9 AM & 1 PM Mon-Fri): {len(prediction_times)}")
        print()

        predictions = []
        evaluations = []
        comparisons = []

        # Phase 1: Generate predictions
        print("=" * 80)
        print("PHASE 1: GENERATING PREDICTIONS")
        print("=" * 80)

        for i, pred_time in enumerate(prediction_times, 1):
            print(f"\n[{i}/{len(prediction_times)}] Processing {pred_time.strftime('%Y-%m-%d %H:%M')}")

            # Generate new prediction
            prediction = self.generate_prediction_at_time(pred_time, portfolio_value, risk_tolerance)
            if prediction:
                self.save_prediction(prediction, pred_time)
                predictions.append((pred_time, prediction))

        print(f"\n✅ Generated {len(predictions)} predictions")

        # Phase 2: Evaluate predictions (12 hours later)
        print("\n" + "=" * 80)
        print("PHASE 2: EVALUATING PREDICTIONS (12 hours later)")
        print("=" * 80)

        for i, (pred_time, prediction) in enumerate(predictions, 1):
            print(f"\n[{i}/{len(predictions)}] Evaluating prediction from {pred_time.strftime('%Y-%m-%d %H:%M')}")

            evaluation = self.evaluate_prediction_12h_later(prediction, pred_time)
            if evaluation:
                evaluations.append((pred_time, evaluation))

        print(f"\n✅ Evaluated {len(evaluations)} predictions")

        # Phase 3: Compare with original system
        print("\n" + "=" * 80)
        print("PHASE 3: COMPARING WITH ORIGINAL SYSTEM")
        print("=" * 80)

        for i, (pred_time, prediction) in enumerate(predictions, 1):
            print(f"\n[{i}/{len(predictions)}] Comparing {pred_time.strftime('%Y-%m-%d %H:%M')}")

            # Load original prediction
            original = self.load_original_prediction(pred_time)
            if original:
                print(f"    Found original prediction")
            else:
                print(f"    No original prediction found")

            # Find evaluations
            new_eval = next((e for t, e in evaluations if t == pred_time), None)

            # For original evaluation, we'd need to check if it exists
            # For simplicity, we'll evaluate the original prediction here too
            original_eval = None
            if original:
                evaluator_orig = TradingEvaluator(results_dir=str(self.original_predictions_dir))
                original_eval = evaluator_orig.evaluate_prediction(original, hours_forward=12)

            # Compare
            comparison = self.compare_predictions(original, prediction, original_eval, new_eval)
            comparisons.append(comparison)

            if comparison['recommendation_changed']:
                print(f"    ⚠️  Recommendation changed: {comparison['original_recommendation']} → {comparison['new_recommendation']}")
            else:
                print(f"    ✓ Recommendation same: {comparison['new_recommendation']}")

        print(f"\n✅ Compared {len(comparisons)} predictions")

        # Phase 4: Generate summary report
        print("\n" + "=" * 80)
        print("PHASE 4: GENERATING SUMMARY REPORT")
        print("=" * 80)

        summary = self.generate_summary_report(evaluations, comparisons)

        # Save all results
        self.save_backtest_results(predictions, evaluations, comparisons, summary, start_date, end_date)

        return summary

    def generate_summary_report(self, evaluations: List[Tuple[datetime, Dict]],
                                 comparisons: List[Dict]) -> Dict:
        """
        Generate summary report comparing old vs new system

        Args:
            evaluations: List of (timestamp, evaluation) tuples
            comparisons: List of comparison dicts

        Returns:
            Summary report dict
        """
        report = {
            'total_predictions': len(comparisons),
            'old_system': {
                'accuracy': 0,
                'correct': 0,
                'incorrect': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            },
            'new_system': {
                'accuracy': 0,
                'correct': 0,
                'incorrect': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            },
            'changes': {
                'recommendations_changed': 0,
                'wyckoff_patterns_differ': 0
            },
            'improvements': {
                'better': 0,
                'worse': 0,
                'same': 0
            },
            'breakdown_by_timeframe': {}
        }

        # Calculate metrics
        old_correct = 0
        old_total_pnl = 0
        old_count = 0
        new_correct = 0
        new_total_pnl = 0
        new_count = 0

        for comp in comparisons:
            # Old system
            if comp['original_correct'] is not None:
                old_count += 1
                if comp['original_correct']:
                    old_correct += 1
                if comp['original_pnl'] is not None:
                    old_total_pnl += comp['original_pnl']

            # New system
            if comp['new_correct'] is not None:
                new_count += 1
                if comp['new_correct']:
                    new_correct += 1
                if comp['new_pnl'] is not None:
                    new_total_pnl += comp['new_pnl']

            # Changes
            if comp['recommendation_changed']:
                report['changes']['recommendations_changed'] += 1
            if comp['wyckoff_changed']:
                report['changes']['wyckoff_patterns_differ'] += 1

            # Improvements
            if comp['improvement'] in ['better', 'better_pnl']:
                report['improvements']['better'] += 1
            elif comp['improvement'] in ['worse', 'worse_pnl']:
                report['improvements']['worse'] += 1
            else:
                report['improvements']['same'] += 1

        # Update report
        if old_count > 0:
            report['old_system']['accuracy'] = round((old_correct / old_count) * 100, 2)
            report['old_system']['correct'] = old_correct
            report['old_system']['incorrect'] = old_count - old_correct
            report['old_system']['total_pnl'] = round(old_total_pnl, 2)
            report['old_system']['avg_pnl'] = round(old_total_pnl / old_count, 2)

        if new_count > 0:
            report['new_system']['accuracy'] = round((new_correct / new_count) * 100, 2)
            report['new_system']['correct'] = new_correct
            report['new_system']['incorrect'] = new_count - new_correct
            report['new_system']['total_pnl'] = round(new_total_pnl, 2)
            report['new_system']['avg_pnl'] = round(new_total_pnl / new_count, 2)

        return report

    def save_backtest_results(self, predictions: List, evaluations: List, comparisons: List,
                              summary: Dict, start_date: datetime, end_date: datetime):
        """Save complete backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}_{timestamp}.json"
        filepath = self.output_dir / filename

        results = {
            'generated_at': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': summary,
            'comparisons': comparisons
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Backtest results saved to: {filepath}")

    def print_summary_report(self, summary: Dict):
        """Print formatted summary report"""
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY REPORT")
        print("=" * 80)

        print(f"\nTotal Predictions: {summary['total_predictions']}")

        print("\n" + "-" * 80)
        print("OLD SYSTEM vs NEW SYSTEM COMPARISON")
        print("-" * 80)

        print(f"\n{'Metric':<25} {'Old System':>20} {'New System':>20} {'Change':>15}")
        print("-" * 80)

        old = summary['old_system']
        new = summary['new_system']

        # Accuracy
        acc_change = new['accuracy'] - old['accuracy']
        print(f"{'Accuracy':<25} {old['accuracy']:>19.2f}% {new['accuracy']:>19.2f}% {acc_change:>+14.2f}%")

        # Correct predictions
        correct_change = new['correct'] - old['correct']
        print(f"{'Correct Predictions':<25} {old['correct']:>20} {new['correct']:>20} {correct_change:>+15}")

        # Total PnL
        pnl_change = new['total_pnl'] - old['total_pnl']
        print(f"{'Total PnL':<25} ${old['total_pnl']:>18,.2f} ${new['total_pnl']:>18,.2f} ${pnl_change:>+13,.2f}")

        # Avg PnL
        avg_pnl_change = new['avg_pnl'] - old['avg_pnl']
        print(f"{'Average PnL':<25} ${old['avg_pnl']:>18,.2f} ${new['avg_pnl']:>18,.2f} ${avg_pnl_change:>+13,.2f}")

        print("\n" + "-" * 80)
        print("CHANGES & IMPROVEMENTS")
        print("-" * 80)

        print(f"\nRecommendations Changed: {summary['changes']['recommendations_changed']}")
        print(f"Wyckoff Patterns Differ: {summary['changes']['wyckoff_patterns_differ']}")

        print(f"\nImprovements:")
        print(f"  Better: {summary['improvements']['better']}")
        print(f"  Worse: {summary['improvements']['worse']}")
        print(f"  Same: {summary['improvements']['same']}")

        # Overall verdict
        print("\n" + "=" * 80)
        if new['accuracy'] > old['accuracy']:
            print("✅ NEW SYSTEM IS MORE ACCURATE")
        elif new['accuracy'] < old['accuracy']:
            print("⚠️  OLD SYSTEM WAS MORE ACCURATE")
        else:
            print("➖ SAME ACCURACY")

        if new['total_pnl'] > old['total_pnl']:
            print("✅ NEW SYSTEM HAS BETTER PNL")
        elif new['total_pnl'] < old['total_pnl']:
            print("⚠️  OLD SYSTEM HAD BETTER PNL")
        else:
            print("➖ SAME PNL")

        print("=" * 80 + "\n")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Backtest Bitcoin Trading AI by re-running predictions on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest Week 3 (Dec 16-20, 2025)
  python scripts/backtest.py --start-date 2025-12-16 --end-date 2025-12-20

  # Backtest with custom output directory
  python scripts/backtest.py --start-date 2025-12-16 --end-date 2025-12-20 --output-dir data/my_backtest
        """
    )

    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD, inclusive)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD, inclusive)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtest_results',
        help='Output directory for backtest results (default: data/backtest_results/)'
    )

    parser.add_argument(
        '--portfolio-value',
        type=float,
        default=10000.0,
        help='Portfolio value in USD (default: 10000.0)'
    )

    parser.add_argument(
        '--risk-tolerance',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Risk tolerance level (default: medium)'
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        # Set end_date to end of day (23:59:59) to include predictions on that day
        end_date = end_date.replace(hour=23, minute=59, second=59)
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("   Expected format: YYYY-MM-DD")
        return

    # Validate date range
    if start_date > end_date:
        print(f"❌ Start date ({args.start_date}) cannot be after end date ({args.end_date})")
        return

    # Initialize backtest runner
    runner = BacktestRunner(output_dir=args.output_dir)

    # Run backtest
    try:
        summary = runner.run_backtest(
            start_date=start_date,
            end_date=end_date,
            portfolio_value=args.portfolio_value,
            risk_tolerance=args.risk_tolerance
        )

        # Print summary
        runner.print_summary_report(summary)

        print("✅ Backtest complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
