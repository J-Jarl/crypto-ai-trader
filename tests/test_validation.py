#!/usr/bin/env python3
"""Test script to validate the recommendation validation logic"""

import sys
sys.path.insert(0, 'scripts')

from trading_ai import TradingRecommendation, MarketData, BitcoinTradingAdvisor, OllamaClient

def test_validation():
    """Test various validation scenarios"""

    # Create advisor instance
    ollama = OllamaClient()
    advisor = BitcoinTradingAdvisor(ollama)

    # Test market data
    market_data = MarketData(
        current_price=95000.0,
        volume_24h=1000000,
        price_change_24h=1000,
        price_change_percentage_24h=1.05,
        fear_greed_index=50,
        fear_greed_classification="Neutral",
        rsi_14=55.0,
        sma_5=94000.0,
        sma_10=93000.0,
        sma_20=92000.0,
        sma_50=90000.0,
        ema_9=94500.0,
        price_vs_sma20="above",
        price_vs_sma50="above",
        exchange_available=True,
        exchange_name="Binance",
        fear_greed_available=True
    )

    print("=" * 80)
    print("VALIDATION TEST SUITE")
    print("=" * 80)
    print()

    # Test 1: Entry price is null
    print("Test 1: Entry price is null (should be fixed to current price)")
    rec1 = TradingRecommendation(
        action="buy",
        confidence=75.0,
        entry_price=None,
        stop_loss=90000.0,
        take_profit=100000.0,
        position_size_percentage=5.0,
        reasoning="Test recommendation"
    )

    fixed_rec1, info1 = advisor.validate_and_fix_recommendation(rec1, market_data, atr_hourly=2000.0)
    print(f"  Original entry: {rec1.entry_price}")
    print(f"  Fixed entry: {fixed_rec1.entry_price}")
    print(f"  Fixes applied: {info1['fixes_applied']}")
    print(f"  PASS" if fixed_rec1.entry_price == 95000.0 else "  FAIL")
    print()

    # Test 2: Unrealistic target (SELL with -55% crash)
    print("Test 2: Unrealistic SELL target ($40k from $90k entry = -55% crash)")
    rec2 = TradingRecommendation(
        action="sell",
        confidence=80.0,
        entry_price=90000.0,
        stop_loss=95000.0,
        take_profit=40000.0,  # -55% crash - unrealistic!
        position_size_percentage=5.0,
        reasoning="Test recommendation"
    )

    fixed_rec2, info2 = advisor.validate_and_fix_recommendation(rec2, market_data, atr_hourly=2000.0)
    print(f"  Original target: ${rec2.take_profit:,.2f}")
    print(f"  Fixed target: ${fixed_rec2.take_profit:,.2f}")
    print(f"  Expected ATR-based: ${90000.0 - (4 * 2000.0):,.2f}")
    print(f"  Fixes applied: {info2['fixes_applied']}")
    print(f"  Target overridden: {info2['target_overridden']}")
    print(f"  [OK] PASS" if info2['target_overridden'] else "  [FAIL] FAIL")
    print()

    # Test 3: Stop-loss in wrong direction (BUY with stop > entry)
    print("Test 3: Stop-loss in wrong direction (BUY with stop > entry)")
    rec3 = TradingRecommendation(
        action="buy",
        confidence=70.0,
        entry_price=95000.0,
        stop_loss=98000.0,  # Wrong direction! Should be below entry
        take_profit=100000.0,
        position_size_percentage=5.0,
        reasoning="Test recommendation"
    )

    fixed_rec3, info3 = advisor.validate_and_fix_recommendation(rec3, market_data, atr_hourly=2000.0)
    print(f"  Original stop: ${rec3.stop_loss:,.2f}")
    print(f"  Fixed stop: ${fixed_rec3.stop_loss:,.2f}")
    print(f"  Expected ATR-based: ${95000.0 - (2.5 * 2000.0):,.2f}")
    print(f"  Fixes applied: {info3['fixes_applied']}")
    print(f"  [OK] PASS" if 'buy_stop_fixed_direction' in info3['fixes_applied'] else "  [FAIL] FAIL")
    print()

    # Test 4: Stop-loss in wrong direction (SELL with stop < entry)
    print("Test 4: Stop-loss in wrong direction (SELL with stop < entry)")
    rec4 = TradingRecommendation(
        action="sell",
        confidence=70.0,
        entry_price=95000.0,
        stop_loss=92000.0,  # Wrong direction! Should be above entry
        take_profit=90000.0,
        position_size_percentage=5.0,
        reasoning="Test recommendation"
    )

    fixed_rec4, info4 = advisor.validate_and_fix_recommendation(rec4, market_data, atr_hourly=2000.0)
    print(f"  Original stop: ${rec4.stop_loss:,.2f}")
    print(f"  Fixed stop: ${fixed_rec4.stop_loss:,.2f}")
    print(f"  Expected ATR-based: ${95000.0 + (2.5 * 2000.0):,.2f}")
    print(f"  Fixes applied: {info4['fixes_applied']}")
    print(f"  [OK] PASS" if 'sell_stop_fixed_direction' in info4['fixes_applied'] else "  [FAIL] FAIL")
    print()

    # Test 5: HOLD action (should skip validation)
    print("Test 5: HOLD action (should skip validation)")
    rec5 = TradingRecommendation(
        action="hold",
        confidence=50.0,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        position_size_percentage=0.0,
        reasoning="Test recommendation"
    )

    fixed_rec5, info5 = advisor.validate_and_fix_recommendation(rec5, market_data, atr_hourly=2000.0)
    print(f"  Validation applied: {info5['validation_applied']}")
    print(f"  Fixes applied: {info5['fixes_applied']}")
    print(f"  [OK] PASS" if not info5['validation_applied'] else "  [FAIL] FAIL")
    print()

    # Test 6: Realistic recommendation (should pass without changes)
    print("Test 6: Realistic BUY recommendation (should pass validation)")
    rec6 = TradingRecommendation(
        action="buy",
        confidence=75.0,
        entry_price=95000.0,
        stop_loss=92000.0,
        take_profit=98000.0,  # ~3.1% gain - realistic
        position_size_percentage=5.0,
        reasoning="Test recommendation"
    )

    fixed_rec6, info6 = advisor.validate_and_fix_recommendation(rec6, market_data, atr_hourly=2000.0)
    print(f"  Entry: ${fixed_rec6.entry_price:,.2f}")
    print(f"  Stop: ${fixed_rec6.stop_loss:,.2f}")
    print(f"  Target: ${fixed_rec6.take_profit:,.2f}")
    print(f"  Validation applied: {info6['validation_applied']}")
    print(f"  Fixes applied: {info6['fixes_applied']}")
    print(f"  [OK] PASS" if len(info6['fixes_applied']) == 0 else "  [FAIL] FAIL")
    print()

    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_validation()
