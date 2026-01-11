"""
Trend Following System - Phase 3
Uses chandelier trailing stops for trend trades
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrendSignal:
    """Trend system recommendation"""
    recommendation: str  # BUY, SELL, or HOLD
    confidence: int  # 0-100
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    trend_direction: str  # 'bullish' or 'bearish'


def detect_trend_direction(ohlcv_data: List[List], ma_period: int = 50) -> Tuple[str, float]:
    """
    Detect trend direction using moving average and price structure

    Args:
        ohlcv_data: OHLCV candles
        ma_period: MA period for trend (default 50)

    Returns:
        Tuple of (direction, strength)
        - direction: 'bullish', 'bearish', or 'neutral'
        - strength: 0-100 confidence score
    """
    if len(ohlcv_data) < ma_period + 10:
        return 'neutral', 0

    # Calculate MA
    closes = [c[4] for c in ohlcv_data[-ma_period:]]
    ma = sum(closes) / len(closes)
    current_price = ohlcv_data[-1][4]

    # Check price position relative to MA
    price_above_ma = current_price > ma
    distance_from_ma_pct = abs((current_price - ma) / ma) * 100

    # Check recent structure (higher highs/lows or lower highs/lows)
    recent_10 = ohlcv_data[-10:]
    highs = [c[2] for c in recent_10]
    lows = [c[3] for c in recent_10]

    # Simple structure check: first half vs second half
    first_half_high = max(highs[:5])
    second_half_high = max(highs[5:])
    first_half_low = min(lows[:5])
    second_half_low = min(lows[5:])

    higher_highs = second_half_high > first_half_high
    higher_lows = second_half_low > first_half_low
    lower_highs = second_half_high < first_half_high
    lower_lows = second_half_low < first_half_low

    # Determine trend
    if price_above_ma and higher_highs and higher_lows:
        # Strong bullish trend
        strength = min(100, int(distance_from_ma_pct * 20) + 60)
        return 'bullish', strength

    elif price_above_ma and (higher_highs or higher_lows):
        # Moderate bullish trend
        strength = min(80, int(distance_from_ma_pct * 15) + 50)
        return 'bullish', strength

    elif not price_above_ma and lower_highs and lower_lows:
        # Strong bearish trend
        strength = min(100, int(distance_from_ma_pct * 20) + 60)
        return 'bearish', strength

    elif not price_above_ma and (lower_highs or lower_lows):
        # Moderate bearish trend
        strength = min(80, int(distance_from_ma_pct * 15) + 50)
        return 'bearish', strength

    else:
        # Neutral/choppy
        return 'neutral', 30


def calculate_chandelier_stop(ohlcv_data: List[List], atr_14: float,
                               direction: str, multiplier: float = 3.0) -> float:
    """
    Calculate chandelier trailing stop

    Args:
        ohlcv_data: OHLCV candles
        atr_14: ATR(14) value
        direction: 'bullish' or 'bearish'
        multiplier: ATR multiplier (default 3.0)

    Returns:
        Stop loss price
    """
    recent_10 = ohlcv_data[-10:]

    if direction == 'bullish':
        # For longs: Stop below recent low - (ATR * multiplier)
        recent_low = min([c[3] for c in recent_10])
        stop = recent_low - (atr_14 * multiplier)
    else:
        # For shorts: Stop above recent high + (ATR * multiplier)
        recent_high = max([c[2] for c in recent_10])
        stop = recent_high + (atr_14 * multiplier)

    return stop


def calculate_trend_target(entry_price: float, stop_loss: float,
                           direction: str, min_rr: float = 0.75) -> float:
    """
    Calculate take profit target based on risk/reward

    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: 'bullish' or 'bearish'
        min_rr: Minimum risk/reward ratio (default 0.75 for trends)

    Returns:
        Take profit price
    """
    risk = abs(entry_price - stop_loss)
    reward = risk * min_rr

    if direction == 'bullish':
        target = entry_price + reward
    else:
        target = entry_price - reward

    return target


def check_trend_entry_conditions(ohlcv_data: List[List], current_price: float,
                                  trend_direction: str, volume_profile: Dict,
                                  order_flow: Dict, liquidity: Dict) -> Tuple[bool, str]:
    """
    Check if conditions are right for trend entry

    Args:
        ohlcv_data: OHLCV candles
        current_price: Current price
        trend_direction: 'bullish' or 'bearish'
        volume_profile: Volume profile analysis
        order_flow: Order flow analysis
        liquidity: Liquidity zone analysis

    Returns:
        Tuple of (should_enter, reason)
    """
    # 1. Must be breaking away from volume nodes (institutional move)
    if not volume_profile['breaking_away']:
        return False, "not_breaking_away"

    # 2. Order flow should be aligned (confirmation)
    if not order_flow['aligned']:
        return False, "order_flow_not_aligned"

    # 3. Should have open space to run (directional check)
    if trend_direction == 'bullish':
        if not liquidity.get('in_open_space_bullish', False):
            return False, "insufficient_upside_space"
    elif trend_direction == 'bearish':
        if not liquidity.get('in_open_space_bearish', False):
            return False, "insufficient_downside_space"
    else:
        if not liquidity['in_open_space']:
            return False, "no_open_space"

    # 4. Recent momentum should match trend
    recent_5 = ohlcv_data[-5:]
    closes = [c[4] for c in recent_5]
    recent_momentum = 'up' if closes[-1] > closes[0] else 'down'

    if trend_direction == 'bullish' and recent_momentum != 'up':
        return False, "momentum_not_bullish"

    if trend_direction == 'bearish' and recent_momentum != 'down':
        return False, "momentum_not_bearish"

    # All conditions met!
    return True, "conditions_met"


def generate_trend_signal(ohlcv_data: List[List], current_price: float, atr_14: float,
                          volume_profile: Dict, order_flow: Dict,
                          liquidity: Dict) -> TrendSignal:
    """
    Generate trend following signal

    Args:
        ohlcv_data: OHLCV candles
        current_price: Current price
        atr_14: ATR(14) value
        volume_profile: Volume profile analysis
        order_flow: Order flow analysis
        liquidity: Liquidity zone analysis

    Returns:
        TrendSignal with recommendation
    """
    print(f"\n  📈 TREND SYSTEM ANALYSIS:")

    # 1. Detect trend direction
    trend_direction, trend_strength = detect_trend_direction(ohlcv_data)
    print(f"     Trend: {trend_direction.upper()} (strength: {trend_strength}%)")

    # 2. Check if trend is strong enough
    if trend_strength < 50:
        print(f"     ⚠️  Trend too weak ({trend_strength}%) - HOLD")
        return TrendSignal(
            recommendation='HOLD',
            confidence=30,
            reason='weak_trend',
            entry_price=current_price,
            stop_loss=current_price,
            take_profit=current_price,
            risk_reward=0,
            trend_direction=trend_direction
        )

    # 3. Check entry conditions
    should_enter, entry_reason = check_trend_entry_conditions(
        ohlcv_data, current_price, trend_direction,
        volume_profile, order_flow, liquidity
    )

    if not should_enter:
        print(f"     ⚠️  Entry blocked: {entry_reason} - HOLD")
        return TrendSignal(
            recommendation='HOLD',
            confidence=40,
            reason=entry_reason,
            entry_price=current_price,
            stop_loss=current_price,
            take_profit=current_price,
            risk_reward=0,
            trend_direction=trend_direction
        )

    # 4. Calculate chandelier stop
    stop_loss = calculate_chandelier_stop(ohlcv_data, atr_14, trend_direction)

    # 5. Calculate target (relaxed R/R for trends)
    take_profit = calculate_trend_target(current_price, stop_loss, trend_direction, min_rr=0.75)

    # 6. Calculate R/R
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    risk_reward = reward / risk if risk > 0 else 0

    # 7. Generate recommendation
    if trend_direction == 'bullish':
        recommendation = 'BUY'
        confidence = min(85, trend_strength)
        reason = 'bullish_trend_entry'
    else:
        recommendation = 'SELL'
        confidence = min(85, trend_strength)
        reason = 'bearish_trend_entry'

    print(f"     ✅ TREND ENTRY: {recommendation} (R/R: {risk_reward:.2f}:1)")
    print(f"     Entry: ${current_price:,.2f}")
    print(f"     Stop: ${stop_loss:,.2f} (Chandelier)")
    print(f"     Target: ${take_profit:,.2f}")

    return TrendSignal(
        recommendation=recommendation,
        confidence=confidence,
        reason=reason,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        trend_direction=trend_direction
    )


# Test function
if __name__ == '__main__':
    print("Trend Following System - Phase 3 Step 4")
    print("\nFunctions available:")
    print("  ✓ detect_trend_direction()")
    print("  ✓ calculate_chandelier_stop()")
    print("  ✓ calculate_trend_target()")
    print("  ✓ check_trend_entry_conditions()")
    print("  ✓ generate_trend_signal()")
    print("\nReady to integrate into trading systems!")
