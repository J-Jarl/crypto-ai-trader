"""
Layer 1 - Regime Detection System
Structure-based regime classification independent of short-term events
"""
from typing import List, Tuple, Dict
import numpy as np


def find_swing_highs(ohlcv_data: List[List], window: int = 5) -> List[Tuple[int, float]]:
    """
    Find swing high points (local peaks)

    Args:
        ohlcv_data: OHLCV candles
        window: Bars to left/right for comparison

    Returns:
        List of (index, price) tuples for swing highs
    """
    swing_highs = []

    for i in range(window, len(ohlcv_data) - window):
        high = ohlcv_data[i][2]  # High price
        is_swing = True

        # Check left side
        for j in range(i - window, i):
            if ohlcv_data[j][2] >= high:
                is_swing = False
                break

        # Check right side
        if is_swing:
            for j in range(i + 1, i + window + 1):
                if ohlcv_data[j][2] >= high:
                    is_swing = False
                    break

        if is_swing:
            swing_highs.append((i, high))

    return swing_highs


def find_swing_lows(ohlcv_data: List[List], window: int = 5) -> List[Tuple[int, float]]:
    """
    Find swing low points (local valleys)

    Args:
        ohlcv_data: OHLCV candles
        window: Bars to left/right for comparison

    Returns:
        List of (index, price) tuples for swing lows
    """
    swing_lows = []

    for i in range(window, len(ohlcv_data) - window):
        low = ohlcv_data[i][3]  # Low price
        is_swing = True

        # Check left side
        for j in range(i - window, i):
            if ohlcv_data[j][3] <= low:
                is_swing = False
                break

        # Check right side
        if is_swing:
            for j in range(i + 1, i + window + 1):
                if ohlcv_data[j][3] <= low:
                    is_swing = False
                    break

        if is_swing:
            swing_lows.append((i, low))

    return swing_lows


def count_higher_highs(swing_highs: List[Tuple[int, float]]) -> int:
    """Count how many swing highs are higher than previous"""
    if len(swing_highs) < 2:
        return 0

    count = 0
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i-1][1]:
            count += 1

    return count


def count_higher_lows(swing_lows: List[Tuple[int, float]]) -> int:
    """Count how many swing lows are higher than previous"""
    if len(swing_lows) < 2:
        return 0

    count = 0
    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] > swing_lows[i-1][1]:
            count += 1

    return count


def count_lower_highs(swing_highs: List[Tuple[int, float]]) -> int:
    """Count how many swing highs are lower than previous"""
    if len(swing_highs) < 2:
        return 0

    count = 0
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] < swing_highs[i-1][1]:
            count += 1

    return count


def count_lower_lows(swing_lows: List[Tuple[int, float]]) -> int:
    """Count how many swing lows are lower than previous"""
    if len(swing_lows) < 2:
        return 0

    count = 0
    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] < swing_lows[i-1][1]:
            count += 1

    return count


def detect_regime_layer1(ohlcv_data: List[List], lookback_days: int = 3) -> Tuple[str, int, Dict]:
    """
    Layer 1: Independent regime detection based on market structure

    NOT influenced by short-term spikes - pure structural analysis

    Args:
        ohlcv_data: OHLCV candles
        lookback_days: Days to analyze (default 3)

    Returns:
        Tuple of (regime, confidence, details)
        - regime: 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'CHOPPY'
        - confidence: 0-100
        - details: Dict with analysis breakdown
    """

    if len(ohlcv_data) < 72:  # Need at least 3 days
        return 'RANGING', 50, {'reason': 'insufficient_data'}

    lookback_hours = lookback_days * 24
    recent_data = ohlcv_data[-lookback_hours:]

    # 1. Calculate net move over lookback period
    start_price = recent_data[0][4]  # Close of first candle
    current_price = recent_data[-1][4]  # Current close
    net_move_pct = ((current_price - start_price) / start_price) * 100

    # 2. Find swing points
    swing_highs = find_swing_highs(recent_data, window=5)
    swing_lows = find_swing_lows(recent_data, window=5)

    # 3. Analyze structure
    higher_highs = count_higher_highs(swing_highs)
    higher_lows = count_higher_lows(swing_lows)
    lower_highs = count_lower_highs(swing_highs)
    lower_lows = count_lower_lows(swing_lows)

    # 4. Calculate volatility context
    closes = [c[4] for c in recent_data]
    price_std = np.std(closes)
    price_mean = np.mean(closes)
    volatility_pct = (price_std / price_mean) * 100

    print(f"\n  🏗️  LAYER 1 - REGIME DETECTION:")
    print(f"     Net move ({lookback_days}d): {net_move_pct:+.1f}%")
    print(f"     Structure: HH={higher_highs}, HL={higher_lows}, LH={lower_highs}, LL={lower_lows}")
    print(f"     Volatility: {volatility_pct:.1f}%")

    # 5. CLASSIFY REGIME

    # TRENDING_UP: Strong upward move + flexible HH/HL structure
    # Multiple valid structure patterns for uptrend:
    # - Strong structure: HH≥2 AND HL≥2
    # - Acceptable structure A: HH≥1 AND HL≥3 (many higher lows)
    # - Acceptable structure B: HH≥3 AND HL≥1 (many higher highs)

    if net_move_pct >= 6.95 and (  # 6.95 to handle floating point precision
        (higher_highs >= 2 and higher_lows >= 2) or  # Strong structure
        (higher_highs >= 1 and higher_lows >= 3) or  # Acceptable: more lows
        (higher_highs >= 3 and higher_lows >= 1)     # Acceptable: more highs
    ):
        confidence = min(95, int(60 + (net_move_pct * 2) + (higher_highs * 5) + (higher_lows * 5)))
        structure_type = 'strong' if (higher_highs >= 2 and higher_lows >= 2) else 'acceptable'
        print(f"     ✅ TRENDING_UP condition MATCHED!")
        print(f"     ✅ REGIME: TRENDING_UP (confidence: {confidence}%, structure: {structure_type})")
        return 'TRENDING_UP', confidence, {
            'net_move': net_move_pct,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'structure_type': structure_type,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

    # TRENDING_DOWN: Strong downward move + flexible LH/LL structure
    # Multiple valid structure patterns for downtrend:
    # - Strong structure: LH≥2 AND LL≥2
    # - Acceptable structure A: LH≥1 AND LL≥3 (many lower lows)
    # - Acceptable structure B: LH≥3 AND LL≥1 (many lower highs)

    if net_move_pct <= -6.95 and (  # -6.95 to handle floating point precision
        (lower_highs >= 2 and lower_lows >= 2) or  # Strong structure
        (lower_highs >= 1 and lower_lows >= 3) or  # Acceptable: more lows
        (lower_highs >= 3 and lower_lows >= 1)     # Acceptable: more highs
    ):
        confidence = min(95, int(60 + (abs(net_move_pct) * 2) + (lower_highs * 5) + (lower_lows * 5)))
        structure_type = 'strong' if (lower_highs >= 2 and lower_lows >= 2) else 'acceptable'
        print(f"     ✅ TRENDING_DOWN condition MATCHED!")
        print(f"     ✅ REGIME: TRENDING_DOWN (confidence: {confidence}%, structure: {structure_type})")
        return 'TRENDING_DOWN', confidence, {
            'net_move': net_move_pct,
            'lower_highs': lower_highs,
            'lower_lows': lower_lows,
            'structure_type': structure_type,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

    # RANGING: Small net move, balanced swings
    if abs(net_move_pct) < 3.0 and volatility_pct < 2.5:
        confidence = 70
        print(f"     ✅ REGIME: RANGING (confidence: {confidence}%)")
        return 'RANGING', confidence, {
            'net_move': net_move_pct,
            'volatility': volatility_pct,
            'reason': 'low_volatility_range'
        }

    # CHOPPY: Medium move or mixed structure (fallback)
    confidence = 60
    reason = 'mixed_structure'

    if abs(net_move_pct) >= 3.0 and abs(net_move_pct) < 7.0:
        reason = 'medium_move_unclear_structure'
    elif higher_highs >= 1 and lower_lows >= 1:
        reason = 'conflicting_swings'

    print(f"     ✅ REGIME: CHOPPY (confidence: {confidence}%, reason: {reason})")
    return 'CHOPPY', confidence, {
        'net_move': net_move_pct,
        'higher_highs': higher_highs,
        'higher_lows': higher_lows,
        'lower_highs': lower_highs,
        'lower_lows': lower_lows,
        'reason': reason
    }


def check_trend_structure_intact(regime: str, ohlcv_data: List[List],
                                  regime_details: Dict) -> Tuple[bool, str]:
    """
    Check if trend structure is still intact (no structural break)

    Used to detect when TRENDING regime should end

    Args:
        regime: Current regime
        ohlcv_data: OHLCV data
        regime_details: Details from regime detection

    Returns:
        Tuple of (intact, reason)
    """

    if regime not in ['TRENDING_UP', 'TRENDING_DOWN']:
        return True, 'not_trending'

    # Get last 24 hours of data
    recent_24h = ohlcv_data[-24:]
    if len(recent_24h) < 24:
        return True, 'insufficient_data'

    # Find recent swing points
    recent_swing_highs = find_swing_highs(recent_24h, window=3)
    recent_swing_lows = find_swing_lows(recent_24h, window=3)

    if regime == 'TRENDING_UP':
        # Check for lower low (trend break)
        if len(recent_swing_lows) >= 2:
            last_low = recent_swing_lows[-1][1]
            prev_low = recent_swing_lows[-2][1]

            if last_low < prev_low:
                return False, 'lower_low_detected'

        return True, 'structure_intact'

    else:  # TRENDING_DOWN
        # Check for higher high (trend break)
        if len(recent_swing_highs) >= 2:
            last_high = recent_swing_highs[-1][1]
            prev_high = recent_swing_highs[-2][1]

            if last_high > prev_high:
                return False, 'higher_high_detected'

        return True, 'structure_intact'


# Test function
if __name__ == '__main__':
    print("Layer 1 - Regime Detection System")
    print("\nFunctions available:")
    print("  ✓ detect_regime_layer1() - Main regime classifier")
    print("  ✓ check_trend_structure_intact() - Trend break detection")
    print("  ✓ find_swing_highs/lows() - Swing point identification")
    print("\nRegimes: TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY")
    print("Independent of short-term spikes and sweeps!")
