"""
Custom Regime Detection System - Phase 3
Analyzes market structure without standard indicators
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def analyze_volume_profile(ohlcv_data: List[List], lookback_hours: int = 720) -> Dict:
    """
    Build volume profile to find high volume nodes (consolidation areas)

    Args:
        ohlcv_data: List of OHLCV candles [timestamp, open, high, low, close, volume]
        lookback_hours: How far back to analyze (default 30 days)

    Returns:
        Dict with volume node analysis
    """
    if len(ohlcv_data) < lookback_hours:
        lookback_hours = len(ohlcv_data)

    recent_candles = ohlcv_data[-lookback_hours:]
    current_price = recent_candles[-1][4]  # Current close

    # Build volume profile - aggregate volume by price level
    # Round prices to nearest $100 for grouping
    volume_by_price = {}

    for candle in recent_candles:
        close = candle[4]
        volume = candle[5]
        price_level = round(close / 100) * 100  # Round to nearest $100

        if price_level not in volume_by_price:
            volume_by_price[price_level] = 0
        volume_by_price[price_level] += volume

    # Find top 3 high volume nodes
    sorted_nodes = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    high_volume_nodes = [node[0] for node in sorted_nodes[:3]]

    # Calculate distance from nearest node
    if high_volume_nodes:
        distances = [abs(current_price - node) for node in high_volume_nodes]
        nearest_distance = min(distances)
        nearest_node = high_volume_nodes[distances.index(nearest_distance)]
        distance_pct = (nearest_distance / current_price) * 100
    else:
        nearest_node = current_price
        distance_pct = 0

    # Classify position
    at_node = distance_pct < 1.0  # Within 1% of high volume node
    breaking_away = distance_pct > 3.0  # More than 3% from all nodes

    return {
        'at_node': at_node,
        'breaking_away': breaking_away,
        'nearest_node': nearest_node,
        'distance_pct': distance_pct,
        'high_volume_nodes': high_volume_nodes
    }


def detect_order_flow_divergence(ohlcv_data: List[List], lookback: int = 10) -> Dict:
    """
    Detect price vs volume divergence (exhaustion signals)

    Args:
        ohlcv_data: List of OHLCV candles
        lookback: Number of candles to analyze

    Returns:
        Dict with divergence signals
    """
    if len(ohlcv_data) < lookback:
        return {'bullish': False, 'bearish': False, 'aligned': True}

    recent_candles = ohlcv_data[-lookback:]

    # Extract prices and volumes
    closes = [c[4] for c in recent_candles]
    volumes = [c[5] for c in recent_candles]

    # Calculate trends (first half vs second half)
    mid_point = lookback // 2

    first_half_price_avg = sum(closes[:mid_point]) / mid_point
    second_half_price_avg = sum(closes[mid_point:]) / (lookback - mid_point)

    first_half_volume_avg = sum(volumes[:mid_point]) / mid_point
    second_half_volume_avg = sum(volumes[mid_point:]) / (lookback - mid_point)

    # Determine trends
    price_trend = 'up' if second_half_price_avg > first_half_price_avg else 'down'
    volume_trend = 'up' if second_half_volume_avg > first_half_volume_avg else 'down'

    # Detect divergences
    # Bullish divergence: Price down but volume also down (selling exhaustion)
    bullish_divergence = (price_trend == 'down' and volume_trend == 'down')

    # Bearish divergence: Price up but volume down (buying exhaustion)
    bearish_divergence = (price_trend == 'up' and volume_trend == 'down')

    # Aligned: Price and volume moving same direction
    aligned = (price_trend == volume_trend)

    return {
        'bullish': bullish_divergence,
        'bearish': bearish_divergence,
        'aligned': aligned,
        'price_trend': price_trend,
        'volume_trend': volume_trend
    }


def find_swing_points(ohlcv_data: List[List], lookback: int, point_type: str = 'high') -> List[float]:
    """
    Find swing highs or lows (potential liquidity zones)

    Args:
        ohlcv_data: List of OHLCV candles
        lookback: How far back to search
        point_type: 'high' or 'low'

    Returns:
        List of swing point prices
    """
    if len(ohlcv_data) < lookback:
        lookback = len(ohlcv_data)

    recent_candles = ohlcv_data[-lookback:]
    swing_points = []

    # Look for local extremes (pivot points)
    window = 5  # Look 5 candles left and right

    for i in range(window, len(recent_candles) - window):
        if point_type == 'high':
            # Check if this is a local high
            center_high = recent_candles[i][2]
            is_swing = True

            # Check left side
            for j in range(i - window, i):
                if recent_candles[j][2] >= center_high:
                    is_swing = False
                    break

            # Check right side
            if is_swing:
                for j in range(i + 1, i + window + 1):
                    if recent_candles[j][2] >= center_high:
                        is_swing = False
                        break

            if is_swing:
                swing_points.append(center_high)

        else:  # 'low'
            # Check if this is a local low
            center_low = recent_candles[i][3]
            is_swing = True

            # Check left side
            for j in range(i - window, i):
                if recent_candles[j][3] <= center_low:
                    is_swing = False
                    break

            # Check right side
            if is_swing:
                for j in range(i + 1, i + window + 1):
                    if recent_candles[j][3] <= center_low:
                        is_swing = False
                        break

            if is_swing:
                swing_points.append(center_low)

    return swing_points


def check_liquidity_zones(ohlcv_data: List[List], lookback_hours: int = 720) -> Dict:
    """
    Check proximity to major support/resistance levels (liquidity zones)

    Args:
        ohlcv_data: List of OHLCV candles
        lookback_hours: How far back to look for swing points (default 30 days)

    Returns:
        Dict with liquidity zone analysis
    """
    if len(ohlcv_data) < 20:
        return {
            'near_resistance': False,
            'near_support': False,
            'in_open_space': True,
            'nearest_resistance': None,
            'nearest_support': None
        }

    current_price = ohlcv_data[-1][4]

    # Find swing highs and lows
    swing_highs = find_swing_points(ohlcv_data, lookback_hours, 'high')
    swing_lows = find_swing_points(ohlcv_data, lookback_hours, 'low')

    # Find nearest resistance (swing high above current price)
    resistances_above = [h for h in swing_highs if h > current_price]
    nearest_resistance = min(resistances_above) if resistances_above else None

    # Find nearest support (swing low below current price)
    supports_below = [l for l in swing_lows if l < current_price]
    nearest_support = max(supports_below) if supports_below else None

    # Calculate distances
    if nearest_resistance:
        distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
    else:
        distance_to_resistance = 100  # Far away

    if nearest_support:
        distance_to_support = ((current_price - nearest_support) / current_price) * 100
    else:
        distance_to_support = 100  # Far away

    # Classify proximity
    near_resistance = distance_to_resistance < 2.0  # Within 2%
    near_support = distance_to_support < 2.0  # Within 2%

    # Open space check - DIRECTIONAL based on trend
    # In uptrends: Need space above (resistance)
    # In downtrends: Need space below (support)
    # Neutral: Need space both ways (original logic)
    in_open_space = distance_to_resistance > 5.0 and distance_to_support > 5.0  # Default
    in_open_space_bullish = distance_to_resistance > 3.0  # Uptrend: space to run up
    in_open_space_bearish = distance_to_support > 3.0  # Downtrend: space to run down

    return {
        'near_resistance': near_resistance,
        'near_support': near_support,
        'in_open_space': in_open_space,
        'in_open_space_bullish': in_open_space_bullish,
        'in_open_space_bearish': in_open_space_bearish,
        'nearest_resistance': nearest_resistance,
        'nearest_support': nearest_support,
        'distance_to_resistance_pct': distance_to_resistance,
        'distance_to_support_pct': distance_to_support
    }


def classify_market_regime_custom(ohlcv_data: List[List], current_price: float,
                                   atr_14: float, liquidity_sweep_result: Tuple = None) -> Tuple[str, str, float]:
    """
    Classify market regime using custom market structure analysis

    NO STANDARD INDICATORS - Pure market structure

    Args:
        ohlcv_data: List of OHLCV candles
        current_price: Current price
        atr_14: ATR(14) for volatility reference
        liquidity_sweep_result: Result from existing liquidity sweep detector (classification, score)

    Returns:
        Tuple of (regime, reason, confidence)
        - regime: 'SPIKE', 'TRENDING', 'CHOPPY', or 'LOW_VOL'
        - reason: Why this regime was selected
        - confidence: 0-100 score
    """

    if len(ohlcv_data) < 50:
        return 'LOW_VOL', 'insufficient_data', 0

    # 1. ANALYZE MARKET STRUCTURE
    volume_profile = analyze_volume_profile(ohlcv_data, lookback_hours=720)
    order_flow = detect_order_flow_divergence(ohlcv_data, lookback=10)
    liquidity = check_liquidity_zones(ohlcv_data, lookback_hours=720)

    # 2. CALCULATE VOLATILITY (for context)
    atr_pct = (atr_14 / current_price) * 100

    # Get recent price action
    recent_10 = ohlcv_data[-10:]
    recent_high = max([c[2] for c in recent_10])
    recent_low = min([c[3] for c in recent_10])
    recent_range_pct = ((recent_high - recent_low) / recent_low) * 100

    print(f"\n  🔬 MARKET STRUCTURE ANALYSIS:")
    print(f"     Volume Profile: at_node={volume_profile['at_node']}, breaking_away={volume_profile['breaking_away']}, distance={volume_profile['distance_pct']:.1f}%")
    print(f"     Order Flow: bullish_div={order_flow['bullish']}, bearish_div={order_flow['bearish']}, aligned={order_flow['aligned']}")
    print(f"     Liquidity: near_support={liquidity['near_support']}, near_resistance={liquidity['near_resistance']}, open_space={liquidity['in_open_space']}")
    print(f"     Recent range: {recent_range_pct:.1f}%, ATR%: {atr_pct:.2f}%")

    # 3. REGIME CLASSIFICATION (PRIORITY ORDER)

    # ========================================
    # PRIORITY 1: SPIKE REGIME
    # ========================================

    # A. Liquidity sweep detected by existing system
    if liquidity_sweep_result and liquidity_sweep_result[0] is not None:
        sweep_classification = liquidity_sweep_result[0]
        print(f"     🚨 SPIKE DETECTED: Liquidity sweep system triggered ({sweep_classification})")
        return 'SPIKE', f'liquidity_sweep_{sweep_classification.lower()}', 90

    # B. Near liquidity zone + divergence (rejection setup)
    if (liquidity['near_support'] or liquidity['near_resistance']) and \
       (order_flow['bearish'] or order_flow['bullish']):

        if liquidity['near_support'] and order_flow['bullish']:
            print(f"     🚨 SPIKE DETECTED: Support zone + bullish divergence")
            return 'SPIKE', 'support_zone_rejection', 80

        if liquidity['near_resistance'] and order_flow['bearish']:
            print(f"     🚨 SPIKE DETECTED: Resistance zone + bearish divergence")
            return 'SPIKE', 'resistance_zone_rejection', 80

    # C. Extreme divergence anywhere (potential reversal)
    if order_flow['bearish'] and recent_range_pct > 4.0:
        print(f"     🚨 SPIKE DETECTED: Bearish divergence + high volatility")
        return 'SPIKE', 'bearish_divergence_spike', 75

    if order_flow['bullish'] and recent_range_pct > 4.0:
        print(f"     🚨 SPIKE DETECTED: Bullish divergence + high volatility")
        return 'SPIKE', 'bullish_divergence_spike', 75

    # ========================================
    # PRIORITY 2: TRENDING REGIME
    # ========================================

    # Breaking away from volume nodes + aligned flow + directional space
    # This is institutional breakout behavior!
    if volume_profile['breaking_away'] and order_flow['aligned']:

        # Check directional open space
        has_space = False
        if order_flow['price_trend'] == 'up' and liquidity.get('in_open_space_bullish', False):
            has_space = True
            direction = 'bullish'
        elif order_flow['price_trend'] == 'down' and liquidity.get('in_open_space_bearish', False):
            has_space = True
            direction = 'bearish'

        if has_space:
            print(f"     📈 TRENDING DETECTED: {direction.upper()} breakout from consolidation")
            return 'TRENDING', f'{direction}_breakout', 85

    # Breaking away + aligned (even without open space)
    if volume_profile['breaking_away'] and order_flow['aligned']:
        direction = 'bullish' if order_flow['price_trend'] == 'up' else 'bearish'
        print(f"     📈 TRENDING DETECTED: {direction.upper()} momentum")
        return 'TRENDING', f'{direction}_momentum', 70

    # ========================================
    # PRIORITY 3: CHOPPY REGIME
    # ========================================

    # At volume node (consolidation)
    if volume_profile['at_node']:
        print(f"     📊 CHOPPY DETECTED: At volume node (consolidation)")
        return 'CHOPPY', 'consolidation_at_node', 75

    # Divergence present (uncertainty)
    if order_flow['bearish'] or order_flow['bullish']:
        div_type = 'bearish' if order_flow['bearish'] else 'bullish'
        print(f"     📊 CHOPPY DETECTED: {div_type.upper()} divergence (uncertainty)")
        return 'CHOPPY', f'{div_type}_divergence', 65

    # Near liquidity zone but no spike yet (waiting)
    if liquidity['near_support'] or liquidity['near_resistance']:
        zone = 'support' if liquidity['near_support'] else 'resistance'
        print(f"     📊 CHOPPY DETECTED: Near {zone} zone (waiting)")
        return 'CHOPPY', f'near_{zone}', 60

    # ========================================
    # PRIORITY 4: LOW_VOL REGIME
    # ========================================

    # Low volatility (nothing clear)
    if atr_pct < 1.5 and recent_range_pct < 2.0:
        print(f"     😴 LOW_VOL DETECTED: Low volatility (ATR {atr_pct:.2f}%)")
        return 'LOW_VOL', 'low_volatility', 70

    # ========================================
    # DEFAULT: CHOPPY
    # ========================================
    print(f"     📊 CHOPPY (DEFAULT): No clear pattern")
    return 'CHOPPY', 'default', 50


def get_regime_trading_mode(regime: str, reason: str) -> str:
    """
    Map regime to trading system

    Args:
        regime: Detected regime
        reason: Reason for regime

    Returns:
        Trading mode: 'liquidity_sweep', 'trend_following', 'volume_blocking', or 'hold'
    """

    if regime == 'SPIKE':
        return 'liquidity_sweep'  # Use existing liquidity sweep system

    elif regime == 'TRENDING':
        return 'trend_following'  # Use new trend system (to be built)

    elif regime == 'CHOPPY':
        return 'volume_blocking'  # Use existing volume blocking system

    else:  # LOW_VOL
        return 'hold'  # Mostly hold, wait for setup


# Test/debug function
if __name__ == '__main__':
    print("Regime Detection System - Phase 3")
    print("\nHelper Functions:")
    print("  [OK] analyze_volume_profile()")
    print("  [OK] detect_order_flow_divergence()")
    print("  [OK] find_swing_points()")
    print("  [OK] check_liquidity_zones()")
    print("\nMain Classifier:")
    print("  [OK] classify_market_regime_custom()")
    print("  [OK] get_regime_trading_mode()")
    print("\nReady to integrate!")
