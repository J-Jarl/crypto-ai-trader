"""
Layer 2 - Event Detection System
Detects local anomalies (sweeps, spikes, divergences) as events within regimes
"""
from typing import List, Tuple, Dict, Optional


def detect_liquidity_sweep_event(ohlcv_data: List[List], current_price: float,
                                  market_data, wyckoff_patterns: Dict) -> Optional[Dict]:
    """
    Detect liquidity sweep as an EVENT (not a regime)

    This is the same logic as before, but returns event data
    instead of overriding regime classification

    Args:
        ohlcv_data: OHLCV candles
        current_price: Current price
        market_data: Market indicators

    Returns:
        Dict with event details or None if no sweep
    """

    if len(ohlcv_data) < 73:
        return None

    # Use 72-hour lookback (same as before)
    recent_candles = ohlcv_data[-73:]

    # Find recent high/low
    recent_high = max([c[2] for c in recent_candles])
    recent_low = min([c[3] for c in recent_candles])

    # Calculate sweep percentages
    upward_spike_pct = ((recent_high - recent_low) / recent_low) * 100
    downward_sweep_pct = ((recent_high - recent_low) / recent_low) * 100

    # Check if we're in recovery zone
    recovery_zone_high = recent_high * 0.99
    recovery_zone_low = recent_low * 1.01

    # Upward spike detection
    if current_price >= recovery_zone_high and upward_spike_pct >= 2.5:
        # Analyze if it's a trap or breakout
        score = 0
        indicators = []

        # RSI check
        if market_data.rsi_14 > 70:
            score -= 1
            indicators.append('overbought_rsi')
        elif market_data.rsi_14 > 60 and market_data.rsi_14 <= 70:
            score += 1
            indicators.append('healthy_rsi')

        # Distribution/accumulation (passed in as parameter)
        # wyckoff_patterns already provided

        distribution_count = sum(1 for p in wyckoff_patterns.values()
                                if p and isinstance(p, dict) and 'distribution' in p.get('pattern', '').lower())
        accumulation_count = sum(1 for p in wyckoff_patterns.values()
                                 if p and isinstance(p, dict) and 'accumulation' in p.get('pattern', '').lower())

        if distribution_count >= 2:
            score -= 1
            indicators.append(f'distribution_{distribution_count}tf')
        if accumulation_count >= 2:
            score += 1
            indicators.append(f'accumulation_{accumulation_count}tf')

        # Volume analysis
        recent_volumes = [c[5] for c in recent_candles[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = recent_candles[-1][5]

        if current_volume > avg_volume * 1.5:
            score += 1
            indicators.append('volume_sustained')
        else:
            score -= 1
            indicators.append('momentum_slowing')

        # Classify
        if score <= -2:
            classification = 'TRAP'
            direction = 'upward'
        elif score >= 2:
            classification = 'BREAKOUT'
            direction = 'upward'
        else:
            classification = 'UNCERTAIN'
            direction = 'upward'

        return {
            'type': 'liquidity_sweep',
            'direction': direction,
            'magnitude': upward_spike_pct,
            'classification': classification,
            'score': score,
            'indicators': indicators,
            'recent_high': recent_high,
            'recent_low': recent_low
        }

    # Downward sweep detection
    elif recent_low * 0.95 <= current_price <= recent_low * 1.05 and downward_sweep_pct >= 2.5:
        # Analyze if it's a bottom trap or breakdown
        score = 0
        indicators = []

        # Volume spike
        recent_volumes = [c[5] for c in recent_candles[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = recent_candles[-1][5]

        if current_volume > avg_volume * 1.5:
            score += 1
            indicators.append('volume_spike_panic')

        # Sharp drop
        if downward_sweep_pct > 4.0:
            score += 1
            indicators.append(f'sharp_{downward_sweep_pct:.1f}%_drop')

        # RSI
        if market_data.rsi_14 < 30:
            score += 1
            indicators.append('rsi_oversold')
        elif market_data.rsi_14 >= 30 and market_data.rsi_14 < 50:
            score -= 1
            indicators.append('rsi_weak')

        # Distribution/accumulation (passed in as parameter)
        # wyckoff_patterns already provided

        distribution_count = sum(1 for p in wyckoff_patterns.values()
                                if p and isinstance(p, dict) and 'distribution' in p.get('pattern', '').lower())
        accumulation_count = sum(1 for p in wyckoff_patterns.values()
                                 if p and isinstance(p, dict) and 'accumulation' in p.get('pattern', '').lower())

        if distribution_count >= 2:
            score -= 1
            indicators.append(f'distribution_{distribution_count}tf')
        if accumulation_count >= 2:
            score += 1
            indicators.append(f'accumulation_{accumulation_count}tf')

        # Quick reversal check
        last_5_closes = [c[4] for c in recent_candles[-5:]]
        if last_5_closes[-1] > last_5_closes[0]:
            score += 1
            indicators.append('quick_reversal')
        else:
            score -= 1
            indicators.append('continuation_lower')

        # Classify
        if score >= 2:
            classification = 'BOTTOM_TRAP'
            direction = 'downward'
        elif score <= -2:
            classification = 'BREAKDOWN'
            direction = 'downward'
        else:
            classification = 'UNCERTAIN'
            direction = 'downward'

        # Check for late entry
        recovery_pct = ((current_price - recent_low) / recent_low) * 100
        late_entry = recovery_pct > 2.0

        if late_entry and classification == 'BOTTOM_TRAP' and distribution_count >= 2:
            # Override to breakdown if late + distribution
            classification = 'BREAKDOWN'
            score = -2
            indicators.append('late_entry_override')

        return {
            'type': 'liquidity_sweep',
            'direction': direction,
            'magnitude': downward_sweep_pct,
            'classification': classification,
            'score': score,
            'indicators': indicators,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'late_entry': late_entry
        }

    return None


def detect_volume_spike_event(ohlcv_data: List[List]) -> Optional[Dict]:
    """
    Detect unusual volume spike as an event

    Args:
        ohlcv_data: OHLCV candles

    Returns:
        Dict with event details or None
    """
    if len(ohlcv_data) < 20:
        return None

    recent_volumes = [c[5] for c in ohlcv_data[-20:]]
    avg_volume = sum(recent_volumes[:-1]) / (len(recent_volumes) - 1)
    current_volume = recent_volumes[-1]

    if current_volume > avg_volume * 2.0:
        spike_ratio = current_volume / avg_volume
        return {
            'type': 'volume_spike',
            'magnitude': spike_ratio,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }

    return None


def detect_divergence_event(ohlcv_data: List[List], market_data) -> Optional[Dict]:
    """
    Detect price/volume divergence as an event

    Args:
        ohlcv_data: OHLCV candles
        market_data: Market indicators

    Returns:
        Dict with event details or None
    """
    if len(ohlcv_data) < 10:
        return None

    recent_10 = ohlcv_data[-10:]

    # Price trend
    closes = [c[4] for c in recent_10]
    price_trend = 'up' if closes[-1] > closes[0] else 'down'

    # Volume trend
    volumes = [c[5] for c in recent_10]
    first_half_vol = sum(volumes[:5]) / 5
    second_half_vol = sum(volumes[5:]) / 5
    volume_trend = 'up' if second_half_vol > first_half_vol else 'down'

    # Divergence
    if price_trend == 'up' and volume_trend == 'down':
        return {
            'type': 'bearish_divergence',
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'severity': 'high' if second_half_vol < first_half_vol * 0.7 else 'medium'
        }
    elif price_trend == 'down' and volume_trend == 'down':
        return {
            'type': 'bullish_divergence',
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'severity': 'high' if second_half_vol < first_half_vol * 0.7 else 'medium'
        }

    return None


# Test function
if __name__ == '__main__':
    print("Layer 2 - Event Detection System")
    print("\nFunctions available:")
    print("  ✓ detect_liquidity_sweep_event() - Sweep as event")
    print("  ✓ detect_volume_spike_event() - Volume anomalies")
    print("  ✓ detect_divergence_event() - Price/volume divergence")
    print("\nEvents refine entry/exit logic within regimes!")
