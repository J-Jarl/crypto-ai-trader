"""
Decision Matrix - Combines Regime (Layer 1) with Events (Layer 2)
Generates context-aware trading signals
"""
from typing import Dict, Tuple, Optional, List


def generate_trading_signal(regime: str, regime_confidence: int, regime_details: Dict,
                            sweep_event: Optional[Dict], divergence_event: Optional[Dict],
                            volume_event: Optional[Dict], current_price: float,
                            market_data) -> Tuple[str, int, str, Dict]:
    """
    Decision Matrix: Regime × Events → Trading Signal

    Args:
        regime: TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY
        regime_confidence: Regime confidence score
        regime_details: Details from regime detection
        sweep_event: Liquidity sweep event data
        divergence_event: Divergence event data
        volume_event: Volume spike event data
        current_price: Current price
        market_data: Market indicators

    Returns:
        Tuple of (signal, confidence, reason, details)
        - signal: BUY, SELL, HOLD
        - confidence: 0-100
        - reason: Human-readable explanation
        - details: Additional context
    """

    print(f"\n  🎯 DECISION MATRIX:")
    print(f"     Regime: {regime} ({regime_confidence}%)")
    if sweep_event:
        print(f"     Event: {sweep_event['type']} - {sweep_event['direction']} {sweep_event['magnitude']:.1f}% ({sweep_event['classification']})")
    if divergence_event:
        print(f"     Event: {divergence_event['type']} ({divergence_event['severity']})")

    # ============================================================
    # TRENDING_UP REGIME
    # ============================================================
    if regime == 'TRENDING_UP':

        # A. Upward spike in uptrend
        if sweep_event and sweep_event['direction'] == 'upward':
            magnitude = sweep_event['magnitude']
            classification = sweep_event['classification']

            # Major exhaustion spike (>8%, RSI>75)
            if magnitude > 8.0 and market_data.rsi_14 > 75:
                # Check for trend break
                from regime_layer1 import check_trend_structure_intact
                structure_intact, break_reason = check_trend_structure_intact(
                    regime, [], regime_details
                )

                if not structure_intact:
                    print(f"     💡 Decision: Exhaustion + Trend Break → SELL")
                    return 'SELL', 85, f'trend_exhaustion_break_{break_reason}', {
                        'sweep_magnitude': magnitude,
                        'rsi': market_data.rsi_14,
                        'structure_break': break_reason
                    }
                else:
                    print(f"     💡 Decision: Exhaustion but structure intact → HOLD (watch for break)")
                    return 'HOLD', 70, 'trend_extended_watching', {
                        'sweep_magnitude': magnitude,
                        'rsi': market_data.rsi_14
                    }

            # Medium spike (continuation pattern)
            elif magnitude >= 4.0 and magnitude <= 8.0:
                if classification == 'BREAKOUT':
                    print(f"     💡 Decision: Breakout continuation in uptrend → BUY")
                    return 'BUY', 80, 'trend_continuation_breakout', {
                        'sweep_magnitude': magnitude
                    }
                elif classification == 'TRAP':
                    print(f"     💡 Decision: Trap signal but in uptrend → HOLD (wait for dip)")
                    return 'HOLD', 60, 'uptrend_trap_signal_cautious', {
                        'sweep_magnitude': magnitude
                    }
                else:  # UNCERTAIN
                    print(f"     💡 Decision: Uncertain spike in uptrend → HOLD")
                    return 'HOLD', 50, 'trend_uncertain_spike', {
                        'sweep_magnitude': magnitude
                    }

            # Small spike (normal volatility)
            else:
                print(f"     💡 Decision: Small spike in uptrend → BUY")
                return 'BUY', 75, 'trend_continuation', {
                    'sweep_magnitude': magnitude
                }

        # B. Downward sweep in uptrend (DIP BUY OPPORTUNITY!)
        elif sweep_event and sweep_event['direction'] == 'downward':
            classification = sweep_event['classification']
            magnitude = sweep_event['magnitude']

            if classification == 'BOTTOM_TRAP':
                print(f"     💡 Decision: Dip in uptrend (trap) → BUY THE DIP!")
                return 'BUY', 85, 'uptrend_dip_buy', {
                    'sweep_magnitude': magnitude,
                    'classification': classification
                }
            elif classification == 'BREAKDOWN':
                # Check structure
                from regime_layer1 import check_trend_structure_intact
                structure_intact, break_reason = check_trend_structure_intact(
                    regime, [], regime_details
                )

                if not structure_intact:
                    print(f"     💡 Decision: Breakdown + trend break → SELL")
                    return 'SELL', 80, f'uptrend_broken_{break_reason}', {
                        'sweep_magnitude': magnitude,
                        'structure_break': break_reason
                    }
                else:
                    print(f"     💡 Decision: Breakdown signal but structure intact → HOLD")
                    return 'HOLD', 60, 'uptrend_pullback_watching', {
                        'sweep_magnitude': magnitude
                    }
            else:  # UNCERTAIN
                print(f"     💡 Decision: Uncertain dip in uptrend → HOLD (wait for clarity)")
                return 'HOLD', 50, 'uptrend_uncertain_dip', {
                    'sweep_magnitude': magnitude
                }

        # C. Bearish divergence in uptrend (WARNING!)
        elif divergence_event and divergence_event['type'] == 'bearish_divergence':
            severity = divergence_event['severity']
            if severity == 'high':
                print(f"     💡 Decision: Strong bearish divergence in uptrend → HOLD (warning)")
                return 'HOLD', 70, 'uptrend_bearish_divergence_warning', {
                    'divergence_severity': severity
                }
            else:
                print(f"     💡 Decision: Mild bearish divergence → Continue trend")
                return 'BUY', 65, 'uptrend_mild_divergence', {
                    'divergence_severity': severity
                }

        # D. No events - pure trend following
        else:
            print(f"     💡 Decision: Clean uptrend, no events → BUY")
            return 'BUY', 80, 'clean_uptrend', {}

    # ============================================================
    # TRENDING_DOWN REGIME
    # ============================================================
    elif regime == 'TRENDING_DOWN':

        # A. Downward sweep in downtrend (continuation)
        if sweep_event and sweep_event['direction'] == 'downward':
            classification = sweep_event['classification']

            if classification == 'BREAKDOWN':
                print(f"     💡 Decision: Breakdown in downtrend → SELL")
                return 'SELL', 85, 'downtrend_continuation', {
                    'sweep_classification': classification
                }
            elif classification == 'BOTTOM_TRAP':
                print(f"     💡 Decision: Trap in downtrend → HOLD (bear trap)")
                return 'HOLD', 60, 'downtrend_bear_trap', {
                    'sweep_classification': classification
                }
            else:
                print(f"     💡 Decision: Uncertain sweep → Follow downtrend SELL")
                return 'SELL', 70, 'downtrend_default', {}

        # B. Upward spike in downtrend (SHORT OPPORTUNITY!)
        elif sweep_event and sweep_event['direction'] == 'upward':
            classification = sweep_event['classification']

            if classification == 'TRAP':
                print(f"     💡 Decision: Bull trap in downtrend → SELL THE RALLY!")
                return 'SELL', 85, 'downtrend_sell_rally', {
                    'sweep_classification': classification
                }
            elif classification == 'BREAKOUT':
                # Check if downtrend broken
                from regime_layer1 import check_trend_structure_intact
                structure_intact, break_reason = check_trend_structure_intact(
                    regime, [], regime_details
                )

                if not structure_intact:
                    print(f"     💡 Decision: Breakout + trend reversal → BUY")
                    return 'BUY', 80, f'downtrend_reversal_{break_reason}', {
                        'structure_break': break_reason
                    }
                else:
                    print(f"     💡 Decision: Breakout but downtrend intact → HOLD")
                    return 'HOLD', 60, 'downtrend_bounce_watching', {}
            else:
                print(f"     💡 Decision: Uncertain rally → Continue downtrend")
                return 'SELL', 70, 'downtrend_default', {}

        # C. No events
        else:
            print(f"     💡 Decision: Clean downtrend → SELL")
            return 'SELL', 80, 'clean_downtrend', {}

    # ============================================================
    # RANGING REGIME
    # ============================================================
    elif regime == 'RANGING':

        # In ranging, liquidity sweeps are CLASSIC trap/breakout plays
        if sweep_event:
            classification = sweep_event['classification']
            direction = sweep_event['direction']

            if classification in ['TRAP', 'BOTTOM_TRAP'] and direction == 'upward':
                print(f"     💡 Decision: Range top trap → SELL (mean reversion)")
                return 'SELL', 75, 'range_top_trap', {
                    'sweep_classification': classification
                }
            elif classification == 'BREAKDOWN' and direction == 'downward':
                print(f"     💡 Decision: Range bottom trap → BUY (mean reversion)")
                return 'BUY', 75, 'range_bottom_trap', {
                    'sweep_classification': classification
                }
            elif classification == 'BREAKOUT':
                print(f"     💡 Decision: Range breakout → BUY/SELL direction")
                signal = 'BUY' if direction == 'upward' else 'SELL'
                return signal, 70, 'range_breakout', {
                    'direction': direction
                }
            else:
                print(f"     💡 Decision: Uncertain in range → HOLD")
                return 'HOLD', 50, 'range_uncertain', {}
        else:
            print(f"     💡 Decision: Ranging, no event → HOLD")
            return 'HOLD', 60, 'range_waiting', {}

    # ============================================================
    # CHOPPY REGIME
    # ============================================================
    else:  # CHOPPY
        # Use volume blocking + distribution logic
        print(f"     💡 Decision: Choppy regime → Use pattern detection")
        return 'HOLD', 50, 'choppy_default_to_patterns', {}


# Test function
if __name__ == '__main__':
    print("Decision Matrix - Regime × Event Signal Generator")
    print("\nCombines:")
    print("  Layer 1 (Regime): TRENDING_UP/DOWN, RANGING, CHOPPY")
    print("  Layer 2 (Events): Sweeps, divergences, volume spikes")
    print("\nOutputs: Context-aware BUY/SELL/HOLD signals!")
