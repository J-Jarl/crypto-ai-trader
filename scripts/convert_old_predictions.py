#!/usr/bin/env python3
"""
Convert Old Predictions to New Format
======================================

Converts Phase 1 predictions to the format expected by evaluation_framework.py

Author: J-Jarl
Created: November 22, 2025
"""

import json
import os
from pathlib import Path


def convert_confidence_to_level(confidence: float) -> str:
    """Convert numeric confidence to string level"""
    if confidence >= 80:
        return "HIGH"
    elif confidence >= 60:
        return "MEDIUM"
    else:
        return "LOW"


def convert_prediction(old_pred: dict) -> dict:
    """Convert old prediction format to new evaluation format"""
    
    # Extract from old format
    recommendation_obj = old_pred.get('recommendation', {})
    sentiment_obj = old_pred.get('sentiment', {})
    position_sizing = old_pred.get('position_sizing', {})
    
    # Extract sentiment direction and confidence
    sentiment = sentiment_obj.get('sentiment', 'neutral')
    confidence = sentiment_obj.get('confidence', 50)
    
    # Convert sentiment direction to score
    if sentiment == 'positive':
        overall_score = confidence / 100.0
    elif sentiment == 'negative':
        overall_score = -confidence / 100.0
    else:  # neutral
        overall_score = 0.0
    
    # Build new format
    new_pred = {
        "timestamp": old_pred.get('timestamp'),
        "recommendation": recommendation_obj.get('action', 'HOLD').upper(),
        "confidence_level": convert_confidence_to_level(recommendation_obj.get('confidence', 50)),
        "sentiment_analysis": {
            "sentiment": sentiment,
            "confidence": confidence,
            "key_points": sentiment_obj.get('key_points', []),
            "overall_score": overall_score
        },
        "position_sizing": {
            "recommended_size": position_sizing.get('percentage_of_portfolio', 0)
        },
        "risk_management": {
            "stop_loss": recommendation_obj.get('stop_loss'),
            "take_profit": recommendation_obj.get('take_profit')
        }
    }
    
    return new_pred


def main():
    """Convert all old predictions"""
    archive_dir = Path("data/archive/phase1_predictions")
    output_dir = Path("data/analysis_results")
    
    if not archive_dir.exists():
        print(f"❌ Archive directory not found: {archive_dir}")
        return
    
    # Get all old prediction files
    old_files = list(archive_dir.glob('btc_analysis_*.json'))
    
    if not old_files:
        print("❌ No old predictions found")
        return
    
    print(f"\n{'='*60}")
    print(f"CONVERTING {len(old_files)} OLD PREDICTIONS")
    print(f"{'='*60}\n")
    
    converted = 0
    
    for filepath in old_files:
        try:
            # Load old prediction
            with open(filepath, 'r') as f:
                old_pred = json.load(f)
            
            # Convert to new format
            new_pred = convert_prediction(old_pred)
            
            # Create new filename
            # Old: btc_analysis_YYYYMMDD_HHMMSS.json
            # New: evaluation_format_YYYYMMDD_HHMMSS.json
            old_name = filepath.stem  # btc_analysis_YYYYMMDD_HHMMSS
            timestamp_part = old_name.replace('btc_analysis_', '')
            new_filename = f"evaluation_format_{timestamp_part}.json"
            new_filepath = output_dir / new_filename
            
            # Save converted prediction
            with open(new_filepath, 'w') as f:
                json.dump(new_pred, f, indent=2)
            
            print(f"✓ Converted: {filepath.name} → {new_filename}")
            converted += 1
            
        except Exception as e:
            print(f"✗ Error converting {filepath.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Converted: {converted}/{len(old_files)} files")
    print(f"Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
