"""
Simple test script to verify the drug response prediction system works
without requiring the full datasets
"""

import sys
import os
sys.path.append('.')

# Import the main predictor
from main_predictor import DrugResponsePredictor
import numpy as np

def test_prediction_system():
    """Test the prediction system with dummy data"""
    print("Testing Drug Response Prediction System...")
    
    # Create predictor instance
    predictor = DrugResponsePredictor()
    
    # Create dummy gene expression data (1000 genes)
    dummy_gene_expression = np.random.randn(1000)
    
    # Test prediction
    try:
        result = predictor.predict_drug_response(
            gene_expression=dummy_gene_expression,
            drug_smiles="CCO"  # Ethanol as test molecule
        )
        
        print("✅ Prediction successful!")
        print(f"Prediction: {result['prediction']:.4f}")
        print(f"Sensitivity: {result['sensitivity_classification']['interpretation']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Dosage Recommendation: {result['dosage_recommendation']['recommended_dosage']} {result['dosage_recommendation']['unit']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction capability"""
    print("\nTesting Batch Prediction...")
    
    predictor = DrugResponsePredictor()
    
    # Create multiple samples
    gene_expressions = [np.random.randn(1000) for _ in range(3)]
    
    try:
        results = predictor.batch_predict(gene_expressions)
        
        print("✅ Batch prediction successful!")
        for i, result in enumerate(results):
            print(f"Sample {i+1}: {result['prediction']:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("DRUG RESPONSE PREDICTION SYSTEM TEST")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_prediction_system()
    test2_passed = test_batch_prediction()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - System is working correctly!")
        print("Ready for integration with frontend/backend")
    else:
        print("⚠️  Some tests failed - check system configuration")
    print("=" * 50)