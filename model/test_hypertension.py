import requests
import os

API_BASE = "http://localhost:5000/api"

def test_hypertension_override():
    print("Testing Hypertension Override...")
    
    # Create a dummy CSV file
    dummy_file = "test_genome.csv"
    with open(dummy_file, "w") as f:
        f.write("gene,mutation\nEGFR,L858R")
        
    try:
        with open(dummy_file, 'rb') as f_in:
            files = {'file': f_in}
            data = {'disease': 'Hypertension'}
            
            response = requests.post(f"{API_BASE}/analyze-genome", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("Response:", result)
            
            drugs = result.get('results', [])
            if not drugs:
                print("FAIL: No drugs returned")
                return
                
            top_drug = drugs[0]
            print(f"Top Drug: {top_drug.get('Drug')}")
            print(f"Dosage: {top_drug.get('dosageGuidance')}")
            
            if "Lisinopril" in top_drug.get('Drug') or "Amlodipine" in top_drug.get('Drug'):
                print("PASS: Correct drug recommended for Hypertension")
            else:
                print(f"FAIL: Expected Hypertension drug, got {top_drug.get('Drug')}")
                
            if top_drug.get('dosageGuidance'):
                print("PASS: Dosage guidance present")
            else:
                print("FAIL: Dosage guidance missing")
                
        else:
            print(f"FAIL: API Error {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    test_hypertension_override()
