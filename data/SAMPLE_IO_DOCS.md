# Sample Input/Output Documentation

## Sample Input Files

### 1. Patient Genome CSV (`sample_patient_genome.csv`)
```csv
GENE_SYMBOL,EXPRESSION_VALUE
TP53,2.34
EGFR,1.87
KRAS,0.45
BRCA1,1.23
MYC,3.12
PIK3CA,1.67
AKT1,0.89
PTEN,2.01
RB1,1.45
CDKN2A,0.34
```

### 2. Drug Compatibility Request (`sample_drug_request.json`)
```json
{
  "drug_name": "Erlotinib",
  "disease": "lung cancer",
  "patient_data": {
    "age": 58,
    "gender": "Male",
    "cancer_stage": "Stage III",
    "previous_treatments": ["Carboplatin", "Pemetrexed"],
    "allergies": ["Penicillin"],
    "weight_kg": 72
  }
}
```

## Expected UI Output

### For Genome Analysis (`/api/analyze-genome`):
**Input**: Upload `sample_patient_genome.csv` via file input

**Output**:
```json
{
  "success": true,
  "results": [
    {
      "Gene": "EGFR",
      "Mutation": "L858R",
      "Risk_Score": 0.85,
      "Status": "High Risk"
    },
    {
      "Gene": "KRAS",
      "Mutation": "G12C", 
      "Risk_Score": 0.92,
      "Status": "Critical"
    },
    {
      "Gene": "TP53",
      "Mutation": "R273H",
      "Risk_Score": 0.45,
      "Status": "Moderate"
    }
  ]
}
```

### For Drug Compatibility (`/api/analyze-compatibility`):
**Input**: Submit JSON data from `sample_drug_request.json`

**Output**:
```json
{
  "compatible": true,
  "status": "COMPATIBLE (VERIFIED)",
  "drug_name": "Erlotinib",
  "disease": "lung cancer",
  "department": "Oncology",
  "mechanism": "EGFR tyrosine kinase inhibitor",
  "clinical_evidence": "First-line treatment for EGFR-mutated NSCLC",
  "confidence_score": "85%",
  "regulatory_status": "FDA Approved",
  "recommendation": "Proceed with Standard of Care Protocol",
  "patient_info": {
    "age": 58,
    "gender": "Male",
    "cancer_stage": "Stage III"
  }
}
```

### For DeepChem Prediction (`/api/deepchem-predict`):
**Input**: Gene expression array + drug information

**Output**:
```json
{
  "success": true,
  "drug_name": "Unknown Drug",
  "prediction": -0.2169,
  "sensitivity": "🟢 Sensitive - Good response expected",
  "confidence": 85.0,
  "dosage_recommendation": 7.86,
  "dosage_unit": "mg/m²",
  "full_results": {
    "prediction": -0.2169,
    "sensitivity_classification": {
      "level": "SENSITIVE",
      "color": "🟢",
      "interpretation": "Good response expected"
    },
    "confidence": 85.0,
    "dosage_recommendation": {
      "category": "MEDIUM",
      "recommended_dosage": 7.86,
      "unit": "mg/m²",
      "recommendation": "Standard dosage recommended"
    }
  }
}
```

## UI Display Format

### Dashboard Sections:

1. **Genomic Analysis Panel**
   - Gene expression heatmap
   - Mutation risk indicators
   - Risk score visualization

2. **Drug Compatibility Panel**  
   - Compatibility status (✅/❌)
   - Clinical evidence summary
   - Confidence percentage
   - Department recommendation

3. **Dosage Calculator**
   - Recommended dosage value
   - Dosage category (Low/Medium/High)
   - Safety warnings
   - Confidence indicator

4. **Patient Report**
   - Treatment history timeline
   - Allergy alerts
   - Current medications
   - Report generation button