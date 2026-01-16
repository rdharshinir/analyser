
# Medical Knowledge Base
MEDICAL_KNOWLEDGE_BASE = {
    "lung cancer": {
        "department": "Oncology (Thoracic)",
        "compatible_drugs": ["Erlotinib", "Gefitinib", "Osimertinib", "Afatinib", "LIG-4920"],
        "mechanism": "EGFR Tyrosine Kinase Inhibitor (TKI) blocking cellular proliferation pathways",
        "clinical_evidence": "Startling effective in EGFR-mutant NSCLC (Phase III FLAURA trial)",
        "confidence_score": 88,
        "regulatory_status": "FDA Approved"
    },
    "non-small cell lung cancer": {
        "department": "Oncology (Thoracic)",
        "compatible_drugs": ["Pembrolizumab", "Cisplatin", "Docetaxel", "LIG-4920"],
        "mechanism": "PD-1 Checkpoint Inhibition / DNA Crosslinking",
        "clinical_evidence": "First-line standard of care for metastatic NSCLC",
        "confidence_score": 85,
        "regulatory_status": "FDA Approved"
    },
    "breast cancer": {
        "department": "Oncology (Breast)",
        "compatible_drugs": ["Tamoxifen", "Trastuzumab", "Paclitaxel"],
        "mechanism": "Estrogen Receptor Antagonist / HER2 Monoclonal Antibody",
        "clinical_evidence": "Gold standard for ER+ / HER2+ subtypes",
        "confidence_score": 82,
        "regulatory_status": "FDA Approved"
    },
    "glioblastoma": {
        "department": "Neurology / Neuro-Oncology",
        "compatible_drugs": ["Temozolomide", "Bevacizumab", "LIG-4920"],
        "mechanism": "DNA Alkylating Agent crossing blood-brain barrier",
        "clinical_evidence": "Stupp Protocol indicates survival benefit",
        "confidence_score": 85,
        "regulatory_status": "FDA Approved"
    },
    "diabetes type 2": {
        "department": "Endocrinology",
        "compatible_drugs": ["Metformin", "Insulin", "Semaglutide"],
        "mechanism": "AMPK Activation / GLP-1 Receptor Agonist",
        "clinical_evidence": "Foundational therapy for glycemic control",
        "confidence_score": 95,
        "regulatory_status": "FDA Approved"
    },
    "hypertension": {
        "department": "Cardiology",
        "compatible_drugs": ["Lisinopril", "Amlodipine", "Losartan"],
        "mechanism": "ACE Inhibition / Calcium Channel Blockade",
        "clinical_evidence": "Proven reduction in stroke and MI risk",
        "confidence_score": 92,
        "regulatory_status": "FDA Approved"
    }
}

DRUG_MECHANISMS = {
    "erlotinib": "Reversible inhibitor of EGFR tyrosine kinase",
    "gefitinib": "Selective inhibitor of EGFR tyrosine kinase",
    "osimertinib": "Irreversible EGFR TKI targeting T790M mutation",
    "lig-4920": "Novel small molecule inhibitor targeting mutant EGFR kinase domain",
    "metformin": "Suppresses hepatic glucose production via AMPK",
    "lisinopril": "Inhibits angiotensin-converting enzyme (ACE)",
    "amlodipine": "Inhibits calcium ion influx across cell membranes",
    "losartan": "Blocks the interaction of angiotensin II and its receptor",
    "tamoxifen": "Selective estrogen receptor modulator (SERM)",
    "trastuzumab": "Monoclonal antibody against HER2 receptor",
    "paclitaxel": "Microtubule stabilizer preventing depolymerization",
    "cisplatin": "Crosslinks DNA inhibiting replication",
    "docetaxel": "Promotes microtubule assembly and inhibits disassembly",
    "pembrolizumab": "PD-1 inhibitor enhancing immune response",
    "temozolomide": "Alkylating agent (DNA methylation)",
    "bevacizumab": "Angiogenesis inhibitor (VEGF-A)",
    "afatinib": "Irreversible ErbB family blocker",
    "insulin": "Regulates glucose metabolism",
    "semaglutide": "GLP-1 receptor agonist"
}

DRUG_DOSAGE = {
    "erlotinib": "150 mg once daily",
    "gefitinib": "250 mg once daily",
    "osimertinib": "80 mg once daily",
    "lisinopril": "10-40 mg once daily",
    "amlodipine": "5-10 mg once daily",
    "losartan": "50-100 mg once daily",
    "metformin": "500-2000 mg daily in divided doses",
    "insulin": "Individualized sliding scale",
    "semaglutide": "0.25-2 mg weekly (subcutaneous)",
    "tamoxifen": "20 mg daily",
    "trastuzumab": "Initial 8 mg/kg, then 6 mg/kg every 3 weeks",
    "paclitaxel": "175 mg/m2 IV over 3 hours every 3 weeks",
    "cisplatin": "75-100 mg/m2 IV every 3-4 weeks",
    "docetaxel": "75 mg/m2 IV every 3 weeks",
    "pembrolizumab": "200 mg IV every 3 weeks",
    "temozolomide": "75 mg/m2 daily during radiation, then 150-200 mg/m2",
    "bevacizumab": "10 mg/kg IV every 2 weeks",
    "afatinib": "40 mg once daily",
    "lig-4920": "Clinical Trial Dosage (Phase 1)"
}
