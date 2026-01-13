
# Medical Knowledge Base
MEDICAL_KNOWLEDGE_BASE = {
    "lung cancer": {
        "department": "Oncology (Thoracic)",
        "compatible_drugs": ["Erlotinib", "Gefitinib", "Osimertinib", "Afatinib", "LIG-4920"],
        "mechanism": "EGFR Tyrosine Kinase Inhibitor (TKI) blocking cellular proliferation pathways",
        "clinical_evidence": "Startling effective in EGFR-mutant NSCLC (Phase III FLAURA trial)",
        "confidence_score": 98,
        "regulatory_status": "FDA Approved"
    },
    "non-small cell lung cancer": {
        "department": "Oncology (Thoracic)",
        "compatible_drugs": ["Pembrolizumab", "Cisplatin", "Docetaxel", "LIG-4920"],
        "mechanism": "PD-1 Checkpoint Inhibition / DNA Crosslinking",
        "clinical_evidence": "First-line standard of care for metastatic NSCLC",
        "confidence_score": 95,
        "regulatory_status": "FDA Approved"
    },
    "breast cancer": {
        "department": "Oncology (Breast)",
        "compatible_drugs": ["Tamoxifen", "Trastuzumab", "Paclitaxel"],
        "mechanism": "Estrogen Receptor Antagonist / HER2 Monoclonal Antibody",
        "clinical_evidence": "Gold standard for ER+ / HER2+ subtypes",
        "confidence_score": 92,
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
        "confidence_score": 99,
        "regulatory_status": "FDA Approved"
    },
    "hypertension": {
        "department": "Cardiology",
        "compatible_drugs": ["Lisinopril", "Amlodipine", "Losartan"],
        "mechanism": "ACE Inhibition / Calcium Channel Blockade",
        "clinical_evidence": "Proven reduction in stroke and MI risk",
        "confidence_score": 97,
        "regulatory_status": "FDA Approved"
    }
}

DRUG_MECHANISMS = {
    "erlotinib": "Reversible inhibitor of EGFR tyrosine kinase",
    "gefitinib": "Selective inhibitor of EGFR tyrosine kinase",
    "osimertinib": "Irreversible EGFR TKI targeting T790M mutation",
    "lig-4920": "Novel small molecule inhibitor targeting mutant EGFR kinase domain",
    "metformin": "Suppresses hepatic glucose production via AMPK",
    "lisinopril": "Inhibits angiotensin-converting enzyme (ACE)"
}
