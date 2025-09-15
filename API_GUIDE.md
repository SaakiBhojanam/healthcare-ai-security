# Security for Health AI Platform - API Guide

## Overview

Professional-grade API for genomic AI security testing and privacy analysis. Built for healthcare AI security researchers, compliance teams, and privacy engineers.

**API Base URL**: `http://localhost:8000`
**Interactive Docs**: `http://localhost:8000/docs`
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Core Capabilities

### **1. AI Model Security Testing**
- **Membership Inference Attacks**: Detect training data membership
- **Model Extraction Attacks**: Query-based model stealing
- **Model Inversion Attacks**: Reconstruct sensitive features
- **Differential Privacy**: Privacy budget tracking and DP models

### **2. Federated Learning Security**
- **Multi-Institution Analysis**: Healthcare collaboration simulation
- **Byzantine Attack Assessment**: Malicious participant detection
- **Privacy Budget Management**: Comprehensive ε-accounting
- **Compliance Framework**: HIPAA, GDPR, FDA guidance alignment

### **3. Genomic Privacy Analysis**
- **Kinship Network Analysis**: IBD matrix calculations
- **Population Structure Risks**: Ancestry inference vulnerabilities
- **Chromosomal Mapping**: High-risk variant identification

---

## API Endpoints

### **AI Model Management**

#### Create AI Model
```bash
POST /ai/models
```

**Description**: Create and train a genomic AI model with optional differential privacy protection.

**Request Body**:
```json
{
  "model_id": "hospital_ancestry_v1",
  "model_type": "ancestry_classifier",
  "use_differential_privacy": true,
  "epsilon": 0.1
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "model_id": "hospital_ancestry_v1",
    "model_type": "ancestry_classifier",
    "training_samples": 500,
    "features": 1000,
    "differential_privacy_enabled": true,
    "epsilon": 0.1,
    "privacy_budget_remaining": 0.9,
    "creation_timestamp": "2025-09-14T19:55:05.677703"
  },
  "message": "Model 'hospital_ancestry_v1' created successfully"
}
```

**Model Types**:
- `ancestry_classifier`: Genetic ancestry prediction
- `disease_predictor`: Disease risk assessment
- `drug_response`: Pharmacogenomic response prediction

---

### **AI Security Attacks**

#### Membership Inference Attack
```bash
POST /ai/attacks/membership-inference
```

**Description**: Execute membership inference attack to detect if individuals were in training dataset.

**Request Body**:
```json
{
  "model_id": "hospital_ancestry_v1",
  "attack_type": "membership_inference",
  "target_samples": 50,
  "query_budget": 500
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "attack_id": "mi_hospital_ancestry_v1_20250914_195505",
    "model_id": "hospital_ancestry_v1",
    "attack_type": "membership_inference",
    "timestamp": "2025-09-14T19:55:05.677703",
    "metrics": {
      "success_rate": 0.42,
      "vulnerability_score": 0.63,
      "privacy_budget_consumed": 2.5,
      "risk_level": "Medium"
    },
    "attack_details": {
      "samples_tested": 50,
      "successful_inferences": 21,
      "average_confidence": 0.257,
      "shadow_models_used": 5,
      "defense_effectiveness": "High"
    },
    "defensive_measures": [
      "Differential Privacy (ε=0.1)"
    ],
    "mitigation_recommendations": [
      "Implement differential privacy with ε ≤ 0.1",
      "Use model ensemble with member sample rotation",
      "Apply output perturbation for high-confidence predictions",
      "Implement query budget limits per user"
    ]
  },
  "message": "Membership inference attack completed with 42.0% success rate"
}
```

**Healthcare Impact**:
- Violates patient privacy by revealing study participation
- Can expose sensitive health conditions
- Particularly dangerous for rare disease cohorts

---

### **Federated Learning Security**

#### Security Analysis
```bash
POST /ai/federated/security-analysis
```

**Description**: Comprehensive security assessment for multi-institutional AI collaboration.

**Request Body**:
```json
{
  "participants": 8,
  "malicious_participants": 2,
  "privacy_budget_per_round": 0.05,
  "total_rounds": 15
}
```

**Response**: *(abbreviated)*
```json
{
  "status": "success",
  "data": {
    "federation_setup": {
      "total_participants": 8,
      "malicious_participants": 2,
      "byzantine_resilience": "Secure",
      "consensus_integrity": 0.95
    },
    "healthcare_specific_risks": {
      "patient_re_identification_risk": 0.2,
      "cross_institutional_linkage": "High",
      "hipaa_compliance_risk": "Manageable",
      "rare_disease_exposure": "High"
    },
    "privacy_economics": {
      "total_privacy_budget": 6.0,
      "sustainability_assessment": "Needs Review",
      "recommended_max_rounds": 2
    },
    "compliance_framework": {
      "gdpr_compliant": false,
      "hipaa_safeguards": ["De-identification", "Secure aggregation"],
      "fda_guidance_alignment": "Compliant"
    }
  }
}
```

**Real-World Applications**:
- Multi-hospital drug response prediction
- Federated genetic association studies
- Collaborative rare disease diagnosis models

---

### **Security Dashboard**

#### AI Security Overview
```bash
GET /ai/security-dashboard
```

**Description**: Comprehensive security metrics across all AI models and attacks.

**Response**:
```json
{
  "status": "success",
  "data": {
    "overview": {
      "models_analyzed": 3,
      "attacks_simulated": 12,
      "avg_vulnerability_score": 0.67,
      "overall_risk_level": "Medium"
    },
    "attack_analysis": {
      "membership_inference": {
        "attempts": 5,
        "avg_success_rate": 0.42,
        "risk_level": "Medium"
      },
      "model_extraction": {
        "attempts": 4,
        "avg_success_rate": 0.78,
        "risk_level": "High"
      }
    },
    "privacy_budget": {
      "total_consumed": 0.35,
      "remaining_budget": 0.65,
      "budget_status": "Healthy"
    }
  }
}
```

---

## Usage Examples

### **1. Healthcare Research Scenario**
```bash
# Create DP-protected model for multi-hospital study
curl -X POST http://localhost:8000/ai/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rare_disease_consortium_v1",
    "model_type": "disease_predictor",
    "use_differential_privacy": true,
    "epsilon": 0.01
  }'

# Test membership inference vulnerability
curl -X POST http://localhost:8000/ai/attacks/membership-inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rare_disease_consortium_v1",
    "target_samples": 100
  }'

# Analyze federated learning security
curl -X POST http://localhost:8000/ai/federated/security-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "participants": 12,
    "malicious_participants": 1,
    "privacy_budget_per_round": 0.01,
    "total_rounds": 50
  }'
```

### **2. Compliance Assessment**
```bash
# Check overall security posture
curl -X GET http://localhost:8000/ai/security-dashboard

# Verify HIPAA compliance for federated setup
curl -X POST http://localhost:8000/ai/federated/security-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "participants": 5,
    "malicious_participants": 0,
    "privacy_budget_per_round": 0.1,
    "total_rounds": 10
  }' | jq '.data.compliance_framework'
```

---

## Security Features

### **Input Validation**
- Pydantic models with field validation
- Epsilon bounds: 0.001 ≤ ε ≤ 10.0
- Sample limits: 1 ≤ samples ≤ 1000
- Participant limits: 2 ≤ participants ≤ 50

### **Error Handling**
- HTTP status codes for different error types
- Detailed error messages with suggestions
- Model existence validation
- Byzantine threshold validation

### **Privacy Protection**
- Real-time privacy budget tracking
- Differential privacy implementation
- Byzantine fault tolerance assessment
- HIPAA/GDPR compliance indicators

---

## Response Format

All endpoints return standardized responses:

```json
{
  "status": "success|error",
  "data": { /* endpoint-specific data */ },
  "message": "Human-readable status message"
}
```

**HTTP Status Codes**:
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Not Found (model doesn't exist)
- `500`: Internal Server Error

---

## Interactive Documentation

Visit `http://localhost:8000/docs` for:
- Complete API specification
- Interactive endpoint testing
- Request/response schemas
- Authentication details

## Perfect for Security Portfolios

This API demonstrates:
- **Healthcare AI Security Expertise**: Real-world attack vectors and defenses
- **Privacy-Preserving ML**: Differential privacy and federated learning
- **Regulatory Compliance**: HIPAA, GDPR, FDA guidance alignment
- **Professional API Design**: Comprehensive validation, documentation, error handling

Built for roles in healthcare AI security, privacy engineering, and medical device compliance.
