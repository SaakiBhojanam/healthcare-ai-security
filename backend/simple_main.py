import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import uuid
from core.ai_security_engine import AISecurityEngine

# Model types
class ModelType:
    ANCESTRY = "ancestry_classifier"
    DISEASE = "disease_predictor"
    DRUG_RESPONSE = "drug_response"

    @classmethod
    def valid_types(cls):
        return [cls.ANCESTRY, cls.DISEASE, cls.DRUG_RESPONSE]

# Request models
class ModelCreationRequest(BaseModel):
    model_id: str
    model_type: str = "ancestry_classifier"
    use_differential_privacy: bool = False
    epsilon: float = 0.1

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ModelType.valid_types():
            raise ValueError(f'Invalid model type. Must be one of: {ModelType.valid_types()}')
        return v

    @validator('epsilon')
    def validate_epsilon(cls, v):
        if not 0.001 <= v <= 10.0:
            raise ValueError('Epsilon must be between 0.001 and 10.0')
        return v

class AttackRequest(BaseModel):
    model_id: str
    attack_type: str
    target_samples: int = 100
    query_budget: int = 1000

    @validator('target_samples')
    def validate_samples(cls, v):
        if not 1 <= v <= 1000:
            raise ValueError('Target samples must be between 1 and 1000')
        return v

    @validator('query_budget')
    def validate_budget(cls, v):
        if not 10 <= v <= 10000:
            raise ValueError('Query budget must be between 10 and 10000')
        return v

class FederatedAnalysisRequest(BaseModel):
    participants: int = 5
    malicious_participants: int = 1
    privacy_budget_per_round: float = 0.1
    total_rounds: int = 10

    @validator('participants')
    def validate_participants(cls, v):
        if not 2 <= v <= 50:
            raise ValueError('Participants must be between 2 and 50')
        return v

    @validator('malicious_participants')
    def validate_malicious(cls, v, values):
        if 'participants' in values and v >= values['participants']:
            raise ValueError('Malicious participants must be less than total participants')
        return v

# Response models
class SecurityMetrics(BaseModel):
    vulnerability_score: float
    privacy_budget_consumed: float
    success_rate: float
    risk_level: str

class AttackResult(BaseModel):
    attack_id: str
    model_id: str
    attack_type: str
    timestamp: str
    metrics: SecurityMetrics
    attack_details: Dict[str, Any]
    mitigation_recommendations: List[str]
    defensive_measures: List[str]

# Initialize FastAPI
app = FastAPI(
    title="Health AI Security Platform",
    description="Security testing platform for healthcare AI models and federated learning systems.",
    version="1.0.0",
    docs_url="/docs"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize security engine
security_engine = AISecurityEngine()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "genomic-privacy-dashboard"}

@app.post("/ai/models")
async def create_ai_model(request: ModelCreationRequest):
    try:
        # Generate synthetic genomic data for demo
        n_samples = 1000
        n_features = 1000
        genomic_data = np.random.randn(n_samples, n_features)

        # Create labels based on model type
        if request.model_type == ModelType.ANCESTRY:
            labels = np.random.randint(0, 5, n_samples)  # 5 ancestry groups
        elif request.model_type == ModelType.DISEASE:
            labels = np.random.randint(0, 2, n_samples)  # Binary disease prediction
        elif request.model_type == ModelType.DRUG_RESPONSE:
            labels = np.random.randint(0, 3, n_samples)  # 3 response categories
        else:
            raise HTTPException(status_code=422, detail="Unsupported model type")

        # Create the model
        model = await security_engine.create_genomic_model(
            model_id=request.model_id,
            model_type=request.model_type,
            genomic_data=genomic_data,
            labels=labels,
            use_differential_privacy=request.use_differential_privacy
        )

        return {
            "status": "success",
            "message": "AI model created successfully",
            "data": {
                "model_id": request.model_id,
                "model_type": request.model_type,
                "training_samples": n_samples,
                "features": n_features,
                "differential_privacy_enabled": request.use_differential_privacy,
                "privacy_budget_remaining": model.privacy_budget if hasattr(model, 'privacy_budget') else 1.0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/attacks/membership-inference")
async def membership_inference_attack(request: AttackRequest):
    try:
        # Generate synthetic target samples
        target_samples = np.random.randn(request.target_samples, 1000)

        # Run the attack
        report = await security_engine.run_membership_inference_attack(
            model_id=request.model_id,
            target_samples=target_samples
        )

        attack_id = str(uuid.uuid4())
        result = AttackResult(
            attack_id=attack_id,
            model_id=request.model_id,
            attack_type="membership_inference",
            timestamp=datetime.now().isoformat(),
            metrics=SecurityMetrics(
                vulnerability_score=report.vulnerability_score,
                privacy_budget_consumed=report.privacy_budget_consumed,
                success_rate=report.success_rate,
                risk_level="High" if report.success_rate > 0.7 else "Medium" if report.success_rate > 0.4 else "Low"
            ),
            attack_details=report.attack_details,
            mitigation_recommendations=report.mitigation_recommendations,
            defensive_measures=report.defensive_measures
        )

        return {
            "status": "success",
            "message": "Membership inference attack completed",
            "data": result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/federated/security-analysis")
async def federated_security_analysis(request: FederatedAnalysisRequest):
    try:
        # Run federated learning security analysis
        analysis = await security_engine.analyze_federated_learning_security(
            participant_count=request.participants,
            malicious_participants=request.malicious_participants
        )

        # Add healthcare-specific compliance analysis
        compliance_data = {
            "hipaa_compliant": request.malicious_participants == 0 and request.privacy_budget_per_round <= 0.1,
            "gdpr_compliant": request.privacy_budget_per_round <= 0.05,
            "fda_guidance_met": analysis["federation_setup"]["byzantine_resilience"] == "Secure",
            "risk_assessment": "Low" if request.malicious_participants == 0 else "High"
        }

        # Calculate healthcare-specific risks
        healthcare_risks = [
            "Patient re-identification through model outputs",
            "Sensitive health data leakage via gradient analysis",
            "Cross-institutional data poisoning attacks"
        ]

        if request.malicious_participants > 0:
            healthcare_risks.extend([
                "Malicious hospital participation detected",
                "Potential for coordinated attacks on patient privacy"
            ])

        return {
            "status": "success",
            "message": "Federated security analysis completed",
            "data": {
                **analysis,
                "healthcare_specific_risks": healthcare_risks,
                "compliance_framework": compliance_data
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/security-dashboard")
async def get_security_dashboard():
    try:
        dashboard_data = await security_engine.get_security_dashboard_data()

        return {
            "status": "success",
            "message": "Security dashboard data retrieved",
            "data": dashboard_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )