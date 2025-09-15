import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import asyncio
import logging

@dataclass
class AISecurityReport:
    model_id: str
    attack_type: str
    success_rate: float
    vulnerability_score: float
    privacy_budget_consumed: float
    defensive_measures: List[str]
    attack_details: Dict[str, Any]
    mitigation_recommendations: List[str]

class GenomicAIModel:
    def __init__(self, model_type: str = "ancestry_classifier"):
        self.model_type = model_type
        self.model = None
        self.training_data_size = 0
        self.privacy_budget = 1.0
        self.differential_privacy_enabled = False

    def train(self, genomic_data: np.ndarray, labels: np.ndarray, use_dp: bool = False):
        self.training_data_size = len(genomic_data)
        self.differential_privacy_enabled = use_dp

        if self.model_type == "ancestry_classifier":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "disease_predictor":
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == "drug_response":
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Apply differential privacy if requested
        if use_dp:
            genomic_data = genomic_data + np.random.laplace(0, 0.1, genomic_data.shape)
            self.privacy_budget -= 0.1

        self.model.fit(genomic_data, labels)
        return self

    def predict(self, genomic_data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(genomic_data)

    def predict_proba(self, genomic_data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained first")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(genomic_data)

        # Fallback for models without probability estimates
        predictions = self.predict(genomic_data)
        probs = np.zeros((len(predictions), 2))
        probs[np.arange(len(predictions)), predictions] = 0.8
        probs[np.arange(len(predictions)), 1-predictions] = 0.2
        return probs

class AISecurityEngine:

    def __init__(self):
        self.models: Dict[str, GenomicAIModel] = {}
        self.attack_history: List[AISecurityReport] = []
        self.logger = logging.getLogger(__name__)

    async def create_genomic_model(self, model_id: str, model_type: str,
                                 genomic_data: np.ndarray, labels: np.ndarray,
                                 use_differential_privacy: bool = False) -> GenomicAIModel:
        model = GenomicAIModel(model_type)
        model.train(genomic_data, labels, use_differential_privacy)
        self.models[model_id] = model
        return model

    async def run_membership_inference_attack(self, model_id: str,
                                            target_samples: np.ndarray,
                                            shadow_models: int = 5) -> AISecurityReport:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        target_model = self.models[model_id]

        attack_success_samples = []
        confidence_scores = []

        for i, sample in enumerate(target_samples):
            pred_proba = target_model.predict_proba(sample.reshape(1, -1))
            max_confidence = np.max(pred_proba)

            # High confidence often indicates membership in training data
            is_member_prediction = max_confidence > 0.8
            actual_membership = np.random.choice([True, False], p=[0.6, 0.4])

            attack_success = (is_member_prediction == actual_membership)
            attack_success_samples.append(attack_success)
            confidence_scores.append(float(max_confidence))

        success_rate = np.mean(attack_success_samples)
        vulnerability_score = min(1.0, success_rate * 1.5)

        # Calculate privacy budget consumed
        privacy_budget_consumed = 0.05 * len(target_samples)
        if target_model.differential_privacy_enabled:
            privacy_budget_consumed *= 0.3  # DP reduces attack effectiveness
            success_rate *= 0.7

        attack_details = {
            "samples_tested": len(target_samples),
            "successful_inferences": int(np.sum(attack_success_samples)),
            "average_confidence": float(np.mean(confidence_scores)),
            "shadow_models_used": shadow_models,
            "defense_effectiveness": "High" if target_model.differential_privacy_enabled else "None"
        }

        defensive_measures = []
        if target_model.differential_privacy_enabled:
            defensive_measures.append("Differential Privacy (ε=0.1)")

        mitigation_recommendations = [
            "Implement differential privacy with ε ≤ 0.1",
            "Use model ensemble with member sample rotation",
            "Apply output perturbation for high-confidence predictions",
            "Implement query budget limits per user"
        ]

        report = AISecurityReport(
            model_id=model_id,
            attack_type="membership_inference",
            success_rate=success_rate,
            vulnerability_score=vulnerability_score,
            privacy_budget_consumed=privacy_budget_consumed,
            defensive_measures=defensive_measures,
            attack_details=attack_details,
            mitigation_recommendations=mitigation_recommendations
        )

        self.attack_history.append(report)
        return report

    async def run_model_extraction_attack(self, model_id: str,
                                        query_budget: int = 1000) -> AISecurityReport:
        """Simulate model extraction attack using query-based methods"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        target_model = self.models[model_id]

        # Generate synthetic query data
        n_features = 1000  # Typical genomic feature count
        query_data = np.random.randn(query_budget, n_features)

        # Query the target model
        target_predictions = target_model.predict_proba(query_data)

        # Train surrogate model using queries
        surrogate_model = RandomForestClassifier(n_estimators=50, random_state=42)
        surrogate_labels = np.argmax(target_predictions, axis=1)
        surrogate_model.fit(query_data, surrogate_labels)

        # Evaluate extraction success
        test_data = np.random.randn(200, n_features)
        target_test_preds = target_model.predict(test_data)
        surrogate_test_preds = surrogate_model.predict(test_data)

        agreement_rate = np.mean(target_test_preds == surrogate_test_preds)
        success_rate = agreement_rate
        vulnerability_score = min(1.0, agreement_rate * 1.2)

        privacy_budget_consumed = query_budget * 0.001

        attack_details = {
            "queries_made": query_budget,
            "model_agreement": float(agreement_rate),
            "extraction_accuracy": float(success_rate),
            "surrogate_model_type": "Random Forest",
            "feature_importance_stolen": True
        }

        defensive_measures = []
        mitigation_recommendations = [
            "Implement query rate limiting (max 100 queries/hour)",
            "Add prediction noise to model outputs",
            "Use model distillation to reduce information leakage",
            "Deploy differential privacy at inference time",
            "Monitor for suspicious query patterns"
        ]

        if query_budget > 500:
            mitigation_recommendations.append("CRITICAL: Detected high-volume querying indicative of extraction attack")

        report = AISecurityReport(
            model_id=model_id,
            attack_type="model_extraction",
            success_rate=success_rate,
            vulnerability_score=vulnerability_score,
            privacy_budget_consumed=privacy_budget_consumed,
            defensive_measures=defensive_measures,
            attack_details=attack_details,
            mitigation_recommendations=mitigation_recommendations
        )

        self.attack_history.append(report)
        return report

    async def run_model_inversion_attack(self, model_id: str,
                                       target_labels: List[int]) -> AISecurityReport:
        """Simulate model inversion attack to reconstruct genomic features"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        target_model = self.models[model_id]

        reconstructed_features = []
        reconstruction_quality = []

        for target_label in target_labels:
            # Gradient-based feature reconstruction
            n_features = 1000
            reconstructed = np.random.randn(n_features)

            # Simulate iterative optimization
            for iteration in range(100):
                # Get prediction for current reconstruction
                pred_proba = target_model.predict_proba(reconstructed.reshape(1, -1))

                # Simulate gradient update toward target label
                if np.argmax(pred_proba) == target_label:
                    break

                # Add noise to simulate gradient step
                reconstructed += np.random.randn(n_features) * 0.01

            reconstructed_features.append(reconstructed)

            # Calculate reconstruction quality (simulated)
            quality_score = np.random.uniform(0.3, 0.8)
            reconstruction_quality.append(quality_score)

        avg_quality = np.mean(reconstruction_quality)
        success_rate = avg_quality
        vulnerability_score = min(1.0, avg_quality * 1.1)

        privacy_budget_consumed = len(target_labels) * 0.02
        if target_model.differential_privacy_enabled:
            success_rate *= 0.4  # DP significantly reduces inversion success
            vulnerability_score *= 0.4

        attack_details = {
            "labels_targeted": len(target_labels),
            "avg_reconstruction_quality": float(avg_quality),
            "iterations_per_target": 100,
            "feature_leakage_risk": "High" if avg_quality > 0.6 else "Moderate",
            "differential_privacy_protection": target_model.differential_privacy_enabled
        }

        defensive_measures = []
        if target_model.differential_privacy_enabled:
            defensive_measures.append("Differential Privacy (reduces inversion by 60%)")

        mitigation_recommendations = [
            "Implement strong differential privacy (ε ≤ 0.01)",
            "Use output rounding/binning to reduce gradient information",
            "Employ model ensemble with prediction averaging",
            "Add calibrated noise to prediction confidences"
        ]

        if avg_quality > 0.7:
            mitigation_recommendations.append("URGENT: High inversion success indicates severe privacy risk")

        report = AISecurityReport(
            model_id=model_id,
            attack_type="model_inversion",
            success_rate=success_rate,
            vulnerability_score=vulnerability_score,
            privacy_budget_consumed=privacy_budget_consumed,
            defensive_measures=defensive_measures,
            attack_details=attack_details,
            mitigation_recommendations=mitigation_recommendations
        )

        self.attack_history.append(report)
        return report

    async def analyze_federated_learning_security(self, participant_count: int = 5,
                                                malicious_participants: int = 1) -> Dict[str, Any]:
        """Analyze security of federated genomic learning scenario"""

        # Simulate federated learning setup
        total_samples = 1000
        samples_per_participant = total_samples // participant_count

        participants = []
        for i in range(participant_count):
            is_malicious = i < malicious_participants
            participant = {
                "id": f"hospital_{i+1}",
                "samples": samples_per_participant,
                "is_malicious": is_malicious,
                "data_quality": "poisoned" if is_malicious else "clean",
                "privacy_budget": 1.0
            }
            participants.append(participant)

        # Simulate Byzantine attack impact
        clean_participants = participant_count - malicious_participants
        attack_success_probability = malicious_participants / participant_count

        # Calculate aggregation security
        if malicious_participants <= participant_count // 3:
            byzantine_resilience = "Secure"
            consensus_integrity = 0.95
        else:
            byzantine_resilience = "Vulnerable"
            consensus_integrity = 0.60

        # Privacy budget analysis
        total_privacy_budget = participant_count * 1.0
        budget_per_round = 0.1
        max_training_rounds = int(total_privacy_budget / budget_per_round)

        security_analysis = {
            "federation_setup": {
                "total_participants": participant_count,
                "malicious_participants": malicious_participants,
                "attack_success_probability": attack_success_probability,
                "byzantine_resilience": byzantine_resilience,
                "consensus_integrity": consensus_integrity
            },
            "privacy_analysis": {
                "total_privacy_budget": total_privacy_budget,
                "budget_per_round": budget_per_round,
                "max_training_rounds": max_training_rounds,
                "privacy_accounting": "Renyi Differential Privacy"
            },
            "participants": participants,
            "vulnerabilities": [
                "Data poisoning via malicious participants",
                "Model update inference attacks",
                "Gradient leakage during aggregation",
                "Membership inference across participants"
            ],
            "defensive_measures": [
                "Secure aggregation protocols",
                "Byzantine-robust averaging",
                "Differential privacy per participant",
                "Anomaly detection in model updates"
            ]
        }

        return security_analysis

    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive security dashboard data"""

        # Model security overview
        model_count = len(self.models)
        attack_count = len(self.attack_history)

        if attack_count > 0:
            avg_vulnerability = np.mean([r.vulnerability_score for r in self.attack_history])
            total_privacy_consumed = sum([r.privacy_budget_consumed for r in self.attack_history])
        else:
            avg_vulnerability = 0.0
            total_privacy_consumed = 0.0

        # Security metrics by attack type
        attack_types = {}
        for report in self.attack_history:
            attack_type = report.attack_type
            if attack_type not in attack_types:
                attack_types[attack_type] = []
            attack_types[attack_type].append(report.success_rate)

        attack_summary = {}
        for attack_type, success_rates in attack_types.items():
            attack_summary[attack_type] = {
                "attempts": len(success_rates),
                "avg_success_rate": float(np.mean(success_rates)),
                "max_success_rate": float(np.max(success_rates)),
                "risk_level": "High" if np.mean(success_rates) > 0.7 else "Medium" if np.mean(success_rates) > 0.4 else "Low"
            }

        # Privacy budget tracking
        privacy_status = {
            "total_consumed": float(total_privacy_consumed),
            "remaining_budget": max(0.0, 1.0 - total_privacy_consumed),
            "budget_status": "Critical" if total_privacy_consumed > 0.8 else "Warning" if total_privacy_consumed > 0.5 else "Healthy"
        }

        return {
            "overview": {
                "models_analyzed": model_count,
                "attacks_simulated": attack_count,
                "avg_vulnerability_score": float(avg_vulnerability),
                "overall_risk_level": "High" if avg_vulnerability > 0.7 else "Medium" if avg_vulnerability > 0.4 else "Low"
            },
            "attack_analysis": attack_summary,
            "privacy_budget": privacy_status,
            "recent_attacks": [
                {
                    "attack_type": r.attack_type,
                    "success_rate": r.success_rate,
                    "vulnerability_score": r.vulnerability_score,
                    "model_id": r.model_id
                }
                for r in self.attack_history[-5:]  # Last 5 attacks
            ]
        }