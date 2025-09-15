#!/usr/bin/env python3
# API test suite for the health AI security platform

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def log_test(self, test_name: str, success: bool, details: str = ""):
        result = "✅ PASS" if success else "❌ FAIL"
        print(f"{result} {test_name}")
        if details:
            print(f"    {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })

    def test_health_check(self):
        try:
            response = self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200
            data = response.json() if success else {}
            details = f"Status: {response.status_code}, Response: {data}"
            self.log_test("Health Check", success, details)
            return success
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False

    def test_api_docs(self):
        try:
            # Test docs endpoint
            docs_response = self.session.get(f"{self.base_url}/docs")
            docs_success = docs_response.status_code == 200

            # Test OpenAPI spec
            openapi_response = self.session.get(f"{self.base_url}/openapi.json")
            openapi_success = openapi_response.status_code == 200

            success = docs_success and openapi_success
            details = f"Docs: {docs_response.status_code}, OpenAPI: {openapi_response.status_code}"
            self.log_test("API Documentation", success, details)
            return success
        except Exception as e:
            self.log_test("API Documentation", False, f"Exception: {str(e)}")
            return False

    def test_ai_security_dashboard(self):
        try:
            response = self.session.get(f"{self.base_url}/ai/security-dashboard")
            success = response.status_code == 200

            if success:
                data = response.json()
                has_required_fields = all(key in data for key in ["status", "data", "message"])
                dashboard_data = data.get("data", {})
                has_overview = "overview" in dashboard_data
                success = has_required_fields and has_overview
                details = f"Response structure valid: {success}"
            else:
                details = f"Status: {response.status_code}"

            self.log_test("AI Security Dashboard", success, details)
            return success
        except Exception as e:
            self.log_test("AI Security Dashboard", False, f"Exception: {str(e)}")
            return False

    def test_create_ai_model(self, model_id: str = "test_model_001"):
        test_cases = [
            {
                "name": "Standard Model",
                "data": {
                    "model_id": f"{model_id}_standard",
                    "model_type": "ancestry_classifier",
                    "use_differential_privacy": False,
                    "epsilon": 1.0
                }
            },
            {
                "name": "DP-Protected Model",
                "data": {
                    "model_id": f"{model_id}_dp",
                    "model_type": "disease_predictor",
                    "use_differential_privacy": True,
                    "epsilon": 0.1
                }
            },
            {
                "name": "Drug Response Model",
                "data": {
                    "model_id": f"{model_id}_drug",
                    "model_type": "drug_response",
                    "use_differential_privacy": True,
                    "epsilon": 0.01
                }
            }
        ]

        all_success = True
        for case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/ai/models",
                    json=case["data"]
                )
                success = response.status_code == 200

                if success:
                    data = response.json()
                    model_data = data.get("data", {})
                    has_model_id = model_data.get("model_id") == case["data"]["model_id"]
                    has_privacy_setting = model_data.get("differential_privacy_enabled") == case["data"]["use_differential_privacy"]
                    success = has_model_id and has_privacy_setting
                    details = f"Model created with correct settings: {success}"
                else:
                    details = f"Status: {response.status_code}, Error: {response.text}"

                self.log_test(f"Create {case['name']}", success, details)
                all_success = all_success and success

                # Store model ID for later tests
                if success:
                    setattr(self, f"test_model_{case['data']['model_type']}", case["data"]["model_id"])

            except Exception as e:
                self.log_test(f"Create {case['name']}", False, f"Exception: {str(e)}")
                all_success = False

        return all_success

    def test_membership_inference_attack(self):
        # Need a model first
        model_id = getattr(self, 'test_model_ancestry_classifier', 'test_model_001_standard')

        test_cases = [
            {
                "name": "Small Sample Attack",
                "data": {
                    "model_id": model_id,
                    "attack_type": "membership_inference",
                    "target_samples": 25,
                    "query_budget": 500
                }
            },
            {
                "name": "Large Sample Attack",
                "data": {
                    "model_id": model_id,
                    "attack_type": "membership_inference",
                    "target_samples": 100,
                    "query_budget": 1000
                }
            }
        ]

        all_success = True
        for case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/ai/attacks/membership-inference",
                    json=case["data"]
                )
                success = response.status_code == 200

                if success:
                    data = response.json()
                    attack_data = data.get("data", {})
                    has_metrics = "metrics" in attack_data
                    has_success_rate = "success_rate" in attack_data.get("metrics", {})
                    has_recommendations = "mitigation_recommendations" in attack_data
                    success = has_metrics and has_success_rate and has_recommendations

                    if success:
                        success_rate = attack_data["metrics"]["success_rate"]
                        details = f"Attack success rate: {success_rate:.1%}"
                    else:
                        details = "Missing required response fields"
                else:
                    details = f"Status: {response.status_code}, Error: {response.text}"

                self.log_test(f"Membership Inference - {case['name']}", success, details)
                all_success = all_success and success

            except Exception as e:
                self.log_test(f"Membership Inference - {case['name']}", False, f"Exception: {str(e)}")
                all_success = False

        return all_success

    def test_federated_security_analysis(self):
        test_cases = [
            {
                "name": "Small Federation",
                "data": {
                    "participants": 3,
                    "malicious_participants": 0,
                    "privacy_budget_per_round": 0.1,
                    "total_rounds": 10
                }
            },
            {
                "name": "Large Federation with Attackers",
                "data": {
                    "participants": 10,
                    "malicious_participants": 2,
                    "privacy_budget_per_round": 0.05,
                    "total_rounds": 20
                }
            },
            {
                "name": "High Privacy Budget",
                "data": {
                    "participants": 5,
                    "malicious_participants": 1,
                    "privacy_budget_per_round": 0.2,
                    "total_rounds": 15
                }
            }
        ]

        all_success = True
        for case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/ai/federated/security-analysis",
                    json=case["data"]
                )
                success = response.status_code == 200

                if success:
                    data = response.json()
                    fed_data = data.get("data", {})
                    has_federation_setup = "federation_setup" in fed_data
                    has_privacy_analysis = "privacy_analysis" in fed_data
                    has_healthcare_risks = "healthcare_specific_risks" in fed_data
                    has_compliance = "compliance_framework" in fed_data
                    success = all([has_federation_setup, has_privacy_analysis,
                                 has_healthcare_risks, has_compliance])

                    if success:
                        byzantine_resilience = fed_data["federation_setup"]["byzantine_resilience"]
                        gdpr_compliant = fed_data["compliance_framework"]["gdpr_compliant"]
                        details = f"Byzantine: {byzantine_resilience}, GDPR: {gdpr_compliant}"
                    else:
                        details = "Missing required analysis components"
                else:
                    details = f"Status: {response.status_code}, Error: {response.text}"

                self.log_test(f"Federated Analysis - {case['name']}", success, details)
                all_success = all_success and success

            except Exception as e:
                self.log_test(f"Federated Analysis - {case['name']}", False, f"Exception: {str(e)}")
                all_success = False

        return all_success

    def test_error_handling(self):
        error_tests = [
            {
                "name": "Invalid Model Type",
                "endpoint": "/ai/models",
                "method": "POST",
                "data": {
                    "model_id": "invalid_test",
                    "model_type": "invalid_type",
                    "use_differential_privacy": False
                },
                "expected_status": 422  # Validation error
            },
            {
                "name": "Missing Model ID for Attack",
                "endpoint": "/ai/attacks/membership-inference",
                "method": "POST",
                "data": {
                    "attack_type": "membership_inference",
                    "target_samples": 50
                },
                "expected_status": 422  # Validation error
            },
            {
                "name": "Attack on Non-existent Model",
                "endpoint": "/ai/attacks/membership-inference",
                "method": "POST",
                "data": {
                    "model_id": "nonexistent_model",
                    "attack_type": "membership_inference",
                    "target_samples": 50
                },
                "expected_status": 404  # Not found
            },
            {
                "name": "Invalid Federation Config",
                "endpoint": "/ai/federated/security-analysis",
                "method": "POST",
                "data": {
                    "participants": 5,
                    "malicious_participants": 10,  # More malicious than total
                    "privacy_budget_per_round": 0.1
                },
                "expected_status": 400  # Bad request
            }
        ]

        all_success = True
        for test in error_tests:
            try:
                if test["method"] == "POST":
                    response = self.session.post(
                        f"{self.base_url}{test['endpoint']}",
                        json=test["data"]
                    )
                else:
                    response = self.session.get(f"{self.base_url}{test['endpoint']}")

                success = response.status_code == test["expected_status"]
                details = f"Expected {test['expected_status']}, got {response.status_code}"

                self.log_test(f"Error Handling - {test['name']}", success, details)
                all_success = all_success and success

            except Exception as e:
                self.log_test(f"Error Handling - {test['name']}", False, f"Exception: {str(e)}")
                all_success = False

        return all_success

    def run_all_tests(self):
        print("🧬 Security for Health AI Platform - API Test Suite")
        print("=" * 60)

        # Test API availability
        if not self.test_health_check():
            print("❌ API is not available. Exiting tests.")
            return False

        # Run all test categories
        tests = [
            ("API Documentation", self.test_api_docs),
            ("AI Security Dashboard", self.test_ai_security_dashboard),
            ("AI Model Creation", self.test_create_ai_model),
            ("Membership Inference Attack", self.test_membership_inference_attack),
            ("Federated Security Analysis", self.test_federated_security_analysis),
            ("Error Handling", self.test_error_handling)
        ]

        print("\n📋 Running Test Categories:")
        print("-" * 30)

        category_results = []
        for category_name, test_func in tests:
            print(f"\n🔬 Testing {category_name}...")
            success = test_func()
            category_results.append((category_name, success))

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

        # Category Summary
        print(f"\n📋 Category Results:")
        for category, success in category_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {category}")

        # Failed tests detail
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")

        overall_success = failed_tests == 0
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

        return overall_success

def main():
    print("Starting API tests...")

    # Wait for API to be ready
    print("Waiting for API to start...")
    time.sleep(2)

    tester = APITester()
    success = tester.run_all_tests()

    # Exit code for CI/CD
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()