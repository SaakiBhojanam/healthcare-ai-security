# Health AI Security Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-100%25-brightgreen.svg)](test_api.py)

A security testing platform for healthcare AI models, focusing on privacy attacks, federated learning vulnerabilities, and compliance assessment.

Built to analyze the security of genomic AI models in healthcare environments, with support for differential privacy and multi-institutional federated learning scenarios.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
cd backend && python3 simple_main.py

# Run tests to verify everything works
python3 test_api.py
```

The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) once the server is running.

---

## Features

### AI Model Security Testing
- **Membership Inference Attacks** - Test whether specific individuals were in the training data
- **Model Extraction Attacks** - Attempt to steal model functionality through queries
- **Model Inversion Attacks** - Try to reconstruct training data from model outputs
- **Differential Privacy** - Evaluate privacy protection mechanisms

### Federated Learning Security
- **Multi-Institution Scenarios** - Simulate collaborative learning between hospitals
- **Byzantine Attack Analysis** - Test resilience against malicious participants
- **Privacy Budget Tracking** - Monitor cumulative privacy loss across training rounds
- **Compliance Assessment** - Check adherence to healthcare regulations (HIPAA, GDPR)

### Security Analysis
- **Vulnerability Scoring** - Quantitative assessment of model privacy risks
- **Attack Success Metrics** - Realistic success rates for different attack types
- **Mitigation Recommendations** - Actionable suggestions for improving security

## Testing

The project includes a comprehensive test suite that validates all API endpoints:

```bash
python3 test_api.py
```

All 15 tests pass, covering model creation, security attacks, federated analysis, and error handling.

## License

MIT License - see [LICENSE](LICENSE) file for details.
