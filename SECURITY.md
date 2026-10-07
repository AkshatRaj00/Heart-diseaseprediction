# Security & Clinical Data Protection Policy

## Supported Versions
Security patches and vulnerability mitigations are actively maintained on the latest stable branch.

| Version | Supported          |
| ------- | ------------------ |
| 11.0.x  | :white_check_mark: |
| 10.x.x  | :x:                |
| < 10.0  | :x:                |

## Reporting Clinical & Code Vulnerabilities
We follow a coordinated disclosure model. If you discover a vulnerability affecting patient data, inference integrity, or telemetry interception:

1. **Do NOT open a public GitHub issue.**
2. Send an advisory to: `security@onepersonai.com` or initiate a Private Security Advisory via GitHub's **Security Advisories** tab.
3. Include:
   - Vector description (e.g., API authentication bypass, zero-day Pydantic deserialization flaw, corrupted HL7/FHIR payload injection).
   - Minimal reproducible proof of concept.
4. Response SLA: We acknowledge submissions within **24 hours** and aim for mitigation within **72 hours**.

## Regulatory Compliance
- **Zero Local PHI Storage**: By default, no Protected Health Information (PHI) is permanently persisted in disk cache unless configured with an encrypted enterprise PostgreSQL instance.
- **Payload Encryption**: All real-time telemetry streaming enforces TLS 1.3 in transit.
