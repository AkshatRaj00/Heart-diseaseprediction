## Clinical Change Summary
Briefly describe what changes are made and the clinical or algorithmic justification.

## Pre-Merge Clinical Verification Checklist
- [ ] No hardcoded API keys or secrets in commit history.
- [ ] Pydantic schemas allow edge vitals (Asystole $BP \le 20$, $HR \le 10$).
- [ ] 60-Minute Prognosis and IV Pump titration math validated against ACC/AHA guidelines.
- [ ] Code formatted with PEP8 (Python) and Prettier (Next.js).
- [ ] All automated multi-bot checks (CodeQL, Dependabot, Scorecard) passed.
