
# Revision Suggestions for Crowd Control Manuscript

## 1. Overall Assessment

The revised manuscript shows **major improvement in unit consistency, physical interpretability, and simulation scaling**, especially:
- Four-channel scenario now includes nominal physical scaling
- Bund scenario includes map-derived spatial scaling
- Most raw metrics are now mapped to `ped`, `ped/s`, `ped/m²`, or simulation-equivalent units

However, several **remaining issues in consistency, interpretation, and completeness** still exist.

---

## 2. Key Remaining Issues

### 2.1 Inconsistent interpretation of "simulation mass unit"

Problem:
- Four-channel: 1 mass unit = 4 ped
- Bund: 1 mass unit ≈ 195 ped

Issue:
- Two different scales exist without a unified explanation
- Readers may think model is inconsistent or recalibrated per experiment

Fix:
- Explicitly state scenario-dependent scaling
- Clarify mass is not globally fixed

Recommended text:
> “Each scenario uses its own nominal physical scaling; simulation mass is a scenario-dependent pedestrian-equivalent unit rather than a universal constant.”

---

### 2.2 CCT / RMT inconsistency (important)

Problem:
- CCT defined as grid-cell-time but reported in m²·s in some places
- RMT defined as mass-time but unit presentation inconsistent

Fix:
- Standardize:
  - RMT → ped·s
  - CCT → rename to CAT (counterflow area-time) → m²·s

---

### 2.3 Average density interpretation issue (Bund experiment)

Problem:
- Average density ≈ 3e-3 ped/m² is extremely small

Issue:
- Not incorrect mathematically, but weak interpretability

Fix options:
- Replace with:
  - 95th percentile density OR
  - occupied-cell mean density
- OR explicitly state domain averaging over empty space

---

### 2.4 Entrance flow = 0 in uncontrolled case

Problem:
- Table shows zero values

Issue:
- Misleading interpretation

Fix:
- Replace with N/A or “not applicable”
- Add note: no internal metering interface active

---

### 2.5 Four-channel scaling ambiguity

Problem:
- Length/time/mass mapping is not fully tied together

Fix:
Add explicit consistency check:
- ρmax, grid area, and mass unit must be coherent

---

### 2.6 ped vs ped-equivalent inconsistency

Fix:
- Use “ped-eq.” everywhere in tables
- Define once:
> simulation mass = pedestrian-equivalent continuous quantity

---

## 3. Missing Unit Coverage

### 3.1 Threshold density ρ_safe
- Must specify unit (ped/m²)
- Must confirm consistency across scenarios

### 3.2 Control variables
- q_c(t): ped/s
- A_c(t): ped/s
- κ: clarify dimension (1/s or dimensionless)

### 3.3 Objective units
Add summary:
- J1: ped·s
- J2: m²·s
- J3: ped²
- J4: ped·s
- J5: (ped/s)²

---

## 4. Figures Still Missing Unit Labels

- Fig. 6: right axis must explicitly show ped or ped-eq
- Fig. 8: y-axis must say “dimensionless objective J”
- Fig. 9: ensure ped/s explicitly stated

---

## 5. Terminology Consistency

Must enforce:

- passage → real-world geometry
- channel → model entity (indexed c)
- walkway → global spatial domain

Add explicit definition section recommended

---

## 6. Strengths of Current Version

✔ Strong normalization framework  
✔ Clear separation of mechanism vs optimization vs transfer  
✔ Good abstraction of Bund scenario  
✔ Improved interpretability of HCMBO results  

---

## 7. Final Recommendation

### High priority fixes
- CCT/RMT unit consistency
- ped vs ped-eq unification
- zero-flow interpretation
- Bund density interpretation

### Medium priority
- ρ_safe clarification
- full unit table
- scaling derivation clarity

### Optional polish
- Replace average density metric
- minor caption improvements

---

## 8. Summary

The manuscript is now **methodologically solid and near publishable**, but still requires:

- strict unit discipline
- clearer physical interpretability layer
- consistent scenario-dependent scaling explanation
