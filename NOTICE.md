# Notice - Forensic Use

This project is released under the [MIT License](LICENSE). This notice does not
modify, restrict, or add conditions to that grant. It states the intended scope
of use, because this software produces material that may be placed before a
court.

## What this software is

Forensic decision support. It assists a human examiner; it does not replace one,
and it does not determine the admissibility or the evidentiary weight of any
exhibit, which are matters for the Court.

Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023 requires a certificate
signed by a person in charge of the device and by an expert. This software
generates *inputs* for that human expert. It signs nothing, and the Part-B sheet
it produces is watermarked **DRAFT - REQUIRES EXPERT REVIEW AND SIGNATURE**.

## What the MIT grant does and does not cover

The permission grant in [LICENSE](LICENSE) covers copying, modification, and
distribution of the source code. It is expressly not a warranty of fitness for
any evidentiary, investigative, or judicial purpose - the MIT text disclaims all
warranties, and that disclaimer is load-bearing here.

## Conditions of responsible use

- No output of this software should be presented as an expert opinion unless a
  qualified examiner has independently reviewed and signed it.
- Automated detection is probabilistic. Absence of detected manipulation does
  not establish authenticity.
- Findings are conditional on the declared validated domain. An exhibit outside
  that domain is reported as inconclusive, and that result must be carried
  forward rather than re-run until a conclusion appears.
- A conclusion whose Evidence Fragility Index is HIGH is one that ordinary
  recompression can erase. The system force-abstains in that case; do not
  override it.

## Third-party components

Model architectures and forensic methodology draw on prior published work,
credited in the [README](README.md#-prior-art) and reproduced on the generated
report's Methods page. The fingerprint paradigm is not original to this project.
