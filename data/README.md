# Data provenance

Raw datasets are NOT committed (sizes). Fetch instructions and citation requirements:

## Primary: CIC-UNSW-NB15 (2024 refresh) — DOWNLOADED 2026-08-18
- Page: https://www.unb.ca/cic/datasets/cic-unsw-nb15.html
- Download form: http://cicresearch.ca//CICDataset/CIC-UNSW/ (link delivered by email after submission)
- Delivered as CIC-UNSW.zip (454 MB), contents verified 2026-08-18:
  - Dataset.csv: 447,915 flows x 76 numeric CICFlowMeter features (IDs removed)
  - Label.csv: integer labels 0-9; 0 = benign (358,332), classes 1-8 = attacks,
    class 9 = 246 rows (extreme minority); binary y = (label != 0) -> 89,583 attacks
  - CICFlowMeter.csv: raw 84-column version incl. IPs/timestamps — provenance only, NOT loaded
- Load validated 2026-08-18 (1.7 s): splits 313,541/67,187/67,187; feature provenance
  split = 23 timing/infrastructure + 53 attacker-controllable (see src/s1/poison.py)
- License: free redistribution/reuse WITH citation (CIC FAQ)
- Required citation: Mohammadian, Lashkari, Ghorbani, "Poisoning and Evasion: Deep
  Learning-Based NIDS under Adversarial Attacks", PST 2024

## Hot spare: NF-ToN-IoT (use ONLY if G1 fails on Aug 20)
- Kaggle (no form): https://www.kaggle.com/datasets/dhoogla/nftoniot
- Official: UQ eSpace UQ:2fa2ed6 (v1), UQ:38a2d07 + https://researchdata.edu.au/nf-ton-iot-v2/ (v2)
- Cite: Sarhan, Layeghy, Moustafa, Portmann (2021), "NetFlow Datasets for Machine
  Learning-Based Network Intrusion Detection Systems"

## Attack harness reference
- PCAP-Backdoor (arXiv:2501.15563): backdoor generator for network traffic, <=1% poison budget
