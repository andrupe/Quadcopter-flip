# 5th AI-Hub Artificial Intelligence Competition (2026)
## University of Patras
### Project Submission Form / Έντυπο Υποβολής Εργασίας

---

### 1. General Project Information / Γενικά Στοιχεία Εργασίας

* **Project Title (English):**  
  **Autonomous Agile Aerobatics on Micro-Aerial Vehicles via Asymmetric Actor-Critic Reinforcement Learning and On-Device Edge Inference**

* **Project Title (Greek / Ελληνικά):**  
  **Αυτόνομη Ευέλικτη Ακροβατική Πτήση Μικρο-Τετρακοπτέρων μέσω Ενισχυτικής Μάθησης Ασύμμετρου Δράστη-Κριτή και Εκτέλεσης Νευρωνικών Δικτύων σε Ενσωματωμένους Μικροελεγκτές (Edge AI)**

* **Scientific Domain / Θεματική Περιοχή:**  
  Artificial Intelligence (AI), Deep Reinforcement Learning (DRL), Cyber-Physical Systems & Robotics, Embedded Edge AI, Sim-to-Real Transfer, Agile Aerial Navigation.

* **Team Representative / Εκπρόσωπος Ομάδας:**  
  * **Full Name / Ονοματεπώνυμο:** Andreas Panagopoulos (Ανδρέας Παναγόπουλος)  
  * **Department / Τμήμα:** Electrical and Computer Engineering (Τμήμα Ηλεκτρολόγων Μηχανικών και Τεχνολογίας Υπολογιστών - ΤΜΗΥΠ)  
  * **Institution / Ίδρυμα:** University of Patras (Πανεπιστήμιο Πατρών)  
  * **Student Status / Ιδιότητα:** Undergraduate Student (Προπτυχιακός Φοιτητής)  
  * **Email:** a.panagopsch@gmail.com  
  * **Academic Email / Ακαδημαϊκό Email:** *(Insert upatras email, e.g., upXXXXXXX@upnet.gr)*  
  * **Student ID / Αριθμός Μητρώου (ΑΜ):** *(Insert AM)*

* **Team Members / Μέλη Ομάδας:**  
  * Andreas Panagopoulos (Undergraduate, Dept. of ECE, University of Patras)  
  *(Add any additional team members if applicable, including Name, Dept, AM, Email)*

* **Project Code Repository:**  
  [https://github.com/andrupe/Quadcopter-flip](https://github.com/andrupe/Quadcopter-flip)

---

### 2. Project Summary & Executive Abstract / Περίληψη Εργασίας

Executing high-rate, aggressive aerobatic manoeuvres (such as full 360-degree power flips, vertical lemniscates, and high-g slaloms) on ultra-lightweight micro-aerial vehicles (MAVs) represents an open, formidable challenge in robotics and artificial intelligence. Conventional cascaded linear controllers fail during extreme envelope excursions because of severe non-linear aerodynamic cross-coupling, actuator saturation, optical tracking occlusions, and severe battery voltage sag under peak load. 

This project designs, trains, and physically deploys an end-to-end intelligent control framework for a resource-constrained nano-quadcopter (Bitcraze Crazyflie 2.1, 33–41 g). Our solution integrates:
1. **Asymmetric Actor-Critic (AAC) Reinforcement Learning:** Trained using Proximal Policy Optimization (PPO) in a highly parallelized, GPU/vector-accelerated MuJoCo MJX physics environment running at >2,330 steps/second. The critic leverages privileged simulation state (unobserved ground-truth wind, drag, and thrust degradation), while the actor receives only deployable observations.
2. **Recurrent Latent History Encoding:** A Gated Recurrent Unit (GRU) encoder maps temporal histories of noisy IMU measurements, optical tracking updates, and previous actions into a compact 16-dimensional latent representation ($z_t$). This latent vector implicitly captures unmodeled dynamics, battery degradation, and rotor time constants without requiring explicit system identification.
3. **Sim-to-Real Domain Randomization (DR):** Comprehensive randomization of vehicle mass ($\pm 20\%$), inertia, motor time constants ($\tau \in [20, 35]\text{ ms}$), battery discharge scaling ($3.0 - 4.2\text{ V}$), and synthetic Lighthouse optical tracking blackouts/occlusions.
4. **Embedded Edge AI Execution on Bare-Metal Microcontroller:** The trained neural network policy (compact MLP with 32 units) and GRU are exported and compiled directly into optimized C code using ST Edge AI for an onboard STM32F405 ARM Cortex-M4 microcontroller (168 MHz, 192 KB RAM). The policy executes at 100 Hz within $<0.38\text{ ms}$ latency ($<4\%$ CPU utilization), commanding collective thrust and body rates into an onboard 1 kHz Rate PID stabilizer.
5. **Robust Safety Architecture:** An Estimator Plausibility Gate filters optical tracking sensor dropouts and prevents state estimation divergence from inducing motor shut-off, complemented by a debounced failsafe descent module.
6. **Physical Experimental Validation:** Full end-to-end hardware deployment, successfully executing live acrobatic flips, vertical figure-eights, and trajectory tracking in the physical flight arena with wireless telemetry logging at 100 Hz.

---

### 3. Evaluation Criteria Alignment / Αντιστοίχιση με τα Κριτήρια Αξιολόγησης

#### (α) Originality of Idea / Πρωτοτυπία Ιδέας
* **Autonomous Aerobatic Micro-Flips on Edge Hardware:** While large research institutions (e.g., ETH Zurich, University of Zurich) have demonstrated drone flips using multi-camera motion capture rooms and offboard ground station computation, our project achieves autonomous flip execution and recovery **entirely onboard** a sub-50g nano-quadcopter using an edge microcontroller.
* **Implicit Latent System Identification:** Instead of brittle, hand-tuned model parameters, our architecture employs a self-supervised GRU history encoder that generates a 16-dimensional latent representation of the vehicle's dynamic state, adapting in real time to motor degradation, payload shifts, and battery voltage sag.
* **Asymmetric Privileged Learning for Compact Deployment:** By training a large, privileged critic in simulation while constraining the deployable actor to a lightweight feedforward topology (32 neurons), we decouple training capacity from MCU inference constraints.

#### (β) Originality of Implementation / Πρωτοτυπία Υλοποίησης
* **Parallel Physics Acceleration (MJX/JAX):** Re-engineered the training stack into JAX-based MuJoCo MJX, accelerating simulation throughput to >2,330 steps/sec on workstation hardware and completing 30 million training steps in under 3.5 hours (a >20x speedup over legacy frameworks).
* **Bare-Metal ST Edge AI Integration:** Converted the trained neural policy to quantized, fixed-memory C arrays executed via an out-of-tree FreeRTOS controller (`controllerOutOfTree`) inside the Crazyflie firmware. Inference runs deterministically at 100 Hz with microsecond-level timing guarantees.
* **Estimator Plausibility Gate & Debounced Failsafe:** Developed a custom filter in firmware that detects Lighthouse optical tracking packet loss and rejects phantom altitude innovations (e.g., preventing runaway altitude estimates from cutting motor thrust mid-flip).

#### (γ) Degree of Completion / Βαθμός Ολοκλήρωσης
* **End-to-End Cyber-Physical System:** The project spans from first-principles symbolic mechanics (PyDy / Kane's equations) and simulator design to RL training, C firmware integration, flashing, and real flight testing.
* **7 Diverse Flight Manoeuvre Families:** Systematically benchmarked across Hover, 3D Polynomial Waypoints, Lemniscate Figure-8, Vertical Figure-8, Slalom Weave, Orbit, Lissajous curves, and 360° Power Flips.
* **Empirical Flight Verification:** The physical Crazyflie 2.1 has flown real test flights, captured via 100 Hz telemetry (`logs/radio_flight_log.csv`, `live_flight_telemetry.png`), demonstrating successful power flips with positive altitude pop (+0.60 m climb, 158° inversion, and upright landing).

#### (δ) Impact & Applications / Αντίκτυπο & Πρακτική Σημασία
* **Advancement of Ultra-Constrained Edge Robotics:** Proves that deep neural network policies can run reliably on low-power, sub-$10 microcontrollers without cloud or offboard compute.
* **Disaster Response & Confined Inspection:** Agile MAVs capable of rapid evasive aerobatics and resilient flight under optical occlusion can navigate collapsed structures, pipelines, and GPS-denied environments.
* **Open Science Contribution:** Fully open-source codebase, comprehensive test suites, reproducible training scripts, and firmware manifests available for the academic and open-source robotics community.

---

### 4. Περιεχόμενα Αρχείου Συμπληρωματικού Υλικού (AI_Hub_2026_Submission_Quadcopter_Flip.zip)

| Α/Α | ΠΕΡΙΕΧΟΜΕΝΑ ΑΡΧΕΙΟΥ ΣΥΜΠΛΗΡΩΜΑΤΙΚΟΥ ΥΛΙΚΟΥ |
|---|---|
| **1** | **`paper.pdf`** : Τεχνικό άρθρο / paper 14 σελίδων όπου περιγράφεται αναλυτικά η μαθηματική μοντελοποίηση, η ενισχυτική μάθηση σε JAX/MuJoCo MJX, ο κωδικοποιητής GRU, η ενσωμάτωση στον μικροελεγκτή (ST Edge AI σε C) και τα πειραματικά αποτελέσματα πτήσης. |
| **2** | **`presentation.pdf`** : Παρουσίαση 14 διαφανειών (slides) του project με τα τεχνικά διαγράμματα, την αρχιτεκτονική ελέγχου και τις πειραματικές μετρήσεις. |
| **3** | **`SCRIPT_8MIN.md`** : Πλήρες σενάριο ομιλίας (~8 λεπτών) με αναλυτικές οδηγίες και επεξηγήσεις ανά διαφάνεια για την παρουσίαση της εργασίας. |
| **4** | **`02_Technical_Project_Description.md`** : Εκτενής επιστημονική και τεχνική περιγραφή της εργασίας, αναλυτική αντιστοίχιση με τα κριτήρια αξιολόγησης και τεκμηρίωση του συστήματος. |
| **5** | **`03_Evaluation_Criteria_Alignment_and_Media_Guide.md`** : Οδηγός αντιστοίχισης κριτηρίων αξιολόγησης και λεπτομερής επεξήγηση των πειραματικών διαγραμμάτων. |
| **6** | **`figures/`** : Φάκελος με όλα τα πρωτότυπα γραφήματα υψηλής ανάλυσης (τηλεμετρία πραγματικής πτήσης 100 Hz `live_flight_telemetry.png`, καμπύλες εκπαίδευσης 30M βημάτων `training_curves_mjx.png`, 3D τροχιές 7 οικογενειών ελιγμών `flight_trajectories_mjx.png`, απόκριση PID `rate_tracking_plot.png`, διανυσματικό πολυσέλιδο `telemetry_plots.pdf`) και φωτογραφίες της διάταξης (Crazyflie 2.1, Lighthouse deck, κινητήρες 16mm vs 20mm, μπαταρία LiPo). |
| **7** | **`code_artifacts/`** : Βασικός εκτελέσιμος πηγαίος κώδικας του ενσωματωμένου ελεγκτή σε C (`controller_app.c`, `policy_guard.c`, `analytic_flip.c`), καθώς και το περιβάλλον προσομοίωσης/εκπαίδευσης σε Python/JAX (`quad_flip_env.py`) και το manifest εξαγωγής της πολιτικής (`policy_export.json`). |
| **8** | **`evaluation_metrics/`** : Αρχεία δεδομένων CSV με τις μετρικές αξιολόγησης της πολιτικής (`eval_metrics_mjx.csv`, `eval_robust_mjx.csv`) σε ονομαστικές συνθήκες και υπό domain randomization σε 1.000 επεισόδια ανά οικογένεια ελιγμών. |
| **9** | **`participant_documents/`** : Βιογραφικό σημείωμα του συμμετέχοντα Ανδρέα Παναγόπουλου (`CV_Andreas_Panagopoulos.pdf`) και συνοδευτικά έγγραφα φοιτητικής ιδιότητας. |
| **10** | **`paper.tex`** : Ο πλήρης πηγαίος κώδικας LaTeX του επιστημονικού άρθρου για πλήρη επαληθευσιμότητα και αναπαραγωγιμότητα. |
