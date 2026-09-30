# Evaluation Criteria Alignment & Media Guide
## 5th AI-Hub Artificial Intelligence Competition (2026) — University of Patras

**Project:** Autonomous Agile Aerobatics on Micro-Aerial Vehicles via Asymmetric Actor-Critic Reinforcement Learning and On-Device Edge Inference  
**Author:** Andreas Panagopoulos (Dept. of ECE, University of Patras)  
**Repository:** [https://github.com/andrupe/Quadcopter-flip](https://github.com/andrupe/Quadcopter-flip)

---

## 1. Direct Alignment with the 4 Evaluation Criteria

| Evaluation Criterion | Project Achievement & Justification | Verified Evidence in Repository |
|---|---|---|
| **(α) Πρωτοτυπία Ιδέας**<br>*(Originality of Idea)* | • **Onboard Autonomous Micro-Flips:** While flips on larger drones have traditionally relied on offboard workstations and mocap cameras, this project achieves fully autonomous aerobatic flips running **onboard** a 33–41 g nano-drone.<br>• **Asymmetric Actor-Critic (AAC) with Privileged Learning:** Decouples privileged simulation training from micro-footprint execution (32 units).<br>• **Latent Representation of Unmodeled Dynamics:** Uses a self-supervised GRU encoder to implicitly infer motor lag, aerodynamic drag, and battery voltage sag without explicit system ID. | • `Simulation/asymmetric_policy.py`<br>• `Simulation/encoder/history_encoder.py`<br>• `Simulation/deploy/manifests/policy_export.json` |
| **(β) Πρωτοτυπία Υλοποίησης**<br>*(Originality of Implementation)* | • **MuJoCo MJX Parallel Acceleration:** Accelerated training to **>2,330 env-steps/s** in JAX, completing 30M steps in ~3.5 hours on workstation hardware.<br>• **Edge AI Bare-Metal Deployment:** Compiles neural policies via ST Edge AI into deterministic C running at 100 Hz on STM32F405 ARM Cortex-M4 (<0.38 ms latency, <4% CPU load).<br>• **Estimator Plausibility Gate:** Firmware-level filter that eliminates optical tracking blackout runaway during inverted flight.<br>• **Debounced Failsafe:** Graceful controlled-descent ramp preventing ballistic crashes. | • `Simulation/quad_mjx/`<br>• `Simulation/deploy/app_policy_controller/src/controller_app.c`<br>• `Simulation/deploy/app_policy_controller/src/policy_guard.c`<br>• `Simulation/deploy/build_app.sh` |
| **(γ) Βαθμός Ολοκλήρωσης**<br>*(Degree of Completion)* | • **Full End-to-End Stack:** Covers symbolic PyDy equations, JAX/MJX simulation, domain randomization, multi-family trajectory generation, FreeRTOS C firmware, and real hardware deployment.<br>• **7 Trajectory Families:** Benchmark scores averaging **85.0%** across Hover, 3D Waypoints, Figure-8, Vertical Figure-8, Slalom, Orbit, Lissajous, and Power Flip.<br>• **Empirical Physical Flight Tests:** Physical Crazyflie 2.1 flights captured at 100 Hz over Crazyradio PA, successfully demonstrating physical power flips (+0.60 m climb, 158° inversion, and upright landing). | • `logs/flight_log_review_20260925.md`<br>• `logs/radio_flight_log.csv`<br>• `live_flight_telemetry.png`<br>• `rate_tracking_plot.png`<br>• `logs/eval_metrics_mjx_prod_best.csv` |
| **(δ) Αντίκτυπο**<br>*(Impact)* | • **Pioneering Edge AI for Micro-Robotics:** Demonstrates that modern DRL is viable on ultra-low-power, sub-$10 microcontrollers.<br>• **Real-World MAV Applications:** Enhances emergency search-and-rescue and industrial inspection in confined, GPS-denied environments.<br>• **Open-Source Reproducibility:** Clean modular code, rigorous test suites, and open documentation for research dissemination. | • [GitHub Repository](https://github.com/andrupe/Quadcopter-flip)<br>• `Simulation/deploy/README.md`<br>• `RETRAIN_AUDIT.txt` |

---

## 2. Guide to Visual Figures & Experimental Media

The accompanying `figures/` directory contains high-resolution empirical plots supporting the submission:

### 1. `live_flight_telemetry.png` & `telemetry_plots.pdf`
- **What it shows:** 100 Hz real-time flight telemetry logged over the Crazyradio PA wireless link during physical flight on the Crazyflie 2.1 nano-quadcopter.
- **Key Takeaways:** 
  - Subplot 1 (Altitude $z$): Shows stable hover regulation at $z = 1.22\text{ m}$, followed by the vertical pop reaching $z = 1.82\text{ m}$ ($+0.60\text{ m}$ climb), through inverted free-fall coast, and smooth recovery touchdown.
  - Subplot 2 (Attitude / Tilt): Measures attitude inversion reaching $158^\circ$, followed by rapid counter-torque settling back to upright flight ($<16^\circ$).
  - Subplot 3 (Battery Voltage): Demonstrates significant load sag ($4.08\text{ V} \to 3.09\text{ V}$), successfully compensated for by the policy's latent adaptation.

### 2. `flight_trajectories_mjx.png`
- **What it shows:** 3D spatial trajectories and tracking performance across all seven benchmark manoeuvre families evaluated in simulation.
- **Key Takeaways:** Demonstrates smooth tracking across Hover, Waypoints, Lemniscate Figure-8, Vertical Figure-8, Slalom Weave, Orbit, and Lissajous curves, strictly complying with the $1.5\text{ m} \times 1.5\text{ m}$ cage footprint.

### 3. `training_curves_mjx.png`
- **What it shows:** Convergence curves over 30 million timesteps in the accelerated JAX MuJoCo MJX environment.
- **Key Takeaways:** Shows steady actor-critic policy reward improvement, robust domain randomization curriculum ramp, and entropy decay.

### 4. `rate_tracking_plot.png`
- **What it shows:** Frequency and time-domain tracking of commanded body rates versus measured gyro feedback in the 1 kHz onboard Rate PID loop.
- **Key Takeaways:** Confirms minimal phase lag and tight rate control up to $20\text{ rad/s}$ ($1,146^\circ/\text{s}$).

---

## 3. Checklist for Submitting via eClass AI-Hub

To submit before the **30/09/2026** deadline:
1. Log in to the **AI-Hub eClass** page at University of Patras.
2. Navigate to the **"Εργασίες" (Assignments)** section and open the **"5ος Διαγωνισμός ΤΝ (2026)"** assignment.
3. Complete the online submission fields using the text from `01_AI_Hub_Submission_Form.md`.
4. In the `participant_documents/` folder, ensure your:
   - **Curriculum Vitae (CV)**
   - **Certificate of Student Status (Βεβαίωση Φοιτητικής Ιδιότητας)**
   are placed.
5. Upload the generated zip file:  
   **`AI_Hub_2026_Submission_Quadcopter_Flip.zip`**
6. Review confirmation receipt on eClass.
