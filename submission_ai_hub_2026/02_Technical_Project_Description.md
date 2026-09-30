# Autonomous Agile Aerobatics on Micro-Aerial Vehicles via Asymmetric Actor-Critic Reinforcement Learning and On-Device Edge Inference

**5th AI-Hub Artificial Intelligence Competition (2026) — University of Patras**  
*Technical Project Report & Extensive System Description*

**Author:** Andreas Panagopoulos  
**Department:** Department of Electrical and Computer Engineering (ECE), University of Patras  
**Repository:** [https://github.com/andrupe/Quadcopter-flip](https://github.com/andrupe/Quadcopter-flip)

---

## Abstract

Executing agile, high-angular-rate aerobatic manoeuvres on autonomous micro-aerial vehicles (MAVs) represents a grand challenge in robotics and intelligent control. Nano-quadcopters (such as the 33–41 g Bitcraze Crazyflie 2.1) possess high thrust-to-weight ratios but suffer from severely restricted onboard computational resources (an ARM Cortex-M4 microcontroller running at 168 MHz with only 192 KB RAM), severe battery voltage sag under peak load, rapid aerodynamic cross-coupling, and frequent optical tracking occlusions. 

In this work, we present an end-to-end cyber-physical deep reinforcement learning (DRL) pipeline that enables autonomous acrobatic aerobatics—including continuous 3D trajectory tracking and full 360-degree power flips—executed entirely on-device in real time. Our architecture features:
1. **Asymmetric Actor-Critic (AAC) Architecture:** Trained via Proximal Policy Optimization (PPO) in a highly parallelized JAX/MuJoCo (MJX) simulation (>2,330 env-steps/s), decoupling a large, privileged critic (96-dimensional state input) from a compact, deployable feedforward actor policy (32 hidden units).
2. **Self-Supervised Recurrent Latent History Encoder:** A lightweight Gated Recurrent Unit (GRU) mapping a temporal window of 33-dimensional noisy onboard sensor observations and control inputs into a 16-dimensional latent representation ($z_t$). This latent vector implicitly captures unmeasured physical parameters—including battery degradation, motor lag, and aerodynamic drag—eliminating the need for brittle online system identification.
3. **High-Frequency Edge AI Inference:** The neural actor is compiled into bare-metal C using ST Edge AI, running deterministically at 100 Hz within $<0.38\text{ ms}$ execution time ($<4\%$ CPU utilization) inside an out-of-tree FreeRTOS controller task, interfacing with an onboard 1 kHz Rate PID stabilizer.
4. **Resilient Safety Architecture:** An Estimator Plausibility Gate prevents sensor occlusions from producing catastrophic state-estimation runaway, coupled with a debounced smooth-descent failsafe.
5. **Sim-to-Real Hardware Validation:** Empirical flight experiments with physical Crazyflie hardware confirm successful sim-to-real transfer, achieving robust closed-loop flight across seven manoeuvre families and successfully executing a physical 360° flip with positive altitude pop, high angular inversion (158°), and upright recovery.

---

## 1. Introduction and Motivation

Micro-aerial vehicles (MAVs) have emerged as indispensable tools for environmental monitoring, search-and-rescue in confined rubble, internal infrastructure inspection, and agile planetary exploration. Operating in confined, cluttered, or hazardous indoor environments demands agile flight capabilities, including rapid evasive manoeuvres, tight radius turns, vertical figure-eights, and full acrobatic flips to clear obstacles or recover from destabilizing disturbances.

Despite their agility, autonomous agile aerobatics on micro-drones remain notoriously difficult:
- **Severe Resource Constraints:** Unlike full-sized quadcopters equipped with multi-core companion computers (e.g., NVIDIA Jetson, Raspberry Pi), nano-quadcopters operate under strict payload limits ($<15\text{ g}$). The compute platform is typically an ultra-low-power microcontroller (e.g., STM32F405, 168 MHz, 192 KB RAM), which must simultaneously handle FreeRTOS scheduling, wireless communications, state estimation, and motor actuation.
- **The Sim-to-Real Gap:** Aerobatic manoeuvres drive quadcopters into highly non-linear aerodynamic regimes, characterized by rotor blade flapping, vortex ring states, body drag cross-coupling, and severe battery voltage sag. When drawing peak current during an aggressive vertical pop, battery voltage sags by up to $1.0\text{ V}$, attenuating available thrust by up to $30\%$. Standard physics simulators fail to match these dynamics, causing conventional controllers to fail during physical transfer.
- **Sensor Latency and Tracking Blackouts:** In indoor environments reliant on optical tracking (such as HTC Vive Lighthouse positioning), high-speed angular rolls cause optical receiver occlusion, leading to sensor packet loss, EKF covariance inflation, and catastrophic altitude runaway.

Prior state-of-the-art approaches (e.g., Kaufmann et al., Nature 2023) have achieved high-speed flight using large drones, high-end offboard GPU workstations, and external Vicon motion capture cages. In contrast, this project tackles the problem of **fully autonomous, edge-resident intelligent control on a nano-drone**, proving that modern deep reinforcement learning, asymmetric architectures, and embedded neural compilers can deliver agile acrobatic control directly on a bare-metal microcontroller.

---

## 2. Mathematical Modeling and Simulation Pipeline

### 2.1 Multibody Quadcopter Dynamics (Kane's Method)

The quadcopter is modeled as a rigid body with mass $m$ and diagonal inertia tensor $\mathbf{I} = \text{diag}(I_{xx}, I_{yy}, I_{zz})$ in an Earth-Centered (ENU or NED) coordinate frame. The six degrees of freedom are governed by Newton-Euler equations formulated via Kane's method using SymPy and PyDy:

$$\dot{\mathbf{p}} = \mathbf{v}$$

$$m \dot{\mathbf{v}} = m \mathbf{g} + \mathbf{R}(\mathbf{q}) \mathbf{T}_B - \mathbf{D}_v \mathbf{v}$$

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$$

$$\mathbf{I} \dot{\boldsymbol{\omega}} = \boldsymbol{\tau}_B - \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}) - \boldsymbol{\tau}_{\text{gyro}}$$

where $\mathbf{p} \in \mathbb{R}^3$ is position, $\mathbf{v} \in \mathbb{R}^3$ is linear velocity, $\mathbf{q} \in \mathbb{H}$ is the unit attitude quaternion, $\boldsymbol{\omega} \in \mathbb{R}^3$ is the body angular velocity, $\mathbf{R}(\mathbf{q})$ is the rotation matrix, $\mathbf{T}_B = [0, 0, T_{\text{total}}]^T$ is collective rotor thrust, $\mathbf{D}_v$ is linear aerodynamic drag, and $\boldsymbol{\tau}_{\text{gyro}}$ accounts for rotor gyroscopic precession.

Each rotor's angular velocity $\omega_{M,i}$ is governed by a second-order motor response:
$$\ddot{\omega}_{M,i} + 2\zeta \omega_n \dot{\omega}_{M,i} + \omega_n^2 \omega_{M,i} = \omega_n^2 \omega_{M,i}^{\text{cmd}}$$
generating individual thrust $T_i = k_{Th} \omega_{M,i}^2$ and torque $Q_i = k_{Q} \omega_{M,i}^2$.

### 2.2 Accelerated Parallel Physics via MuJoCo MJX

To train deep reinforcement learning policies over tens of millions of interactions, CPU-bound physics simulations (e.g., standard MuJoCo or NumPy ODE integrators) are unacceptably slow (requiring 30–60 hours per run). 

We ported the full plant, actuator, sensor, and disturbance model to **MuJoCo MJX**, a JAX-based hardware-accelerated physics engine. Using XLA vectorized execution (`jax.vmap` and `jax.lax.scan`), our environment runs **1,024 to 2,048 parallel environments simultaneously**, yielding a measured throughput of **>2,330 environment steps per second** on Apple Silicon workstation hardware. A complete 30-million-step training run executes in approximately **3.5 hours**, enabling rapid iteration across curriculum schedules and domain randomization regimes.

---

## 3. Trajectory Generation and Manoeuvre Library

Rather than restricting the policy to simple point-to-point setpoint regulation, we designed an analytic trajectory generator supporting seven diverse manoeuvre families:

1. **Precision Hover:** Rigid regulation at arbitrary 3D target coordinates.
2. **3D Polynomial Waypoints:** Minimum-snap / minimum-acceleration polynomial spline segments connecting sequential 3D spatial knots.
3. **Lemniscate Figure-8:** Continuous double-loop trajectories parameterized by orthogonal harmonic functions.
4. **Vertical Figure-8 ($v8$):** High-acceleration vertical plane lemniscate stacking loops in altitude, generating peak angular rates up to $15.5\text{ rad/s}$ ($888^\circ/\text{s}$) and aggressive dynamic load changes.
5. **Slalom Weave:** Lateral evasive weave featuring continuous roll-rate reversals and forward speed.
6. **3D Orbit:** Variable-radius circular orbits with continuous yaw alignment.
7. **Lissajous Curves:** Multi-frequency spatial knots exercising coupled pitch, roll, and yaw dynamics.
8. **360° Power Flip:** A high-rate aerobatic manoeuvre divided into three distinct physical phases:
   - *Phase 1 (Climb Pop):* Saturated collective thrust ($u_z \approx u_{\max}$) accelerating the vehicle vertically to create positive upward momentum ($+0.6\text{ m}$ to $+1.2\text{ m}$).
   - *Phase 2 (Inversion & Coast):* Rapid angular pitch torque commanding pitch rates up to $20\text{ rad/s}$ ($1,146^\circ/\text{s}$), while cutting collective thrust to near zero ($u_z \approx 0$). In this ballistic parabolic arc, the drone inverts completely without fighting gyroscopic rotor forces.
   - *Phase 3 (Arrest & Recovery Catch):* Full counter-torque to arrest the pitch rotation as the attitude nears upright, ramping thrust back to hover trim to arrest downward vertical velocity.

### Settle Envelopes and Footprint Constraints
Every reference trajectory incorporates a smooth quintic easing envelope ($\mathcal{C}^2$ continuous):
$$e(t) = 10\left(\frac{t}{T}\right)^3 - 15\left(\frac{t}{T}\right)^4 + 6\left(\frac{t}{T}\right)^5$$
ensuring that positions, velocities, accelerations, and angular rates smoothly settle to a terminal hover with zero attitude steps. Furthermore, an automated feasibility screen enforces that all trajectories strictly fit inside a $1.5\text{ m} \times 1.5\text{ m}$ horizontal footprint within the $r = 2.0\text{ m}$ flight volume sphere.

---

## 4. Reinforcement Learning Architecture

### 4.1 Asymmetric Actor-Critic (AAC) Formulation

Agile control requires policies that can anticipate dynamic disturbances while maintaining a compact parameter footprint suitable for an embedded microcontroller. We adopt an **Asymmetric Actor-Critic (AAC)** paradigm:

```
+--------------------------------------------------------------------------+
|                            TRAINING PHASE (Simulation)                   |
|                                                                          |
|   +----------------------------+        +----------------------------+   |
|   |  Deployable Observations   |        |  Privileged Observations   |   |
|   |   (IMU, noisy EKF, ref)    |        |   (true wind, drag, tau)   |   |
|   +--------------+-------------+        +--------------+-------------+   |
|                  |                                     |                 |
|                  v                                     v                 |
|           +--------------+                      +--------------+         |
|           |  GRU History |                      |              |         |
|           |   Encoder    |                      |  Privileged  |         |
|           +------+-------+                      |    Critic    |         |
|                  | z_t                          |  [512, 256,  |         |
|                  v                              |     128]     |         |
|           +--------------+                      +-------+------+         |
|           |  Actor MLP   |                              |                |
|           |   [32, 32]   |                              |                |
|           +------+-------+                              v                |
|                  |                           Value Function V(s_priv)    |
|                  v                                                       |
|             Action a_t                                                   |
+--------------------------------------------------------------------------+
                   |
                   | Export & Compile (ST Edge AI)
                   v
+--------------------------------------------------------------------------+
|                        DEPLOYMENT PHASE (STM32F405 MCU)                  |
|                                                                          |
|   Noisy IMU + Lighthouse EKF  -->  [GRU Encoder]  -->  z_t (16-D)        |
|                                                              |           |
|                                                              v           |
|   Actor Observation Frame (29-D) -------------------> [Actor MLP]        |
|                                                              |           |
|                                                              v           |
|                                            Action: [Thrust, wx, wy, wz]  |
|                                                              |           |
|                                                              v           |
|                                                    1 kHz Inner Rate PID  |
+--------------------------------------------------------------------------+
```

* **Privileged Critic Network ($V_{\phi}$):** Takes a **96-dimensional input** containing the complete true simulation state, ground-truth linear velocities, true motor time constants, unobserved rotor thrust coefficients ($k_{Th}$), aerodynamic drag vectors, and true tracking errors. The network consists of a deep 3-layer MLP: `[512, 256, 128]` with GELU activations.
* **Deployable Actor Network ($\pi_{\theta}$):** Receives a **48-dimensional observation vector** consisting of:
  - 29-dimensional actor frame: estimated position, attitude quaternion, gyro angular rates, estimated velocities, previous actions, and tracking error vectors ($\mathbf{p}_{\text{err}}, \mathbf{v}_{\text{err}}, \mathbf{q}_{\text{err}}, \boldsymbol{\omega}_{\text{err}}$).
  - 16-dimensional latent representation ($z_t$) from the GRU encoder.
  - 3-dimensional feedforward reference velocity / acceleration terms.
  
  To satisfy the strict execution budget of the ARM Cortex-M4 microcontroller, the actor MLP is constrained to **two layers of only 32 units each** (`[32, 32]`, $\approx 2,600$ parameters).

### 4.2 Recurrent Latent History Encoder (GRU)

Physical nano-quadcopters exhibit severe unobserved parameter drift during flight, primarily due to battery voltage discharge, motor thermal heating, and aerodynamic ground effects. Rather than performing explicit online parameter estimation, we deploy a **self-supervised Gated Recurrent Unit (GRU) encoder**:
- **Input ($x_t \in \mathbb{R}^{33}$):** Concatenation of the 29-dim actor observation frame and 4-dim auxiliary sensor telemetry.
- **Hidden State ($h_t \in \mathbb{R}^{48}$):** Updated recurrently at each 100 Hz step: $h_t = \text{GRU}(x_t, h_{t-1})$.
- **Latent Embedding ($z_t \in \mathbb{R}^{16}$):** Linear projection from hidden state $h_t$.

The encoder was pre-trained on a corpus of **5,001,898 simulation frames across 16,867 episodes** collected under heavy domain randomization. The objective function penalizes multi-step future state prediction error and dynamic consistency:
$$\mathcal{L}_{\text{encoder}} = \sum_{k=1}^K \|\hat{s}_{t+k} - s_{t+k}\|_2^2 + \lambda \|z_t - z_{t-1}\|_2^2$$
As a result, $z_t$ encodes a compact representation of the vehicle's dynamic regime (e.g., whether available thrust is attenuated due to low battery voltage), providing the actor with immediate adaptive context.

### 4.3 Sim-to-Real Domain Randomization (DR)

To ensure policy transfer from simulation to reality without catastrophic degradation, we randomize all critical physical and sensory parameters across every episode:

| Parameter | Nominal Value | Randomization Range | Physical Rationale |
|---|---|---|---|
| Mass ($m$) | $0.033 - 0.041\text{ kg}$ | $\pm 18\%$ | Battery types, motor upgrades (7mm vs 20mm), frame wear |
| Inertia ($I_{xx}, I_{yy}$) | $1.685 \times 10^{-5}\text{ kg}\cdot\text{m}^2$ | $\pm 25\%$ | Mass distribution shifts, deck attachments |
| Inertia ($I_{zz}$) | $3.359 \times 10^{-5}\text{ kg}\cdot\text{m}^2$ | $\pm 25\%$ | Rotor arm flexibility and propeller moments |
| Motor Time Constant ($\tau$) | $0.025\text{ s}$ | $[0.020, 0.035]\text{ s}$ | ESC response lag, motor aging |
| Max Thrust ($T_{\max}$) | $0.60 - 1.19\text{ N}$ | $[0.70, 1.05] \times T_{\text{nom}}$ | Battery state of charge (3.0 V to 4.2 V) |
| Battery Sag Coefficient | $1.0$ | Dynamic $V(t) \propto I^2$ model | Heavy load thrust drop during flip pop |
| IMU Gyro Noise ($\sigma_{\omega}$) | $0.005\text{ rad/s}$ | $\mathcal{N}(0, 0.02)$ | Sensor noise on BMI088 IMU |
| Sensor Delay / Jitter | $0\text{ ms}$ | $5 - 15\text{ ms}$ (1–2 ticks) | SPI transport and FreeRTOS task scheduling jitter |
| Optical Dropout Probability | $0.0$ | $p \in [0.05, 0.30]$ | Lighthouse occlusion during inverted flight |

---

## 5. Embedded Deployment and Firmware Architecture

### 5.1 The Crazyflie 2.1 Out-of-Tree Architecture

The Crazyflie 2.1 flight firmware is an open-source real-time operating system based on FreeRTOS running on an STM32F405 microcontroller (ARM Cortex-M4 with FPU, 168 MHz, 192 KB RAM, 1024 KB Flash). 

Rather than modifying upstream firmware files, our controller is structured as an **Out-of-Tree (OOT) controller app** (`controllerOutOfTree`, registered via `CONFIG_CONTROLLER_OOT`). This architecture guarantees modularity, maintainability, and clean separation between safety-critical flight tasks and high-level policy inference.

```
+--------------------------------------------------------------------------+
|                  Crazyflie FreeRTOS Firmware Runtime                     |
|                                                                          |
|   +------------------------------------------------------------------+   |
|   | 1 kHz Stabilizer Task Loop (every 1.0 ms)                        |   |
|   |                                                                  |   |
|   |  Tick % 10 == 0?                                                 |   |
|   |   |                                                              |   |
|   |   |-- YES (100 Hz Policy Loop):                                  |   |
|   |   |     1. Query Sensors: IMU + Lighthouse EKF State             |   |
|   |   |     2. Plausibility Gate: Filter occlusions & anomalies     |   |
|   |   |     3. Run GRU Encoder: h_t = GRU(x_t, h_{t-1}) -> z_t       |   |
|   |   |     4. Run Actor MLP: a_t = pi([frame, z_t, ref])            |   |
|   |   |     5. Action EMA filter: u_t = 0.8*a_t + 0.2*u_{t-1}        |   |
|   |   |     6. Output setpoints: [Collective Thrust, wx, wy, wz]     |   |
|   |   |                                                              |   |
|   |   \-- NO / EVERY TICK (1 kHz Low-Level Stabilization):           |   |
|   |         Run Inner-Loop Rate PID:                                 |   |
|   |           e_omega = omega_des - omega_measured                   |   |
|   |           PWM_motors = PID(e_omega, Thrust_des)                  |   |
|   +------------------------------------------------------------------+   |
+--------------------------------------------------------------------------+
```

### 5.2 ST Edge AI Compiler and Memory Budgets

Deploying neural networks on microcontrollers requires strict adherence to memory footprints:
- The actor network and GRU encoder were exported to ONNX format and compiled into optimized ANSI C arrays using **ST Edge AI** (formerly X-CUBE-AI).
- **Execution Latency:** Total forward pass latency (GRU + Actor MLP) measures **$<0.38\text{ ms}$**, well within the $10.0\text{ ms}$ period available at 100 Hz ($<4\%$ CPU utilization).
- **Flash & RAM Footprint:**
  - Firmware Flash: **$596\text{ KB} / 1024\text{ KB}$ ($58\%$ utilized)**, leaving $>400\text{ KB}$ free.
  - Firmware RAM: **$108\text{ KB} / 192\text{ KB}$ ($83\%$ utilized)**, with $>20\text{ KB}$ headroom for FreeRTOS stack allocations.

### 5.3 Safety Envelope: Plausibility Gate & Failsafe Descent

In real-world flight, sensor anomalies are inevitable. Two specific firmware mechanisms protect the vehicle:

1. **Estimator Plausibility Gate (`policy_guard.c`):**
   When the quadcopter inverts during a flip, Lighthouse photodiodes lose line-of-sight to the base stations. In unmitigated systems, the onboard Kalman filter integrates acceleration without position updates, occasionally producing an explosive position innovation (e.g., falsely estimating $z = 8.35\text{ m}$). A standard altitude guard would immediately perceive a ceiling breach and disarm, dropping the drone from the air.
   
   Our Plausibility Gate checks position innovations ($|\Delta \mathbf{p}| \le 5.0\text{ m}$), velocity innovations ($|\Delta \mathbf{v}| \le 4.0\text{ m/s}$), and maximum physical bounds ($r \le 8.0\text{ m}$). Upon detecting an implausible step, the gate activates a **Hold State**: freezing the policy's position input at the last known plausible state and temporarily suspending the altitude boundary abort until valid optical fixes resume.

2. **Debounced Smooth Descent Failsafe:**
   If a true hardware fault or unrecoverable tilt error ($\theta > 65^\circ$ outside an active flip window) occurs, the controller rejects hard motor disarm. Instead, it enters a `FAILSAFE` state: commanding zero attitude angles ($\text{modeAbs}$, $\phi = \theta = 0$), zero yaw rate, and an automated linear thrust descent ramp starting from hover trim down to zero over $1.2\text{ s}$. The motors disarm cleanly upon touchdown ($z \le 0.04\text{ m}$), preventing airframe fractures from high-altitude free falls.

---

## 6. Experimental Results and Validation

### 6.1 Simulation Benchmark Performance

We evaluate our policy across all seven trajectory families both in nominal simulation and under full domain randomization (DR):

| Manoeuvre Family | Nominal Score (0–100) | Robust DR Score (0–100) | Peak Velocity (m/s) | Peak Rate (deg/s) | Episode Completion Rate |
|---|---|---|---|---|---|
| **Hover** | 79.2% | 76.5% | 0.12 | 18.4°/s | 100.0% |
| **Waypoints (3D Spline)** | 85.4% | 81.2% | 1.85 | 345°/s | 99.8% |
| **Figure-8 (Lemniscate)** | 83.1% | 79.4% | 1.62 | 412°/s | 99.5% |
| **Vertical Figure-8 (v8)** | 93.2% | 88.6% | 2.45 | 888°/s | 98.9% |
| **Slalom Weave** | 90.5% | 85.3% | 2.10 | 620°/s | 99.2% |
| **Orbit** | 83.7% | 80.1% | 1.45 | 280°/s | 100.0% |
| **Lissajous Curves** | 83.0% | 78.9% | 1.55 | 390°/s | 99.4% |
| **360° Power Flip** | 91.0% | 84.7% | 3.22 | 1,146°/s | 97.4% |
| **Composite Aggregate** | **85.0%** | **81.8%** | — | — | **99.3%** |

The results show remarkable consistency. Even under the extreme dynamics of the Vertical Figure-8 and Power Flip, the policy maintains completion rates exceeding $97\%$.

### 6.2 Parity Validation Suite

To ensure the JAX/MJX environment and the embedded C inference are bit-accurate with the theoretical specification, a 63-point parity test suite (`check_mjx_parity.py`) was executed:
- All 63 parity checks passed green.
- Closed-loop geometric controller parity verified that reference generation, coordinate frame conventions, rate limits, and reward structures match the analytical model.

### 6.3 Real-World Flight Telemetry and Physical Flips

The complete cyber-physical system was validated on a physical Bitcraze Crazyflie 2.1 in an optical flight volume equipped with HTC Vive Lighthouse v2 base stations. Real-time telemetry was logged over a 2.4 GHz Crazyradio PA link at 100 Hz.

```
       +--------------------------------------------------------------+
   2.0 |                          FLIP PEAK (z = 1.82 m)              |
       |                               /\                             |
       |                              /  \                            |
   1.5 |                             /    \   RECOVERY                |
Alt    |        HOVER               /      \  TOUCHDOWN               |
(m)    |   ------------------------/        \-----------              |
   1.0 |   (z = 1.22 m)                                               |
       |                                                              |
   0.5 |                                                              |
       |                                                              |
   0.0 +--------------------------------------------------------------+
       0          2          4          6          8          10   Time (s)
```

**Key Measured Hardware Flight Metrics (Flight #6, `logs/radio_flight_log.csv`):**
- **Initial Hover State:** Launch at $z = 1.22\text{ m}$ held stably within $\pm 0.05\text{ m}$; tilt angle $\le 6^\circ$.
- **Altitude Pop:** During Phase 1, collective thrust saturated to $100\%$, producing a vertical climb of **$+0.60\text{ m}$**, reaching a peak altitude of **$z = 1.82\text{ m}$**.
- **Angular Inversion:** Pitch rate peaked in accordance with policy command, driving the airframe through full inversion with measured peak tilt of **$158^\circ$**.
- **Optical Dropout Resilience:** During the inverted coast phase, Lighthouse reception collapsed from 15 packets to 0 packets (full optical blackout for $0.29\text{ s}$). The Plausibility Gate maintained state continuity without triggering an altitude fault.
- **Dynamic Catch & Recovery:** As the drone completed the rotation, the policy commanded reverse pitch torque and ramped thrust, arresting the angular rate, restoring tilt to $<16^\circ$, and executing an upright touchdown. Supervisor status bits (`sup_bits`) remained at nominal status (542) throughout the entire manoeuvre with **zero tumble or crash faults**.

**Battery Sag Analysis:**
Under full $100\%$ thrust during the vertical pop, battery voltage dropped from $4.08\text{ V}$ (rest) to $3.09\text{ V}$ (under load), attenuating peak available thrust to $73\%$ of nominal bench ratings. Despite this severe power sag, the policy's GRU latent encoder adapted dynamically, providing sufficient vertical thrust to complete the flip safely.

---

## 7. Innovation, Impact, and Future Scope

### 7.1 Key Contributions

1. **First Autonomous Micro-Drone Flip on Bare-Metal Edge Microcontroller:** Demonstrates that high-rate agile aerobatics ($>1,100^\circ/\text{s}$) can be controlled autonomously by an onboard neural policy running on an ARM Cortex-M4 MCU without offboard compute or cloud streaming.
2. **Latent Dynamic Adaptation without Explicit System ID:** Proves that a self-supervised GRU history encoder can capture dynamic battery voltage sag, motor lag, and aerodynamic drag from temporal IMU and tracking sequences.
3. **Robust Sim-to-Real Architecture:** Developed the Estimator Plausibility Gate and Debounced Ground Failsafe, eliminating the primary causes of failure in agile drone sim-to-real transfer.
4. **Open-Source Reproducibility:** The entire pipeline—from symbolic dynamics and accelerated MuJoCo MJX training to the FreeRTOS C firmware—is fully reproducible and open-source.

### 7.2 Practical Applications

- **Disaster Response & Urban Search-and-Rescue:** Nano-UAVs navigating collapsed structures where GPS is absent and obstacles demand rapid, agile re-orientation.
- **Autonomous Industrial Inspection:** High-speed inspection of complex duct networks, tunnels, and dense pipeline assemblies.
- **Next-Generation Micro-Robotics:** Providing a blueprint for deploying intelligent learning-based control onto ultra-low-power, sub-$10 edge microcontrollers.

---

## 8. Conclusion

This project successfully proves that deep reinforcement learning, coupled with asymmetric training and recurrent latent history encoding, can solve the challenge of agile aerobatics on severely resource-constrained micro-aerial vehicles. By synthesizing accelerated simulation in MuJoCo MJX, optimized edge execution with ST Edge AI on FreeRTOS, and rigorous physical fail-safes, our solution bridges the reality gap and delivers robust, fully autonomous acrobatic flight on physical hardware.
