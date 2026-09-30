# Video Presentation Script: Autonomous Agile Aerobatics on Micro Quadcopters
## Side-by-Side Spoken Script for Video Recording & Presentation (13 Slides)
**Author:** Andreas Panagopoulos  
**Department:** Electrical & Computer Engineering, University of Patras  
**Accompanying Slide Deck:** `presentation/slides.pdf` (13 Widescreen 16:9 Slides)  
**Total Target Duration:** 7:30 – 7:45 (Strict limit: Max 8:00)  
**Pacing:** ~130 words per minute (Natural, conceptual, conversational delivery)

---

## Timing Overview & Slide Map

| Slide # | Slide Title | Start Time | End Time | Duration | Target Words |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Slide 1** | Title & Overview | 0:00 | 0:30 | 0:30 | ~70 words |
| **Slide 2** | Motivation: An Extreme Control & Engineering Challenge | 0:30 | 1:05 | 0:35 | ~85 words |
| **Slide 3** | Experimental Setup: The Bitcraze Crazyflie 2.1 Platform | 1:05 | 1:40 | 0:35 | ~85 words |
| **Slide 4** | Experimental Setup: Lighthouse Tracking & Inversion Dilemma | 1:40 | 2:15 | 0:35 | ~85 words |
| **Slide 5** | Project Timeline: From Simulation Struggles to Hardware | 2:15 | 2:55 | 0:40 | ~95 words |
| **Slide 6** | Simulation Parallelization: JAX Rewrite & Training Dynamics | 2:55 | 3:30 | 0:35 | ~85 words |
| **Slide 7** | Controller Architecture: Actor-Critic, Asymmetry & GRU Rationale | 3:30 | 4:25 | 0:55 | ~125 words |
| **Slide 8** | Embedded Deployment: Deterministic Execution on FreeRTOS | 4:25 | 5:00 | 0:35 | ~80 words |
| **Slide 9** | Sim-to-Real Transfer: Overcoming the Physical Reality Gap | 5:00 | 5:40 | 0:40 | ~95 words |
| **Slide 10** | Firmware Safeguards & Motor Transfer Adaptation | 5:40 | 6:20 | 0:40 | ~95 words |
| **Slide 11** | Experimental Flight Results: Hardware Flight Log Telemetry | 6:20 | 7:00 | 0:40 | ~100 words |
| **Slide 12** | Looking Ahead: Technical Improvements & Practical Use Cases | 7:00 | 7:30 | 0:30 | ~70 words |
| **Slide 13** | Project Summary: Autonomous Edge AI Control | 7:30 | 7:45 | 0:15 | ~40 words |
| **TOTAL** | | **0:00** | **7:45** | **~7:45** | **~1,050 words** |

---

## Full Word-for-Word Spoken Script

> **Speaker Note:** Keep this document open side-by-side with your presentation. Do not read the bullet points directly off the slides. Use this script to explain concepts intuitively so that a master's student appreciates the depth and a professor can follow effortlessly even without prior Crazyflie experience.

---

### [0:00 - 0:30] Slide 1: Title & Overview
*(Visual on screen: Slide 1 showing title, author Andreas Panagopoulos, University of Patras, and repository link)*

> "Hello everyone. My name is Andreas Panagopoulos, from the Department of Electrical and Computer Engineering at the University of Patras.
> 
> Today, I'm presenting my project: **Autonomous Agile Aerobatics on Micro Quadcopters via Deep Reinforcement Learning and Onboard Embedded Inference**.
> 
> In this project, I investigated whether a micro quadcopter,the crazyflie 2.1, can perform aerobatics specifically flips using neural networks trained with reinfrocmeent learning, running entirely onboard.

---

### [0:30 - 1:05] Slide 2: Motivation: An Extreme Control & Engineering Challenge
*(Action: [ADVANCE TO SLIDE 2])*  
*(Visual on screen: Slide 2 with Why Agile Micro Quadcopters on left and Primary Engineering Hurdle on right)*

> "To understand why this is useful and what technical difficulties I faced, we first consider where these drones operate. Small quadcopters like the Crazyflie 2.1 are safe to fly around humans and are usually flown indoors. They are well suited for exploring confined environments where traditional positioning, like gps, isn't available.
> 
> From a control perspective, their low rotational inertia allows extreme angular acceleration—exceeding 1,100 degrees per second. But they are also underactuated: four fixed propellers only push up relative to the drone's body. To turn or stop, the entire vehicle must tilt.
> 
> Agile aerobatic maneuvers are exceptionally difficult on micro quadcopters: they push flight dynamics beyond linear, involving violent rotational rates, severe aerodynamic drag, and minimal thrust control authority during inverted flight. My goal was to start from sensing, continue to state estimation, and finally perform inference, onboard a microcontroller.


---

### [1:05 - 1:40] Slide 3: Experimental Setup: The Bitcraze Crazyflie 2.1 Platform
*(Action: [ADVANCE TO SLIDE 3])*  
*(Visual on screen: Slide 3 showing the palm-sized Crazyflie hardware and embedded microcontroller constraints)*


> The crazyflie is an open-source mini quadcopter weighing only around 30 grams. It uses four brushed coreless DC motors powered by a small 1-cell LiPo battery.
> The difference between the crazyflie and a Full-sized drone is that full sized drones usually carry a significant computer running Linux, like a Raspberry Pi. Here, everything runs on a single STM32 microcontroller. That chip has to handle radio communications, IMU sensor decoding, state estimation, motor PWM generation, and our neural policy inference simultaneously under FreeRTOS."

---

### [1:40 - 2:15] Slide 4: Experimental Setup: Lighthouse Tracking & Inversion Dilemma
*(Action: [ADVANCE TO SLIDE 4])*  
*(Visual on screen: Slide 4 showing how Lighthouse sweeps work on left and the inverted tracking blackout on right)*

> "For indoor positioning, the setup uses the Lighthouse optical tracking system.
> 
> Four base stations mounted in the room sweep rotating infrared laser lines across space. Mounted on top of the Crazyflie is a lightweight sensor with four optical photodiodes. The drone measures pulse arrival times, solves angles, and feeds them into an onboard Extended Kalman Filter to calculate a somewhat accurate, but noisy 3D position in real time.
> 
> However, there is a catch during aerobatics. The sensors sit on the top deck and when the drone flips upside down past 90 degrees tilt, the sensors point toward the floor and lose line of sight to the base stations. Without optical updates, standard Kalman filters integrate noisy IMU data, hallucinate altitude jumps, and cause standard controllers to shut off motors mid-air."

---

### [2:15 - 2:55] Slide 5: Project Timeline: From Simulation Struggles to Hardware
*(Action: [ADVANCE TO SLIDE 5])*  
*(Visual on screen: Slide 5 showing the 4 project phases on left and simulation roadblocks & solutions on right)*

> "Looking at the project timeline, getting to a working flight on real hardware was a long journey that started with significant struggles in simulation.
> 
> The first major breakthrough was rebuilding the entire project around continuous unit quaternions, eliminating singularities caused by euler angles limitations and across all orientations and handling ambiguities that have to do with the double state representation of quaternions, where we would do a 350 degree rotation needlessly, where you could have done a 10 degree rotation.
> 
> Furthermore, on trajectory generation I had initially tried pure reward shaping to get a flip, but that was unstable during learning. I designed a geometric controller that generates feasible reference trajectories on the fly across multiple curve families, from hover and waypoints to acrobatic flips."

---

### [2:55 - 3:30] Slide 6: Simulation Parallelization: JAX Rewrite & Training Dynamics
*(Action: [ADVANCE TO SLIDE 6])*  
*(Visual on screen: Slide 6 showing JAX parallelization on left and 30M-step training curves on right)*

> "To make reinforcement learning practical, I needed to parallelize environment rollouts. I rewrote a primative simulation enviornment in JAX, Jax is a computational framework that compiles python code.
> 
> If we look at the training dynamics on the right, the episode rewards climb steadily from around 160 up toward 1,500 over 30 million steps. 
> 
> Notice that the curve is still trending upward at the end of the run rather than plateauing. This indicates that more training steps and additional compute would benefit the policy significantly. Throughout these runs, I kept the PPO hyperparameters: a clip range of 0.2 and generalized advantage estimation lambda of 0.95 and while linearly decaying the entropy coefficient.

---

### [3:30 - 4:25] Slide 7: Controller Architecture: Actor-Critic, Asymmetry & GRU Rationale
*(Action: [ADVANCE TO SLIDE 7])*  
*(Visual on screen: Slide 7 showing 3 architectural steps and the clean block diagram at the bottom)*

> "Designing the controller architecture went through three distinct evolutions.
> 
> In Generation 1, I tried pure end-to-end Proximal Policy Optimization, mapping states directly to motor PWM. (PPO) is a policy gradient algorithm that stably improves an agent's policy by clipping updates to prevent destructive large steps away from the previous policy. That was partly successful in simulation by tuning the reward multiple times, but the resulting policy produced high-frequency chattering and ultimately was not feasable in hardware.
> 
> In the second generation, I decoupled the architecture:  meaning the neural policy ran at 100 Hertz to command angular rates and collective thrust, while the drone's native 1-kilohertz PID stabilizer tracked those rates.
> 
> Finally, in the third iteration, I combined an Asymmetric Actor-Critic architecture with a Recurrent Latent History Encoder. In general, we say that the actor controls the system while the critic evaluates its performance. We call it an asymmetric actor critic because, during simulation, a large critic network observes hidden physical states like drag and motor delay to guide policy learning, something that is impossible to do on the real drone. This teaches the policy to compensate for the physical characteristics of the system.
> 
> The actor does not see previous history directly; instead, a  Gated Recurrent Unit encoder converts recent sensor history into a compact 16-dimensional vector. Why a GRU? I doubted that a vanilla RNN could capture long-term dynamics, while a standard LSTM has too many gates and cannot run effectively on the STM32 microcontroller.
---

### [4:25 - 5:00] Slide 8: Embedded Deployment: Deterministic Execution on FreeRTOS
*(Action: [ADVANCE TO SLIDE 8])*  
*(Visual on screen: Slide 8 showing the Two-Rate FreeRTOS hierarchy and bare-metal C highlights)*

> "To deploy this on the Crazyflie, the large privileged critic is discarded. Only the compact MLP actor  and GRU encoder are exported as static C arrays. 
> The firmware runs a two-rate control loop inside the real time operating system:
> Every 10 milliseconds, the high-level controller validates sensor estimates, updates the GRU latent state, and does a forward pass of the actor neural network to output desired angular rates and collective thrust.
> Every millisecond, the lower-level PID controller updates motor PWM signals to reach the desired attitude and thrust.

---

### [5:00 - 5:40] Slide 9: Sim-to-Real Transfer: Overcoming the Physical Reality Gap
*(Action: [ADVANCE TO SLIDE 9])*  
*(Visual on screen: Slide 9 detailing motor lag, battery voltage sag, and optical tracking loss)*

> "When I took the drone to the flight cage for physical testing, I ran into three major physical roadblocks:
> 
> First, motor lag: the DC motors have a 25-millisecond electromechanical delay. At flip rotation speeds of 15 to 20 radians per second, this creates a 25-to-30-degree phase lag, causing severe over-rotation and ultimately crashing. This lead to 
> 
> The second and most significant problem was battery voltage sag. The battery would drop from 4.1 volts down to 3.1 volts, severely reducing thrust output, something that, when accounted for, boosted performance significantly.
> 
> The third issue was optical tracking data loss. During the inversion required for a flip, the top-mounted sensors lose line of sight to the base stations, causing the Kalman filter to drift into false estimates. This was partly solved with the GRU encoder, which was trained to be resistant to noise, and partly with a pre-processing step that prevents the drone from following implausible estimates, like teleporting.

---

### [5:40 - 6:20] Slide 10: Firmware Safeguards & Motor Transfer Adaptation
*(Action: [ADVANCE TO SLIDE 10])*  
*(Visual on screen: Slide 10 showing firmware precautions on left and motor upgrade adaptation on right)*

> "This process took the form of an iterative loop: flying in the safety cage, logging telemetry over Crazyradio, inspecting logs, and updating simulation physics.
> 
> To protect the drone during physical testing, I took practical firmware precautions that occasionally save the drone from crashing during aggressive maneuvers. In particular, a filter catches optical tracking dropouts during inversion, freezing state estimates at the last reading rather than letting sensor spikes alter motor throttle significantly.
> 
> After installing the upgraded 20-millimeter motors because the flip was unfeasible with baseline motors, I did not need to retrain the policy. The GRU latent vector absorbed the altered mass and higher thrust online, achieving rapid motor adaptation on hardware."

---

### [6:20 - 6:55] Slide 11: Experimental Flight Results: Hardware Flight Log Telemetry
*(Action: [ADVANCE TO SLIDE 11])*  
*(Visual on screen: Slide 11 showing the 3 physical flip phases and the real hardware telemetry plot from Flight #6)*

> "Here you can see the actual empirical flight telemetry recorded during Flight 6 in the lab, capturing a complete autonomous flip.
> 
> Notice the three distinct phases: first, the vertical climb pop where the drone ascends to 1.82 meters; second, the inverted ballistic coast where the vehicle pitches past 158 degrees inversion while optical tracking drops to zero packets; and third, the upright recovery where the low-level PID controller counter-torques, arrests the descent, and stabilizes the drone upright."

---

### [6:55 - 7:25] Slide 12: Empirical Flight Demonstration: Simulation vs. Real Flip
*(Action: [ADVANCE TO SLIDE 12])*  
*(Visual on screen: Slide 12 side-by-side video demonstration showing MuJoCo simulation rollout on the left and physical Crazyflie flight on the right)*

> "Here is the side-by-side video demonstration comparing the simulation with a physical hardware flight.
> 
> On the left, the policy performs a flip and then an orbit maneuver.
> 
> On the right is the physical Crazyflie executing the flip inside the lab safety cage. You can see the initial climb pop, the rapid rotation about the pitch axis, and the clean upright catch.

---

### [7:25 - 7:50] Slide 13: Looking Ahead: Technical Improvements & Practical Use Cases
*(Action: [ADVANCE TO SLIDE 13])*  
*(Visual on screen: Slide 13 showing future technical extensions on left and practical applications on right)*

> "Looking ahead, I think there are a couple technical directions this work could go.
> 
> First, integrating lightweight optical flow camera sensors directly on the vehicle would eliminate external Lighthouse tracking.
> 
> Second, extending training to recover from arbitrary impulse distrubances will make the drone more robust to external impacts, like hitting an obstacle.
> 
> Third, given the upward trend in the training curves, scaling training compute will yield even tighter trajectory tracking.
> 
> In terms of use cases, this work might be used in applications that require agile robotics in cluttered environments or narrow industrial inspections."

---

### [7:50 - 8:05] Slide 14: Project Summary: Autonomous Edge AI Control
*(Action: [ADVANCE TO SLIDE 14])*  
*(Visual on screen: Slide 14 displaying summary takeaways, author contact, and open-source repository link)*

> "To conclude, this project demonstrates that deep reinforcement learning can enable a tiny drone to perform aerobatic maneuvers. entirely onboard.
> 
Thank you!"

---

## Speaker's Quick Cheat Sheet (Keep On Screen During Recording)

```
[0:00 - 0:30] SLIDE 1: Intro (Andreas Panagopoulos, ECE UPatras, Crazyflie onboard flip).
[0:30 - 1:05] SLIDE 2: Motivation (Control challenge, underactuated dynamics, onboard vs offboard).
[1:05 - 1:40] SLIDE 3: Setup - Drone (Crazyflie 2.1, 30g, STM32F405 MCU, 192KB RAM, 1S LiPo, no Linux).
[1:40 - 2:15] SLIDE 4: Setup - Lighthouse (IR sweeps, top deck photodiodes, EKF, Inverted blackout problem).
[2:15 - 2:55] SLIDE 5: Timeline (Euler gimbal lock -> Quaternions, Reward shaping -> Geometric controller).
[2:55 - 3:30] SLIDE 6: JAX (3-5x CPU speedup, Upward training curves -> compute-bound dynamics).
[3:30 - 4:25] SLIDE 7: Architecture (PPO+PID decoupling, Asymmetric Actor-Critic, GRU vs TCN, RMA).
[4:25 - 5:00] SLIDE 8: Embedded (FreeRTOS 100 Hz / 1 kHz, static C arrays, no malloc, Cortex-M4 FPU).
[5:00 - 5:40] SLIDE 9: Reality Gap (25ms motor lag, 4.1V -> 3.1V/2.8V sag, optical tracking loss).
[5:40 - 6:20] SLIDE 10: Firmware (Plausibility filter, anti-teleporting, Motor transfer adaptation via RMA).
[6:20 - 6:55] SLIDE 11: Real Flight Telemetry (Flight #6: z=1.82m pop, 158° tilt, Rx=0 blackout, sup_bits 542).
[6:55 - 7:25] SLIDE 12: Video Demonstration (Sim vs. Real flip side-by-side video playback).
[7:25 - 7:50] SLIDE 13: Future Work (Onboard VIO, Tumble recovery, Compute scaling, Micro-robotics).
[7:50 - 8:05] SLIDE 14: Conclusion (Wrap up, GitHub repository, Thank you).
```
