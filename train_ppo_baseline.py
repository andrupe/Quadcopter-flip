"""
Εκπαίδευση PPO ως baseline σύγκρισης με τον cascaded-PID controller.

Χρησιμοποιεί το υπάρχον Simulation/train.py, αλλά με:
  - ξεχωριστό MODEL_NAME (δεν πατάει τα υπάρχοντα αρχεία σου)
  - ρυθμιζόμενο budget βημάτων και αριθμό workers

Χρήση:
    # Εκπαίδευση από το μηδέν
    py -3.11 train_ppo_baseline.py --steps 5000000 --workers 4 --name ppo_5m

    # Συνέχεια από υπάρχον μοντέλο (το παλιό ΔΕΝ πειράζεται - σώζει σε νέο όνομα)
    py -3.11 train_ppo_baseline.py --steps 10000000 --workers 4 \
        --resume-from ppo_5m --name ppo_10m
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
for _p in (_ROOT, _SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", type=str, default="ppo_smoke")
    ap.add_argument("--dr", action="store_true",
                    help="ενεργοποίηση της ράμπας domain randomization")
    ap.add_argument("--dr-start", type=int, default=None,
                    help="βήμα στο οποίο ΞΕΚΙΝΑΕΙ η ράμπα DR. ΑΠΑΡΑΙΤΗΤΟ σε συνέχεια "
                         "εκπαίδευσης: η προεπιλογή του train.py (4.5M) είναι ήδη πίσω "
                         "μας, οπότε το dr_level θα πεταγόταν αμέσως στο 1.0")
    ap.add_argument("--dr-end", type=int, default=None,
                    help="βήμα στο οποίο η ράμπα DR φτάνει στο 1.0")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="όνομα υπάρχοντος μοντέλου (χωρίς .zip) για συνέχεια εκπαίδευσης")
    ap.add_argument("--ft-lr", type=float, default=1e-4,
                    help="learning rate εκκίνησης στη συνέχεια εκπαίδευσης")
    ap.add_argument("--ft-lr-floor", type=float, default=3e-5,
                    help="learning rate τερματισμού στη συνέχεια εκπαίδευσης")
    ap.add_argument("--clean", action="store_true",
                    help="εκπαίδευση στο ΚΑΘΑΡΟ benchmark του project: χωρίς θόρυβο "
                         "παρατήρησης και χωρίς άνεμο (το env τα κρατά ενεργά ακόμα "
                         "και με dr_level=0)")
    ap.add_argument("--shaped", action="store_true",
                    help="πειραματικός σχηματισμός reward (reward_shaping.py): ποινή "
                         "μεγέθους δράσης + σχηματισμός ύψους στη φάση flip")
    ap.add_argument("--ent", type=float, default=None,
                    help="entropy coefficient (προεπιλογή train.py: 0.01). "
                         "Βάλε 0.0 ώστε η πολιτική να συγκλίνει και να δουλεύει ντετερμινιστικά")
    args = ap.parse_args()

    import train as T

    resume_path = None
    if args.resume_from:
        resume_path = os.path.join(_ROOT, args.resume_from + ".zip")
        if not os.path.isfile(resume_path):
            print(f"[!] Δεν βρέθηκε το μοντέλο: {resume_path}")
            return
        if args.resume_from == args.name:
            print("[!] Το --name πρέπει να διαφέρει από το --resume-from, "
                  "αλλιώς το παλιό μοντέλο θα αντικατασταθεί.")
            print(f"    Πρότεινε π.χ.:  --resume-from {args.resume_from} "
                  f"--name {args.resume_from}_cont")
            return

    # Το ADR ξεκινά στα 4.5M βήματα. Σε μικρό budget δεν προλαβαίνει καν να
    # ενεργοποιηθεί, οπότε το απενεργοποιούμε ρητά για να μην μπερδεύει.
    if not args.dr:
        T.DR_ENABLED = False
    else:
        # Ο DomainRandomizationCallback υπολογίζει το dr_level από τα ΣΥΝΟΛΙΚΑ
        # βήματα. Σε συνέχεια εκπαίδευσης πέρα από το DR_END_STEPS (14.5M) θα
        # ξεκινούσε κατευθείαν στο 1.0, δηλαδή απότομο σοκ πλήρους
        # τυχαιοποίησης πάνω σε πολιτική που δεν έχει δει ποτέ διαταραχή.
        if args.dr_start is not None:
            T.DR_START_STEPS = int(args.dr_start)
        if args.dr_end is not None:
            T.DR_END_STEPS = int(args.dr_end)
        if resume_path is not None and args.dr_start is None:
            print("[!] Συνέχεια εκπαίδευσης με --dr χωρίς --dr-start.")
            print(f"    Η ράμπα του train.py ({T.DR_START_STEPS:,} -> {T.DR_END_STEPS:,}) "
                  f"είναι ήδη πίσω μας: το dr_level θα ξεκινήσει στο 1.0.")
            print("    Όρισε --dr-start / --dr-end, π.χ. για συνέχεια από 20M:")
            print("      --dr-start 20000000 --dr-end 28000000")
            return

    # Το χρονοδιάγραμμα learning rate είναι δεμένο με 20M βήματα. Σε μικρό
    # budget θα έμενε κολλημένο στην αρχή της Φάσης 1, οπότε το κλιμακώνουμε.
    # Στη συνέχεια εκπαίδευσης δεν ισχύει: εκεί χρησιμοποιείται το ft_schedule.
    if resume_path is None:
        scale = args.steps / 20_000_000
        T.LR_WARMUP_STEPS = max(1, int(T.LR_WARMUP_STEPS * scale))

    print(f"{'='*65}")
    print(f"PPO baseline | steps={args.steps:,} | workers={args.workers} | "
          f"name={args.name} | DR={'on' if args.dr else 'off'}")
    if resume_path:
        print(f"Συνέχεια από : {os.path.basename(resume_path)}")
        print(f"Έξοδος σε    : {args.name}.zip  (το παλιό παραμένει ανέπαφο)")
        print(f"Fine-tune LR : {args.ft_lr:.1e} -> {args.ft_lr_floor:.1e}")
    else:
        print(f"LR warmup κλιμακώθηκε σε {T.LR_WARMUP_STEPS:,} βήματα")
    ent = T.ENT_COEF if args.ent is None else args.ent
    print(f"ent_coef     : {ent}")
    print(f"Συνθήκες     : {'ΚΑΘΑΡΕΣ (χωρίς θόρυβο/άνεμο)' if args.clean else 'με θόρυβο + άνεμο'}")
    print(f"Reward       : {'ΣΧΗΜΑΤΙΣΜΕΝΟ (reward_shaping.py)' if args.shaped else 'αρχικό του env'}")
    if args.dr:
        print(f"ADR ραμπα    : dr_level 0->1 στα βήματα "
              f"{T.DR_START_STEPS:,} -> {T.DR_END_STEPS:,}")
    else:
        print("ADR          : απενεργοποιημένο")
    print(f"{'='*65}\n")

    t0 = time.time()
    T.train(
        total_timesteps=args.steps,
        num_workers=args.workers,
        model_name=args.name,
        device="cpu",
        ent_coef=ent,
        obs_noise=not args.clean,
        random_wind=not args.clean,
        shaped_reward=args.shaped,
        load_previous_model=resume_path is not None,
        previous_model_path=resume_path,
        ft_lr_start=args.ft_lr,
        ft_lr_floor=args.ft_lr_floor,
    )
    print(f"\nΔιάρκεια: {(time.time() - t0)/60:.1f} λεπτά")


if __name__ == "__main__":
    main()
