# Judge-Ready Checklist

Use this as the final submission checklist for the OpenEnv Hackathon.

## Core Requirements

- [x] OpenEnv-compliant environment with `reset`, `step`, `state`, and `health`
- [x] Hosted app entrypoint via FastAPI in `main.py`
- [x] Docker-ready deployment for Hugging Face Spaces
- [x] README explaining the environment and how it works
- [ ] Public Hugging Face Space URL added to the README
- [ ] Mini-blog, video, or slide deck linked from the README
- [ ] Training evidence linked from the README

## Judging Story

- [x] Clear problem statement: customer support operations
- [x] Multi-agent interactions: triage, research, resolver, QA, escalation, critic
- [x] World-modeling / professional workflow fit
- [x] Self-improvement loop via critic memory
- [x] Live dashboard to tell the story visually
- [ ] 2-minute demo video or short blog post

## Reward And Environment Quality

- [x] Dense shaped reward
- [x] Policy violation penalties
- [x] Hallucination penalties
- [x] Frustration/churn dynamics
- [x] Difficulty-aware tasks
- [x] Critic lesson injection into future episodes
- [ ] Ablation or baseline comparison in README

## Training Evidence

- [x] Local `training_loop.py` for repeated episodes
- [x] Minimal HF TRL example in `trl_training_example.py`
- [ ] Colab notebook version of the training flow
- [ ] Reward curve image committed to repo
- [ ] Baseline vs trained comparison table
- [ ] Real run screenshots or exported plots

## README Links To Add Before Submission

- [ ] Hugging Face Space URL
- [ ] YouTube demo URL or HF blog URL
- [ ] Plot image links
- [ ] WandB or Colab run link if available

## Suggested Final Deliverables

1. `README.md` with links, plots, and a short results section.
2. Hugging Face Space deployed and tested.
3. One Colab notebook showing data collection plus TRL or Unsloth training.
4. One figure showing reward improvement over episodes.
5. One short demo video showing the dashboard and what changed after training.
