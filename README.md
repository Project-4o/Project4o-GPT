# Project-4o

An open-source project to build a 4o-like AI, focused on personality, not parameters.

**Website:** [github.com/Project-4o/project-4o](https://github.com/Project-4o/project-4o)
**Community:** [r/project4o](https://www.reddit.com/r/project4o/)
**License:** MIT

---

## What is this?

Project-4o is an attempt to recreate the feel of a 4o-style AI, one that actually talks like a person, not a manual. Think natural conversation, personality, roleplay support, and less of that stiff, overly-safe corporate tone.

This isn't about building the biggest model. It's about building the best *experience*.

## Goals

- Natural, fluid conversation
- Real personality (not a persona bolted on top)
- Roleplay (RP) mode that actually works
- Less rigid, more human behavior
- Small and efficient, not everything needs 70B parameters

## Approach

- Start small — under 5B params so the model can run on mobile
- Fine-tune using QLoRA adapters (pipeline on the `finetuning` branch)
- Base model: **not chosen yet** — open call for candidates (custom architectures welcome) in the [r/project4o megathread](https://www.reddit.com/r/project4o/comments/1wcteap/megathread_which_model/)
- Focus training on personality, tone, and interaction quality
- Iterate fast, keep it lightweight

## Contributing

This is a community project. Everyone's welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

- **Developers** — model training, tooling, infrastructure
- **Writers** — training data, conversation examples, personality design
- **Testers** — try the models, report what works and what doesn't
- **Ideas** — if you have thoughts on what makes an AI feel natural, share them

Check the [issues](https://github.com/Project-4o/project-4o/issues) or join the conversation on [Reddit](https://www.reddit.com/r/project4o/).

## License

Project-4o is released under the [MIT License](LICENSE).
