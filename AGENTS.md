# Working conventions

- Keep this a simple research project. Implement the requested experiment directly.
- Use the sibling Forge library and Hydra configs for every pipeline stage.
- Do not add tests, defensive wrappers, speculative abstractions, or production infrastructure.
- Add guards sparingly, primarily at I/O boundaries. Let ordinary errors surface.
- Use PyTorch tensors for numerical work; vectorize over tokens, masks, and examples.
- Use DataLoaders and explicit collate functions. Configure devices and GPU parallelism through runtime configs.
- Keep representation processing separate from classifiers in `tagger/`.
- When asked to run an experiment, launch it in a tmux pane, report the pane, and stop watching. The user will report completion. Do not poll or tail it.
- Do not modify `../exploration`, `../forge`, or `../old_lion` as part of work here.

These are the user's general preferences for their research projects.
