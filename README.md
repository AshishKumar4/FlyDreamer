# FlyDreamer: Learning to Fly FPV in Dreams

> **Note**: This project is forked and modified from the official [DreamerV3](https://github.com/danijar/dreamerv3) repository by Danijar Hafner. Full credit for the base code structure and DreamerV3 algorithm goes to the original authors and Google DeepMind.

---

<div align="center">
  <img src="assets/demo.gif" alt="FlyDreamer Demo" width="600">
</div>

## Overview

FlyDreamer trains a reinforcement learning agent to fly an FPV (First-Person View) drone in simulation using world model-based learning. The agent learns entirely from imagination—dreaming trajectories in a learned world model before executing in the real environment.

**Supported Simulator**: [Custom Colosseum Fork](https://github.com/AshishKumar4/Colosseum) (based on Colosseum, which is itself a fork of Microsoft AirSim)

## Quick Start

### Manual Controller Test

Test the FPV environment with a physical controller (TX16S and other joysticks supported):

```bash
python test_manual_control.py
```

### Training

Train an FPV navigation agent:

```bash
python dreamerv3/main.py \
  --configs fpv \
  --logdir logs/fpv_run
```

### World Model Pretraining (Experimental)

Pretrain the world model on demonstration data before policy learning:

```bash
python dreamerv3/main.py \
  --configs fpv pretrain_wm \
  --logdir logs/pretrain \
  --demo_data_dir demo_data/chunks
```

### TSSM Transformer Dynamics (Experimental)

Use a Transformer-based state-space model instead of the default RSSM:

```bash
python dreamerv3/main.py \
  --configs fpv \
  --agent.dyn.typ tssm \
  --logdir logs/tssm_run
```

## Development Status

This project is under **active development and experimentation**. Documentation, features, and APIs may change frequently. More detailed documentation will be added in the future.

## References

### Core Algorithm

- **DreamerV3**: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). *Mastering diverse control tasks through world models*. Nature.
  [[Paper]](https://arxiv.org/abs/2301.04104) [[Website]](https://danijar.com/dreamerv3) [[Code]](https://github.com/danijar/dreamerv3)

- **DreamerV2**: Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with Discrete World Models*. ICLR.
  [[Paper]](https://arxiv.org/abs/2010.02193)

### Related Drone Learning Work

- **SkyDreamer**: Wu, X., et al. (2024). *SkyDreamer: Autonomous Drone Racing with World Models*.
  [[Paper]](https://arxiv.org/abs/2404.14648)

- **Dream to Fly**: Zhou, P., et al. (2024). *Learning to Fly FPV Drones in Virtual Reality*.
  [[Paper]](https://rpg.ifi.uzh.ch/docs/RSS24_Bhattacharyya.pdf)

- **Learning to Fly in Seconds**: Kaufmann, E., et al. (2023). *Champion-level drone racing using deep reinforcement learning*. Nature.
  [[Paper]](https://www.nature.com/articles/s41586-023-06419-4)

### Simulators

- **Colosseum (Custom Fork)**: Modified Colosseum with FPV-specific enhancements.
  [[GitHub]](https://github.com/AshishKumar4/Colosseum)

- **Colosseum**: CodexLabs fork of AirSim with continued maintenance.
  [[GitHub]](https://github.com/CodexLabsLLC/Colosseum)

- **AirSim**: Microsoft Research drone and car simulation platform.
  [[GitHub]](https://github.com/microsoft/AirSim) [[Docs]](https://microsoft.github.io/AirSim/)

## Citation

If you use this code, please cite the original DreamerV3 paper:

```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## License

This project follows the license of the original DreamerV3 repository.
