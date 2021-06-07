# MetaCURE: Meta Reinforcement Learning with Empowerment-Driven Exploration

This repository is the official implementation of [MetaCURE: Meta Reinforcement Learning with Empowerment-Driven Exploration]. Please create an issue if you have any problems!


## Requirements

To install requirements:

```setup
conda env create -n metacure -f  environment.yaml
```

> This will create a new conda env called metacure.

You may also need to install Meta-World: https://github.com/rlworkgroup/metaworld


## Training

To train MetaCURE, run this command:

```train
python launch_experiment_metacure.py ./configs/sparse-point-robot-metacure.json --gpu 0
```


To train PEARL as a baseline, run this command:

```train
python launch_experiment_pearl.py ./configs/sparse-point-robot-pearl.json --gpu 0
```

> You can also run additional experiments by specifying certain .json files in the 'configs' folder, and some example commands are available in commands.sh.

## Evaluation

Evaluation is automatically done after each epoch of training. 

Results are stored in the 'outputmetacure'and 'outputpearl' folders, respectively.

 You can visualize learning curves with viskit: https://github.com/vitchyr/viskit.


## Results

Refer to the original paper and Appendix for results.