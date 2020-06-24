#######################################
# Bayesian Optimization

For Bayesian optimization, we used the scripts from https://github.com/mkusner/grammarVAE

## Usage
First generate the latent representation of all training molecules:
```
python gen_latent.py --data ../data/covid/fulltrain.txt --vocab ../data/covid/autogen_vocab.txt --model ../fast_molvae/covid_model/autogen_vocab/model.iter-8300 --output './descriptors/' --qed 1 --logp 0.5 --sa 1 --cycle 0 --pic50 1 --sim 0.7
```
This generates `latent_features.txt` for latent vectors and other files for scoring parameters. The weights for the scoring function are stored in score_weights.txt in the order of pIC50_weight, QED_weight, logP_weight, SA_weight, cycle_weight, sim_weight.

To run Bayesian optimization:

```
python run_bo.py --vocab ../data/covid/autogen_vocab.txt --save_dir ./results/ --seed 1 --model ../fast_molvae/covid_model/autogen_vocab/model.iter-8300 --descriptors './descriptors/' --sampling 50 --iteration 10
```
The score weights input while generating the latent representation are read in automatically.

This command performs 10 iterations of Bayesian optimization with EI heuristics over a sample size of 50, and saves discovered molecules in `results/`
Following previous work, we set `seed = 1`.

Finally, run postprocessing scripts to get txt files of the SMILES, scores, and pIC50 values of the discovered molecules.
```
python postprocess.py --path ./results/
```
