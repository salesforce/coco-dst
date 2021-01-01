


# CoCo: Controllable Counterfactuals for Evaluating Dialogue State Trackers
Authors: [Shiyang Li*](http://shiyangli.me/), [Semih Yavuz*](https://scholar.google.co.uk/citations?user=krh3p8AAAAAJ&hl=en), Kazuma Hashimoto, Jia Li, Tong Niu, Nazneen Rajani, Xifeng Yan, Yingbo Zhou and Caiming Xiong (*Equal Contribution)

## Abstract

Dialogue state trackers have made significant progress on benchmark datasets, but their generalization capability to novel and realistic scenarios beyond the heldout conversations is less understood. We propose controllable counterfactuals (COCO) to bridge this gap and evaluate dialogue state tracking (DST) models on novel scenarios, i.e., would the system successfully tackle the request if the user responded differently but still consistently with the dialogue flow? COCO leverages turn-level belief states as counterfactual conditionals to produce novel conversation scenarios in two steps: (i) counterfactual goal generation at turnlevel by dropping and adding slots followed by replacing slot values, (ii) counterfactual conversation generation that is conditioned on (i) and consistent with the dialogue flow. Evaluating state-of-the-art DST models on MultiWOZ dataset with COCO-generated counterfactuals results in a significant performance drop of up to 30.8% (from 49.4% to 18.6%) in absolute joint goal accuracy. In comparison, widely used techniques like paraphrasing only affect the accuracy by at most 2%. Human evaluations show that COCO-generated conversations perfectly reflect the underlying user goal with more than 95% accuracy and are as human-like as the original conversations, further strengthening its reliability and promise to be adopted as part of the robustness evaluation of DST models.

Paper link: https://arxiv.org/pdf/2010.12850.pdf

## Model Architecture

![coco](/plot/coco.png)

The overall pipeline of CoCo. The very left part represents the training phase of utterance generation model, where the concatenation of system utterance and turn-level belief state is processed by the encoder, which the decoder then conditions on to generate the user utterance. The input and output of this model is shown within the box at the lower-left. The right part depicts the inference phase, where the counterfactual goal generator first modifies the original belief state fed from the left part into a new one, which is then fed to the trained utterance generator along with the same conversation history to generate new user utterances by beam search followed by filtering undesired utterances. Note that conversational turns in inference phase don’t have to originate from training phase.
## Installation

The package general requirements are

- Python >= 3.7
- Pytorch >= 1.5 (installation instructions [here](https://pytorch.org/))
- Transformers >= 3.0.2 (installation instructions [here](https://huggingface.co/transformers/))
 
The package can be installed by running the following command. Run

```sh setup.sh```

## Usage
This section explains steps to prepare for MultiWOZ dataset and how to train CoCo model and run it for evaluation and data augmentation. 

### Data
It includes preprocessed MultiWOZ 2.1 and MultiWOZ 2.2 dataset. 
[Download](https://storage.cloud.google.com/sfr-coco-dst-research/multiwoz.zip), uncompress it, and place the 
resulting ```multiwoz``` folder under the root of the repository as ```./multiwoz```.

### Details of CoCo: 
See ```./coco-dst/README.md```
### Details of TRADE: 
See ```./trade-dst/README.md```
### Details of SimpleTOD: 
See ```./simpletod/README.md```
### Details of TripPy: 
See ```./trippy-public/README.md```
## Citation
```
@article{SHIYANG2020CoCoCC, 
title={CoCo: Controllable Counterfactuals for Evaluating Dialogue State Trackers}, 
author={Shiyang Li, Semih Yavuz, Kazuma Hashimoto, Jia Li, Tong Niu, Nazneen Rajani, Xifeng Yan, Yingbo Zhou and Caiming Xiong}, 
journal={ArXiv}, 
year={2020}, 
volume={abs/2010.12850} }
```

## Questions?
For any questions, feel free to open issues, or shoot emails to
- Semih Yavuz (syavuz@salesforce.com)
- [Shiyang Li](http://shiyangli.me/)

## License
The code is released under MIT license - see [LICENSE](LICENSE.txt) for details. 

This code includes other open source software components: 
[trade-dst](https://github.com/jasonwu0731/trade-dst), 
[simpletod](https://github.com/salesforce/simpletod/), and 
[trippy-public](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public).
Each of these software components have their own license. Please see them under 
```./trade-dst```, ```./simpletod```, and ```./trippy-public``` folders. 
