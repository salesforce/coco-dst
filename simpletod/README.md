## SimpleTOD: A Simple Language Model for Task-Oriented Dialogue

This code is an adaptation of its [**original implementation**](https://github.com/salesforce/simpletod/). Here, we 
only include its training for DST task on whole MultiWOZ dataset and its re-training for CoCo-generated data.
## Train baseline 
As ```SimpleTOD``` needs to preprocess MultiWOZ dataset, we include it as part of release. 
[**Download**](https://storage.cloud.google.com/sfr-coco-dst-research/simpletod-resources/resources.zip) and 
uncompress the preprocessed ```resources``` folder under the current module (```coco-dst/simpletod/```). Then, run
```console
❱❱❱ sh train_baseline.sh
```
After training, modify ```MODEL_CHECKPOINT``` in line 16 of ```interface4eval_simpletod.py``` to the path of your 
checkpoint you want to do evaluation. You can also [**download**](https://storage.cloud.google.com/sfr-coco-dst-research/simpletod-resources/baseline.zip) 
the checkpoint we use in our paper.

Then, you can simply evaluate the desired SimpleTOD model checkpoint on CoCo-generated examples by running
```console
❱❱❱ sh run_eval.sh
```
after setting the required arguments as instructed under section ```Run Evaluation on CoCo Examples``` in 
```../coco-dst/README.md```.
 
## Re-Train with CoCo-generated data 
Before running the re-training script, make sure that 
- You generated the data using CoCo for re-training (see further details under section 
```CoCo+ as data augmentation defense``` in ```../coco-dst/README.md```.
- You moved the data for re-training into ```../coco-dst/coco_data```.

Then run
```console
❱❱❱ sh format_aug_data.sh
```
to combine original training data with CoCo-generated data for ```SimpleTOD```training. You can use our post-processed 
[data](https://storage.cloud.google.com/sfr-coco-dst-research/simpletod-resources/resources.zip) (```train.coco-vs_rare_aug_history_belief```) 
for re-training. After obtaining re-training data, run the following command to launch re-training:
```console
❱❱❱ sh train_coco-aug.sh
```
After training, modify ```MODEL_CHECKPOINT``` in line 16 of ```interface4eval_simpletod.py``` to the path of your 
checkpoint you want to do evaluation. You can also [**download**](https://storage.cloud.google.com/sfr-coco-dst-research/simpletod-resources/coco-vs_rare.zip) 
the retrained checkpoint we use in our paper to do the evaluation.

Then, you can simply re-evaluate SimpleTOD on CoCo-generated examples by running
```console
❱❱❱ sh run_eval.sh
```
after setting the required arguments as instructed under section ```Run Evaluation on CoCo Examples``` in 
```../coco-dst/README.md```.

## License
The code is released under the BSD-3 License - see [LICENSE](LICENSE.txt) for details
