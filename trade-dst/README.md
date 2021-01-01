


## TRADE Multi-Domain and Unseen-Domain Dialogue State Tracking

This code is an adaptation of its [**original implementation**](https://github.com/jasonwu0731/trade-dst/).   Here, we only include its training on whole MultiWOZ dataset and its re-training for CoCo-generated data.
## Train baseline 
```console
❱❱❱ sh train_baseline.sh
```
After training the model, run
```console
❱❱❱ cp baseline/TRADE-multiwozdst/*.pkl baseline/
```
to copy .pkl files under the parent directory (```baseline/```). Modify ```MODEL_PATH``` in line 5 of 
```interface4eval_trade.py``` to the model checkpoint you want to evaluate. By default, it points to the checkpoint we 
evaluate in our paper, which you can achieve by following the steps below:
1. [Download](https://storage.cloud.google.com/sfr-coco-dst-research/trade-dst-resources/baseline.zip) and uncompress it,
2. Place the resulting ```baseline``` folder under the current module (```coco-dst/trade-dst/```).

Then, you can simply evaluate the desired TRADE model checkpoint on CoCo-generated examples by running
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
❱❱❱ sh train_coco-aug.sh
```
After training the model, run
```console
❱❱❱ cp coco-vs_rare/TRADE-multiwozdst/*.pkl coco-vs_rare/
```
Modify ```MODEL_PATH``` in line 5 of ```interface4eval_trade.py``` to the path of the checkpoint trained on 
CoCo-generated data. 

OR, if you prefer to evaluate the checkpoint we used in our paper
1. [Download](https://storage.cloud.google.com/sfr-coco-dst-research/trade-dst-resources/coco-vs_rare.zip) and uncompress it,
2. Place the resulting ```coco-vs_rare``` folder under the current module (```coco-dst/trade-dst/```).
3. Instead of Line 5, use ```MODEL_PATH``` in line 6 of ```interface4eval_trade.py``` which by default points to the 
right checkpoint.

Then, you can simply re-evaluate it on CoCo-generated examples by running
```console
❱❱❱ sh run_eval.sh
```
after setting the required arguments as instructed under section ```Run Evaluation on CoCo Examples``` in 
```../coco-dst/README.md```.


## License
```
copyright 2019-present https://jasonwu0731.github.io/

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
