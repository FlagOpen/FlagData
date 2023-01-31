# Text Condensation

## Description

FlagData Condensation provides a fast, extensible data condensation tool for the text classification task. It provides most of the commonly used text condensation modules which can be easily configured by a human-readable YAML config file. Meanwhile, users can define their processors and add to the whole pipeline.

## Installation

Python >= 3.9

Install the Condensation modules via pip. By doing this, only the specified module's dependencies will be installed. This is for users who only want to use the Condensation module and do not want to install the other dependencies:
```bash
pip install flagdata[condensation]
```
If you want to install all the dependencies, use the following command:
```bash
pip install flagdata[all]
```

## Usage

We currently provide a data distillation algorithm. It can set the data path and model path in the YAML config file.
Note: You may need at least 20GB of GPU memory in order to train data distillation model.

### Data Format

We preset the data format as JSONL in which each piece of data contains two fields: text and label. Data examples are as follows：

```json
{
   "text": "Fears for T N pension after talks Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.", 
   "label": 3
}
```

### Quick Start

There are basically 2 steps in order to use our FlagData Condensation:

1. Modify the YAML config file. We have written detailed comments in the configuration file to explain the meaning of each parameter.

2. Specify the path to the config file and run!
   
   ```python
   from flagdata.condensation.data_distillation import DataDistillationTrainer
   # you need to specify your own configuration file path here
   trainer = DataDistillationTrainer("flagdata/config/distillation_config.yaml") 
   # data should be in jsonl format with keys: "text", "label"
   trainer.load_data()
   # fit() will run data condensation training and save the distilled data in binary format which can be read by torch.load()
   # you can specify the save path by setting "distilled_save_path" in config file
   trainer.fit()
   ```
   
### Advanced Usage

1. If you want to define your own dataset, following these steps:
   
   - define your own dataset class
   
   ```
   from torch.utils.data import Dataset
   class CustomDataSet(Dataset):
       def __init__(self, **args):
           pass # your code here
   
       def __getitem__(self, idx):
           # your code here
           return input_ids.squeeze(), attention_mask.squeeze(), torch.tensor(self.targets[idx])
   ```
   
   - pass your custom dataset class to `trainer.load_data()`, like:
   
   ```
      # ...
      trainer.load_data(self, dataset_class=CustomDataset)
   ```
   
   ## Config
   
   We use YAML format config file for readability. For more information about YAML, you can check <https://yaml.org/> . 
   
   Descriptions and details of each option are commented in our [config template](https://dorc.baai.ac.cn/resources/projects/FlagData/distillation_config.yaml) .
   
   ## Reference
   
   1. Dataset Distillation for Text Classification [[paper\]](https://arxiv.org/abs/2104.08448)
   
   2. <https://github.com/arumaekawa/text-dataset-distillation>
   
   3. <https://github.com/huggingface/transformers>
