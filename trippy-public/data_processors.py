# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import dataset_multiwoz21


class DataProcessor(object):
    def __init__(self, dataset_config):
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        self.class_types = raw_config['class_types']
        self.slot_list = raw_config['slots']
        self.label_maps = raw_config['label_maps']

    def get_train_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, **args):
        raise NotImplementedError()

class Multiwoz21Processor(DataProcessor):
  
    def get_train_examples(self, data_dir, aug_file, args):

        if(aug_file): 

          return dataset_multiwoz21.create_aug_examples(os.path.join(data_dir, 'train_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, aug_file , **args)
        else:

          return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'train_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'dev_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'dev', self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'test_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'test', self.slot_list, self.label_maps, **args)


class Multiwoz21GUIProcessor(DataProcessor):


    def __init__(self,dataset_config,data_dir):
      
        super(Multiwoz21GUIProcessor, self).__init__(dataset_config = dataset_config)
        self.sys_inform_dict = dataset_multiwoz21.load_acts(os.path.join(data_dir, 'dialogue_acts.json'))
        with open(os.path.join(data_dir, 'test_dials.json'), "r", encoding='utf-8') as reader:
            self.input_data = json.load(reader)

    def get_example(self, args , dialog_id, turn_id, new_usr_utter, new_turn_label):
        return dataset_multiwoz21.create_GUI_examples(self.input_data, dialog_id, turn_id, new_usr_utter,new_turn_label,
                self.sys_inform_dict,'test', self.slot_list, self.label_maps, **args)


PROCESSORS = {"multiwoz21": Multiwoz21Processor,
              "multiwoz21gui":Multiwoz21GUIProcessor}
