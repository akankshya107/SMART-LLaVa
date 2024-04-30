#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('ls /mnt/disks/1/SMART101-release-v1/SMART101-Data')


# In[2]:


import torch
print(torch.__version__)


# In[3]:


# get_ipython().system('pip install -q -U transformers==4.37.2')
# get_ipython().system('pip install peft')
# get_ipython().system('pip install -q bitsandbytes==0.41.3 accelerate==0.25.0')


# In[4]:


from transformers import get_linear_schedule_with_warmup, AutoTokenizer, PreTrainedModel
import requests
from PIL import Image
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import gc


# In[5]:


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 0
print(device)


# In[6]:


from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)

class CustomLLaVAModel(LlavaForConditionalGeneration):
  def __init__(self, config):
    super().__init__(config)
    self.word_embeddings = self.get_input_embeddings()

model = CustomLLaVAModel.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")


# In[7]:


print(model)


# In[8]:


DATASET_DIR = "/mnt/disks/1/SMART101-release-v1/SMART101-Data/"


# In[9]:


def read_csv(csvfilename, puzzle_id):
    import csv
    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
    return qa_info

SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
SIGNS = np.array(["+", "-", "x", "/"])
MAX_DECODE_STEPS = 10

def get_puzzle_class_info(puzzle_ids, icon_class_ids):
    #    global SEQ_PUZZLES, puzzle_diff_str, puzzle_diff
    puzzle_classes = {}
    for puzzle_id in puzzle_ids:
        puzzle_root = puzzle_id
        csv_file = "puzzle_%s.csv" % (puzzle_id)
        qa_info = read_csv(os.path.join(DATASET_DIR, puzzle_root, csv_file), puzzle_id)

        pid = int(puzzle_id)
        if pid not in SEQ_PUZZLES:
            num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids) for qa in qa_info]).max() + 1
        else:
            if pid in [16, 39, 100]:
                num_classes = 26 + 1  # if the output is a string of numbers, and the max classes is - max val.
            elif pid in [18, 35]:
                num_classes = 5 + 1  # the minus one is for end of items.
            elif pid in [63]:
                num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids).max() for qa in qa_info]).max() + 1
        puzzle_classes[str(puzzle_id)] = num_classes
    return puzzle_classes

def get_icon_dataset_classes(icon_path):
    """returns the classes in ICONs-50 dataset"""
    with open(icon_path, "r") as f:
        icon_classes = f.readlines()
    return [ii.rstrip() for ii in icon_classes]

def str_replace(ans):
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    return ans

def pad_with_max_val(gt_list, val):
    """if the number of elements in gt is less than MAX_DECODE_STEPS, we pad it with the max value in a class"""
    if len(gt_list) < MAX_DECODE_STEPS:
        gt_list = (
            gt_list
            + (
                np.ones(
                    MAX_DECODE_STEPS - len(gt_list),
                )
                * val
            ).tolist()
        )
    return gt_list

def get_val(qinfo, ans_opt, num_classes_per_puzzle, icon_class_ids, is_one_of_option=False):
    """get the value of the answer option. This code also encodes the value into a number by removing extreneous strings"""
    """ is_one_of_option is True, when ans_opt is one of the options, need not be the correct answer option."""
    where = lambda x, y: np.where(np.array(x) == y)[0][0]
    pid = int(qinfo["puzzle_id"])
    if pid in SEQ_PUZZLES:
        ans = qinfo[ans_opt]
        if pid == 16:
            ans_opt_val = [int(ii) for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 18:
            ans_opt_val = [int(ii) for ii in ans.split("-")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 35:
            ans_opt_val = [
                ord(ii) - ord("A") for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")
            ]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 39:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 63:
            ans_opt_val = [
                int(ii)
                for ii in ans.replace("and", ",")
                .replace("or", ",")
                .replace(", ,", ",")
                .replace("only", "")
                .replace(" ", "")
                .split(",")
            ]
            key = str(63)
            if key in num_classes_per_puzzle:
                ans_opt_val = pad_with_max_val(ans_opt_val, num_classes_per_puzzle[key] - 1)
        elif pid == 100:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        ans_opt_val = np.array(ans_opt_val)

    elif pid == 58:
        # puzzle 58 has answers as <operator><one digit number>, e.g./4,-5, etc.
        # we use +=1, -=2, x=3, /=4. so /4 will be 44, -5=25, +2= 2.
        ans_opt_val = qinfo[ans_opt]
        ans_opt_val = (where(SIGNS, ans_opt_val[0]) + 1) * 10 + int(ans_opt_val[1:])
    elif pid == 25:
        # we need to fix the time in AM/PM format properly.
        ans = qinfo[ans_opt]
        ans_opt_val = int(ans.replace(":00 AM", "").replace(":00 PM", ""))
        if ans.find("PM") > -1:
            ans_opt_val += 12
    else:
        try:
            ans_opt_val = int(qinfo[ans_opt])
        except:
            if len(qinfo[ans_opt]) > 0:
                try:
                    ans_opt_val = ord(qinfo[ans_opt]) - ord("A")
                except:
                    try:
                        ans_opt_val = str_replace(qinfo[ans_opt])
                        ans_opt_val = ans_opt_val.replace("Impossible", "0")  # puzzle 58.
                        if int(qinfo["puzzle_id"]) == 1:  # if the puzzle id is 1, then the options are icon classes.
                            ans_opt_val = "_".join(ans_opt_val.split(" "))
                            if ans_opt_val in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val)
                            elif ans_opt_val + "s" in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val + "s")
                        ans_opt_val = int(ans_opt_val)
                    except:
                        print(qinfo)
                        pdb.set_trace()
            else:
                ans_opt_val = ord(ans_opt) - ord("A")
    if not is_one_of_option:  # implies we are encoding the correct answer.
        qinfo["AnswerValue"] = ans_opt_val
    return ans_opt_val


# In[10]:


def split_data(info, split):
    """
    split_type=standard is to use the split_ratio in the instance order
    split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
    split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
    """
    split_ratio = "80:5:15"
    splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
    n = len(info)
    if split == "train":
        st = 0
        en = int(np.floor(n * splits[0] / 100.0))
        info = info[st:en]
    elif split == "val":
        st = int(np.ceil(n * splits[0] / 100.0))
        en = int(np.floor(n * splits[1] / 100.0))
        info = info[st:en]
    else:
        st = int(np.ceil(n * splits[1] / 100.0))
        info = info[st:]
    return info


# In[11]:


import random
random.seed(1007)
from torch.utils.data import Dataset, DataLoader

PS_VAL_IDX = [7, 43, 64]
PS_TEST_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]
tokenizer = AutoTokenizer.from_pretrained(model_id)

def str_replace_(info, ans_opt):
    ans = info[ans_opt]
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    ans = ans.replace("Impossible", "0")
    info[ans_opt] = ans
    return ans

class SMARTData(Dataset):
    def __init__(self, split):
        super(SMARTData, self).__init__()
        MAX_VAL = 0
        self.qa_info = []
        self.icon_class_ids = get_icon_dataset_classes(DATASET_DIR + "icon-classes.txt")

        if split == "train":
            puzzle_ids = os.listdir(DATASET_DIR)
            puzzle_ids = np.array(puzzle_ids)[np.array([x.find(".") == -1 for x in puzzle_ids])]
            puzzle_ids = puzzle_ids.tolist()
            val_test = PS_VAL_IDX + PS_TEST_IDX
            val_test = set([str(ii) for ii in val_test])
            puzzle_ids = list(set(puzzle_ids).difference(val_test))
        elif split == "val":
            puzzle_ids = [str(ii) for ii in PS_VAL_IDX]
        else:
            puzzle_ids = [str(ii) for ii in PS_TEST_IDX]

        self.split = split
        self.num_classes_per_puzzle = get_puzzle_class_info(puzzle_ids, self.icon_class_ids)
        print("number of train puzzles = %d" % (len(puzzle_ids)))
        for puzzle_id in puzzle_ids:
          csv_file = "puzzle_%s.csv" % (puzzle_id)
          tqa_info = read_csv(os.path.join(DATASET_DIR, puzzle_id, csv_file), puzzle_id)
          for t in range(len(tqa_info)):
              tqa_info[t]["AnswerValue"] = get_val(tqa_info[t], tqa_info[t]["Answer"], self.num_classes_per_puzzle, self.icon_class_ids)
          self.qa_info += split_data(tqa_info, split)
        print(len(self.qa_info))

    def __len__(self):
        return len(self.qa_info)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/"
        im = Image.open(os.path.join(DATASET_DIR, puzzle_root, "img", info["image"]))
        qa = info["Question"]
        _ = [str_replace_(info, key) for key in ["A", "B", "C", "D", "E"]]
        opt_vals = [get_val(info, key, self.num_classes_per_puzzle, self.icon_class_ids, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
        lbl = info["Answer"]
        answer_value = info["AnswerValue"]
        answer = np.zeros(MAX_DECODE_STEPS,)
        if int(pid) not in SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value
        inputs = []
        len_data = len(self.qa_info)
        opts = ["A", "B", "C", "D", "E"]
        prompt = "USER: <image>\n {} \nOptions: {} \nASSISTANT: ".format(qa, " ".join(["({}) {}".format(opt, str(val)) for opt, val in zip(opts, opt_vals)]))
        target = "{} </s>".format(lbl)
        processed_inputs = processor(images=im, text=prompt, padding=True, return_tensors="pt")
        processed_targets = tokenizer(target)
        processed_targets['input_ids'] = processed_targets['input_ids'][1:]
        processed_targets['attention_mask'] = processed_targets['attention_mask'][1:]
        processed_targets = {k: torch.tensor(v) for k, v in processed_targets.items()}

        model_inputs = dict()
        model_inputs["input_ids"] = torch.cat((processed_inputs["input_ids"][0], processed_targets["input_ids"]), dim=0)
        model_inputs["labels"] = torch.cat((torch.tensor([-100] * processed_inputs["input_ids"][0].shape[0]), processed_targets["input_ids"]), dim=0)
        model_inputs["attention_mask"] = torch.cat((processed_inputs["attention_mask"][0], processed_targets["attention_mask"]), dim=0)
        model_inputs["pixel_values"] = processed_inputs["pixel_values"][0]
#         model_inputs["lbl"] = lbl
#         model_inputs["answer_value"] = answer_value
        
#         if self.split == "test":
#             model_inputs["qa"] = qa
#             model_inputs["opts"] = opts
#             model_inputs["puzzle_id"] = info["puzzle_id"]
#             model_inputs["im"] = info["image"]
        return model_inputs


# In[12]:


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    # Find maximum length in batch
    max_length = max(len(ids) for ids in input_ids) + 1

    # Pad sequences
    for item in batch:
        item["input_ids"] = torch.cat((item["input_ids"], torch.tensor([tokenizer.pad_token_id] * (max_length - len(item["input_ids"])))), dim=0)
        item["labels"] = torch.cat((item["labels"], torch.tensor([-100] * (max_length - len(item["labels"])))), dim=0)
        item["attention_mask"] = torch.cat((item["attention_mask"], torch.tensor([0] * (max_length - len(item["attention_mask"])))), dim=0)
        
    padded_input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True)
    padded_labels = pad_sequence([item["labels"] for item in batch], batch_first=True)
    padded_attention_masks = pad_sequence([item["attention_mask"] for item in batch], batch_first=True)
    padded_pixel_values = pad_sequence([item["pixel_values"] for item in batch], batch_first=True)
    
    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels,
        'attention_mask': padded_attention_masks,
        'pixel_values': padded_pixel_values
    }


# In[13]:


train_data = SMARTData("train")
train_loader = DataLoader(train_data, batch_size=4, collate_fn=collate_fn, shuffle=True)
val_data = SMARTData("val")
val_loader = DataLoader(val_data, batch_size=4, collate_fn=collate_fn)
test_data = SMARTData("test")
test_loader = DataLoader(test_data, batch_size=4, collate_fn=collate_fn)


# In[14]:


def test_extreme_generalization(model, test_loader):
    accuracy = 0.0
    num_correct = 0
    num_total = 0
    with open(DATASET_DIR + "test_data_llava_dpt.json", "w+") as fh:
        print("[", file=fh)
        for i, inputs in enumerate(tqdm(test_loader)):
            for inp in inputs:
                tdict = inp.copy()
                with torch.no_grad():
                    prediction = model.generate(**inp, max_new_tokens=512)
                answer = processor.batch_decode(prediction, skip_special_tokens=True)
                tdict["answer_llava"] = answer.split("ASSISTANT: ")[-1].strip()
                if (tdict["answer_llava"].isnumeric() and tdict["answer_llava"] == str(tdict["answer_value"])) or tdict["lbl"] == tdict["answer_llava"]:
                    num_correct += 1
                json.dump(tdict, fh)
                print(",", file=fh)
            num_total += len(inputs)
            accuracy = num_correct / num_total
            print("Accuracy: " + format(accuracy, '.2f'))
        print("]", file=fh)


# In[15]:


def train_fn(model, train_loader, val_loader, num_epochs=10, lr=1e-5):
#     model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
      optimizer=optimizer,
      num_warmup_steps=2000,
      num_training_steps=(len(train_data) * num_epochs),
    )
    gc.collect()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(1, num_epochs+1):
        model.train()
#         model.print_trainable_parameters()
        total_loss = 0
        train_acc = 0
        for i, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device).long() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.detach().float()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i % 1000 == 0:
                print("Training loss: ", total_loss.item() / (i+1))
        
        model.push_to_hub(f"akankshya107/llava_pt_{epoch}")
        
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for _, batch in enumerate(tqdm(val_loader)):
            with torch.no_grad():
              batch = {k: v.to(device).long() for k, v in batch.items()}
              outputs = model(**batch)
            loss = outputs[0]
            eval_loss += loss.detach().float()

        eval_epoch_loss = eval_loss.item() / len(val_loader)
#         train_acc = train_acc / len(train_qa_info)
        train_epoch_loss = total_loss.item() / len(train_loader)
#         eval_acc = eval_acc / len(val_qa_info)
        print(f"{epoch=}: {train_epoch_loss=}  {eval_epoch_loss=}")
        train_losses.append(train_epoch_loss)
        val_losses.append(eval_epoch_loss)
#         train_accuracies.append(train_acc)
#         val_accuracies.append(eval_acc)

    return model


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_features, 9)
        checkpoint = torch.load('resnet_smart_types_1.pt')
        resnet.load_state_dict(checkpoint)
        resnet.fc = torch.nn.Linear(num_features, config.token_dim)
        self.embedding = resnet

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings

from peft import PeftModelForCausalLM, PromptTuningConfig, PeftConfig
import packaging.version
import transformers

class PromptBeforeInstruction(PeftModelForCausalLM):
  def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
      super().__init__(model, peft_config, adapter_name)
  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      token_type_ids=None,
      task_ids=None,
      pixel_values=None,
      position_ids=None,
      past_key_values=None,
      vision_feature_layer=-2,
      vision_feature_select_strategy="default",
      use_cache=None,
      **kwargs,
  ):
    peft_config = self.active_peft_config
    batch_size = input_ids.shape[0]
    if attention_mask is not None:
      prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
      kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )
    # 1. Extra the input embeddings
    insert_idx = torch.where(input_ids[0] == self.base_model.config.image_token_index)[0][0].item()
    inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

    # 2. Get prompt embedding and labels
    if labels is not None:
      prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
      kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
    prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
    prompts = prompts.to(inputs_embeds.dtype)
    inputs_embeds = torch.cat((inputs_embeds[:, :insert_idx+1], prompts[:, :peft_config.num_virtual_tokens], inputs_embeds[:, insert_idx+1:]), dim=1)
    # 3. Merge text and images and trainable prompt
    if pixel_values is not None and input_ids.shape[1] != 1:
        image_outputs = self.base_model.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.base_model.config.vision_feature_select_strategy}"
            )

        image_features = self.base_model.multi_modal_projector(selected_image_feature)
        inputs_embeds, attention_mask, labels, position_ids = self.base_model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )
        if labels is None:
            labels = torch.full_like(attention_mask, self.base_model.config.ignore_index).to(torch.long)
    gc.collect()
    outputs = self.base_model.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    out = dict()
    out["loss"] = loss
    out["logits"] = logits
    out["past_key_values"] = outputs.past_key_values
    out["hidden_states"] = outputs.hidden_states
    out["attentions"] = outputs.attentions
    return out

  def _setup_prompt_encoder(self, adapter_name: str):
    config = self.peft_config[adapter_name]
    if not hasattr(self, "prompt_encoder"):
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
    transformer_backbone = None
    for name, module in self.base_model.named_children():
        for param in module.parameters():
            param.requires_grad = False
        if isinstance(module, PreTrainedModel):
            # Make sure to freeze Tranformers model
            if transformer_backbone is None:
                transformer_backbone = module
                self.transformer_backbone_name = name
    if transformer_backbone is None:
        transformer_backbone = self.base_model

    if config.num_transformer_submodules is None:
        config.num_transformer_submodules = 1

    for named_param, value in list(transformer_backbone.named_parameters()):
        # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape [0]
        # the actual unsharded shape is stored in "ds_shape" attribute
        # special handling is needed in case the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig
        # has been called before
        # For reference refer to issue: https://github.com/huggingface/peft/issues/996
        deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

        if value.shape[0] == self.base_model.config.vocab_size or (
            deepspeed_distributed_tensor_shape is not None
            and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
        ):
            self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
            break

    prompt_encoder = PromptEmbedding(config)

    prompt_encoder = prompt_encoder.to(self.device)
    self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
    self.prompt_tokens[adapter_name] = torch.arange(
        config.token_dim
    ).float()
    
  def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
    """
    prompt_encoder = self.prompt_encoder[self.active_adapter]
    prompt_tokens = (
        self.prompt_tokens[self.active_adapter]
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(prompt_encoder.embedding.fc.device)
    )
    prompts = prompt_encoder(prompt_tokens)
    return prompts

  def prepare_inputs_for_generation(
        self, input_ids=None, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, task_ids=None, **kwargs
    ):
    peft_config = self.active_peft_config
    model_kwargs = self.base_model_prepare_inputs_for_generation(input_ids, **kwargs)

    # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
    # for some architectures which requires a special fix for prompt tuning etc.
    # TODO: starting with transformers 4.38, all architectures should support caching.
    uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
    uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
    transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
    uses_cache = uses_transformers_4_38 or (
        uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
    )

    if uses_cache and (model_kwargs["past_key_values"] is not None):
        # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
        # In prompt learning methods, past key values are longer when compared to the `input_ids`.
        # As such only consider the last input ids in the autogressive generation phase.
        if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
            model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

    if model_kwargs.get("attention_mask", None) is not None:
        size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
        prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
        model_kwargs["attention_mask"] = torch.cat(
            (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
        )

    model_kwargs["position_ids"] = None

    if model_kwargs["past_key_values"] is None:
        insert_idx = torch.where(model_kwargs["input_ids"][0] == self.base_model.config.image_token_index)[0][0].item()
        inputs_embeds = self.base_model.get_input_embeddings()(model_kwargs["input_ids"])
        prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((inputs_embeds[:, :insert_idx], prompts[:, :peft_config.num_virtual_tokens], inputs_embeds[:, insert_idx:]), dim=1)
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["input_ids"] = None

    _ = model_kwargs.pop("cache_position", None)
    return model_kwargs

def prompt_tuning(model):
  peft_config = PromptTuningConfig(
      peft_type="PROMPT_TUNING",
      task_type="CAUSAL_LM",
      num_virtual_tokens=9,
      tokenizer_name_or_path=model_id,
  )
  model_config = getattr(model, "config", {"model_type": "custom"})
  if hasattr(model_config, "to_dict"):
      model_config = model_config.to_dict()['text_config']
  print(model_config)
  peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
  if peft_config.num_layers is None:
      if "num_hidden_layers" in model_config:
          num_layers = model_config["num_hidden_layers"]
      elif "num_layers" in model_config:
          num_layers = model_config["num_layers"]
      elif "n_layer" in model_config:
          num_layers = model_config["n_layer"]
      else:
          raise ValueError("Please specify `num_layers` in `peft_config`")
      peft_config.num_layers = num_layers

  if peft_config.token_dim is None:
      if "hidden_size" in model_config:
          token_dim = model_config["hidden_size"]
      elif "n_embd" in model_config:
          token_dim = model_config["n_embd"]
      elif "d_model" in model_config:
          token_dim = model_config["d_model"]
      else:
          raise ValueError("Please specify `token_dim` in `peft_config`")
      peft_config.token_dim = token_dim

  if peft_config.num_attention_heads is None:
      if "num_attention_heads" in model_config:
          num_attention_heads = model_config["num_attention_heads"]
      elif "n_head" in model_config:
          num_attention_heads = model_config["n_head"]
      elif "num_heads" in model_config:
          num_attention_heads = model_config["num_heads"]
      elif "encoder_attention_heads" in model_config:
          num_attention_heads = model_config["encoder_attention_heads"]
      else:
          raise ValueError("Please specify `num_attention_heads` in `peft_config`")
      peft_config.num_attention_heads = num_attention_heads

  if getattr(peft_config, "encoder_hidden_size", None) is None:
      setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)
  peft_model = PromptBeforeInstruction(model, peft_config)
  peft_model.print_trainable_parameters()
  return peft_model


# In[ ]:


baseline_selection = "PT" # @param ["None", "ZS", "PT"]

if baseline_selection == "ZS":
  # Zero-shot extreme generalization
  test_extreme_generalization(model, test_loader)
elif baseline_selection == "PT":
  # Prompt tuning
  import huggingface_hub
  import os
  huggingface_hub.login(os.getenv("HUGGINGFACE_TOKEN"))
  peft_model = prompt_tuning(model)
  peft_model = train_fn(peft_model, train_loader, val_loader, num_epochs=1)

