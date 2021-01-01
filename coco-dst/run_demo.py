"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import logging
import numpy as np
import torch
import random
from utility.data import *
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "t5": (T5ForConditionalGeneration, T5Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--eval_data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Dataset to do eval")

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_sample", action="store_true", help="Do sample during decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--shuffle_turn_label", action='store_true', help="if we shuffle conditional turn label")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    dataset = MultiWOZForT5_Interact(data_dir=args.eval_data_file, tokenizer=tokenizer,
                                     shuffle_turn_label=args.shuffle_turn_label)
    logger.info(args)
    play = True
    while (play):
        print("------------------------------------------")
        print("----------------- NEXT -------------------")
        print("Dataset len: ", len(dataset))
        valid_input_flag = True
        while (valid_input_flag):
            try:
                data_idx = input("Dataset index >>> ")
                idx = int(data_idx.strip())
                assert idx >= 0 and idx < len(dataset)
                dataset.print_value(idx)
                valid_input_flag = False
            except:
                if ("exit" in data_idx or "quit" in data_idx):
                    import sys
                    sys.exit()
                print("Index out of boundary or not valid")

        valid_input_flag = True
        while (valid_input_flag):
            try:
                print(
                    "Input your belief state as "'domain1-name1-value1 , domain2-name2-value2'" or ENTER to use default belief state")
                prompt_text = input(">>> ").strip().lower()
                dataset.prompt_text = prompt_text
                valid_input_flag = False
            except:
                valid_input_flag = True

        encoded_prompt = dataset[idx]
        print("-------------------------")
        print("Input Tokens:")
        print(tokenizer.decode(encoded_prompt))
        print("-------------------------")
        print("Generated User Utterence Candidates:")
        print("-------------------------")
        encoded_prompt = torch.tensor(encoded_prompt, dtype=torch.long).view(1, -1)
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
        )
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            generated_sequences.append(text)
            print("(" + str(generated_sequence_idx) + "): " + text)


if __name__ == "__main__":
    main()
