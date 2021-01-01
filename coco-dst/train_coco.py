"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import glob
import logging
import os
import pickle
import random
import re
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
from transformers import T5Config, WEIGHTS_NAME
import math
from utility.checkpoint import *
from utility.configure import *
from utility.data import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}


def prepare(args):
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("--should_continue is true, but no checkpoint found in --output_dir")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_optimizer_scheduler(args, model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler


def prepare_for_training(args, model, train_dataloader):
    # total iteration and batch size
    if args.num_train_steps > 0:
        t_total = args.num_train_steps
        args.num_train_epochs = args.num_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.num_train_steps = t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    return args, model, optimizer, scheduler


def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    return model, tokenizer, model_class, tokenizer_class, args


def get_training_info(dataloader, args):
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    return global_step, epochs_trained, steps_trained_in_current_epoch


def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step,
                steps_trained_in_current_epoch, tb_writer, args):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        src_ids = batch["src_ids"].to(args.device)
        src_mask = batch["src_mask"].to(args.device)
        tgt_ids = batch["tgt_ids"].to(args.device)
        tgt_mask = batch["tgt_mask"].to(args.device)
        outputs = model(input_ids=src_ids, attention_mask=src_mask, decoder_attention_mask=tgt_mask, labels=tgt_ids)
        loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            # save checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, tokenizer, global_step, args)

                if (
                        args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer, prefix="{}-{}".format("checkpoint", global_step))
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        if args.num_train_steps > 0 and global_step > args.num_train_steps:
            epoch_iterator.close()
            break

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    # Prepare dataloader
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args, split='train')
    args, model, optimizer, scheduler = prepare_for_training(args, model, train_dataloader)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(train_dataset)))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(args.num_train_steps))

    global_step, epochs_trained, steps_trained_in_current_epoch = get_training_info(train_dataloader, args)

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for _ in train_iterator:

        model, optimizer, scheduler, global_step, tr_loss, logging_loss = train_epoch(model, tokenizer, optimizer,
                                                                                      scheduler, train_dataloader,
                                                                                      tr_loss, logging_loss,
                                                                                      global_step,
                                                                                      steps_trained_in_current_epoch,
                                                                                      tb_writer, args)

        if args.num_train_steps > 0 and global_step > args.num_train_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        if args.local_rank in [-1, 0]:
            save_checkpoint(model, optimizer, scheduler, tokenizer, global_step, args)
            if (
                    args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer, prefix="{}-{}".format("checkpoint", global_step))
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = MultiWOZForT5(args.max_src_len, args.max_tgt_len, args.eval_data_file, tokenizer,
                                 args.shuffle_turn_label)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare dataloader
    eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='eval')

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            src_ids = batch["src_ids"].to(args.device)
            src_mask = batch["src_mask"].to(args.device)
            tgt_ids = batch["tgt_ids"].to(args.device)
            tgt_mask = batch["tgt_mask"].to(args.device)
            outputs = model(input_ids=src_ids, attention_mask=src_mask, decoder_attention_mask=tgt_mask, labels=tgt_ids)
            loss = outputs[0]

        eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity}
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    args = ArgsParser().parse()
    prepare(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, tokenizer_class, args = get_model_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    logger.info("Training/evaluation parameters {}".format(args))

    # Training
    if args.do_train:
        train_dataset = MultiWOZForT5(args.max_src_len, args.max_tgt_len, args.train_data_file, tokenizer,
                                      args.shuffle_turn_label, args.train_data_ratio)
        with open(args.output_dir + '/train_data.json', 'w') as outfile:
            json.dump(train_dataset.data, outfile, indent=4)
        global_step, train_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = {}, average loss = {}".format(global_step, train_loss))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = []

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("models.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: {}".format(checkpoints))

            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                model = model_class.from_pretrained(checkpoint)
                tokenizer = tokenizer_class.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, model, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

            print("**** Evaluation results on all checkpoint ****")
            best_metric = math.inf
            best_key = ""
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                if (results[key] < best_metric):
                    best_key = key
                    best_metric = results[key]

            print("**** Best results are achieved on " + str(best_key) + " and its perplexity is: " + str(
                best_metric.item()))

        else:
            checkpoint = args.output_dir
            model = model_class.from_pretrained(checkpoint)
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
