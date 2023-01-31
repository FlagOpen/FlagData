# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import time
from loguru import logger
from .evaluate import evaluate
from tqdm import tqdm
from torch.optim import SGD, Adam
from transformers import get_linear_schedule_with_warmup
from torch.cuda import amp
from collections import OrderedDict


class DistilledData:
    def __init__(self, config, model, train_loader, initial_model_path):
        self.num_classes = config["data"]["num_classes"]
        self.data_size = config["distill"]["data_size"]
        self.device = config["basic"]["device"]
        self.use_amp = config["basic"]["use_amp"]
        self.dtype = config["basic"]["dtype"]
        self.random_init = config["distill"]["random_init"]
        self.n_inner_steps = config["distill"]["n_inner_steps"]
        self.accum_loss = config["distill"]["accum_loss"]
        self.distill_max_grad_norm = config["distill"]["distill_max_grad_norm"]
        self.logging_steps = config["basic"]["logging_steps"]
        self.initial_model_path = initial_model_path
        self.model = model
        self.inputs_embeds = torch.randn(
            self.num_classes * self.data_size,
            *(model.bert_config.max_position_embeddings, model.bert_config.dim,)
        )
        label_classes = torch.tensor(
            [[c] * self.data_size for c in range(self.num_classes)]
        ).view(-1)
        self._labels = torch.eye(self.num_classes)[label_classes]
        self.model_lr, self.step_lr_gamma = None, None
        if self.model_lr is None:
            self.model_lr = torch.tensor(config["distill"]["distill_model_lr"])
        if self.step_lr_gamma is None:
            self.step_lr_gamma = torch.tensor(
                config["distill"]["distill_step_lr_gamma"])

        # set on device
        self.inputs_embeds = self.inputs_embeds.to(self.device)
        self._labels = self._labels.to(self.device)

        self.model_lr = self.model_lr.to(self.device)
        self.step_lr_gamma = self.step_lr_gamma.to(self.device)

        # train data loader
        self.train_loader = train_loader
        # number of traning steps
        num_tot_train_steps = len(train_loader) * \
            config["distill"]["n_distill_epochs"]
        # set optimizer of distilled data
        self.optimize_param_list = [self.inputs_embeds]

        if config["distill"]["optimize_lr"]:
            self.optimize_param_list += [self.model_lr, self.step_lr_gamma]
        for param in self.optimize_param_list:
            param.requires_grad = True
        self.d_optimizer = Adam(self.optimize_param_list,
                                lr=config["distill"]["distill_lr"])
        # scheduler (linear decay with linear warmup)
        self.d_scheduler = get_linear_schedule_with_warmup(
            self.d_optimizer,
            int(num_tot_train_steps * config["distill"]["distill_warmup_ratio"]),
            num_tot_train_steps,
        )
        # gradient scaler for mixed precision
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        # initial model parameters
        self.initial_state_dict = torch.load(self.initial_model_path)

    def train_distilled_data(self, epoch):
        # init model with original params each epoch
        self.model.load_state_dict(self.initial_state_dict)
        # training loop
        cur_num, cur_before_loss, cur_after_loss = 0, 0, 0
        cur_before_correct, cur_after_correct = 0, 0
        with tqdm(self.train_loader, ncols=140, desc=f"Epoch[{epoch+1}]") as pbar:
            for outer_step, (input_ids, attention_mask, labels) in enumerate(pbar):
                # initialize model parameters
                if self.random_init:
                    self.model.reset_additional_parameters()
                # model parameters
                weights = OrderedDict(self.model.named_parameters())

                batch_size = len(input_ids)
                cur_num += batch_size

                # acc & loss of initial parameters (before updating with distilled data)
                with torch.no_grad():
                    with amp.autocast(dtype=self.dtype, enabled=self.use_amp):
                        before_losses, before_logits, _ = self.model.forward_with_params(
                            input_ids=input_ids.to(self.device),
                            attention_mask=attention_mask.to(self.device),
                            labels=labels.to(self.device),
                            weights=weights,
                        )
                        cur_before_loss += before_losses.mean().item() * batch_size
                        cur_before_correct += (
                            before_logits.cpu().argmax(1).eq(labels).sum().item()
                        )

                # update model parameters with distilled data
                loss = 0
                for inner_step in range(self.n_inner_steps):
                    # forward
                    d_losses, _, bert_outputs = self.model.forward_with_params(
                        inputs_embeds=self.inputs_embeds,
                        labels=self._labels,
                        weights=weights,
                        output_attentions=True,
                    )
                    d_loss = d_losses.mean()
                    d_loss = d_loss * self.model_lr * \
                        (self.step_lr_gamma**inner_step)

                    # backward
                    grads = torch.autograd.grad(
                        d_loss, weights.values(), create_graph=True, allow_unused=True
                    )
                    # update parameters (SGD)
                    weights = OrderedDict(
                        (name, param - grad) if grad is not None else (name, param)
                        for ((name, param), grad) in zip(weights.items(), grads)
                    )

                    if self.accum_loss or (inner_step + 1) == self.n_inner_steps:
                        # loss of updated parameters (after each gradient step)
                        with amp.autocast(dtype=self.dtype, enabled=self.use_amp):
                            after_losses, after_logits, _ = self.model.forward_with_params(
                                input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                                labels=labels.to(self.device),
                                weights=weights,
                            )
                            after_loss = after_losses.mean()
                            loss += after_loss

                cur_after_loss += after_loss.item() * batch_size
                cur_after_correct += (
                    after_logits.cpu().argmax(1).eq(labels).sum().item()
                )

                self.d_optimizer.zero_grad()
                # backward
                self.scaler.scale(loss).backward()
                # unscale gradients (for gradient clipping)
                self.scaler.unscale_(self.d_optimizer)
                # gradient cliping
                torch.nn.utils.clip_grad_norm_(
                    self.optimize_param_list, self.distill_max_grad_norm
                )
                self.scaler.step(self.d_optimizer)
                self.scaler.update()
                self.d_scheduler.step()

                # logging
                if (outer_step + 1) % self.logging_steps == 0:
                    logger.info(
                        "Epoch[{:.2f}] | (before) loss: {:>6.4f}, acc: {:5.2%}"
                        " -> (after) loss: {:>6.4f}, acc: {:5.2%}"
                        " | lr={:.2E}, gamma={:.2f}".format(
                            epoch + (outer_step + 1) / len(pbar),
                            cur_before_loss / cur_num,
                            cur_before_correct / cur_num,
                            cur_after_loss / cur_num,
                            cur_after_correct / cur_num,
                            self.model_lr.item(),
                            self.step_lr_gamma.item(),
                        )
                    )
                    cur_num, cur_before_loss, cur_after_loss = 0, 0, 0
                    cur_before_correct, cur_after_correct = 0, 0

                # update infomation of progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{after_loss.item():.4}",
                        "lr": f"{self.d_scheduler.get_last_lr()[0]:.1E}",
                        "gd_scale": f"{self.scaler.get_scale()}",
                    }
                )

    def train_model_on_distilled_data(self):
        # optimizer
        model_opt = SGD(self.model.parameters(), lr=1.0)
        # gradient updating with distilled data
        start_time = time.time()
        for inner_step in range(self.n_inner_steps):
            losses, _, bert_outputs = self.model(
                inputs_embeds=self.inputs_embeds,
                labels=self._labels,
                output_attentions=True,
            )
            loss = losses.mean()
            loss = loss * self.model_lr * (self.step_lr_gamma**inner_step)
            # backward
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
        end_time = time.time() - start_time
        logger.info(f"Time for model traning : {end_time:.2f}s")

    @property
    def data_dict(self):
        return {
            "config": {
                "input_embeds_shape": self.inputs_embeds.shape[1:],
                "num_classes": self.num_classes,
                "data_size": self.data_size,
            },
            "inputs_embeds": self.inputs_embeds.cpu().data,
            "labels": self._labels.cpu().data,
            "lr": self.model_lr.cpu().data,
            "gamma": self.step_lr_gamma.cpu().data,
        }

    def save_distilled_data(self, path):
        # save data as dict
        torch.save(self.data_dict, path)

    def load_distilled_data(self, distill_data_path):
        data_dict = torch.load(distill_data_path)
        self.inputs_embeds = data_dict["inputs_embeds"].to(self.device)
        self._labels = data_dict["labels"].to(self.device)
        self.model_lr = data_dict["lr"].to(self.device)
        self.step_lr_gamma = data_dict["gamma"].to(self.device)

    def test_distilled_data(self, test_loader):
        # init model with original params
        self.model.load_state_dict(self.initial_state_dict)
        self.train_model_on_distilled_data()
        test_acc, test_loss = evaluate(
            self.model, test_loader, self.dtype, self.use_amp, self.device)
        logger.info(
            "Evaluate on Test dataset | loss: {:>6.4f}, acc: {:5.2%}".format(
                test_loss,
                test_acc,
            )
        )
        return test_loss, test_acc
