# Based on Diffusers' training example:
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from ..pipelines.cprm_pipeline import CPRMPipeline

logger = getLogger(__name__)


@dataclass
class CPRMTrainer:
    train_batch_size: int = 16
    eval_batch_size: int = 4
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 50
    save_sample_epochs: int = 5
    save_model_epochs: int = 25
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    logdir: str = "logdir"
    seed: int = 0

    def fit(self, model, train_dataset):
        tbdir = Path(self.logdir).joinpath("tb")
        tbdir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=tbdir,
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_example")

        # Prepare everything
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * self.num_epochs),
        )
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        logger.info("Prepared everything")
        global_step = 0

        # Now you train the model
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            train_loss = 0
            for step, (sample_images, destinations, images) in enumerate(
                train_dataloader
            ):
                with accelerator.accumulate(model):
                    pred_maps = model(images)
                    loss = F.binary_cross_entropy_with_logits(pred_maps, sample_images)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                train_loss += logs["loss"]
                progress_bar.set_postfix(**logs)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = CPRMPipeline(unet=accelerator.unwrap_model(model))

                if (
                    epoch + 1
                ) % self.save_sample_epochs == 0 or epoch == self.num_epochs - 1:
                    self.evaluate(epoch, pipeline, test_dataloader)

                if (
                    epoch + 1
                ) % self.save_model_epochs == 0 or epoch == self.num_epochs - 1:
                    pipeline.save_pretrained(self.logdir)
                accelerator.log(
                    {
                        "loss": train_loss / len(train_dataloader),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

    def evaluate(self, epoch, pipeline, test_dataloader):
        batch = next(iter(test_dataloader))
        data = batch[0]
        destinations = batch[1].to(pipeline.device)
        images = batch[2].to(pipeline.device)
        pout = pipeline(
            destinations=destinations,
            images=images,
            num_samples=1000,
        )
        samples = pout.samples
        destinations = pout.destinations
        images = pout.images

        fig, axes = plt.subplots(1, self.eval_batch_size, figsize=(16, 4))
        for k in range(self.eval_batch_size):
            axes[k].imshow(images[k, 0].T, cmap="gray_r")
            axes[k].imshow(data[k].T, alpha=0.5)
            axes[k].scatter(samples[k, 0], samples[k, 1], s=1, c="r")
        fig.tight_layout()
        sampledir = Path(self.logdir).joinpath("samples")
        sampledir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{sampledir}/{epoch:04d}.png")
