# Based on Diffusers' training example:
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from ..pipelines.ddpm1d_pipeline import DDPM1DPipeline
from ..viz import visualize

logger = getLogger(__name__)


@dataclass
class DDPMTrainer:
    train_batch_size: int = 16
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 50
    num_train_timesteps: int = 100  # for noise scheduling
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

    def fit(self, unet, train_dataset):
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
        optimizer = torch.optim.AdamW(unet.parameters(), lr=self.learning_rate)
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
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)

        logger.info("Prepared everything")
        global_step = 0

        # Now you train the model
        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            train_loss = 0
            for step, (clean_samples, destinations, images) in enumerate(
                train_dataloader
            ):
                # Sample noise to add to the images
                noise = torch.randn(clean_samples.shape).to(clean_samples.device)
                bs = clean_samples.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_samples.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_samples = noise_scheduler.add_noise(
                    clean_samples, noise, timesteps
                )

                with accelerator.accumulate(unet):
                    # Predict the noise residual
                    noise_pred = unet(
                        noisy_samples,
                        timesteps,
                        images=images,
                        return_dict=False,
                    )[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
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
                pipeline = DDPM1DPipeline(
                    unet=accelerator.unwrap_model(unet), scheduler=noise_scheduler
                )

                if (
                    epoch + 1
                ) % self.save_sample_epochs == 0 or epoch == self.num_epochs - 1:
                    self.evaluate(epoch, pipeline, test_dataloader)

                if (
                    epoch + 1
                ) % self.save_model_epochs == 0 or epoch == self.num_epochs - 1:
                    if train_loss / len(train_dataloader) < best_loss:
                        best_loss = train_loss / len(train_dataloader)
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
        clean_samples = batch[0].to(pipeline.device).cpu().numpy()
        destinations = batch[1].to(pipeline.device)
        images = batch[2].to(pipeline.device)
        pout = pipeline(
            destinations=destinations,
            images=images,
            generator=torch.manual_seed(self.seed),
            num_inference_steps=5,  # assumes fewer steps
            guidance_type=None,  # currently guidance supports only single samples
        )
        samples = pout.samples
        destinations = pout.destinations
        images = pout.images
        # renormalize samples and destinations
        image_size = images.shape[-1]
        clean_samples = (clean_samples + 1) * image_size / 2
        samples = (samples + 1) * image_size / 2
        destinations = (destinations + 1) * image_size / 2

        fig, axes = plt.subplots(1, self.eval_batch_size, figsize=(16, 4))
        for k in range(self.eval_batch_size):
            visualize(
                images[k, 0].squeeze(),
                destinations[k].T,
                path=clean_samples[k].T,
                pred_path=samples[k].T,
                ax=axes[k],
            )
        fig.tight_layout()
        sampledir = Path(self.logdir).joinpath("samples")
        sampledir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{sampledir}/{epoch:04d}.png")
