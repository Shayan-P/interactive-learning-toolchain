from dataclasses import dataclass
from typing import List
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from tqdm.notebook import tqdm
from utils import save_model
import torch


@dataclass
class Config:
    batch_size: int = 128
    checkpoint_interval: int = 50
    inspection_interval: int = 50
    lr: float = 1e-3


@dataclass
class InputChannel:
    config: Config
    stop_now: bool = False
    checkpoint_now: bool = False
    inspect_func = lambda input_channel: None
    inspect_now: bool = False


@dataclass
class OutputChannel:
    losses: List[float]



def train(input_channel: InputChannel, output_channel: OutputChannel, model, loss_fn, device, dataloader, optimizer, epochs, trial_name, scheduler=None): # todo we can do infinite epochs...
    model.train()

    pbar = tqdm(range(len(dataloader)))
    
    output_channel.losses = []
    
    last_epoch_loss = 0
    for epoch in tqdm(range(epochs)):
        if input_channel.stop_now:
            input_channel.stop_now = False
            break
            
        cur_epoch_loss = 0
        pbar.reset()
        for batch_idx, (data, target) in enumerate(dataloader):
            if input_channel.stop_now:
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            max_grad = max(
                [
                    torch.linalg.norm(p.grad).item()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
            pbar.set_description(
                f'Epoch {epoch+1} Batch_Idx {batch_idx} Loss: {loss.item():.6f} Last_Epoch_Loss={last_epoch_loss:.6f} MaxGrad={max_grad:.4f} Lr={optimizer.param_groups[0]["lr"]}'
            )
            pbar.update(1)

            output_channel.losses.append(loss.item())
            cur_epoch_loss += loss.item()
            
        last_epoch_loss = cur_epoch_loss / len(dataloader)

        if (
            (input_channel.config.inspection_interval is not None)
            and (input_channel.inspect_func is not None)
            and (input_channel.inspect_now or (epoch % input_channel.config.inspection_interval == 0))
        ):
            input_channel.inspect_func()
        if input_channel.inspect_now:
            input_channel.inspect_now = False

        if (epoch % input_channel.config.checkpoint_interval == 0) or input_channel.checkpoint_now:
            name = f"{trial_name}_epoch_{epoch}"
            save_model(model, name)
            print(f"saved {name}")
        if input_channel.checkpoint_now:
            input_channel.checkpoint_now = False

        if scheduler is not None:
            scheduler.step()            


class Tuner:
    def __init__(self, optimizer, config) -> None:
        self.optimizer = optimizer
        self.config = config

        self.input_channel = InputChannel(config=self.config)
        self.output_channel = OutputChannel(losses=[])

    def show(self):
        def tune(lr, stop_now, checkpoint_now, inspect_now):
            self.config.lr = lr
            self.input_channel.stop_now = stop_now
            self.input_channel.checkpoint_now = checkpoint_now
            self.input_channel.inspect_now = inspect_now
            
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        interact(tune,
                lr=widgets.BoundedFloatText(
                    value=self.config.lr,
                    min=0,
                    max=1,
                    description='LR:',
                    disabled=False),
                stop_now=widgets.ToggleButton(
                    value=self.input_channel.stop_now,
                    description='Stop Now',
                    disabled=False,
                    button_style='danger',
                    tooltip='Stops the Training',
                    # icon='check'
                ),
                checkpoint_now=widgets.ToggleButton(
                    value=self.input_channel.checkpoint_now,
                    description='Checkpoint Now',
                    disabled=False,
                    button_style='danger',
                    tooltip='Makes a checkpoint now',
                    # icon='check'
                ),
                inspect_now=widgets.ToggleButton(
                    value=self.input_channel.checkpoint_now,
                    description='Inspect Now',
                    disabled=False,
                    button_style='danger',
                    tooltip='Inspects the model now',
                    # icon='check'
                ),
        )
