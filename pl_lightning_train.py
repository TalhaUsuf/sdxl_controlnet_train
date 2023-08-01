
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
from rich.console import Console
import torchvision.transforms as transforms



p = Console().log

class lineart_DM(pl.LightningDataModule):
    def __init__(self, 
                 hf_dataset_name: str = "zbulrush/lineart", 
                 batch_size: int = 32,  # Added the equal sign (=) for default value
                 num_workers: int = 4,
                 resolution : int = 512,
                 image_column : str = None,
                 conditioning_image_column : str = None,
                 caption_column : str = None,
                 
                 ):  
        super().__init__() 

        # Initialize the arguments as class attributes
        self.hf_dataset_name = hf_dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        # needed to check the dataset columns
        self.image_column = image_column
        self.conditioning_image_column = conditioning_image_column
        self.caption_column = caption_column
        self.resolution = resolution

        self.save_hyperparameters() 

    def setup(self, stage=None):
        # load the dataset from hf_hub
        dataset = load_dataset(
                self.hf_dataset_name,
            )
        p(f"dataset has been loaded from hf_hub: {self.hf_dataset_name}", style="bold green")
        column_names = dataset["train"].column_names
        p(f"column names are {column_names}", style="bold green")

        # ==========================================================================
        #                             prase the dataset columns                                   
        # ==========================================================================
        if self.image_column is None:
            image_column = column_names[0]
            
        else:
            image_column = self.image_column
            p(f"image column defaulting to {image_column}", style="bold green")
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{self.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            
        else:
            caption_column = self.caption_column
            p(f"caption column defaulting to {caption_column}", style="bold green")
            if caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.conditioning_image_column is None:
            conditioning_image_column = column_names[2]
        else:
            conditioning_image_column = self.conditioning_image_column
            p(f"conditioning image column defaulting to {conditioning_image_column}", style="bold green")
            if conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{self.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        
        # confirm that dataset is loaded correctly
        assert dataset is not None, "dataset is not loaded correctly"
        
        
        # shuffle the dataset
        self.train_dataset = dataset["train"].shuffle(seed=1000)
        
        p(f"dataset has been shuffled", style="bold green")
        
        # preprocess the dataset
        self.process_dataset()
        
        
        
    def process_dataset(self):
        
        # define the transforms for both the images and the conditioning images
        image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(), # [0, 1] with shape (3, H, W)
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[self.image_column]]
            images = [image_transforms(image) for image in images]

            conditioning_images = [image.convert("RGB") for image in examples[self.conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

            examples["pixel_values"] = images
            examples["conditioning_pixel_values"] = conditioning_images

            return examples


        self.train_dataset = self.train_dataset.with_transform(preprocess_train)

        return self.train_dataset


    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

        prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

        add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
        add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompt_ids": prompt_ids,
            "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        }


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)



if __name__ == '__main__':
    dm = lineart_DM(image_column="image",
               conditioning_image_column="conditioning_image",
               caption_column="text",
               )
    dm.setup()
    
    # test the samples from dataloader
    for k in dm.train_dataloader():
        print(k)
        break