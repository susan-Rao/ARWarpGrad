import random
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image
import torchvision.transforms as transforms

# Define a training image loader for Omniglot that specifies transforms on images.
train_transformer_Omniglot = transforms.Compose([
    transforms.Resize((28, 28)),
    # transforms.RandomRotation(90),  # NOTE DO or DO NOT? need to be consistent
    transforms.ToTensor()
])

# Define a evaluation loader, no random rotation.
eval_transformer_Omniglot = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor()])

# Define a training image loader for ImageNet (miniImageNet, tieredImageNet)
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize((84, 84)),
    # transforms.RandomRotation([0, 90, 180, 270]),
    transforms.ToTensor()
])

# Define a evaluation loader, no random rotation.
eval_transformer_ImageNet = transforms.Compose(
    [transforms.Resize((84, 84)),
     transforms.ToTensor()])


def split_omniglot_characters(data_dir, SEED):
    if not os.path.exists(data_dir):
        raise Exception("Omniglot data folder does not exist.")

    character_folders = [os.path.join(data_dir, family, character) \
                        for family in os.listdir(data_dir) \
                        if os.path.isdir(os.path.join(data_dir, family)) \
                        for character in os.listdir(os.path.join(data_dir, family))]
    random.seed(SEED)
    random.shuffle(character_folders)

    test_ratio = 0.2  # against total data
    val_ratio = 0.2  # against train data
    num_total = len(character_folders)
    num_test = int(num_total * test_ratio)
    num_val = int((num_total - num_test) * val_ratio)
    num_train = num_total - num_test - num_val

    train_chars = character_folders[:num_train]
    val_chars = character_folders[num_train:num_train + num_val]
    test_chars = character_folders[-num_test:]
    return train_chars, val_chars, test_chars


def load_imagenet_images(data_dir):
    if not os.path.exists(data_dir):
        raise Exception("ImageNet data folder does not exist.")

    train_classes = [os.path.join(data_dir, 'train', family)\
                    for family in os.listdir(os.path.join(data_dir, 'train')) \
                    if os.path.isdir((os.path.join(data_dir, 'train', family)))]
    val_classes = [os.path.join(data_dir, 'val', family)\
                     for family in os.listdir(os.path.join(data_dir, 'val')) \
                     if os.path.isdir((os.path.join(data_dir, 'val', family)))]
    test_classes = [os.path.join(data_dir, 'test', family)\
                   for family in os.listdir(os.path.join(data_dir, 'test')) \
                   if os.path.isdir((os.path.join(data_dir, 'test', family)))]

    return train_classes, val_classes, test_classes


class Task(object):
    def __init__(self, character_folders, num_classes, support_num, query_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.support_num = support_num
        self.query_num = query_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = list(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:support_num]
            self.test_roots += samples[c][support_num:support_num + query_num]

        samples = dict()
        self.meta_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.meta_roots += samples[c][:support_num]

        self.train_labels = [
            labels[self.get_class(x)] for x in self.train_roots
        ]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]
        self.meta_labels = [labels[self.get_class(x)] for x in self.meta_roots]

    def get_class(self, sample):
        # raise NotImplementedError("This is abstract class")
        return os.path.join('/',*sample.split('/')[:-1])


class OmniglotTask(Task):
    """
    Class for defining a single few-shot task given Omniglot dataset.
    """

    def __init__(self, *args, **kwargs):
        super(OmniglotTask, self).__init__(*args, **kwargs)


class ImageNetTask(Task):
    """
    Class for defining a single few-shot task given ImageNet dataset.
    """

    def __init__(self, *args, **kwargs):
        super(ImageNetTask, self).__init__(*args, **kwargs)


class FewShotDataset(Dataset):

    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self._train = False

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(types, task):
    if isinstance(task, OmniglotTask):
        train_transformer = train_transformer_Omniglot
        eval_transformer = eval_transformer_Omniglot
    else:
        train_transformer = train_transformer_ImageNet
        eval_transformer = eval_transformer_ImageNet

    dataloaders = {}
    for split in ['train', 'test', 'meta']:
        if split in types:
            # use the train_transformer if training data,
            # else use eval_transformer without random flip
            if split == 'train':
                train_filenames = task.train_roots
                train_labels = task.train_labels
                train_batch_size = len(train_filenames)
                train_data = FewShotDataset(train_filenames, train_labels,train_transformer)
                dl = DataLoader(
                    train_data,
                    batch_size=train_batch_size,  # full-batch in episode
                    shuffle=True)  # TODO args: num_workers, pin_memory
            elif split == 'test':
                test_filenames = task.test_roots
                test_labels = task.test_labels
                test_batch_size = len(test_filenames)
                test_data = FewShotDataset(test_filenames, test_labels, eval_transformer)
                dl = DataLoader(
                    test_data,
                    batch_size=test_batch_size,  # full-batch in episode
                    shuffle=True)
            elif split == 'meta':
                meta_filenames = task.meta_roots
                meta_labels = task.meta_labels
                meta_batch_size = len(meta_filenames)
                meta_data = FewShotDataset(meta_filenames, meta_labels, train_transformer)
                dl = DataLoader(
                    meta_data,
                    batch_size=meta_batch_size,  # full-batch in episode
                    shuffle=True)  # TODO args: num_workers, pin_memory
            else:
                raise NotImplementedError()
            dataloaders[split] = dl

    return dataloaders

