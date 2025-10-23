from torch.utils.data import Dataset
import torchvision as tv
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[index])
        else:
            return data


def process_pixmo_data(img, pts):
    """
    Add 0-padding to the image to make it square and transform the points accordingly
    
    Args:
        img: numpy array (H, W, C)
        pts: numpy array (x, y) in original image
    
    Returns:
        img_square: numpy array (H, W, C)
        pts_square: numpy array (x, y) in padded image
    """
    img_h, img_w = img.shape[:2]
    size = max(img_h, img_w)

    # Padding amounts
    pad_top = (size - img_h) // 2
    pad_bottom = (size - img_h) - pad_top
    pad_left = (size - img_w) // 2
    pad_right = (size - img_w) - pad_left

    # Pad image to square
    img_square = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    ).astype(np.float32)

    # Transform points
    pts_square = np.asarray(pts, dtype=np.float32)
    # Shift by padding
    pts_square[..., 0] += pad_left
    pts_square[..., 1] += pad_top

    return img_square, pts_square


class PixmoPointsDataset(Dataset):
    def __init__(self, data, split="train", size=224):
        self.split = split
        self.data = data

        # only consider square images
        self.height = size
        self.width = size

        self.transforms = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.Resize(size, antialias=True),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load images, labels, points from the data
        item = self.data[idx]
        image = item["image"]
        if 'label' in item:
            label = item["label"]
        else:
            label = -1 # no label provided for test set
        point = item["point"]

        # process the image and points to make it square
        image, point = process_pixmo_data(image, point)

        # rescale points to self.width and self.height
        img_h, img_w = image.shape[:2] # 224x224
        sx = float(self.width) / float(img_w)
        sy = float(self.height) / float(img_h)
        point[..., 0] *= sx
        point[..., 1] *= sy
        
        # apply transforms to resize the image to self.height and self.width and normalize it
        image = image / 255.0 # normalize to [0, 1]
        image = self.transforms(image)

        # flip the image with a probability of 0.5
        if self.split == "train":
            if np.random.rand() < 0.5:
                # flip the image and points
                image = tv.transforms.functional.hflip(image)
                point[..., 0] = self.width - point[..., 0]


        #cropping around point 
        cropx, cropy = int(point[0]), int(point[1])
        crop = 32
        half_crop = crop //2
        x1 = max(0, cropx - half_crop)
        y1 = max(0, cropy - half_crop)
        x2 = min(self.width, cropx + half_crop)
        y2 = min(self.height, cropy + half_crop) #dimensions of cropped image

        cropped_image = image[:, y1:y2, x1:x2]
        cropped_image = tv.transforms.functional.resize(cropped_image, (self.height, self.width)) #resize back to original size

        
        return image, cropped_image, label, point