from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset): 
    def __init__(self, data, label, transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        #self.data = torch.from_numpy(data).float()
        #self.label = torch.from_numpy(label).long()
        self.data = data
        self.label = label
        self.transform = transform
        self.img_shape = data.shape
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        
        img = Image.fromarray(self.data[index])
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = self.transforms.ToTensor()
            img = img_to_tensor(img)
            #label = torch.from_numpy(label).long()
        return img, label
        
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.data)
