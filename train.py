import torch
import torchvision
import json
import os
from PIL import Image

# Define the device to use for training and inference
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the classes to detect
classes = ['1', '2']

# Define the model to use for training and inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(progress=True, num_classes=len(classes))
model.to(device)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the dataset and data loader
class LabelmeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.json_files = [f for f in os.listdir(root) if f.endswith('.json')]
        self.image_files = [os.path.splitext(f)[0] + '.png' for f in self.json_files]

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root, image_file)
        json_path = os.path.join(self.root, json_file)
        with open(json_path, 'r') as f:
            labelme_data = json.load(f)
        image = Image.open(image_path).convert('RGB')
        boxes = []
        labels = []
        for shape in labelme_data['shapes']:
            label = shape['label']
            if label not in classes:
                continue
            points = shape['points'][0]
            x_min = points[0]
            y_min = points[1]
            x_max = points[0]
            y_max = points[1]
            if x_min >= x_max or y_min >= y_max:
                continue  # Skip invalid bounding box
            box = [x_min, y_min, x_max, y_max]
            boxes.append(box)
            label_index = classes.index(label)
            labels.append(label_index)
        if len(boxes) == 0:
            print(f"No bounding boxes found for image {image_file}")
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.json_files)

def collate_fn(batch):
    return tuple(zip(*batch))

transform = torchvision.transforms.ToTensor()
dataset = LabelmeDataset('D:/NEWWORLD/markdowntex/Computervision/dataset2/train', transforms=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Define the training function
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(losses).all(), "Loss overflow"
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += losses.item()
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = total_loss / print_freq
            print(f"Epoch {epoch}, batch {batch_idx+1}/{num_batches}, loss={avg_loss}, lr={optimizer.param_groups[0]['lr']}")
            total_loss = 0

# Train the model for several epochs
num_epochs = 20
print_freq = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
    lr_scheduler.step()

# Save the trained model
torch.save(model.state_dict(), './')