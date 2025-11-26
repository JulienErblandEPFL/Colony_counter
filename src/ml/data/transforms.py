from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(3),
        transforms.ToTensor(),           #NO normalization for now
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])