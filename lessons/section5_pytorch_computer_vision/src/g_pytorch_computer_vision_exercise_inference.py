import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchmetrics

import pathlib

from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt

from common.device import get_best_device, print_device_info

# prepare the data for inference/prediction
test_data = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None,
)

class_name = test_data.classes
print(f"class names are: {class_name}")

BATCH_SIZE = 32

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class TinyVGG(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int) -> None:
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(1, in_features, 28, 28)  # [batch=1, C, H, W]
            dummy = self.conv_stack_2(self.conv_stack_1(dummy))
            # dummy shape after both conv stacks: [1, hidden_units, H_out, W_out]
            linear_in_features = dummy.flatten(start_dim=1).shape[1]
            # flatten(start_dim=1): collapse all dims except batch → [1, hidden_units*H_out*W_out]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=linear_in_features,  # auto-computed above
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))


"""
10. Make predictions using your trained model and visualize 
    at least 5 of them comparing the prediction to the target label.
"""
device = get_best_device()
print_device_info(device)

my_tinyvgg_gpu_inference = TinyVGG(
    in_features=1,
    hidden_units=10,
    out_features=len(class_name),
)

loss_fn = torch.nn.CrossEntropyLoss()

# load the model trained in f_pytorch_computer_vision_exercise_gpu.py
MODEL_PATH = pathlib.Path("lessons/section5_pytorch_computer_vision/models")
MODEL_TINYVGG_MNIST_PATH = MODEL_PATH / "section5_model_tinyvgg_mnist.pth"

my_tinyvgg_gpu_inference.load_state_dict(
    torch.load(f=MODEL_TINYVGG_MNIST_PATH, weights_only=True)
)
my_tinyvgg_gpu_inference.to(device=device)

# Make predictions and get the prediction label
prediction_labels: list[torch.Tensor] = []

my_tinyvgg_gpu_inference.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)

        # Logits and prediction labels
        y_logits: torch.Tensor = my_tinyvgg_gpu_inference(X)
        y_prediction_labels_batch = y_logits.argmax(dim=1)
        prediction_labels.append(y_prediction_labels_batch.cpu())

print(f"prediction_labels[0] shape: {prediction_labels[0].shape}")
prediction_labels_tensor = torch.cat(prediction_labels, dim=0)
print(prediction_labels_tensor[:10])

# Plot prediction and compare with origin
plt.figure(figsize=(12, 12))
nrows = 5
ncolunms = 5
for index in range(nrows * ncolunms):
    plt.subplot(nrows, ncolunms, index + 1)

    # get a random index of image and plot its true and prediction class name
    image_rand_index = torch.randint(low=0, high=len(test_data), size=(1,)).item()

    raw_image, raw_label = test_data[int(image_rand_index)]
    prediction_label = prediction_labels_tensor[int(image_rand_index)]

    true_class_name = class_name[raw_label]
    prediction_class_name = class_name[prediction_label]

    plt.imshow(
        raw_image.squeeze(dim=0), cmap="gray"
    )  # remove the first dim -> (1, 28, 28)
    title_text = f"Pred: {prediction_class_name} | Truth: {true_class_name}"

    if prediction_label == raw_label:
        plt.title(title_text, fontsize=10, c="g") # correct prediction -- green
    else:
        plt.title(title_text, fontsize=10, c="r") # wrong prediction -- red
    plt.axis(False)

plt.tight_layout()
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/g_line_173_visualize_random_predictions.png"
)

"""
11. Plot a confusion matrix comparing your model's predictions to the truth labels.
"""
confusion_matrix_calculator = torchmetrics.ConfusionMatrix(
    task="multiclass",
    num_classes=len(class_name)
)

confusion_matrix = confusion_matrix_calculator(
    preds=prediction_labels_tensor, target=test_data.targets
)

fig, ax = plot_confusion_matrix(
    conf_mat=confusion_matrix.numpy(), class_names=class_name, figsize=(10, 7)
)
plt.tight_layout()
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/g_line_202_plot_confusion_matrix.png"
)

