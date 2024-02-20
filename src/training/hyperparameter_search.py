import itertools
from collections import Counter

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from importer.filters.is_connected import IsConnected
from importer.filters.min_size import MinNumberNodes
from importer.filters.min_sparsity import MinSparsity
from importer.importer import AIFDataset
from importer.transforms.embed_node_text import EmbedNodeText
from importer.transforms.extract_additional_features import ExtractAdditionalFeatures
from importer.transforms.keep_selected_nodes import KeepSelectedNodeTypes
from importer.transforms.remove_link_nodes import RemoveLinkNodeTypes
from models.single.gcn import GCNModel
from training.tainer import Trainer


class HyperparameterSearch:
    def __init__(self):
        self.hidden_sizes = [512, 1024]
        self.dropout_rates = [0.1, 0.3]
        self.learning_rates = [0.0001, 0.001]
        self.momentum_values = [0.3, 0.9]
        self.weight_decays = [0.0, 0.2]
        self.batch_sizes = [128, 256]

    def search_hyperparameters(self):
        best_f1_score = 0
        best_hyperparameters = {}

        for (
            hidden_size,
            dropout_rate,
            learning_rate,
            momentum_value,
            weight_decay,
            batch_size,
        ) in itertools.product(
            self.hidden_sizes,
            self.dropout_rates,
            self.learning_rates,
            self.momentum_values,
            self.weight_decays,
            self.batch_sizes,
        ):
            transforms = Compose(
                [
                    KeepSelectedNodeTypes(),
                    RemoveLinkNodeTypes(),
                    EmbedNodeText(),
                    ExtractAdditionalFeatures(),
                ]
            )

            filters = Compose([MinNumberNodes(), MinSparsity(), IsConnected()])

            aif_dataset = AIFDataset(
                root="../data", pre_transform=transforms, pre_filter=filters
            )

            train_size = int(0.7 * len(aif_dataset))
            val_size = int(0.2 * len(aif_dataset))
            test_size = len(aif_dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                aif_dataset, [train_size, val_size, test_size]
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            class_distribution = Counter(
                [label for data in train_dataset for label in data.y.numpy()]
            )
            total_samples = sum(class_distribution.values())
            class_weights = {
                label: total_samples / count
                for label, count in class_distribution.items()
            }
            weights = torch.tensor(
                [class_weights[i] for i in range(len(class_distribution))]
            )

            min_lr = 0.0000000001
            max_lr = 0.001
            step_size_up = int(len(train_loader) * 2)

            model = GCNModel(input_size=773, hidden_size=hidden_size)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum_value,
                weight_decay=weight_decay,
            )
            criterion = nn.CrossEntropyLoss(weight=weights)
            clr_scheduler = CyclicLR(
                optimizer,
                base_lr=min_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode="triangular",
            )

            trainer = Trainer(
                model,
                criterion,
                optimizer,
                train_loader,
                valid_loader,
                test_loader,
                batch_size,
            )
            accuracy, precision, recall, f1 = trainer.train(epochs=20000)

            if f1 > best_f1_score:
                best_f1_score = f1
                best_hyperparameters = {
                    "hidden_size": hidden_size,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate,
                    "momentum_value": momentum_value,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                }
                print("New best F1 score found:", best_f1_score)

        print("Grid search complete.")
        print("Best hyperparameters:", best_hyperparameters)
        print("Best F1 score:", best_f1_score)
