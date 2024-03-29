import os
import threading

import matplotlib.pyplot as plt
import networkx as nx


class ThreadUtils:
    def __init__(self):
        self.print_lock = threading.Lock()
        self.save_lock = threading.Lock()

    def thread_safe_print(self, message):
        with self.print_lock:
            print(message)

    def acquire_save_lock(self):
        self.save_lock.acquire()

    def release_save_lock(self):
        self.save_lock.release()


class PlotUtils:
    def __init__(self):
        pass

    def visualize_graph(data, output_dir):
        graph = data.graph
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(graph, seed=42, k=0.2)
        nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=200)
        nx.draw_networkx_edges(graph, pos, edge_color="gray", width=0.5)
        edge_labels = {
            (edge[0], edge[1]): graph.edges[edge]["type"] for edge in graph.edges
        }
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Graph Visualization: {data.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{data.name}_graph.png"))
        plt.close()

    def visualize_training_losses(train_losses, valid_losses, output_dir):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-")
        plt.plot(
            epochs, valid_losses, label="Validation Loss", marker="o", linestyle="-"
        )
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_validation_losses.png"))
        plt.close()

    def visualize_training_accuracy(train_accuracy, valid_accuracy, output_dir):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accuracy) + 1)
        plt.plot(
            epochs, train_accuracy, label="Training Accuracy", marker="o", linestyle="-"
        )
        plt.plot(
            epochs,
            valid_accuracy,
            label="Validation Accuracy",
            marker="o",
            linestyle="-",
        )
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_validation_accuracy.png"))
        plt.close()
