# Graph-Based Argument Structure Prediction

## Model Descriptions

#### Single GNN Architectures

| Architecture  | Description                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| GCN           | GCN captures local node patterns by aggregating information from immediate neighbors. It employs a graph convolution operation that considers the feature information from neighboring nodes, allowing each node to update its representation based on its local context. GCN is effective for capturing localized patterns and dependencies within the graph.        |
| GAT           | GAT employs attention mechanisms to assign varying importance to different neighbors during aggregation. Unlike GCN, GAT dynamically weights the contribution of each neighbor based on their relevance to the current node. This attention-driven approach allows GAT to focus more on informative neighbors, enabling it to capture more nuanced relationships within the graph.|
| GIN           | GIN generalizes graph convolution operations by incorporating information from the entire neighborhood. It employs isomorphic mapping to capture global patterns and dependencies within the graph. GIN is effective for learning representations that consider both local and longer-range relationships, making it suitable for tasks that require a more holistic understanding of the graph structure.  |
| GAE           | GAE learns a low-dimensional representation of the graph structure, capturing both global and local features. It operates as an autoencoder, with an encoder mapping the input graph to a lower-dimensional latent space and a decoder reconstructing the graph from this representation. GAE is particularly useful for unsupervised learning tasks and capturing the intrinsic structure of the entire graph.     |

#### Hybrid GNN Architectures

| Architecture            | Composition                                            | Description                                                                                                                     |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Sequential Stacking     | GCN → GAT → GIN                                      | This architecture employs a sequential stacking of GCN, GAT, and GIN layers. The initial GCN captures localized context for language semantics in the neighborhood. Subsequently, the GAT introduces an attention mechanism to assign importance to different connected local neighborhoods. Finally, the GIN layer generalizes over longer-range neighborhood dependencies, allowing the model to understand global patterns in the argumentative graph. The sequential stacking fosters a hierarchical learning process, with each layer building upon the insights of the previous one.                |
| Parallel Fusion         | Parallel Fusion of GCN, GAT, and GIN                   | In this architecture, GCN, GAT, and GIN operate independently and simultaneously on the initial node embeddings. Each model captures distinct aspects of the graph's local and global features. The final fusion step combines these independent insights, creating a comprehensive representation of both local and global graph structures. The parallel approach allows models to specialize in different aspects of the graph simultaneously, fostering a holistic understanding.       |
| Comprehensive Fusion    | GCN + GAT + GIN + GAE                                 | This architecture combines GCN, GAT, and GIN with global insights from GAE. GCN captures local patterns, GAT introduces attention-guided processing, and GIN generalizes over longer-range dependencies. The global insights from GAE contribute to a balanced representation of both local and global structures in the argumentative graph. Comprehensive Fusion leverages the strengths of each model to provide a holistic and nuanced understanding of the graph. |
| Attention-Guided Fusion  | Attention-Guided Fusion of GCN, GAT, and GIN           | This architecture incorporates attention mechanisms to dynamically weigh the contributions of GCN, GAT, and GIN based on their relevance. The attention-guided fusion enhances the model's ability to focus on key aspects of the graph, allowing it to adaptively allocate attention to different types of information. This attention-driven approach promotes a more selective and nuanced learning process.           |
| Hierarchical Fusion      | GCN → (GAT + GIN) → GAE                               | Hierarchical Fusion combines local patterns from GCN with the fusion of GAT and GIN, and global insights from GAE. The initial GCN captures localized context, followed by a hierarchical fusion of GAT and GIN, addressing both local and longer-range dependencies. The final incorporation of global insights from GAE ensures a comprehensive understanding of the argumentative graph at different scales. The hierarchical fusion promotes a structured learning process with a focus on both local and global features.              |

## Results

#### GCN Architecture with Different Text Embeddings

| Text Embedding Method | Accuracy | F1 Score | AUC-ROC |
| ---------------------- | -------- | -------- | ------- |
| Word2Vec              | x        | x        | x       |
| Doc2Vec               | x        | x        | x       |
| Bag-of-Words                | x        | x        | x       |
| BERT  (generic)                 | x        | x        | x       |
| BERT  (trained)                 | x        | x        | x       |


#### Single GNN Architecture Performance Based on Sparsity Level

| Sparsity Level | GCN Accuracy | GAT Accuracy | GIN Accuracy | GAE Accuracy |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| Low            | x_low        | x_low        | x_low        | x_low        |
| Medium         | x_mid        | x_mid        | x_mid        | x_mid        |
| High           | x_high       | x_high       | x_high       | x_high       |

#### Single GNN Architecture Performance Based on Graph Size

| Graph Size     | GCN Accuracy | GAT Accuracy | GIN Accuracy | GAE Accuracy |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| Small          | x_low        | x_low        | x_low        | x_low        |
| Medium         | x_mid        | x_mid        | x_mid        | x_mid        |
| Large          | x_high       | x_high       | x_high       | x_high       |

#### Single GNN Architecture Performance Based on Connectivity

| Graph Size     | GCN Accuracy | GAT Accuracy | GIN Accuracy | GAE Accuracy |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| Small          | x_low        | x_low        | x_low        | x_low        |
| Medium         | x_mid        | x_mid        | x_mid        | x_mid        |
| Large          | x_high       | x_high       | x_high       | x_high       |

#### Single GNN Architecture Performance Based on Node Features

| Graph Size     | GCN Accuracy | GAT Accuracy | GIN Accuracy | GAE Accuracy |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| Small          | x_low        | x_low        | x_low        | x_low        |
| Medium         | x_mid        | x_mid        | x_mid        | x_mid        |
| Large          | x_high       | x_high       | x_high       | x_high       |

#### Single GNN Model Performance on Dataset 1

| Architecture | Accuracy | F1 Score | AUC-ROC |
| ------------ | -------- | -------- | ------- |
| GCN          | x        | x        | x       |
| GAT          | x        | x        | x       |
| GIN          | x        | x        | x       |
| GAE          | x        | x        | x       |

#### Single GNN Model Performance on Dataset 2

| Architecture | Accuracy | F1 Score | AUC-ROC |
| ------------ | -------- | -------- | ------- |
| GCN          | x        | x        | x       |
| GAT          | x        | x        | x       |
| GIN          | x        | x        | x       |
| GAE          | x        | x        | x       |

#### Single GNN Model Performance on Dataset 3

| Architecture | Accuracy | F1 Score | AUC-ROC |
| ------------ | -------- | -------- | ------- |
| GCN          | x        | x        | x       |
| GAT          | x        | x        | x       |
| GIN          | x        | x        | x       |
| GAE          | x        | x        | x       |

#### Single GNN Model Average Performance

| Architecture | Average Accuracy | Average F1 Score | Average AUC-ROC |
| ------------ | ---------------- | ---------------- | --------------- |
| GCN          | x                | x                | x               |
| GAT          | x                | x                | x               |
| GIN          | x                | x                | x               |
| GAE          | x                | x                | x               |

#### Hybrid GNN Model Performance on Dataset 1

| Architecture            | Accuracy | F1 Score | AUC-ROC |
| ----------------------- | -------- | -------- | ------- |
| Sequential Stacking     | x        | x        | x       |
| Parallel Fusion         | x        | x        | x       |
| Comprehensive Fusion    | x        | x        | x       |
| Attention-Guided Fusion  | x        | x        | x       |
| Hierarchical Fusion      | x        | x        | x       |

#### Hybrid GNN Model Performance on Dataset 2

| Architecture            | Accuracy | F1 Score | AUC-ROC |
| ----------------------- | -------- | -------- | ------- |
| Sequential Stacking     | x        | x        | x       |
| Parallel Fusion         | x        | x        | x       |
| Comprehensive Fusion    | x        | x        | x       |
| Attention-Guided Fusion  | x        | x        | x       |
| Hierarchical Fusion      | x        | x        | x       |

#### Hybrid GNN Model Performance on Dataset 3

| Architecture            | Accuracy | F1 Score | AUC-ROC |
| ----------------------- | -------- | -------- | ------- |
| Sequential Stacking     | x        | x        | x       |
| Parallel Fusion         | x        | x        | x       |
| Comprehensive Fusion    | x        | x        | x       |
| Attention-Guided Fusion  | x        | x        | x       |
| Hierarchical Fusion      | x        | x        | x       |

#### Hybrid GNN Model Average Performance

| Architecture            | Average Accuracy | Average F1 Score | Average AUC-ROC |
| ----------------------- | ----------------- | ---------------- | --------------- |
| Sequential Stacking     | x                 | x                | x               |
| Parallel Fusion         | x                 | x                | x               |
| Comprehensive Fusion    | x                 | x                | x               |
| Attention-Guided Fusion  | x                 | x                | x               |
| Hierarchical Fusion      | x                 | x                | x               |

#### Overall GNN Model Average Performance

| Architecture            | Average Accuracy | Average F1 Score | Average AUC-ROC |
| ----------------------- | ----------------- | ---------------- | --------------- |
| GCN          | x                | x                | x               |
| GAT          | x                | x                | x               |
| GIN          | x                | x                | x               |
| GAE          | x                | x                | x               |
| Sequential Stacking     | x                 | x                | x               |
| Parallel Fusion         | x                 | x                | x               |
| Comprehensive Fusion    | x                 | x                | x               |
| Attention-Guided Fusion  | x                 | x                | x               |
| Hierarchical Fusion      | x                 | x                | x               |

#### Single GNN Architecture Computational Efficiency

| Architecture  | Training Time (hours) | Memory Usage (GB) | Inference Speed (predictions/second) |
| ------------- | ---------------------- | ------------------ | ------------------------------------- |
| GCN           | x                      | x                  | x                                     |
| GAT           | x                      | x                  | x                                     |
| GIN           | x                      | x                  | x                                     |
| GAE           | x                      | x                  | x                                     |

#### Hybrid GNN Architecture Computational Efficiency

| Architecture            | Training Time (hours) | Memory Usage (GB) | Inference Speed (predictions/second) |
| ----------------------- | ---------------------- | ------------------ | ------------------------------------- |
| Sequential Stacking     | x                      | x                  | x                                     |
| Parallel Fusion         | x                      | x                  | x                                     |
| Comprehensive Fusion    | x                      | x                  | x                                     |
| Attention-Guided Fusion  | x                      | x                  | x                                     |
| Hierarchical Fusion      | x                      | x                  | x                                     |
