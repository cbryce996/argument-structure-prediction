# PyTorch Geometric AIF Data Importer

This module contains the data importer for converting AIF raw data to a PyTorch Geometric Dataset for compatibility with PyTorch Geometric neural networks. The module features a number of pre-processing utilities which have proven useful for AIF data importing.

## Example:

```
# Define transformation steps to be performed on imported data
transforms = Compose([
    KeepSelectedNodeTypes(types_to_keep=["I", "RA", "MA"]),
    RemoveLinkNodeTypes(types_to_remove=["RA", "MA"]),
    EdgeLabelEncoder(),
    CreateBertEmbeddings(tokenizer, model, 128),
    GraphToPyGData()
])

# Define selecion criteria for the imported data
filters = Compose([MinNodesAndEdges()])

# Construct the PyTorch Geometric Dataset
qt30_dataset = AIFDataset(root="./QT30", pre_transform=transforms, pre_filter=filters)
```

## Transforms:

Each transform is applied to the raw data before being imported, chaining together transforms creates a pre-processing pipeline.

Transforms operate as simple functions when used within a PyTorch Geometric transform object such as `Compose`. Data is received as input, transformed and then returned, allowing pre-processing pipelines to be created.

### KeepSelectedNodeTypes:

Keeps only selected node types and removes all other node types including connected edges.

**Input:**
Array of AIF node labels.
```
types_to_keep=["I", "RA", "MA"]
```

**Usage:**
```
KeepSelectedNodeTypes(types_to_keep=["I", "RA", "MA"])
```

Useful for reducing the AIF data to a specific subset of node types.

### RemoveLinkNodeTypes:

Removes selected node types and restores the connections as labeled edges.

**Input:**
Array of AIF node labels.
```
types_to_remove=["RA", "MA"]
```

**Usage:**
```
RemoveLinkNodeTypes(types_to_remove=["RA", "MA"]),
```

Useful for modifying the graph structure without changing the connectivity. For example, removing the link nodes in AIF data but keeping the connections they represent.

### EdgeLabelEncoder:

Simple edge label encoder which uses 