#from .single.gae import GAEModel
from .single.gat import GATModel
from .single.gcn import GCNModel
from .single.gin import GINModel

#from .hybrid.attention_guided_fusion import AttentionGuidedFusionModel
#from .hybrid.comprehensive_fusion import ComprehensiveFusionModel
#from .hybrid.hierarchical_fusion import HierarchicalFusionModel
#from .hybrid.parallel_fusion import ParallelFusionModel
from .hybrid.sequential_stacking import SequentialStackingModel