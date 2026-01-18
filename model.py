import torch
import torch.nn as nn
from .encoders import GraphKANEncoder, FingerprintEncoder
from .encoders import LSSEncoder, MambaKANEncoder  

class MambaKANModel(nn.Module):
    def __init__(self, atom_in_dim= int, edge_attr_dim= int,
                 fp_dim= int, desc_dim=int,
                 graph_hidden=128, fp_out=128, seq_hidden=128,
                 num_tasks=1, task_type="binary", dropout=0.4,
                 seq_encoder_type="mamba_kan",   
                 lss_depth=3, lss_kernel=256):
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks

        self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2,3), dropout=dropout)

        if seq_encoder_type.lower() == "lss":
            self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
                                      dropout=0.1)

        elif seq_encoder_type.lower() == "mamba":
           
            self.seq_enc = MambaKANEncoder(
                in_dim=atom_in_dim,
                hidden=seq_hidden,
                depth=lss_depth,
                d_state=16,  
                dropout=0.1
            )

        self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out, dropout=dropout)

        fusion_dim = graph_hidden + seq_hidden + fp_out
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, num_tasks)

    def forward(self, data):
        g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
        s_repr = self.seq_enc(data.x, data.batch)
        fp_repr = self.fp_enc(data.fp, data.desc)
        h = torch.cat([g_repr, s_repr, fp_repr], dim=-1)
        h = self.fuse(h)
        logits = self.head(h)
        return logits


class AblationMambaKANModel(nn.Module):
    def __init__(self, atom_in_dim, edge_attr_dim, fp_dim, desc_dim,
                 graph_hidden=128, fp_out=128, seq_hidden=128,
                 num_tasks=1, task_type="binary", dropout=0.4,
                 seq_encoder_type="lss", lss_depth=4, lss_kernel=128,
                 remove_graph_encoder=False, remove_seq_encoder=False, remove_fp_encoder=False):
        super().__init__()

        self.task_type = task_type
        self.num_tasks = num_tasks


        if not remove_graph_encoder:
            self.graph_enc = GraphKANEncoder(atom_in_dim, edge_attr_dim, hidden=graph_hidden, layers=(1, 2, 2, 3),
                                             dropout=dropout)


        if not remove_seq_encoder:
            if seq_encoder_type.lower() == "lss":
                self.seq_enc = LSSEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=lss_depth, kernel_len=lss_kernel,
                                          dropout=0.1)
            else:
                from .encoders import MambaEncoder
                self.seq_enc = MambaEncoder(in_dim=atom_in_dim, hidden=seq_hidden, depth=2, dropout=0.1)


        if not remove_fp_encoder:
            self.fp_enc = FingerprintEncoder(fp_dim=fp_dim, desc_dim=desc_dim, hidden=256, out_dim=fp_out,
                                             dropout=dropout)


        fusion_dim = 0
        if not remove_graph_encoder:
            fusion_dim += graph_hidden
        if not remove_seq_encoder:
            fusion_dim += seq_hidden
        if not remove_fp_encoder:
            fusion_dim += fp_out

        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, num_tasks)

    def forward(self, data):
        features = []

        if hasattr(self, 'graph_enc'):
            g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
            features.append(g_repr)
        else:
            g_repr = None

        if hasattr(self, 'seq_enc'):
            s_repr = self.seq_enc(data.x, data.batch)
            features.append(s_repr)
        else:
            s_repr = None

        if hasattr(self, 'fp_enc'):
            fp_repr = self.fp_enc(data.fp, data.desc)
            features.append(fp_repr)
        else:
            fp_repr = None


        h = torch.cat(features, dim=-1)
        h = self.fuse(h)
        logits = self.head(h)
        return logits

    @torch.no_grad()
    def encode_modalities(self, data):
       
        self.eval()
        features = []

        if hasattr(self, 'graph_enc'):
            g_repr = self.graph_enc(data.x, data.edge_index, data.edge_attr, data.batch)
            features.append(g_repr)
        else:
            g_repr = None

        if hasattr(self, 'seq_enc'):
            s_repr = self.seq_enc(data.x, data.batch)
            features.append(s_repr)
        else:
            s_repr = None

        if hasattr(self, 'fp_enc'):
            fp_repr = self.fp_enc(data.fp, data.desc)
            features.append(fp_repr)
        else:
            fp_repr = None

        h = torch.cat(features, dim=-1)
        h = self.fuse(h)
        return g_repr, s_repr, fp_repr, h
