import torch
import hydra
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
import math

DIFFUSION_CONFIG = {
    "dataset": "scannet",
    "size_scale": 2.0,
    "label_scale": 4.0,
    "timesteps": 1000,
    "sampling_timesteps": 2,
    "iterative_train": 2,
    "renewal_obj": 0.9,
    "renewal_iou": 0.25,
    "renewal_sem_cls": 0.9,
    "size_mean_bias": 0.25,
    "center_mean_bias": 0.5,
    "label_mean_bias": 1 / 18,
    "label_loss_weight": 0.1,
}
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def nn_distance_topk(pc1, pc2, k=1, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        k: int, return top-k
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B, N, k) torch float32 tensor
        idx1: (B, N, k) torch int64 tensor
        dist2: (B, k, M) torch float32 tensor
        idx2: (B, k, M) torch int64 tensor
    """
    #print(pc1.shape, pc2.shape)
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1,-1,M,-1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1,N,-1,-1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    
    
    dist1, idx1 = torch.topk(pc_dist, k=k, dim=2, largest=False, sorted=True) # (B,N,k)
    dist2, idx2 = torch.topk(pc_dist, k=k, dim=1, largest=False, sorted=True) # (B,k,M)
    
    return dist1, idx1, dist2, idx2

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

class Mask3D(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type

        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]
        self.diffusion_config = DIFFUSION_CONFIG
        self.build_diffusion()
        self.num_proposal = 110

        self.mask_features_head = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(
                mask, p2s, dim=dim
            )[0]
        else:
            assert False, "Scatter function not known"

        assert (
            not use_np_features
        ) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )
        else:
            assert False, "pos enc type not known"

        self.pooling = MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3
        )

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(sizes[hlevel], self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def build_diffusion(self):

        timesteps = self.diffusion_config["timesteps"]
        sampling_timesteps = self.diffusion_config["sampling_timesteps"]

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        (timesteps,) = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1

        self.size_scale = self.diffusion_config["size_scale"]
        self.label_scale = self.diffusion_config["label_scale"]

        self.center_bias = self.diffusion_config["center_mean_bias"]
        self.size_bias = self.diffusion_config["size_mean_bias"]
        self.label_bias = self.diffusion_config["label_mean_bias"]

        self.center_sigma = self.center_bias / 3
        self.size_sigma = self.size_bias / 3
        self.label_sigma = self.label_bias / 3

        self.label_init_topk = 3
        self.label_assign_thres = 0.3

        self.betas = betas.cuda()
        self.alphas_cumprod = alphas_cumprod.cuda()
        self.alphas_cumprod_prev = alphas_cumprod_prev.cuda()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).cuda()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).cuda()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).cuda()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).cuda()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).cuda()

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start).cuda()

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    def prepare_targets(self, gt_centers, label_data):

        batch_size = len(gt_centers)

        centers = torch.zeros(
            (batch_size, self.num_proposal, 3)
        ).cuda()

        

        final_mask = None


        for batch_ind in range(batch_size):

            (
                center,
                t,
            ) = self.prepare_diffusion_concat(
                gt_centers[batch_ind],
                batch_ind,
                centers,
                label_data
            )

            centers[batch_ind] = center

        return centers

    def prepare_diffusion_concat(
        self,
        gt_centers,
        batch_ind,
        centers,
        label_data
    ):

        #print(gt_centers.shape)
        #print(centers.shape)
        #print(ema_pred_center.shape)
        num_gt = len(gt_centers)

        # Initialize -> denominator, adjusting to comply with the 3 sigma principle
        diffusion_boxes_center = (
            torch.randn(self.num_proposal, 3) * self.center_sigma + self.center_bias
        ).cuda()

        fps_seed = label_data["fps_seed"][batch_ind].unsqueeze(0)
        gt_center = gt_centers.unsqueeze(0)

        k = min(self.label_init_topk, num_gt)

        _, _, _, ind2 = nn_distance_topk(fps_seed, gt_center, k=k)


        #dist2 = dist2.squeeze(0).permute(1, 0)
        ind2 = ind2.squeeze(0).permute(1, 0)
        
        mins = label_data["mins"][batch_ind]
        maxs = label_data["maxs"][batch_ind]
        #print(mins, maxs)
        #exit()
        for gt_ind in range(num_gt):

            # Regardless of whether it exceeds the label assignment threshold or not,
            # each ground truth (GT) should be updated to at least one bounding box.
            gt_box_id = ind2[gt_ind][0]

            diffusion_boxes_center[gt_box_id] = gt_centers[gt_ind]

            
            # Normalization
            diffusion_boxes_center[gt_box_id][0] = (
                diffusion_boxes_center[gt_box_id][0] - mins[0]
            ) / (maxs[0] - mins[0])
            diffusion_boxes_center[gt_box_id][1] = (
                diffusion_boxes_center[gt_box_id][1] - mins[1]
            ) / (maxs[1] - mins[1])
            diffusion_boxes_center[gt_box_id][2] = (
                diffusion_boxes_center[gt_box_id][2] - mins[2]
            ) / (maxs[2] - mins[2])
        

        t = torch.randint(0, self.num_timesteps, (1,)).long().cuda()

        # 1) Perform calculations with the scale (SNR) ####
        diffusion_boxes_center = (diffusion_boxes_center * 2.0 - 1.0) * self.size_scale

        # 2) Add Noise
        diffusion_boxes_center = self.q_sample(diffusion_boxes_center, t, None)

        # 3) De-normalization 0~1
        diffusion_boxes_center = torch.clamp(
            diffusion_boxes_center,
            min=-0.9999 * self.size_scale,
            max=0.9999 * self.size_scale,
        )


        diffusion_boxes_center = ((diffusion_boxes_center / self.size_scale) + 1) / 2.0

        diffusion_boxes_center[:, 0] = diffusion_boxes_center[:, 0] * (maxs[0] - mins[0]) + mins[0]
        diffusion_boxes_center[:, 1] = diffusion_boxes_center[:, 1] * (maxs[1] - mins[1]) + mins[1]
        diffusion_boxes_center[:, 2] = diffusion_boxes_center[:, 2] * (maxs[2] - mins[2]) + mins[2]
        #print('waak', gt_centers[0], diffusion_boxes_center[ind2[0][0]])
        return diffusion_boxes_center, t
    
    def ddim_sample(
        self,
        xyz,
        features,
    ):

        batch_size = len(xyz)

        total_timesteps, sampling_timesteps, eta = (
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Initialization
        diffused_boxes = torch.randn(batch_size, self.num_proposal, 3).cuda()
        #exit()

        mins = torch.stack(
            [
                xyz[i].min(dim=0)[0]
                for i in range(len(xyz))
            ]
        )
        maxs = torch.stack(
            [
                xyz[i].max(dim=0)[0]
                for i in range(len(xyz))
            ]
        )

        x_start, x_start_label = None, None

        for time_id, (time, time_next) in enumerate(time_pairs):

            # Initialization
            time_cond = torch.full((batch_size,), time, dtype=torch.long).cuda()
            #end_points["ts"] = time_cond
            tmp_diffused_boxes = diffused_boxes.clone()

            # De-normalization
            diffused_boxes = torch.clamp(
                diffused_boxes,
                min=-0.9999 * self.size_scale,
                max=0.9999 * self.size_scale,
            )
            diffused_boxes = ((diffused_boxes / self.size_scale) + 1) / 2
            #print(mins[:, 0].shape)
            for i in range(3):  
                diffused_boxes[:, :, i] = diffused_boxes[:, :, i] * (maxs[:, i] - mins[:, i]).unsqueeze(1) + mins[:, i].unsqueeze(1)

            center = diffused_boxes


            pred_center = center.clone()


            # Utilize the results from Pnet and update diffusion boxes by adding noise.

            x_start = torch.zeros((batch_size, self.num_proposal, 3)).cuda()

            x_start_center = pred_center
            for i in range(3): 
                x_start[:, :, i] = x_start_center[:, :, i] - mins[:, 0].unsqueeze(1) / (maxs[:, 0] - mins[:, 0]).unsqueeze(1)

            x_start = (x_start * 2.0 - 1.0) * self.size_scale

            if time_next < 0:
                continue

            # Calculation with alpha
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x_start = torch.clamp(
                x_start, min=-0.9999 * self.size_scale, max=0.9999 * self.size_scale
            )
            pred_noise = self.predict_noise_from_start(
                tmp_diffused_boxes, time_cond, x_start
            )
            noise = torch.randn_like(x_start)

            x_start = (
                x_start * alpha_next.sqrt().to(x_start) + c.to(x_start) * pred_noise.to(x_start) + sigma.to(x_start) * noise.to(x_start)
            ).float()

            # Update diffusion boxes for next round
            diffused_boxes = x_start.clone()

        return center

    def predict_noise_from_start(self, x_t, t, x0):
        #print(self.sqrt_recip_alphas_cumprod.shape, t.shape, x_t.shape, x0.shape)
        return (
            extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)


    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False, target = None#, centers_init = None, #query_pos_init = None
    ):
        

        batch_size = len(x.decomposed_coordinates)
        pcd_features, aux = self.backbone(x)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)
        
        if self.train_on_segments:
            mask_segments = []
            #coord_segments = []
            for i, mask_feature in enumerate(
                mask_features.decomposed_features
            ):
                mask_segments.append(
                    self.scatter_fn(mask_feature, point2segment[i], dim=0)
                )

        fps_idx = [
            furthest_point_sample(
                x.decomposed_coordinates[i][None, ...].float(),
                self.num_queries,
            )
            .squeeze(0)
            .long()
            for i in range(len(x.decomposed_coordinates))
        ]

                

            

        sampled_coords = torch.stack(
            [
                coordinates.decomposed_features[i][fps_idx[i].long(), :]
                for i in range(len(fps_idx))
            ]
        )
        #print(sampled_coords[0])
        #exit()

        if target is not None:
            for i in range(len(sampled_coords)):
                mask = target[i]['masks'].float()
                center = (mask @ coordinates.decomposed_features[i]) 
                center = center / (mask.sum(1).unsqueeze(1) + 1e-7)
                rand_idx = torch.randperm(120)[:len(center)]
                sampled_coords[i][rand_idx] = center

        mins = torch.stack(
            [
                coordinates.decomposed_features[i].min(dim=0)[0]
                for i in range(len(coordinates.decomposed_features))
            ]
        )
        maxs = torch.stack(
            [
                coordinates.decomposed_features[i].max(dim=0)[0]
                for i in range(len(coordinates.decomposed_features))
            ]
        )

        query_pos = self.pos_enc(
            sampled_coords.float(), input_range=[mins, maxs]
        )  # Batch, Dim, queries
        query_pos = self.query_projection(query_pos)
        

        queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))


        #print(mask_features.decomposed_features[0].shape, mask_segments[0].shape, point2segment[0].shape)

        sampled_coords = None

        predictions_class = []
        predictions_mask = []
        queries_ = []
        query_embs = []

        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                if self.train_on_segments:
                    output_class, outputs_mask, attn_mask, query_emb = self.mask_module(
                        queries,
                        mask_features,
                        mask_segments,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=point2segment,
                        coords=coords,
                    )
                else:
                    output_class, outputs_mask, attn_mask, query_emb = self.mask_module(
                        queries,
                        mask_features,
                        None,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=None,
                        coords=coords,
                    )

                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(
                        curr_sample_size, self.sample_sizes[hlevel]
                    )

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(
                            pcd_size, device=queries.device
                        )

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(
                            decomposed_aux[k].shape[0], device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [
                        decomposed_aux[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn = torch.stack(
                    [
                        decomposed_attn[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](
                    batched_aux.permute((1, 0, 2))
                )
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)
                queries_.append(queries)
                query_embs.append(query_emb)

        if self.train_on_segments:
            output_class, outputs_mask, query_emb = self.mask_module(
                queries,
                mask_features,
                mask_segments,
                0,
                ret_attn_mask=False,
                point2segment=point2segment,
                coords=coords,
            )
        else:
            output_class, outputs_mask, query_emb = self.mask_module(
                queries,
                mask_features,
                None,
                0,
                ret_attn_mask=False,
                point2segment=None,
                coords=coords,
            )
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        queries_.append(queries)
        query_embs.append(query_emb)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask, queries_, query_embs
            ),
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
            "queries": queries_[-1],
            "query_embs": query_embs[-1],
            "segment_features": mask_segments
        }

    def mask_module(
        self,
        query_feat,
        mask_features,
        mask_segments,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        coords=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                #print(mask_segments[i].shape, mask_embed[i].shape)
                #print(mask_segments[i][0][:5], mask_embed[i][0][:5])
                #exit()
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(
                    mask_features.decomposed_features[i] @ mask_embed[i].T
                )

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            if point2segment is not None:
                return outputs_class, output_segments, attn_mask, mask_embed
            else:
                return (
                    outputs_class,
                    outputs_mask.decomposed_features,
                    attn_mask,
                    mask_embed
                )

        if point2segment is not None:
            return outputs_class, output_segments, mask_embed
        else:
            return outputs_class, outputs_mask.decomposed_features, mask_embed

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_queries, outputs_query_embs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b, "queries": c, "query_embs": d}
            for a, b ,c,d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_queries[:-1], outputs_query_embs[:-1])
        ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")