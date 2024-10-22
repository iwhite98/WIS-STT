# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

empty_weight_cls = torch.ones(21)#torch.full((21), 1.0)
empty_weight_cls[-1] = 0.1
empty_weight_cls[:1] = 0.3
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    #print(inputs.shape, targets.shape)

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()


        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        '''
        print('1', src_logits.shape, target_classes.shape) ### 
        print('target : ',targets[0]['labels'].shape, targets[0]['labels'])
        print(indices[0][1].shape, indices)
    
        1 torch.Size([1, 100, 19]) torch.Size([1, 100])
        target :  torch.Size([28]) tensor([  0,   4,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,  17,
          4, 253,  12,   0, 253,   6,   6,   9,  15,  12,   0,   5,   5,  17], device='cuda:0')
        torch.Size([28]) [(tensor([ 3,  8, 13, 23, 24, 34, 36, 38, 43, 45, 46, 47, 49, 51, 56, 61, 65, 71,
        74, 76, 81, 82, 87, 88, 90, 94, 96, 98]), tensor([15, 18, 14,  8,  1, 12, 26, 22,  0, 16,  2,  6, 23, 10, 27, 19,  9, 11,
        25, 24, 21, 17, 20,  3,  5,  7, 13,  4]))]
        '''
        
        #print(targets[0]['labels'], target_classes)
        #exit()
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            ignore_index=253,
        )
        #print('label', loss_ce, src_logits.transpose(1, 2)[0][0][:5])
        #print(loss_ce, src_logits[0][0], target_classes[0][0])
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_acc(self, s_outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval):
        macc = []
        for batch_id in range(len(targets)):

            #print(targets[batch_id][mask_type].shape, s_outputs['segment_features'][batch_id].shape)
            #exit()
            undet_idx = ~targets[batch_id]["segment_det_idx"] 
            
            if undet_idx.float().sum() == 0:
                continue
            else:
                target_id = target_ids[batch_id]
                map = s_outputs["pred_masks"][batch_id][:, s_map_ids[batch_id]].T
                target = targets[batch_id][mask_type][target_id]
                target_mask = torch.zeros_like(map).bool()
                #print(target_mask, target)
                #exit()
                target_mask[:, undet_idx] = target[:, undet_idx]#.float() #######check
                
                t_mask = t_outputs["pred_masks"][batch_id][undet_idx][:, t_map_ids[batch_id]].T.clone().detach()
                undet_mask = targets[batch_id][mask_type][:, undet_idx][target_id]
                
                mask_score = torch.argmax(t_mask.sigmoid() * undet_mask.float(), dim = 0)
                queries = torch.arange(undet_mask.shape[0]).unsqueeze(1)  
                ps_mask = (queries.to(mask_score.device) == mask_score)
                matches = (undet_mask.float().argmax(0) == ps_mask.float().argmax(0))
                acc = torch.sum(matches) / undet_idx.float().sum()

            macc.append(acc)
        if len(macc) == 0:
            return None
        loss_macc = torch.mean(torch.stack(macc))
        
        
        
        return loss_macc

    def loss_masks(self, s_outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in s_outputs

        loss_masks = []
        loss_dices = []

        #print('output:', outputs['pred_masks'][0].shape) ## [1611, 100]= [M, K]
        #print(mask_type, targets[0][mask_type].shape)
        ##segment_mask torch.Size([28, 1611])  = [K_gt, M]
        ##target_mask :  torch.Size([28, 1611]) = [K_gt, M]

        for batch_id in range(len(targets)):

            #print(targets[batch_id][mask_type].shape, s_outputs['segment_features'][batch_id].shape)
            #exit()
            det_idx = targets[batch_id]["segment_det_idx"] 
            
            if is_eval or (~det_idx).float().sum() == 0:
                map_id, target_id = indices[batch_id]
                map = s_outputs["pred_masks"][batch_id][:, map_id].T
                target_mask = targets[batch_id][mask_type][target_id]
            else:
                target_id = target_ids[batch_id]
                map = s_outputs["pred_masks"][batch_id][:, s_map_ids[batch_id]].T
                target = targets[batch_id][mask_type][target_id]
                target_mask = torch.zeros_like(map).bool()
                #print(target_mask, target)
                #exit()
                target_mask[:, det_idx] = target[:, det_idx]#.float() #######check
                
                t_mask = t_outputs["pred_masks"][batch_id][~det_idx][:, t_map_ids[batch_id]].T.clone().detach()
                undet_mask = targets[batch_id][mask_type][:, ~det_idx][target_id]
                
                mask_score = torch.argmax(t_mask.sigmoid() * undet_mask.float(), dim = 0)
                queries = torch.arange(undet_mask.shape[0]).unsqueeze(1)  
                ps_mask = (queries.to(mask_score.device) == mask_score)
                target_mask[:, ~det_idx] = ps_mask
                '''
                score = (t_mask * undet_mask.float()).softmax(0)
                mask_score = torch.argmax(score, dim = 0)
                queries = torch.arange(undet_mask.shape[0]).unsqueeze(1)  
                ps_mask = score * (queries.to(mask_score.device) == mask_score)
                target_mask[:, ~det_idx] = ps_mask
                '''
                

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            #print('target_mask : ', target_mask.shape)

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        loss_mask = torch.sum(torch.stack(loss_masks))
        #print('mask', loss_mask)
        # del target_mask
        return {
            "loss_mask": loss_mask,
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = s_outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
            "loss_dice": dice_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
        }

        del src_masks
        del target_masks
        return losses
  

    def loss_query(self, t_queries, s_queries, t_map_ids, s_map_ids):
        loss_query = []
        for batch_id in range(len(t_queries)):
            loss_query.append(F.l1_loss(
                t_queries[batch_id][t_map_ids[batch_id]],
                s_queries[batch_id][s_map_ids[batch_id]]
            ))  

        loss = torch.sum(torch.stack(loss_query))     
        return loss               

    def loss_center(self, t_outputs, t_pcd, target_ids, t_map_ids, targets):
        loss_center = []
        for batch_id in range(len(targets)):
            
            #query = t_outputs['query_embs'][batch_id][t_map_ids[batch_id]]
            
            pred_mask = (t_outputs["pred_masks"][batch_id][:, t_map_ids[batch_id]].sigmoid()>0.5).T.float()


            det_idx = targets[batch_id]["segment_det_idx"] 
            pcd = t_pcd[batch_id]
            point_pcd = pcd[det_idx].detach() # P * 128
            target = targets[batch_id]['segment_mask'][target_ids[batch_id]][:, det_idx].float() # N * P
            point_counts = target.sum(dim=1, keepdim=True).clamp(min=1)
            feature_sums = torch.mm(target, point_pcd)
            #print(feature_sums.shape, point_counts.shape, query.shape)
            #exit()
            feature_centers = feature_sums / point_counts
            #print(feature_sums.shape, point_counts.shape)
            #print(feature_sums[0][0], point_counts[0], feature_centers[0][0])
            mask_counts = pred_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mask_sums = torch.mm(pred_mask, pcd)
            mask_centers = mask_sums / mask_counts
            #print(pred_mask.shape, pcd.shape, mask_sums.shape, mask_counts.shape)

            #print(mask_centers.shape, feature_centers.shape)
            #print(mask_centers[0][:5], feature_centers[0][:5], pcd[0][:5])
            #exit()
            ## F[P * C] -> FC[N * C]
            ## Q[N * C] * F[P * C] -> MC[N * P] -> 
            loss_center.append(F.mse_loss(mask_centers, feature_centers))
        
        loss = torch.sum(torch.stack(loss_center))
        
        return loss   
    
    def loss_ps(self, t_outputs, s_outputs, t_map_ids, s_map_ids, target_ids, targets):
        loss_masks = []
        loss_dices = []
        for batch_id in range(len(targets)):
            t_mask = t_outputs["pred_masks"][batch_id][~targets[batch_id]["segment_det_idx"]][:, t_map_ids[batch_id]].T.clone().detach()
            if (~targets[batch_id]["segment_det_idx"]).float().sum() == 0:
                loss_masks.append(torch.tensor(0.0).to(t_mask.device))
                continue
            #t_mask = t_outputs["pred_masks"][batch_id][~targets[batch_id]["segment_det_idx"]][:, t_map_ids[batch_id]].T
            s_mask = s_outputs["pred_masks"][batch_id][~targets[batch_id]["segment_det_idx"]][:, s_map_ids[batch_id]].T
            undet_mask = targets[batch_id]['segment_mask'][:, ~targets[batch_id]["segment_det_idx"]][target_ids[batch_id]]
            mask_score = torch.argmax(t_mask.sigmoid() * undet_mask.float(), dim = 0)

            queries = torch.arange(undet_mask.shape[0]).unsqueeze(1)  
            ps_mask = (queries.to(mask_score.device) == mask_score)

            
            num_masks = ps_mask.shape[0]
            loss = sigmoid_ce_loss_jit(s_mask, ps_mask.float(), num_masks)
            loss_masks.append(loss)
            #print(s_mask.shape, loss)

        loss_mask = torch.sum(torch.stack(loss_masks))
        #print('mask', loss_mask)
        # del target_mask
        #print(loss_mask)
        #exit()
        return loss_mask  


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}#, "cls_masks": self.loss_cls_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        losses = loss_map[loss](outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval)
        return losses

    def forward(self, outputs, targets, mask_type, is_eval = False, t_outputs = None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        #print('outputs : ', outputs.keys()) ##outputs :  dict_keys(['pred_logits', 'pred_masks', 'aux_outputs', 'sampled_coords', 'backbone_features'])
        #print('targets : ', targets[0]['labels'].shape, targets[0]['masks'].shape, targets[0]['segment_mask'].shape)

        ##targets :  torch.Size([28]) torch.Size([28, 164833]) torch.Size([28, 1611])
        ## targets=[labels : [K_gt](element : label class), masks : [K_gt, M?], 'segmentation_mask': [K_gt, M]]

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type, is_eval, is_s = True)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        if t_outputs is None:
            losses = {}
            for loss in self.losses:
                losses.update(
                    self.get_loss(
                        loss, outputs, None, targets, indices, None, None, None, num_masks, mask_type, is_eval
                    )
                )
        else:
            t_outputs_without_aux = {
                k: v for k, v in t_outputs.items() if k != "aux_outputs"
            }
            # Retrieve the matching between the outputs of the last layer and the targets
            t_indices = self.matcher(t_outputs_without_aux, targets, mask_type, is_eval, is_s = False)

            t_map_ids = []
            s_map_ids = []
            target_ids = []
            for batch_id in range(len(indices)):
                t_map_id, t_target_id = t_indices[batch_id]
                s_map_id, s_target_id = indices[batch_id]
                t_sort_order_target = torch.argsort(t_target_id)
                s_sort_order_target = torch.argsort(s_target_id)
                target_ids.append(t_target_id[t_sort_order_target])
                t_map_ids.append(t_map_id[t_sort_order_target])
                s_map_ids.append(s_map_id[s_sort_order_target])
            
            losses = {}
            for loss in self.losses:
                losses.update(
                    self.get_loss(
                        loss, outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval
                    )
                )
            losses['loss_query'] = self.loss_query(t_outputs["queries"], outputs["queries"], t_map_ids, s_map_ids)
            losses['loss_center'] = self.loss_center(t_outputs, t_outputs['segment_features'], target_ids, t_map_ids, targets)
            #losses['loss_acc'] = self.loss_acc(
            #            outputs, t_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval
            #        )

        if "aux_outputs" in outputs:
            #print(len(outputs["aux_outputs"]))
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type, is_eval, is_s = True)
                if t_outputs is None:
                    for loss in self.losses:
                        l_dict = self.get_loss(
                            loss, aux_outputs, None, targets, indices, None, None, None, num_masks, mask_type, is_eval
                        )
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        #print(l_dict)
                        losses.update(l_dict)
                
                else:
                    t_aux_outputs = t_outputs["aux_outputs"][i]

                    # Retrieve the matching between the outputs of the last layer and the targets
                    t_indices = self.matcher(t_aux_outputs, targets, mask_type, is_eval, is_s = False)

                    t_map_ids = []
                    s_map_ids = []
                    target_ids = []
                    for batch_id in range(len(indices)):
                        t_map_id, t_target_id = t_indices[batch_id]
                        s_map_id, s_target_id = indices[batch_id]
                        t_sort_order_target = torch.argsort(t_target_id)
                        s_sort_order_target = torch.argsort(s_target_id)
                        target_ids.append(t_target_id[t_sort_order_target])
                        t_map_ids.append(t_map_id[t_sort_order_target])
                        s_map_ids.append(s_map_id[s_sort_order_target])
                    
                    for loss in self.losses:
                        l_dict = self.get_loss(
                            loss, aux_outputs, t_aux_outputs, targets, indices, s_map_ids, t_map_ids, target_ids, num_masks, mask_type, is_eval
                        )
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        #print(l_dict)
                        losses.update(l_dict)
                    losses[f"loss_query_{i}"] = self.loss_query(t_aux_outputs["queries"], aux_outputs["queries"], t_map_ids, s_map_ids)
                    losses[f"loss_center_{i}"] = self.loss_center(t_aux_outputs,t_outputs['segment_features'], target_ids, t_map_ids, targets)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
