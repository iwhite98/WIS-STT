import MinkowskiEngine as ME
import numpy as np
import torch
from random import random
from torch_scatter import scatter_mean

class VoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries

    def __call__(self, batch):
        if ("train" in self.mode) and (
            self.small_crops or self.very_small_crops
        ):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            num_queries=self.num_queries,
        )


class VoxelizeCollateMerge:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        scenes=2,
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        make_one_pc_noise=False,
        place_nearby=False,
        place_far=False,
        proba=1,
        probing=False,
        task="instance_segmentation",
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba
        self.probing = probing

    def __call__(self, batch):
        if (
            ("train" in self.mode)
            and (not self.make_one_pc_noise)
            and (self.proba > random())
        ):
            if self.small_crops or self.very_small_crops:
                batch = make_crops(batch)
            if self.very_small_crops:
                batch = make_crops(batch)
            if self.batch_instance:
                batch = batch_instances(batch)
            new_batch = []
            for i in range(0, len(batch), self.scenes):
                batch_coordinates = []
                batch_features = []
                batch_labels = []

                batch_filenames = ""
                batch_raw_color = []
                batch_raw_normals = []

                offset_instance_id = 0
                offset_segment_id = 0

                for j in range(min(len(batch[i:]), self.scenes)):
                    batch_coordinates.append(batch[i + j][0])
                    batch_features.append(batch[i + j][1])

                    if j == 0:
                        batch_filenames = batch[i + j][3]
                    else:
                        batch_filenames = (
                            batch_filenames + f"+{batch[i + j][3]}"
                        )

                    batch_raw_color.append(batch[i + j][4])
                    batch_raw_normals.append(batch[i + j][5])

                    # make instance ids and segment ids unique
                    # take care that -1 instances stay at -1
                    batch_labels.append(
                        batch[i + j][2]
                        + [0, offset_instance_id, offset_segment_id]
                    )
                    batch_labels[-1][batch[i + j][2][:, 1] == -1, 1] = -1

                    max_instance_id, max_segment_id = batch[i + j][2].max(
                        axis=0
                    )[1:]
                    offset_segment_id = offset_segment_id + max_segment_id + 1
                    offset_instance_id = (
                        offset_instance_id + max_instance_id + 1
                    )

                if (len(batch_coordinates) == 2) and self.place_nearby:
                    border = batch_coordinates[0][:, 0].max()
                    border -= batch_coordinates[1][:, 0].min()
                    batch_coordinates[1][:, 0] += border
                elif (len(batch_coordinates) == 2) and self.place_far:
                    batch_coordinates[1] += (
                        np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                    )
                new_batch.append(
                    (
                        np.vstack(batch_coordinates),
                        np.vstack(batch_features),
                        np.concatenate(batch_labels),
                        batch_filenames,
                        np.vstack(batch_raw_color),
                        np.vstack(batch_raw_normals),
                    )
                )
            # TODO WHAT ABOUT POINT2SEGMENT AND SO ON ...
            batch = new_batch
        elif ("train" in self.mode) and self.make_one_pc_noise:
            new_batch = []
            for i in range(0, len(batch), 2):
                if (i + 1) < len(batch):
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    batch[i][2],
                                    np.full_like(
                                        batch[i + 1][2], self.ignore_label
                                    ),
                                )
                            ),
                        ]
                    )
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    np.full_like(
                                        batch[i][2], self.ignore_label
                                    ),
                                    batch[i + 1][2],
                                )
                            ),
                        ]
                    )
                else:
                    new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
            batch = new_batch
        # return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode)
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
        )


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch


def voxelize(
    batch,
    ignore_label,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
):
    (
        coordinates,
        coordinates_,
        coordinates_org,
        features,
        labels,
        labels_,
        original_labels,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
        probs,
        probs_det,
        unique_maps
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []

    for sample in batch:
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[4])
        original_normals.append(sample[5])
        
        train_mode = sample[8]

        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )
        inverse_maps.append(inverse_map)
        unique_maps.append(unique_map)
        

        
            
        sample_coordinates = coords[unique_map]
        #coordinates_.append(torch.from_numpy(coords).int())
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        coordinates_.append(torch.from_numpy(sample_coordinates))
        coordinates_org.append(torch.from_numpy(sample[6]))
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())


        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())
            labels_.append(torch.from_numpy(sample[2]).long())
        '''
        if train_mode:
            sample_prob = torch.from_numpy(sample[-1][unique_map]).float()
            probs.append(sample_prob)
        '''
                    
    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}

    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])
    #print(features.shape)
    #exit()
    if probing:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
            ),
            labels,
        )

    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(
                input_dict["labels"][i][:, 0],
                return_index=True,
                return_inverse=True,
            )
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, -1],
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                input_dict["segment2label"].append(
                    input_dict["labels"][i][ret_index][:, :-1]
                )
        
        

    if "labels" in input_dict:
        list_labels = input_dict["labels"]
        list_labels_ = labels_

        target = []
        target_full = []

        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id]
                        == label_ids.unsqueeze(1),
                    }
                )

        
        else:
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append(
                        {"point2segment": input_dict["labels"][i][:, 0]}
                    )
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long()
                        }
                    )
            else:
                for i in range(len(input_dict["labels"])):
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long()
                        }
                    )
                #exit()
                #print(inverse_map.shape, unique_map.shape, list_labels[-1][:, 2].shape, inverse_map.unique().shape, unique_map.unique().shape, original_labels[-1][:, 2].shape)
                #print(inverse_map[:10], unique_map[:10])
                #exit()
                target = get_instance_masks(
                    list_labels_,
                    list_segments=input_dict["segment2label"],
                    task=task,
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                    train_mode = train_mode,
                    coordinates = coordinates_org,
                    unique_map = unique_maps,
                    vox_label = list_labels,
                    #probs = probs,
                )


                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]
                    
                
                
                #if "train" not in mode:
                target_full = get_instance_masks(
                    [torch.from_numpy(l) for l in original_labels],
                    task=task,
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                    train_mode = train_mode,
                    coordinates = coordinates_org,
                )
                for i in range(len(target_full)):
                    target_full[i]["point2segment"] = torch.from_numpy(
                        original_labels[i][:, 2]
                    ).long()
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    if "train" not in mode:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                target_full,
                original_colors,
                original_normals,
                original_coordinates,
                idx,
            ),
            target,
            [sample[3] for sample in batch],
        )
    else:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                target_full,
                original_colors,
                original_normals,
                original_coordinates,
                idx,
            ),
            target,
            [sample[3] for sample in batch],
        )

def batch_giou_cross(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: M, 6
    # out: N, M
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]
    intersection = torch.prod(
        torch.clamp(
            (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)


    return iou

def is_within_bb_torch(points, bb_min, bb_max):
    return torch.all(points >= bb_min, dim=-1) & torch.all(points <= bb_max, dim=-1)

def is_box1_in_box2(box1, box2, offset=0.05):
    return torch.all((box1[:3] + offset) >= box2[:3]) & torch.all((box1[3:] - offset) <= box2[3:])
    
def get_instance_masks(
    list_labels,
    task,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
    train_mode = False,
    coordinates = None,
    unique_map = None,
    vox_label = None
    #probs = None
):
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        empty_masks = []
        ignore_masks = []
        segment_masks_ = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()
        mins = []
        maxs = []
        #filter_out_idx = []
        #print(train_mode)
        if train_mode:
            bbox_data = []
            weak_idx = []
            instance_box = []
            ignore_idx = []
            for instance_id in instance_ids:
                masking = list_labels[batch_id][:, 1] == instance_id
                if instance_id == -1:
                    #filter_out_idx.append(masking)
                    continue

                # TODO is it possible that a ignore class (255) is an instance???
                # instance == -1 ???
                
                tmp = list_labels[batch_id][
                    masking
                ]
                label_id = tmp[0, 0]

                if (
                    label_id in filter_out_classes
                ):  # floor, wall, undefined==255 is not included
                    continue
                if label_id == 255:
                    ignore_idx.append(1)
                    ignore_masks.append(masking)
                else:
                    ignore_idx.append(0)
                coord = coordinates[batch_id].float()
                coord -= coord.mean(axis=0)
                obj_coords = coord[masking].float()
                
                obj_center = obj_coords.mean(0)
                max_coords = obj_coords.max(0)[0]
                min_coords = obj_coords.min(0)[0]
                mins.append(min_coords)
                maxs.append(max_coords)
                
                distances = torch.full((coordinates[batch_id].shape[0], ),float("inf"))
                distances[masking] = torch.norm(obj_coords.float() - obj_center, dim=1)
                #print(distances.shape, distances[masking].shape, torch.norm(points - center, dim=1).shape)
                #exit()
                rand_idx = torch.argmin(distances).item()
                weak_idx.append(rand_idx)
                
                empty_masks.append(masking)
                label_ids.append(label_id) 
                
                instance_box.append(torch.cat([min_coords, max_coords], axis=0))


            weak_idx = torch.tensor(weak_idx, dtype=torch.long).unique()
            empty_masks = torch.stack(empty_masks)
            ignore_idx = torch.tensor(ignore_idx)
            #ignore_masks = torch.stack(ignore_masks)

            instance_box = torch.stack(instance_box)
            bb_occupancy = is_within_bb_torch(
                coordinates[batch_id][:, None, :], instance_box[None, :, :3], instance_box[None, :, 3:]
            ) 
            

            cross_box_iou = batch_giou_cross(instance_box, instance_box) 
            cross_box_iou.fill_diagonal_(0.0)
            n_boxes = len(instance_box)
            box_visited = torch.zeros(n_boxes)
            #undet_point = 
            
            for b1 in range(n_boxes):
                b1_ious = cross_box_iou[b1]
                overlap_cond = (b1_ious > 0.0001) & (box_visited == 0)
                overlap_inds = torch.nonzero(overlap_cond).view(-1).int()

                n_overlap_ = len(overlap_inds)

                if n_overlap_ == 0:
                    box_visited[b1] = 1
                    continue

                for b2 in overlap_inds:
                    intersect_cond = (bb_occupancy[:, b1] == 1) & (bb_occupancy[:, b2] == 1)

                    intersect_inds = torch.nonzero(intersect_cond).view(-1)
                    num_intersect_points = len(intersect_inds)

                    if num_intersect_points == 0:
                        continue

                    if is_box1_in_box2(instance_box[b1], instance_box[b2], offset=0.05):
                        bb_occupancy[intersect_inds, b2] = 0
                        box_visited[b1] = 1
                        break

                    if is_box1_in_box2(instance_box[b2], instance_box[b1], offset=0.05):
                        bb_occupancy[intersect_inds, b1] = 0
                        box_visited[b2] = 1
                        continue


                box_visited[b1] = 1
            #exit()
            
            
            #masks = empty_masks.clone().T
            #masks[:, ~ignore_idx] = bb_occupancy[:, ~ignore_idx]
            empty_idx = torch.all(~empty_masks, dim = 0)
            bb_occupancy[empty_idx] = empty_masks.T[empty_idx] # num_point X instance
            ## ignore_idx = instance
            bb_occupancy[:, ignore_idx] = empty_masks[ignore_idx].T
            for i_mask in ignore_masks:
                bb_occupancy[i_mask] = empty_masks.T[i_mask]
            
            #print(bb_occupancy.shape, empty_masks.shape, ignore_idx.shape)
            #exit()
            spp = list_labels[batch_id][:, 2].long()
            #segment_masks = scatter_mul(bb_occupancy.float(), spp, dim = 0).T            
            #segment_masks = scatter_mean(bb_occupancy.float(), spp, dim = 0).T >= 0.7


            
            if unique_map is not None:
                bb_occupancy = bb_occupancy[unique_map[batch_id]]
                #[batch_id][:, 2][unique_map[batch_id]]
                #print(point2seg.shape, vox_label[batch_id][:, 2].shape)
                spp = vox_label[batch_id][:, 2].long()
                segment_masks = scatter_mean(bb_occupancy.float(), spp, dim = 0).T >= 0.8
            else:
                spp = list_labels[batch_id][:, 2].long()
                segment_masks = scatter_mean(bb_occupancy.float(), spp, dim = 0).T >= 0.8
            

            bbox_counts = torch.sum(bb_occupancy, dim=1)
            det_idx = (bbox_counts <= 1)#.nonzero(as_tuple=True)[0]
            undet_idx = ~det_idx
            
            #segment_masks = scatter_mul(bb_occupancy.float(), list_labels[batch_id][:, 2].long(), dim = 0).T
            masks = bb_occupancy.T
            
            if len(label_ids) == 0:
                return list()
            label_ids = torch.stack(label_ids)
            #mins = torch.stack(mins)
            #maxs = torch.stack(maxs)


        else:
            det_idx, undet_idx, undet_mask, weak_idx, spp = None, None, None, None, None
            for instance_id in instance_ids:
                if instance_id == -1:
                    continue

                # TODO is it possible that a ignore class (255) is an instance???
                # instance == -1 ???
                masking = list_labels[batch_id][:, 1] == instance_id
                tmp = list_labels[batch_id][
                    masking
                ]
                label_id = tmp[0, 0]

                if (
                    label_id in filter_out_classes
                ):  # floor, wall, undefined==255 is not included
                    continue
                '''
                if (
                    255 in filter_out_classes
                    and label_id.item() == 255
                    and tmp.shape[0] < ignore_class_threshold
                ):
                    continue
                '''
                #if label_id == 255:
                #    continue                

                label_ids.append(label_id)
                masks.append(masking)
                '''
                if list_segments:
                    segment_mask = torch.zeros(
                        list_segments[batch_id].shape[0]
                    ).bool()
                    segment_mask[
                        list_labels[batch_id][
                            list_labels[batch_id][:, 1] == instance_id
                        ][:, 2].unique()
                    ] = True
                
                    segment_masks.append(segment_mask)     
                '''      
            if len(label_ids) == 0:
                return list()

            label_ids = torch.stack(label_ids)
            masks = torch.stack(masks)
            spp = list_labels[batch_id][:, 2].long()
            segment_masks = scatter_mean(masks.float(), spp, dim = 1) >= 0.8
        if list_segments:
            if train_mode:
                #prob_idx = scatter_mul((probs[batch_id] == 1.0).float(), list_labels[batch_id][:, 2], dim = 0).T
                segment_det_idx = (segment_masks.float().sum(0) <= 1).bool()
                #print(segment_det_idx.float().sum(), prob_idx.sum())
                #exit()
            else:
                #segment_masks = torch.stack(segment_masks)
                segment_det_idx = None
                segment_undet_idx = None
                point2segment_undet_mask = None
                #segment_det_idx = (segment_masks.float() * torch.from_numpy(probs[batch_id])) == 1.0
                #print(segment_masks.shape, segment_det_idx.shape)

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = label_ids == label_id

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(
                        segment_masks[masking, :].sum(dim=0).bool()
                    )

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                #segment_masks = torch.stack(new_segment_masks)

                target.append(
                    {
                        "labels": label_ids,
                        "masks": masks,
                        "segment_mask": segment_masks,
                        "det_idx": det_idx,
                        "undet_idx": undet_idx,
                        "undet_mask": undet_mask,
                        "segment_det_idx": segment_det_idx,
                        #"segment_undet_idx": segment_undet_idx
                    }
                )
            else:
                target.append({"labels": label_ids, "masks": masks, "det_idx": det_idx})
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append(
                    {
                        "labels": l,
                        "masks": masks,
                        "segment_mask": segment_masks,
                        "det_idx": det_idx,
                        "undet_idx": undet_idx,
                        "segment_det_idx": segment_det_idx,
                        "weak_idx": weak_idx,
                        "org_masks": empty_masks,
                        "mins": mins,
                        "maxs": maxs,
                        "spp": spp
                    }
                )
            else:
                target.append({"labels": l, "masks": masks, "det_idx": det_idx, "undet_idx": undet_idx, "org_masks": empty_masks, "mins": mins, "maxs": maxs, "spp": spp})#, "undet_mask": undet_mask})
    return target

def get_weak_idx(
    list_labels,
    filter_out_classes=[],
    coordinates = None
):
    weak_idxs = []

    for batch_id in range(len(list_labels)):
        weak_idx = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            masking = list_labels[batch_id][:, 1] == instance_id
            tmp = list_labels[batch_id][
                masking
            ]
            label_id = tmp[0, 0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue
            distances = torch.full((coordinates[batch_id].shape[0], ),float("inf"))
            points = coordinates[batch_id][masking].float()
            center = points.mean(dim=0)
            distances[masking] = torch.norm(points - center, dim=1)
            #print(distances.shape, distances[masking].shape, torch.norm(points - center, dim=1).shape)
            #exit()
            rand_idx = torch.argmin(distances).item()
            weak_idx.append(rand_idx)

        # It's assumed unique operation is still necessary due to the potential of rand_idx being duplicated.
        weak_idx = torch.tensor(weak_idx, dtype=torch.long).unique()
        weak_idxs.append(weak_idx)
    return weak_idxs

def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate(
            (scene[2], np.full_like((scene[2]), 255)[:4])
        )

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        full_res_coords=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx


class NoGpuMask:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        masks=None,
        labels=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels