import hydra
import torch.nn as nn
import torch

class Detector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.detector = hydra.utils.instantiate(config)
        self.ema_detector = hydra.utils.instantiate(config)
        for param in self.ema_detector.parameters():
            param.detach_()

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False, target = None#, queries_init = None, query_pos_init = None
    ):
        if not is_eval:
            
            #print('ssssssss')
            with torch.no_grad():
                t_outputs = self.ema_detector(x, point2segment, raw_coordinates, is_eval, target)#, s_outputs['query_pos_init'])
            s_outputs = self.detector(x, point2segment, raw_coordinates, is_eval)
            return s_outputs, t_outputs
        else:
            outputs = self.detector(x, point2segment, raw_coordinates, is_eval)#ema_outputs)
            #print(outputs)
            #exit()
            #outputs = self.detector.evaluate(x, point2segment, raw_coordinates, is_eval, target, ema = False)

        return outputs