import torch.nn as nn

class LateralBlock(nn.Module):
    '''
    A implementation of lateral connection module across different columns
    '''
    def __init__(self,col_id,depth,block):
        super(LateralBlock,self).__init__()
        self.col_id = col_id     # expert id
        self.depth = depth # layer index of the current block
        self.block = block # block should contain nn.Conv2d
        
        if self.depth > 0: # from the second layer
            # determine the shape of lateral connection
            if isinstance(self.block,list) or isinstance(self.block,nn.ModuleList) or isinstance(self.block,nn.Sequential):
                for layer in self.block:
                    if isinstance(layer, nn.Conv2d):
                        weight_shape = layer.weight.shape
                        break
            else:
                assert isinstance(self.block,nn.Conv2d)
                weight_shape = self.block.weight.shape
            out_channels = weight_shape[0]
            in_channels = weight_shape[1] if self.col_id != 0 else 0
            
            if in_channels > 0:
                self.lateral_connection = nn.Conv2d(in_channels,out_channels,3,padding=1)
                # When sel.col_id > 0, we add lateral connection for the block
                self.lateral_prelu = nn.PReLU()
            else:
                self.lateral_connection = None
                # When sel.col_id == 0, No lateral connection

    def forward(self, inputs, is_1st_layer=False, prev_lateral_connections=None):
        if is_1st_layer:
            v,a = inputs[0],inputs[1]
            return self.block(v,a)
        
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        cur_column_out = self.block(inputs[-1])
        # if lateral connection exists, calculate it
        if self.lateral_connection is not None:
            prev_columns_out = self.lateral_connection(inputs[-2])
            if prev_lateral_connections is not None:
                for i, lc in prev_lateral_connections.items():
                    assert i < len(inputs)
                    prev_columns_out += lc(inputs[i])
            prev_columns_out = self.lateral_prelu(prev_columns_out)
        else:
            prev_columns_out = 0

        res = cur_column_out + prev_columns_out
        return res