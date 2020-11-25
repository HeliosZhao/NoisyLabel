import torch
import torch.nn as nn

class CleanSelect(nn.Module):
    def __init__(self, args):
        super(CleanSelect, self).__init__()
        self.num_instances = args.num_instances
        self.clean = int((1 - args.eta) * self.num_instances)
    def forward(self, x):
        B = x.size(0)
        num_split = int(B // self.num_instances)
        split_data = x.view(num_split, self.num_instances, -1)
        sim = torch.bmm(split_data, split_data.permute(0,2,1))
        weight = torch.arange(self.num_instances).float().cuda()
        clean_data = []

        for i in range(num_split):
            mask_i = torch.zeros(self.num_instances).cuda()
            for j in range(sim.size(0)):
                sim_j = sim[j]
                _, indices = sim_j.sort()
                mask = torch.zeros(self.num_instances).cuda()
                print(indices)
                mask = mask.scatter(dim=0, index=indices, src=weight)
                mask_i = mask_i + mask

            _, image_indices = mask_i.sort(descending=True)
            image_indices = image_indices[:self.clean]
            clean_data.append(split_data[i][image_indices])


        clean_data = torch.cat(clean_data, dim=0)

        return clean_data
