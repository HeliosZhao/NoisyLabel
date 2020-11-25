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
        # clean_data = []
        # clean_indices = []
        # for i in range(num_split):
        #     mask_i = torch.zeros(self.num_instances).cuda()
        #     sim_class = sim[i]
        #     for j in range(sim_class.size(0)):
        #         sim_j = sim_class[j]
        #         # print('sim j : ',sim_j)
        #         # print('sim j shape :', sim_j.shape)
        #         _, indices = sim_j.sort()
        #         mask = torch.zeros(self.num_instances).cuda()
        #
        #         mask = mask.scatter(dim=0, index=indices, src=weight)
        #         # print('mask : ', mask)
        #         mask_i = mask_i + mask
        #     # print('mask i : ', mask_i)
        #
        #     _, image_indices = mask_i.sort(descending=True)
        #     image_indices = image_indices[:self.clean]
        #     # print(' image indices : ', image_indices)
        #     clean_indices.append(image_indices)
        #     clean_data.append(split_data[i][image_indices])
        #
        #
        # clean_data = torch.cat(clean_data, dim=0)
        # old_ = clean_data.view(-1,256)
        # print('old shape = ',old_.shape)
        ## pytorch mode
        _, sim_ind = sim.sort(dim=-1)
        # print(sim_ind)
        mask_new = torch.zeros(num_split, self.num_instances, self.num_instances).cuda()
        weight_new = weight.unsqueeze(0).unsqueeze(0).repeat(num_split, self.num_instances, 1)
        mask_new.scatter_(dim=-1, index=sim_ind, src=weight_new)

        mask_new = mask_new.sum(dim=1)

        _, image_indices_new = mask_new.sort(dim=-1, descending=True)
        clean_indices = image_indices_new[:,:self.clean]
        clean_data = []
        for i in range(num_split):
            clean_data.append(split_data[i][clean_indices[i]]) #split_data.index_select(1, clean_indices) #[clean_indices, :]

        clean_data = torch.cat(clean_data, dim=0)

        clean_data = clean_data.view(-1, x.size(1))


        return clean_data, clean_indices
