import torch
import numpy as np

SMALL_NUM = np.log(1e-45)

class Decoupled_Contrastive_Loss(torch.nn.Module):
    """
    A interface of Decoupled Contrastive Loss
    proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(Decoupled_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.sim = Similarity(temperature)

    def _count_loss(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector (batch_size,embedding dim)
        :param z2: second embedding vector
        :return: loss
        """
        total_dim = z1.size()[0]*z1.size()[0]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_sim = torch.log_softmax(cos_sim, 1)
        diag = torch.diag(cos_sim)
        diag_marix = torch.diag_embed(diag)
        positive_sim_value = diag.sum()
        negative_value = (cos_sim - diag_marix).sum()
        if self.weight_fn is not None:
            positive_sim_value = positive_sim_value * self.weight_fn(z1, z2)
        count_loss = torch.exp(positive_sim_value/total_dim)/torch.exp(negative_value/ total_dim)
        count_loss = torch.log(count_loss)
        return count_loss


    def one_way_loss(self,z1,z2):
        """
        Calculate one way InfoNCE loss
        :param z1:
        :param z2:
        :return:
        """
        total_dim = z1.size()[0]*z1.size()[0]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_sim = torch.log_softmax(cos_sim, 1)
        diag = torch.diag(cos_sim)
        diag_marix = torch.diag_embed(diag)
        positive_sim_value = diag.sum()
        negative_value = (cos_sim-diag_marix).sum()
        count_value = torch.exp(positive_sim_value/total_dim)/torch.exp((positive_sim_value+negative_value)/total_dim)
        count_value = torch.log(count_value)
        return count_value

    def other_way_to_loss(self,z1,z2):
        """
        Calculate other way InfoNCE loss
        @param z1:
        @param z2:
        @return:
        """
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim,labels)
        return loss
    def __call__(self, z1, z2):
        return self._count_loss(z1, z2)

class Similarity(torch.nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Decoupled_Contrastive_Loss_with_weight(Decoupled_Contrastive_Loss):
	"""
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

	def __init__(self, sigma=0.5, temperature=0.1):
		weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma,
		                                                                        dim=0).squeeze()
		super(Decoupled_Contrastive_Loss_with_weight, self).__init__(weight_fn=weight_fn, temperature=temperature)

if __name__ == '__main__':
    DCL = Decoupled_Contrastive_Loss(temperature=0.05)
    DCLW = Decoupled_Contrastive_Loss_with_weight(temperature=0.5, sigma=0.5)
    dim1 = 64
    dim2 = 768
    z1 = torch.rand(size=[dim1,dim2])
    z2 = torch.rand(size=[dim1,dim2])
    z3 = torch.rand(size=[dim1,dim2])

    print(DCL(z1,z2))
    print(DCL.one_way_loss(z1,z2))
    print(DCL.other_way_to_loss(z1,z2))
    print(DCL.other_way_to_loss(z1,z2)/DCL.one_way_loss(z1,z2))