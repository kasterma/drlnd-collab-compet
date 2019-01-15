from collabcompet.model import Actor
import torch

filename = "data/trained_model-actor_local_2-agent-A-28.pth"

actor = Actor(24, 2)
dat = torch.load(filename)
actor.load_state_dict(dat)

pars = dict(actor.named_parameters())
pars['fc1.weight']
