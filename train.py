import argparse, torch
from stylegan2 import Generator
from coralstyle import CoralAttnNet

parser = argparse.ArgumentParser(description="Generate samples from the generator")

parser.add_argument(
    "--size", type=int, default=256, help="output image size of the generator"
)
parser.add_argument(
    "--device", type=str, default='cuda', help="output image size of the generator"
)
parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
parser.add_argument(
    "--truncation_mean",
    type=int,
    default=4096,
    help="number of vectors to calculate mean for the truncation",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="stylegan2-ffhq-config-f.pt",
    help="path to the model checkpoint",
)
parser.add_argument(
    "--channel_multiplier",
    type=int,
    default=2,
    help="channel multiplier of the generator. config-f = 2, else = 1",
)

args = parser.parse_args()

args.latent = 512
args.n_mlp = 8
device = torch.device(args.device)

g_ema = Generator(
    args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
).to(device)
checkpoint = torch.load(args.ckpt, map_location=device)

g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

model = CoralAttnNet(g_ema)

with torch.no_grad():
    w_plus = torch.randn((1, 14, 512))
    image, image_star, image_bar, masks, delta_w_plus = model(w_plus)
